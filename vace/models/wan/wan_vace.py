# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import gc
import math
import time
import random
import types
import logging
import traceback
from contextlib import contextmanager
from functools import partial

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
from .modules.model import VaceWanModel
from ..utils.preprocessor import VaceVideoProcessor


class WanVace(WanT2V):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward,
                                                            usp_dit_forward_vace)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    # def decode_latent(self, zs, ref_images=None, vae=None):
    #     vae = self.vae if vae is None else vae
    #     if ref_images is None:
    #         ref_images = [None] * len(zs)
    #     else:
    #         assert len(zs) == len(ref_images)

    #     trimed_zs = []
    #     for z, refs in zip(zs, ref_images):
    #         if refs is not None:
    #             z = z[:, len(refs):, :, :]
    #         trimed_zs.append(z)

    #     return vae.decode(trimed_zs)
    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae

    # No need to check ref_images length or trim anymore
        return vae.decode(zs)


    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        #
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size
        

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # vace context encode
        z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
        m0 = self.vace_encode_masks(input_masks, input_ref_images)
        z = self.vace_latent(z0, m0)

        target_shape = list(z0[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,**arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    
    def generate_with_framepack(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=240,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=1,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        # F = frame_num
        # target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
        #                 size[1] // self.vae_stride[1],
        #                 size[0] // self.vae_stride[2])
        #
        # seq_len = math.ceil((target_shape[2] * target_shape[3]) /
        #                     (self.patch_size[1] * self.patch_size[2]) *
        #                     target_shape[1] / self.sp_size) * self.sp_size
        section_window=41
        frame_num=121
        section_num= math.ceil(frame_num/section_window)
        history_latent=[]
        generated_latent=[]
        frame_list=None
        mask_list=None
        print('total frames', frame_num)
        print('total sections', section_num)
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        # seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        # seed_g = torch.Generator(device=self.device)
        # seed_g.manual_seed(seed)
        if seed == -1:
        # Use current time for true randomness across runs
            import time
            base_seed = int(time.time() * 1000) % (1 << 32)
        else:
            base_seed = seed
        
        for section_id in range(section_num):
            print(f"\n🔷🔷🔷 [SECTION START] — Processing Section {section_id+1} / {section_num}🔷🔷🔷\n ")
            section_seed = base_seed + section_id * 12345  # Large prime for good distribution
        
            # Create fresh generator for each section
            section_generator = torch.Generator(device=self.device)
            section_generator.manual_seed(section_seed)
            if frame_list is None or mask_list is None:
                # vace context encode
                frame_offset =0
                context_scale=1.0
                z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks)
                print('zo shape', z0[0].shape)
                m0 = self.vace_encode_masks(input_masks, input_ref_images)
                z = self.vace_latent(z0, m0)
                target_shape = list(z0[0].shape)
                target_shape[0] = int(target_shape[0] / 2)
                noise = [
                torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=section_generator)
            ]
                
            else:
                
                print('here with context frames')
                context_variation = torch.rand(1).item() * 0.6 + 0.5
                context_scale = context_scale * context_variation
                frame_offset=22 + (section_id - 1) * 14 if section_id > 0 else 0
                guide_scale = 5.0 + (torch.rand(1).item() - 0.5) * 0.8
                ref_image=None
                # Now encode
                z0 = self.vace_encode_frames(frame_list, ref_image, masks=mask_list)
                m0 = self.vace_encode_masks(mask_list, ref_image)
                z = self.vace_latent(z0, m0)
                noise_seed = seed + section_id * 1000  # Different seed per section
                # seed_g = torch.Generator(device=self.device)
                # seed_g.manual_seed(noise_seed)
                target_shape = list(z0[0].shape)
                target_shape[0] = int(target_shape[0] / 2)
                noise = [
                    torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=self.device,
                        generator=section_generator)
                ]
                
                            
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            
            
            seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                (self.patch_size[1] * self.patch_size[2]) *
                                target_shape[1] / self.sp_size) * self.sp_size

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)
            sample_solver='dpm++'
            sampling_steps=20
            # evaluation mode
            with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")

                # sample videos
                latents = noise
                for i, tensor in enumerate(noise):
                    print(f"\n🔷🔷🔷 noise shape🔷🔷🔷\n",  {tensor.shape} )
                    print(f"\n🔷🔷🔷 context shape🔷🔷🔷\n",  {z[0].shape} )
                arg_c = {'context': context, 'seq_len': seq_len,  'frame_offset': frame_offset, 'sectionId': section_id}
                arg_null = {'context': context_null, 'seq_len': seq_len,  'frame_offset': frame_offset, 'sectionId': section_id}

                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents
                    timestep = [t]

                    timestep = torch.stack(timestep)

                    self.model.to(self.device)
                   
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,**arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=section_generator)[0]
                    latents = [temp_x0.squeeze(0)]
                
                    # break
                    print(f"\n🔷🔷🔷 prediicted noise shape {latents[0].shape} 🔷🔷🔷\n")
                    
                    
                
                if section_id == 0:
                    history_latent.append(latents[0])
                    generated_latent.append(latents[0])
                else:
                    history_latent.append(latents[0][:, -14:])
                    generated_latent.append(latents[0][:, -14:])
                        
                contexts= torch.cat(generated_latent, dim=1)
                result = self.pick_context(contexts)
               
                context_videos = self.decode_latent([result], input_ref_images)
                if section_id > 0:  # Only for subsequent sections
                    # Apply frequency separation
                    appearance, motion = self.separate_appearance_and_motion(context_videos[0])
                    motion_corrupted = motion + torch.randn_like(motion) * 0.5
                    modified_video = appearance + motion_corrupted * 0.3
                    
                    # Re-encode the modified video
                    frame_list = [modified_video]
                else:
                    frame_list = context_videos
                import torchvision
                import torchvision.transforms.functional as TF
                import numpy as np
                import imageio
                
                output_path = f"output-section-{section_id}.mp4"
                
                video_tensor = context_videos[0].cpu().detach()
        
                # Normalize from [-1, 1] to [0, 1]
                video_tensor = (video_tensor + 1.0) / 2.0
                video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
                
                # Convert to NumPy: [T, H, W, C]
                video_np = video_tensor.permute(1, 2, 3, 0).numpy()
                video_np_uint8 = (video_np * 255).astype(np.uint8)
                
                # Save MP4
                imageio.mimsave(output_path, video_np_uint8, fps=24)
                # frame_list = []
                # for t in range(context_videos[0].shape[1]):  # 81 frames
                #     frame = context_videos[0][:, t]  # Extract single frame [3, 832, 480]
                #     frame_list.append(frame)
                frame_list=[context_videos[0]]
                context_list=context_videos
                print(f"\n🔷🔷🔷 prediicted frame shape {frame_list[0].shape} 🔷🔷🔷\n")
                mask_list=self.create_temporal_blend_mask_for_context(context_list[0].shape)
                print(f"\n🔷🔷🔷 mask shape {mask_list[0].shape} 🔷🔷🔷\n")
                
                    
        full_history = torch.cat(history_latent, dim=1)
        for i, latent in enumerate(history_latent):
                print(f"history_latent[{i}] shape: {latent.shape}")
                
        print(f"full history shape: {full_history.shape}")

        x0 = full_history
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            input_ref_images=None
            print('here')
            videos = self.decode_latent([x0], input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    
    def separate_appearance_and_motion(self, frames):
        """Use frequency domain to separate appearance from motion"""
        # Store original shape
        C, T, H, W = frames.shape
        
        # FFT to frequency domain
        fft_frames = torch.fft.rfft2(frames, dim=(-2, -1))
        
        # Get FFT dimensions
        fft_h = H
        fft_w = W // 2 + 1  # rfft2 reduces the last dimension
        
        # Create frequency grids
        # For height: use full frequency range
        h_freqs = torch.fft.fftfreq(H, device=frames.device)
        # For width: use rfft frequency range
        w_freqs = torch.fft.rfftfreq(W, device=frames.device)
        
        # Create 2D grid
        h_grid, w_grid = torch.meshgrid(h_freqs, w_freqs, indexing='ij')
        
        # Calculate frequency magnitude
        freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)
        
        # Create low-pass mask
        cutoff = 0.1  # Adjust this value
        low_pass_mask = (freq_magnitude < cutoff).float().to(frames.device)
        
        # Ensure mask shape matches fft_frames
        # fft_frames shape: [C, T, H, W//2+1, 2] (complex) or [C, T, H, W//2+1]
        if low_pass_mask.shape != fft_frames.shape[-2:]:
            print(f"Mask shape: {low_pass_mask.shape}, FFT shape: {fft_frames.shape}")
            # Adjust mask if needed
            low_pass_mask = low_pass_mask[:fft_h, :fft_w]
        
        # Expand mask dimensions to match fft_frames
        while low_pass_mask.dim() < fft_frames.dim():
            low_pass_mask = low_pass_mask.unsqueeze(0)
        
        # Apply masks
        appearance_fft = fft_frames * low_pass_mask
        motion_fft = fft_frames * (1 - low_pass_mask)
        
        # Back to spatial domain
        appearance = torch.fft.irfft2(appearance_fft, s=(H, W))
        motion = torch.fft.irfft2(motion_fft, s=(H, W))
        
        return appearance, motion
    def pick_context(self,context_frames):
        
        def pick_exactly_7_frames(latent, step=4, num_frames=7):
            T = latent.shape[1]
            indices = []
            for i in range(num_frames):
                idx = T - 1 - step * i
                if idx < 0:
                    idx = 0  # clamp to first frame if out of range
                indices.append(idx)
           
            
            selected_frames = latent[:, indices]
            return selected_frames
        
        # zero_frame = torch.zeros_like(context_frames[:, :1, :, :])
        gen_frames= context_frames[:, -10:]
        context_frames=context_frames[:,:-10]
        overlap=context_frames[:, -4:]
        context_frames=context_frames[:,:-4]
        recent= context_frames[:, -1:]
        context_frames=context_frames[:,:-1]
        mid_ind = torch.tensor([context_frames.shape[1] - 1, context_frames.shape[1] - 3])
        mid = context_frames[:, mid_ind]
        context_frames=context_frames[:,:-3]
        long= pick_exactly_7_frames(context_frames, step=4, num_frames=5)
        final_latents = torch.cat([long, mid, recent, overlap ,gen_frames], dim=1)
        
        
        
        return final_latents
    
    def create_temporal_blend_mask_for_context(self, frame_shape, device='cuda'):
        """
        Creates temporal blend masks matching the pick_context structure.
        
        Context structure from pick_context:
        - long: 9 frames (sampled at 4x distance) - mostly keep
        - mid: 2 frames (mid-range sampling) - start blending
        - recent: 1 frame (most recent) - light blend
        - gen_frames: 6 frames (recently generated) - gradual blend
        - overlap: 4 frames (overlap region) - full blend to generate
        
        Total: 22 frames with progressive blending
        
        Args:
            context_structure: Dict with frame counts for each segment
            frame_shape: Shape of a single frame (C, H, W)
            device: Device to create tensors on
        
        Returns:
            List of mask tensors, one per frame
        """
        
        C, T, H, W = frame_shape # e.g., [3, 81, 832, 480]

        # Create a single mask tensor for all frames
        mask_tensor = torch.zeros(1, T, H, W, device=self.device)  # [1, 81, 832, 480]

        # Apply your masking pattern to the tensor
        context_structure = {
            'long': 5,
            'mid': 2,
            'recent': 1,
            'gen_frames': 10,
            'overlap': 4
        }

        # Map structure to frame indices (assuming linear mapping)
        frames_per_latent = T / 22  # If 22 latent frames map to T pixel frames

        current_frame = 0
        # Context frames - keep (mask = 0)
        context_frame_count = int((context_structure['long'] + context_structure['mid'] + 
                                context_structure['recent']) * frames_per_latent)
        mask_tensor[:, :context_frame_count] = 0.0

        # Generation frames - replace (mask = 1)
        gen_start = context_frame_count
        gen_end = gen_start + int(context_structure['gen_frames'] * frames_per_latent)
        mask_tensor[:, gen_start:gen_end] = 1.0

        # Overlap frames - blend
        for i in range(gen_end, T):
            progress = (i - gen_end) / (T - gen_end - 1) if T > gen_end + 1 else 1.0
            mask_tensor[:, i] = 0.4 + progress * 0.6

        # Wrap mask tensor in a list
        masks_for_vae = [mask_tensor]  # List containing [1, T, H, W]
        
       
        return masks_for_vae


    
class WanVaceMP(WanVace):
    def __init__(
            self,
            config,
            checkpoint_dir,
            use_usp=False,
            ulysses_size=None,
            ring_size=None
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.use_usp = use_usp
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        self.in_q_list = None
        self.out_q = None
        self.inference_pids = None
        self.ulysses_size = ulysses_size
        self.ring_size = ring_size
        self.dynamic_load()

        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.vid_proc = VaceVideoProcessor(
            downsample=tuple([x * y for x, y in zip(config.vae_stride, config.patch_size)]),
            min_area=720 * 1280,
            max_area=720 * 1280,
            min_fps=config.sample_fps,
            max_fps=config.sample_fps,
            zero_start=True,
            seq_len=75600,
            keep_last=True)


    def dynamic_load(self):
        if hasattr(self, 'inference_pids') and self.inference_pids is not None:
            return
        gpu_infer = os.environ.get('LOCAL_WORLD_SIZE') or torch.cuda.device_count()
        pmi_rank = int(os.environ['RANK'])
        pmi_world_size = int(os.environ['WORLD_SIZE'])
        in_q_list = [torch.multiprocessing.Manager().Queue() for _ in range(gpu_infer)]
        out_q = torch.multiprocessing.Manager().Queue()
        initialized_events = [torch.multiprocessing.Manager().Event() for _ in range(gpu_infer)]
        context = mp.spawn(self.mp_worker, nprocs=gpu_infer, args=(gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, self), join=False)
        all_initialized = False
        while not all_initialized:
            all_initialized = all(event.is_set() for event in initialized_events)
            if not all_initialized:
                time.sleep(0.1)
        print('Inference model is initialized', flush=True)
        self.in_q_list = in_q_list
        self.out_q = out_q
        self.inference_pids = context.pids()
        self.initialized_events = initialized_events

    def transfer_data_to_cuda(self, data, device):
        if data is None:
            return None
        else:
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            elif isinstance(data, list):
                data = [self.transfer_data_to_cuda(subdata, device) for subdata in data]
            elif isinstance(data, dict):
                data = {key: self.transfer_data_to_cuda(val, device) for key, val in data.items()}
        return data

    def mp_worker(self, gpu, gpu_infer, pmi_rank, pmi_world_size, in_q_list, out_q, initialized_events, work_env):
        try:
            world_size = pmi_world_size * gpu_infer
            rank = pmi_rank * gpu_infer + gpu
            print("world_size", world_size, "rank", rank, flush=True)

            torch.cuda.set_device(gpu)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=rank,
                world_size=world_size
            )

            from xfuser.core.distributed import (initialize_model_parallel,
                                                 init_distributed_environment)
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())

            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=self.ring_size or 1,
                ulysses_degree=self.ulysses_size or 1
            )

            num_train_timesteps = self.config.num_train_timesteps
            param_dtype = self.config.param_dtype
            shard_fn = partial(shard_model, device_id=gpu)
            text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
                tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
                shard_fn=shard_fn if True else None)
            text_encoder.model.to(gpu)
            vae_stride = self.config.vae_stride
            patch_size = self.config.patch_size
            vae = WanVAE(
                vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                device=gpu)
            logging.info(f"Creating VaceWanModel from {self.checkpoint_dir}")
            model = VaceWanModel.from_pretrained(self.checkpoint_dir)
            model.eval().requires_grad_(False)

            if self.use_usp:
                from xfuser.core.distributed import get_sequence_parallel_world_size
                from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                                usp_dit_forward,
                                                                usp_dit_forward_vace)
                for block in model.blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                for block in model.vace_blocks:
                    block.self_attn.forward = types.MethodType(
                        usp_attn_forward, block.self_attn)
                model.forward = types.MethodType(usp_dit_forward, model)
                model.forward_vace = types.MethodType(usp_dit_forward_vace, model)
                sp_size = get_sequence_parallel_world_size()
            else:
                sp_size = 1

            dist.barrier()
            model = shard_fn(model)
            sample_neg_prompt = self.config.sample_neg_prompt

            torch.cuda.empty_cache()
            event = initialized_events[gpu]
            in_q = in_q_list[gpu]
            event.set()

            while True:
                item = in_q.get()
                input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale, \
                shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model = item
                input_frames = self.transfer_data_to_cuda(input_frames, gpu)
                input_masks = self.transfer_data_to_cuda(input_masks, gpu)
                input_ref_images = self.transfer_data_to_cuda(input_ref_images, gpu)

                if n_prompt == "":
                    n_prompt = sample_neg_prompt
                seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
                seed_g = torch.Generator(device=gpu)
                seed_g.manual_seed(seed)

                context = text_encoder([input_prompt], gpu)
                context_null = text_encoder([n_prompt], gpu)

                # vace context encode
                z0 = self.vace_encode_frames(input_frames, input_ref_images, masks=input_masks, vae=vae)
                m0 = self.vace_encode_masks(input_masks, input_ref_images, vae_stride=vae_stride)
                z = self.vace_latent(z0, m0)

                target_shape = list(z0[0].shape)
                target_shape[0] = int(target_shape[0] / 2)
                noise = [
                    torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=gpu,
                        generator=seed_g)
                ]
                seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                                    (patch_size[1] * patch_size[2]) *
                                    target_shape[1] / sp_size) * sp_size

                @contextmanager
                def noop_no_sync():
                    yield

                no_sync = getattr(model, 'no_sync', noop_no_sync)

                # evaluation mode
                with amp.autocast(dtype=param_dtype), torch.no_grad(), no_sync():

                    if sample_solver == 'unipc':
                        sample_scheduler = FlowUniPCMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sample_scheduler.set_timesteps(
                            sampling_steps, device=gpu, shift=shift)
                        timesteps = sample_scheduler.timesteps
                    elif sample_solver == 'dpm++':
                        sample_scheduler = FlowDPMSolverMultistepScheduler(
                            num_train_timesteps=num_train_timesteps,
                            shift=1,
                            use_dynamic_shifting=False)
                        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                        timesteps, _ = retrieve_timesteps(
                            sample_scheduler,
                            device=gpu,
                            sigmas=sampling_sigmas)
                    else:
                        raise NotImplementedError("Unsupported solver.")

                    # sample videos
                    latents = noise

                    arg_c = {'context': context, 'seq_len': seq_len}
                    arg_null = {'context': context_null, 'seq_len': seq_len}

                    for _, t in enumerate(tqdm(timesteps)):
                        latent_model_input = latents
                        timestep = [t]

                        timestep = torch.stack(timestep)

                        model.to(gpu)
                        noise_pred_cond = model(
                            latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale, **arg_c)[
                            0]
                        noise_pred_uncond = model(
                            latent_model_input, t=timestep, vace_context=z, vace_context_scale=context_scale,
                            **arg_null)[0]

                        noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond)

                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]

                    torch.cuda.empty_cache()
                    x0 = latents
                    if rank == 0:
                        videos = self.decode_latent(x0, input_ref_images, vae=vae)

                del noise, latents
                del sample_scheduler
                if offload_model:
                    gc.collect()
                    torch.cuda.synchronize()
                if dist.is_initialized():
                    dist.barrier()

                if rank == 0:
                    out_q.put(videos[0].cpu())

        except Exception as e:
            trace_info = traceback.format_exc()
            print(trace_info, flush=True)
            print(e, flush=True)



    def generate(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 size=(1280, 720),
                 frame_num=81,
                 context_scale=1.0,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):

        input_data = (input_prompt, input_frames, input_masks, input_ref_images, size, frame_num, context_scale,
                      shift, sample_solver, sampling_steps, guide_scale, n_prompt, seed, offload_model)
        for in_q in self.in_q_list:
            in_q.put(input_data)
        value_output = self.out_q.get()

        return value_output


