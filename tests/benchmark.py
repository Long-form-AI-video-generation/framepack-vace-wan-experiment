# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings
import csv
import traceback
warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

from vace.models.wan import WanVace
from vace.models.wan import FramepackVace
from vace.models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

def benchmark_generate_with_framepack(
    output_csv="framepack_benchmark.csv"
):
    print('benchmarking framepack')
    results = []
    frame_nums = [121,250]

 
    for frame_num in frame_nums:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        logging.info("Creating WanT2V pipeline.")
        
        model_name=  "vace-1.3B"
        cfg = WAN_CONFIGS[model_name]
        ckpt_dir= "models/Wan2.1-VACE-1.3B"
        device='0'
        rank=1
        framepack_vace = FramepackVace(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=None,
            dit_fsdp=None,
            use_usp=None,
            t5_cpu=None,
        )
        src_video=None
        src_mask=None
        src_ref_images='start_img.png'
        size='480*832'
    
        shift=5
        sample_solver='dpm++'
        sampling_steps= 5
        seed= -1
        guide_scale=5
        offload_model=True
        src_video, src_mask, src_ref_images = framepack_vace.prepare_source([src_video],
                                                                    [src_mask],
                                                                    [None if src_ref_images is None else src_ref_images.split(',')],
                                                                    81, SIZE_CONFIGS[size], 'cuda:0')

        print(f"\n=== Running frame_num={frame_num} ===")
        success = True
        peak_mem = None
        error_msg = None
        elapsed_time = None
        prompt='a girl talking to the camera'
        try:
            start_time = time.time()

            video = framepack_vace.generate_with_framepack(
                prompt,
                src_video,
                src_mask,
                src_ref_images,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=offload_model,
            )

            elapsed_time = time.time() - start_time

            # record peak memory (in MB)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                success = False
                error_msg = "OOM"
            else:
                success = False
                error_msg = f"RuntimeError: {str(e)}"
            traceback.print_exc()

        results.append({
            "frame_num": frame_num,
            "success": success,
            "peak_memory_MB": peak_mem,
            "time_seconds": elapsed_time,
            "error": error_msg
        })

        torch.cuda.empty_cache()

    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_num", "success", "peak_memory_MB", "time_seconds", "error"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Benchmark complete ===")
    for r in results:
        print(r)

    return results
def benchmark_generate_with_vace(
    output_csv="vace_benchmark.csv"
):
    print('benchmarking vace')
    results = []
    frame_nums = [121,250]

   
    for frame_num in frame_nums:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        logging.info("Creating WanT2V pipeline.")
        
        model_name=  "vace-1.3B"
        cfg = WAN_CONFIGS[model_name]
        ckpt_dir= "models/Wan2.1-VACE-1.3B"
        device='0'
        rank=1
        wan_vace = WanVace(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=None,
            dit_fsdp=None,
            use_usp=None,
            t5_cpu=None,
        )
        src_video=None
        src_mask=None
        src_ref_images='start_img.png'
        size='480*832'
    
        shift=5
        sample_solver='dpm++'
        sampling_steps= 5
        seed= -1
        guide_scale=5
        offload_model=True
        src_video, src_mask, src_ref_images = wan_vace.prepare_source([src_video],
                                                                    [src_mask],
                                                                    [None if src_ref_images is None else src_ref_images.split(',')],
                                                                    frame_num, SIZE_CONFIGS[size], 'cuda:0')

        print(f"\n=== Running frame_num={frame_num} ===")
        success = True
        peak_mem = None
        error_msg = None
        elapsed_time = None
        prompt='a girl talking to the camera'
        try:
            start_time = time.time()

            video = wan_vace.generate(
                prompt,
                src_video,
                src_mask,
                src_ref_images,
                size=size,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=seed,
                offload_model=offload_model,
            )

            elapsed_time = time.time() - start_time

            # record peak memory (in MB)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                success = False
                error_msg = "OOM"
            else:
                success = False
                error_msg = f"RuntimeError: {str(e)}"
            traceback.print_exc()

        results.append({
            "frame_num": frame_num,
            "success": success,
            "peak_memory_MB": peak_mem,
            "time_seconds": elapsed_time,
            "error": error_msg
        })

        torch.cuda.empty_cache()

    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_num", "success", "peak_memory_MB", "time_seconds", "error"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Benchmark complete ===")
    for r in results:
        print(r)

    return results
def main():
    benchmark_generate_with_framepack()
    benchmark_generate_with_vace()
   
    import pandas as pd
    import matplotlib.pyplot as plt

    df_fp = pd.read_csv("framepack_benchmark.csv")
    df_base = pd.read_csv("vace_benchmark.csv")

    plt.figure(figsize=(10,6))
    plt.plot(df_fp["frame_num"], df_fp["peak_memory_MB"], "-o", label="FramePack")
    plt.plot(df_base["frame_num"], df_base["peak_memory_MB"], "-o", label="Baseline")
    plt.xlabel("Frame Number")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("memory_comparison.png")

    plt.figure(figsize=(10,6))
    plt.plot(df_fp["frame_num"], df_fp["time_seconds"], "-o", label="FramePack")
    plt.plot(df_base["frame_num"], df_base["time_seconds"], "-o", label="Baseline")
    plt.xlabel("Frame Number")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_comparison.png")  # saves the plot
    plt.close()
   

    
if __name__ == "__main__":
    
    main()
