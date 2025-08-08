import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class FramePackCompressor:
    """
    FramePack compression for video generation models to handle long sequences
    without context length explosion.
    """

    def __init__(self, 
                 lambda_compression: float = 2.0,
                 base_kernel_sizes: List[Tuple[int, int, int]] = [(2, 4, 4), (4, 8, 8), (8, 16, 16)],
                 max_history_frames: int = 100):
        """
        Initialize FramePack compressor.
        
        Args:
            lambda_compression: Compression parameter (λ > 1)
            base_kernel_sizes: List of (frame, height, width) kernel sizes for compression
            max_history_frames: Maximum number of frames to keep in history
        """
        self.lambda_compression = lambda_compression
        self.base_kernel_sizes = sorted(base_kernel_sizes, key=lambda x: x[0] * x[1] * x[2])
        self.max_history_frames = max_history_frames

       
        self.compression_schedule = self._create_compression_schedule()

    def _create_compression_schedule(self) -> List[Tuple[int, Tuple[int, int, int]]]:
        """
        Create compression schedule mapping frame indices to kernel sizes.
        Returns list of (compression_rate, kernel_size) tuples.
        """
        schedule = []

        schedule.append((1, (1, 1, 1)))

        for i in range(1, self.max_history_frames):
            compression_rate = int(self.lambda_compression ** i)
            kernel_size = self._get_kernel_for_compression_rate(compression_rate)
            schedule.append((compression_rate, kernel_size))

        return schedule

    def _get_kernel_for_compression_rate(self, rate: int) -> Tuple[int, int, int]:
        """
        Get appropriate kernel size for given compression rate.
        Prioritizes temporal compression over spatial for video data.
        """
        if rate == 1:
            return (1, 1, 1)

        for kernel in self.base_kernel_sizes:
            kernel_rate = kernel[0] * kernel[1] * kernel[2]
            if kernel_rate >= rate:
                return kernel

        largest_kernel = self.base_kernel_sizes[-1]
        remaining_compression = rate // (largest_kernel[0] * largest_kernel[1] * largest_kernel[2])

        if remaining_compression <= 8:
         
            return (min(remaining_compression * largest_kernel[0], 16), 
                   largest_kernel[1], largest_kernel[2])
        else:
      
            return (16, 32, 32)

    def compress_frame(self, frame: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Compress a single frame using the specified kernel size.
        Safely handles cases where frame is too small for the kernel.
        
        Args:
            frame: Input frame tensor of shape [C, T, H, W]
            kernel_size: (frame, height, width) compression kernel
            
        Returns:
            Compressed frame tensor
        """
        if kernel_size == (1, 1, 1):
            return frame

        if frame.dim() == 4 and frame.shape[-1] < frame.shape[0]:
           
            frame = frame.permute(3, 0, 1, 2)

        C, T, H, W = frame.shape
        
        adjusted_kernel = list(kernel_size)
        
        if T < kernel_size[0]:
            adjusted_kernel[0] = min(T, kernel_size[0])
            if adjusted_kernel[0] < 1:
                adjusted_kernel[0] = 1
        
     
        if H < kernel_size[1]:
            adjusted_kernel[1] = min(H, kernel_size[1])
            if adjusted_kernel[1] < 1:
                adjusted_kernel[1] = 1
        
     
        if W < kernel_size[2]:
            adjusted_kernel[2] = min(W, kernel_size[2])
            if adjusted_kernel[2] < 1:
                adjusted_kernel[2] = 1
        
        adjusted_kernel = tuple(adjusted_kernel)
      
        if adjusted_kernel == (1, 1, 1):
            print(f"Warning: Frame too small for compression. Shape: {frame.shape}, "
                f"Original kernel: {kernel_size}")
            return frame
    
        if adjusted_kernel != kernel_size:
            print(f"Adjusted kernel from {kernel_size} to {adjusted_kernel} "
                f"for frame shape [C={C}, T={T}, H={H}, W={W}]")
        
        try:
    
            compressed = F.avg_pool3d(
                frame.unsqueeze(0), 
                kernel_size=adjusted_kernel,
                stride=adjusted_kernel
            ).squeeze(0) 
            
         
            if compressed.shape[2] < 1 or compressed.shape[3] < 1:
                print(f"Warning: Compression resulted in degenerate dimensions: {compressed.shape}")
                
                return frame
                
            return compressed
            
        except RuntimeError as e:
            print(f"Compression failed with error: {e}")
            print(f"Frame shape: {frame.shape}, Kernel: {adjusted_kernel}")
            
            
            if "size" in str(e).lower():
            
                compressed = torch.mean(frame, dim=(1, 2, 3), keepdim=True)
             
                compressed = compressed.expand(C, 1, 1, 1)
                print(f"Applied global pooling fallback. Output shape: {compressed.shape}")
                return compressed
            else:
                raise e

    def add_new_section(self, 
                   current_history: List[torch.Tensor], 
                   new_section: torch.Tensor) -> List[torch.Tensor]:
        """
        Add a new section to the compression history without re-compressing old sections.
        This prevents channel dimension corruption from multiple compressions.
        
        Args:
            current_history: List of previously compressed sections (already compressed, don't recompress!)
            new_section: New section to add (uncompressed)
            
        Returns:
            Updated compressed history
        """
        if len(current_history) == 0:
          
            return [new_section]
        
        new_section_compressed = new_section 
  
        new_history = [new_section_compressed]
     
        for i, old_section in enumerate(current_history):
       
            new_history.append(old_section)
            
            if len(new_history) >= self.max_history_frames // 20:
                break
        
        
        final_history = []
        for age_index, section in enumerate(new_history):
            if age_index == 0:
                
                final_history.append(section)
            elif age_index == 1 and section.shape[2] > 50:  
                
                kernel = (1, 2, 2)
                compressed = self.safe_compress_once(section, kernel)
                final_history.append(compressed)
            elif age_index == 2 and section.shape[2] > 25:  
               
                kernel = (1, 4, 4)
                compressed = self.safe_compress_once(section, kernel)
                final_history.append(compressed)
            elif age_index >= 3 and section.shape[2] > 10: 
                
                kernel = (1, 8, 8)
                compressed = self.safe_compress_once(section, kernel)
                final_history.append(compressed)
            else:
               
                final_history.append(section)
        
        return final_history

    def safe_compress_once(self, frame: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Safely compress a frame ONCE, preserving channel dimensions.
        
        Args:
            frame: Input frame tensor of shape [C, T, H, W]
            kernel_size: (temporal, height, width) compression kernel
            
        Returns:
            Compressed frame tensor with preserved channel count
        """
        if kernel_size == (1, 1, 1):
            return frame
        
        C, T, H, W = frame.shape
        original_channels = C 
      
        adjusted_kernel = list(kernel_size)
       
        if T < kernel_size[0]:
            adjusted_kernel[0] = min(T, 1)
        if H < kernel_size[1]:
            adjusted_kernel[1] = min(H, 1)
        if W < kernel_size[2]:
            adjusted_kernel[2] = min(W, 1)
        
        adjusted_kernel = tuple(adjusted_kernel)
        
        if adjusted_kernel == (1, 1, 1):
            return frame
      
        if H <= 4 or W <= 4:
            print(f"Skipping compression for small frame: {frame.shape}")
            return frame
        
        try:
        
            frame_reshaped = frame.unsqueeze(0) 
            
            compressed = F.avg_pool3d(
                frame_reshaped,
                kernel_size=adjusted_kernel,
                stride=adjusted_kernel
            )
            
            compressed = compressed.squeeze(0)  
         
            if compressed.shape[0] != original_channels:
                print(f"WARNING: Channel dimension changed from {original_channels} to {compressed.shape[0]}")
                print(f"Frame shape before: {frame.shape}, after: {compressed.shape}")
                print(f"Kernel used: {adjusted_kernel}")
                
               
                if compressed.shape[0] < original_channels:
                   
                    padding = torch.zeros(
                        original_channels - compressed.shape[0],
                        compressed.shape[1],
                        compressed.shape[2],
                        compressed.shape[3],
                        device=compressed.device,
                        dtype=compressed.dtype
                    )
                    compressed = torch.cat([compressed, padding], dim=0)
                else:
                   
                    compressed = compressed[:original_channels]
            
            return compressed
            
        except Exception as e:
            print(f"Compression failed: {e}")
            print(f"Returning original frame")
            return frame
    def compress_history(self, history_frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compress history frames according to FramePack schedule.
        
        Args:
            history_frames: List of frame tensors, ordered from newest to oldest
            
        Returns:
            List of compressed frame tensors
        """
        if not history_frames:
            return []

        compressed_history = []

        for i, frame in enumerate(history_frames):
            if i >= len(self.compression_schedule):
              
                kernel_size = self.base_kernel_sizes[-1]
            else:
                _, kernel_size = self.compression_schedule[i]

            try:
                compressed_frame = self.compress_frame(frame, kernel_size)
                compressed_history.append(compressed_frame)
            except RuntimeError as e:
               
                if "size" in str(e).lower():
                   
                    compressed_frame = torch.mean(frame, dim=(1, 2, 3), keepdim=True)
                    compressed_history.append(compressed_frame)
                else:
                    raise e

        return compressed_history

    def get_compression_stats(self, history_frames: List[torch.Tensor]) -> dict:
        """
        Get compression statistics for debugging.
        
        Returns:
            Dictionary with compression statistics
        """
        if not history_frames:
            return {"total_frames": 0, "total_elements": 0}

        original_elements = sum(frame.numel() for frame in history_frames)
        compressed_history = self.compress_history(history_frames)
        compressed_elements = sum(frame.numel() for frame in compressed_history)

        return {
            "total_frames": len(history_frames),
            "original_elements": original_elements,
            "compressed_elements": compressed_elements,
            "compression_ratio": original_elements / compressed_elements if compressed_elements > 0 else 0,
            "memory_saved_mb": (original_elements - compressed_elements) * 4 / (1024 * 1024)  # Assuming float32
        }

    def select_context_frames(self, 
                    compressed_history: List[torch.Tensor], 
                    num_context_frames: int = 11,
                    add_generation_frames: bool = True) -> torch.Tensor:
        """
        Fixed version that handles dimension mismatches properly
        """
        if not compressed_history:
            return torch.empty(0)

        
        LONG_FRAMES = 5
        MID_FRAMES = 3
        RECENT_FRAMES = 1
        OVERLAP_FRAMES = 2
        GEN_FRAMES = 30 
        TOTAL_FRAMES = 41 
        CONTEXT_FRAMES = 11

        all_frames = []
        frame_to_section_map = []

        for section_idx, section in enumerate(compressed_history):
            num_frames_in_section = section.shape[1]
            for frame_idx in range(num_frames_in_section):
                frame = section[:, frame_idx:frame_idx+1, :, :]
                all_frames.append(frame)
                frame_to_section_map.append(section_idx)

        total_available_frames = len(all_frames)

        if total_available_frames == 0:
            return torch.empty(0)

       
        selected_frame_indices = []

        
        if total_available_frames >= 40:
            step = max(1, (total_available_frames - 20) // LONG_FRAMES)
            for i in range(LONG_FRAMES):
                idx = i * step
                selected_frame_indices.append(idx)
        else:
            if total_available_frames >= LONG_FRAMES:
                step = total_available_frames // LONG_FRAMES
                for i in range(LONG_FRAMES):
                    selected_frame_indices.append(i * step)
            else:
                selected_frame_indices.extend(range(total_available_frames))
                while len(selected_frame_indices) < LONG_FRAMES:
                    selected_frame_indices.append(total_available_frames - 1)

        mid_start = max(LONG_FRAMES, total_available_frames - 15)
        for i in range(MID_FRAMES):
            idx = min(mid_start + i * 2, total_available_frames - 1)
            selected_frame_indices.append(idx)

        recent_idx = max(0, total_available_frames - 5)
        selected_frame_indices.append(recent_idx)

        for i in range(OVERLAP_FRAMES):
            idx = max(0, total_available_frames - OVERLAP_FRAMES + i)
            selected_frame_indices.append(idx)

        seen = set()
        unique_indices = []
        for idx in selected_frame_indices:
            if idx not in seen and idx < total_available_frames:
                unique_indices.append(idx)
                seen.add(idx)

        if len(unique_indices) > CONTEXT_FRAMES:
            unique_indices = unique_indices[:CONTEXT_FRAMES]
        elif len(unique_indices) < CONTEXT_FRAMES:
            while len(unique_indices) < CONTEXT_FRAMES:
                unique_indices.append(total_available_frames - 1)

        selected_frames = [all_frames[i] for i in unique_indices]

        print(f"Selected {len(selected_frames)} frames with dimensions:")
        for i, frame in enumerate(selected_frames):
            print(f"  Frame {i}: {frame.shape}")

        max_c = max(f.shape[0] for f in selected_frames)
        max_h = max(f.shape[2] for f in selected_frames)
        max_w = max(f.shape[3] for f in selected_frames)
        
        print(f"Target dimensions: C={max_c}, H={max_h}, W={max_w}")

        normalized_frames = []
        for i, frame in enumerate(selected_frames):
            C, T, H, W = frame.shape
 
            if C != max_c:
                if C < max_c:

                    padding = frame[:, :, :, :].repeat(1, 1, 1, 1)
                    while padding.shape[0] < max_c:
                        frame = torch.cat([frame, padding[:1]], dim=0)
                else:
 
                    frame = frame[:max_c]
            

            if H != max_h or W != max_w:
 
                reshaped = frame.view(C * T, H, W).unsqueeze(0) 
                
                resized = F.interpolate(
                    reshaped, 
                    size=(max_h, max_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
 
                frame = resized.squeeze(0).view(C, T, max_h, max_w)
                
            normalized_frames.append(frame)
            print(f"  Normalized frame {i}: {frame.shape}")

  
        try:
            context_tensor = torch.cat(normalized_frames, dim=1)  
            print(f"Context tensor shape: {context_tensor.shape}")
        except RuntimeError as e:
            print(f"Error concatenating frames: {e}")
  
            device = selected_frames[0].device
            context_tensor = torch.zeros((max_c, CONTEXT_FRAMES, max_h, max_w), device=device)
            print(f"Using fallback zero tensor: {context_tensor.shape}")

        if add_generation_frames:
            C, T, H, W = context_tensor.shape
            
 
            if T != CONTEXT_FRAMES:
                if T < CONTEXT_FRAMES:
                  
                    padding_needed = CONTEXT_FRAMES - T
                    last_frame = context_tensor[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                    context_tensor = torch.cat([context_tensor, last_frame], dim=1)
                else:
                   
                    context_tensor = context_tensor[:, :CONTEXT_FRAMES, :, :]

 
            gen_placeholder = torch.zeros((C, GEN_FRAMES, H, W), device=context_tensor.device)
            final_tensor = torch.cat([context_tensor, gen_placeholder], dim=1)

            print(f"Final tensor shape: {final_tensor.shape}")
            assert final_tensor.shape[1] == TOTAL_FRAMES, f"Expected {TOTAL_FRAMES} frames, got {final_tensor.shape[1]}"

            return final_tensor

        return context_tensor