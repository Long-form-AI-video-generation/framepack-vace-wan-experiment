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
        
        # Create compression schedule
        self.compression_schedule = self._create_compression_schedule()
    
    def _create_compression_schedule(self) -> List[Tuple[int, Tuple[int, int, int]]]:
        """
        Create compression schedule mapping frame indices to kernel sizes.
        Returns list of (compression_rate, kernel_size) tuples.
        """
        schedule = []
        
        # Frame 0 (most recent) has no compression
        schedule.append((1, (1, 1, 1)))
        
        # Create geometric progression for older frames
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
        
        # Find the best matching kernel from predefined sizes
        for kernel in self.base_kernel_sizes:
            kernel_rate = kernel[0] * kernel[1] * kernel[2]
            if kernel_rate >= rate:
                return kernel
        
        # For very high compression rates, use the largest kernel
        # and apply additional downsampling
        largest_kernel = self.base_kernel_sizes[-1]
        remaining_compression = rate // (largest_kernel[0] * largest_kernel[1] * largest_kernel[2])
        
        if remaining_compression <= 8:
            # Apply additional temporal compression
            return (min(remaining_compression * largest_kernel[0], 16), 
                   largest_kernel[1], largest_kernel[2])
        else:
            # Use maximum compression available
            return (16, 32, 32)
    
    def compress_frame(self, frame: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Compress a single frame using the specified kernel size.
        
        Args:
            frame: Input frame tensor of shape [C, T, H, W] or [T, H, W, C]
            kernel_size: (frame, height, width) compression kernel
            
        Returns:
            Compressed frame tensor
        """
        if kernel_size == (1, 1, 1):
            return frame
        
        # Ensure frame is in [C, T, H, W] format
        if frame.dim() == 4 and frame.shape[-1] < frame.shape[0]:
            # Likely [T, H, W, C] format, transpose to [C, T, H, W]
            frame = frame.permute(3, 0, 1, 2)
        
        # Apply 3D average pooling for compression
        compressed = F.avg_pool3d(
            frame.unsqueeze(0),  # Add batch dimension
            kernel_size=kernel_size,
            stride=kernel_size
        ).squeeze(0)  # Remove batch dimension
        
        return compressed
    
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
                # Handle tail frames - use maximum compression or skip
                kernel_size = self.base_kernel_sizes[-1]
            else:
                _, kernel_size = self.compression_schedule[i]
            
            try:
                compressed_frame = self.compress_frame(frame, kernel_size)
                compressed_history.append(compressed_frame)
            except RuntimeError as e:
                # Handle cases where frame is too small for compression
                if "size" in str(e).lower():
                    # Apply global average pooling as fallback
                    compressed_frame = torch.mean(frame, dim=(1, 2, 3), keepdim=True)
                    compressed_history.append(compressed_frame)
                else:
                    raise e
        
        return compressed_history
    
    def add_new_section(self, 
                       current_history: List[torch.Tensor], 
                       new_section: torch.Tensor) -> List[torch.Tensor]:
        if len(current_history) == 0:
            # First section - no compression needed
            return [new_section]
        
        # Add new section as a single unit
        updated_history = [new_section] + current_history
        
        # Apply compression based on age
        compressed_history = []
        for i, section in enumerate(updated_history):
            if i >= len(self.compression_schedule):
                # Maximum compression for old sections
                _, kernel_size = self.compression_schedule[-1]
            else:
                _, kernel_size = self.compression_schedule[i]
            
            # Compress the entire section
            compressed_section = self.compress_frame(section, kernel_size)
            compressed_history.append(compressed_section)
            
            # Stop if we've reached max history
            if len(compressed_history) >= self.max_history_frames // 20:  # Assuming ~20 frames per section
                break
        
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
        Fixed version that only selects 11 frames instead of concatenating all sections
        """
        if not compressed_history:
            return torch.empty(0)
        
        # Constants
        LONG_FRAMES = 5
        MID_FRAMES = 3
        RECENT_FRAMES = 1
        OVERLAP_FRAMES = 2
        GEN_FRAMES = 11
        TOTAL_FRAMES = 22
        CONTEXT_FRAMES = 11  # Should match LONG + MID + RECENT + OVERLAP
        
        # First, we need to understand what compressed_history contains
        # It's a list of compressed sections, each with shape [C, T_compressed, H, W]
        
        # Flatten all frames from all sections to pick individual frames
        all_frames = []
        frame_to_section_map = []  # Track which section each frame came from
        
        for section_idx, section in enumerate(compressed_history):
            # section shape: [C, T, H, W]
            num_frames_in_section = section.shape[1]
            for frame_idx in range(num_frames_in_section):
                # Extract individual frame
                frame = section[:, frame_idx:frame_idx+1, :, :]  # [C, 1, H, W]
                all_frames.append(frame)
                frame_to_section_map.append(section_idx)
        
        total_available_frames = len(all_frames)
        
        if total_available_frames == 0:
            return torch.empty(0)
        
        # Now select exactly 11 frames hierarchically
        selected_frame_indices = []
        
        # Long-term frames (5 frames from older history)
        if total_available_frames >= 40:
            # Sample sparsely from older frames
            step = max(1, (total_available_frames - 20) // LONG_FRAMES)
            for i in range(LONG_FRAMES):
                idx = i * step
                selected_frame_indices.append(idx)
        else:
            # Not enough frames - distribute evenly
            if total_available_frames >= LONG_FRAMES:
                step = total_available_frames // LONG_FRAMES
                for i in range(LONG_FRAMES):
                    selected_frame_indices.append(i * step)
            else:
                # Use what we have and repeat last
                selected_frame_indices.extend(range(total_available_frames))
                while len(selected_frame_indices) < LONG_FRAMES:
                    selected_frame_indices.append(total_available_frames - 1)
        
        # Mid-term frames (3 frames)
        mid_start = max(LONG_FRAMES, total_available_frames - 15)
        for i in range(MID_FRAMES):
            idx = min(mid_start + i * 2, total_available_frames - 1)
            selected_frame_indices.append(idx)
        
        # Recent frame (1 frame)
        recent_idx = max(0, total_available_frames - 5)
        selected_frame_indices.append(recent_idx)
        
        # Overlap frames (2 most recent frames)
        for i in range(OVERLAP_FRAMES):
            idx = max(0, total_available_frames - OVERLAP_FRAMES + i)
            selected_frame_indices.append(idx)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in selected_frame_indices:
            if idx not in seen and idx < total_available_frames:
                unique_indices.append(idx)
                seen.add(idx)
        
        # Select exactly CONTEXT_FRAMES frames
        if len(unique_indices) > CONTEXT_FRAMES:
            unique_indices = unique_indices[:CONTEXT_FRAMES]
        elif len(unique_indices) < CONTEXT_FRAMES:
            # Pad with most recent frame
            while len(unique_indices) < CONTEXT_FRAMES:
                unique_indices.append(total_available_frames - 1)
        
        # Gather selected frames
        selected_frames = [all_frames[i] for i in unique_indices]
        
        # Find max dimensions for padding
        max_h = max(f.shape[2] for f in selected_frames)
        max_w = max(f.shape[3] for f in selected_frames)
        
        # Pad frames to same spatial size
        padded_frames = []
        for frame in selected_frames:
            if frame.shape[2] < max_h or frame.shape[3] < max_w:
                pad_h = max_h - frame.shape[2]
                pad_w = max_w - frame.shape[3]
                frame = F.pad(frame, (0, pad_w, 0, pad_h))
            padded_frames.append(frame)
        
        # Concatenate only the 11 selected frames
        context_tensor = torch.cat(padded_frames, dim=1)  # [C, 11, H, W]
        
        if add_generation_frames:
            C, T, H, W = context_tensor.shape
            assert T == CONTEXT_FRAMES, f"Expected {CONTEXT_FRAMES} context frames, got {T}"
            
            # Add generation placeholder
            gen_placeholder = torch.zeros((C, GEN_FRAMES, H, W), device=context_tensor.device)
            final_tensor = torch.cat([context_tensor, gen_placeholder], dim=1)
            
            # Should now be exactly 85 frames (11 + 74)
            assert final_tensor.shape[1] == TOTAL_FRAMES, f"Expected {TOTAL_FRAMES} frames, got {final_tensor.shape[1]}"
            
            
            return final_tensor
        
        return context_tensor