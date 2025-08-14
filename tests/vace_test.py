
import os
import sys
import gc
import time
import torch
import numpy as np
import logging
from pathlib import Path
import json
from PIL import Image
import psutil
import GPUtil
from contextlib import contextmanager
 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from vace.models.wan.wan_vace  import WanVace  
from wan.text2video import WanT2V, T5EncoderModel, WanVAE
from vace.models.wan.modules.model import VaceWanModel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConfig:
    """Configuration matching your actual model requirements"""
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
       
        self.num_train_timesteps = 1000
        self.param_dtype = torch.float16
        self.text_len = 256
        self.t5_dtype = torch.float32
        self.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"  
        self.t5_tokenizer = "google/umt5-xxl"  
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.vae_checkpoint = "Wan2.1_VAE.pth" 
        self.sample_neg_prompt = "low quality, blurry, distorted"
        self.sample_fps = 12



class VaceTester:
    """Test the actual FramepackVace implementation"""
    
    def __init__(self, checkpoint_dir, device_id=0):
        """
        Initialize with your actual model checkpoints
        
        Args:
            checkpoint_dir: Path to your model checkpoints directory
            device_id: GPU device ID to use
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing Vace with checkpoints from {checkpoint_dir}")
        logger.info(f"Using device: {self.device}")
        
        # Check if checkpoint files exist
        self._verify_checkpoints()
        
        # Initialize configuration
        self.config = TestConfig(checkpoint_dir)
        
        # Initialize the REAL model
        self.model = self._initialize_model()
        
    def _verify_checkpoints(self):
        """Verify that required checkpoint files exist"""
        required_files = [
            "Wan2.1_VAE.pth",  # VAE checkpoint
            "diffusion_pytorch_model.safetensors",  # Main model checkpoint
        ]
        
        required_dirs = [
            "google/umt5-xxl",  # T5 tokenizer directory
        ]
        
        missing = []
        
        # Check files
        for file in required_files:
            if not (self.checkpoint_dir / file).exists():
                missing.append(f"File: {file}")
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = self.checkpoint_dir / dir_name
            if not dir_path.exists():
                missing.append(f"Directory: {dir_name}")
            elif not dir_path.is_dir():
                missing.append(f"{dir_name} exists but is not a directory")
            elif not any(dir_path.iterdir()):  # Check if directory is empty
                missing.append(f"Directory {dir_name} is empty")
        
        if missing:
            logger.error("Missing checkpoints:")
            for item in missing:
                logger.error(f"  - {item}")
            raise FileNotFoundError(f"Missing checkpoint files/directories: {missing}")
        
        logger.info("✓ All checkpoint files and directories found")
        
        # Log what was found
        logger.info("  Found files:")
        for file in required_files:
            file_path = self.checkpoint_dir / file
            size_mb = file_path.stat().st_size / (1024**2)
            logger.info(f"    - {file} ({size_mb:.1f} MB)")
        
        logger.info("  Found directories:")
        for dir_name in required_dirs:
            dir_path = self.checkpoint_dir / dir_name
            num_files = len(list(dir_path.glob('*')))
            logger.info(f"    - {dir_name}/ ({num_files} files)")
    
    def _initialize_model(self):
        """Initialize the actual Vace model"""
        try:
            model = WanVace(
                config=self.config,
                checkpoint_dir=str(self.checkpoint_dir),
                device_id=self.device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=True  # Set based on your memory constraints
            )
            logger.info("✓ Model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    @contextmanager
    def memory_tracking(self):
        """Track memory usage during operations"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            
            logger.info(f"  Memory used: {(end_mem - start_mem) / 1024**3:.2f} GB")
            logger.info(f"  Peak memory: {peak_mem / 1024**3:.2f} GB")
    
    # ============================================================================
    # Test Methods
    # ============================================================================
    
    def test_model_components(self):
        """Test that all model components are loaded correctly"""
        logger.info("\n=== Testing Model Components ===")
        
        tests_passed = 0
        tests_failed = 0
        
        # Test T5 Encoder
        try:
            assert self.model.text_encoder is not None
            assert hasattr(self.model.text_encoder, 'model')
            logger.info("✓ T5 Encoder loaded")
            tests_passed += 1
        except AssertionError:
            logger.error("✗ T5 Encoder not loaded properly")
            tests_failed += 1
        
        # Test VAE
        try:
            assert self.model.vae is not None
            logger.info("✓ VAE loaded")
            tests_passed += 1
        except AssertionError:
            logger.error("✗ VAE not loaded properly")
            tests_failed += 1
        
        # Test Main Model
        try:
            assert self.model.model is not None
            logger.info("✓ VaceWanModel loaded")
            tests_passed += 1
        except AssertionError:
            logger.error("✗ VaceWanModel not loaded properly")
            tests_failed += 1
        
        # Test Video Processor
        try:
            assert self.model.vid_proc is not None
            logger.info("✓ Video Processor initialized")
            tests_passed += 1
        except AssertionError:
            logger.error("✗ Video Processor not initialized")
            tests_failed += 1
        
        logger.info(f"\nComponent tests: {tests_passed} passed, {tests_failed} failed")
        return tests_failed == 0
    
    def test_encoding_pipeline(self):
        """Test the encoding pipeline with real data"""
        logger.info("\n=== Testing Encoding Pipeline ===")
        
        # Create test data
        test_frames = [torch.randn(3, 41, 480, 832, device=self.device) * 0.5]  # Normalized [-1, 1]
        test_masks = [torch.ones(1, 41, 480, 832, device=self.device)]
        
        with self.memory_tracking():
            try:
                # Test frame encoding
                start = time.time()
                encoded_frames = self.model.vace_encode_frames(test_frames, None, test_masks)
                encode_time = time.time() - start
                
                logger.info(f"✓ Frame encoding successful")
                logger.info(f"  Input shape: {test_frames[0].shape}")
                logger.info(f"  Output shape: {encoded_frames[0].shape}")
                logger.info(f"  Encoding time: {encode_time:.2f}s")
                
                # Test mask encoding
                start = time.time()
                encoded_masks = self.model.vace_encode_masks(test_masks, None)
                mask_time = time.time() - start
                
                logger.info(f"✓ Mask encoding successful")
                logger.info(f"  Mask shape: {encoded_masks[0].shape}")
                logger.info(f"  Mask encoding time: {mask_time:.2f}s")
                
                # Test latent combination
                latents = self.model.vace_latent(encoded_frames, encoded_masks)
                logger.info(f"✓ Latent combination successful")
                logger.info(f"  Combined shape: {latents[0].shape}")
                
                return True
                
            except Exception as e:
                logger.error(f"✗ Encoding pipeline failed: {e}")
                return False
    
   
    def test_mini_generation(self, prompt="A serene beach at sunset", num_frames=82):
        """Test a minimal generation pipeline"""
        logger.info(f"\n=== Testing Mini Generation ('{prompt}', {num_frames} frames) ===")
        
        # Create minimal test inputs
        test_frames = [torch.randn(3, num_frames, 480, 832, device=self.device) * 0.5]
        test_masks = [torch.ones(1, num_frames, 480, 832, device=self.device)]
        
        try:
            with self.memory_tracking():
                start = time.time()
                
              
                result = self.model.generate(
                    input_prompt=prompt,
                    input_frames=test_frames,
                    input_masks=test_masks,
                    input_ref_images=[None],
                    size=(832, 480),
                    frame_num=num_frames,
                    sampling_steps=5,  
                    guide_scale=3.0,
                    seed=42,
                    offload_model=False 
                )
                
                gen_time = time.time() - start
                
                if result is not None:
                    logger.info(f"✓ Generation successful!")
                    logger.info(f"  Output shape: {result.shape}")
                    logger.info(f"  Generation time: {gen_time:.2f}s")
                    logger.info(f"  FPS: {num_frames/gen_time:.2f}")
                    
                    # Save a sample frame
                    self._save_sample_frame(result)
                    
                    return True
                else:
                    logger.warning("Generation returned None (might be non-rank-0 process)")
                    return True
                    
        except Exception as e:
            logger.error(f"✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_sample_frame(self, video_tensor):
        """Save a sample frame from generated video"""
        try:
            # Take middle frame
            frame_idx = video_tensor.shape[1] // 2
            frame = video_tensor[:, frame_idx, :, :].cpu()
            
            # Convert to PIL Image
            frame = (frame + 1.0) / 2.0  # Denormalize to [0, 1]
            frame = torch.clamp(frame, 0, 1)
            frame = (frame * 255).byte()
            frame = frame.permute(1, 2, 0).numpy()
            
            img = Image.fromarray(frame)
            img.save("test_generation_sample.png")
            logger.info(f"  Sample frame saved to test_generation_sample.png")
            
        except Exception as e:
            logger.warning(f"Could not save sample frame: {e}")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING ALL TESTS")
        logger.info("="*60)
        
        test_results = {
            "components": self.test_model_components(),
            "encoding": self.test_encoding_pipeline(),
            # Uncomment for full generation test (takes longer)
            "generation": self.test_mini_generation()
        }
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name:15s}: {status}")
        
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        # Save results
        with open("test_results.json", "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
                "results": test_results,
                "passed": passed,
                "total": total
            }, f, indent=2)
        
        return passed == total


class PerformanceBenchmark:
    """Benchmark actual model performance"""
    
    def __init__(self, tester):
        self.model = tester.model
        self.device = tester.device
        self.results = []
    
    def benchmark_encoding_speed(self):
        """Benchmark encoding at different resolutions and frame counts"""
        logger.info("\n=== Benchmarking Encoding Speed ===")
        
        configs = [
            {"frames": 41, "height": 480, "width": 832, "name": "Default"},
            {"frames": 81, "height": 480, "width": 832, "name": "Long"},
            {"frames": 41, "height": 720, "width": 1280, "name": "HD"},
        ]
        
        for config in configs:
            logger.info(f"\nTesting {config['name']}: {config['frames']} frames @ {config['height']}x{config['width']}")
            
          
            frames = [torch.randn(3, config['frames'], config['height'], config['width'], 
                                 device=self.device) * 0.5]
            masks = [torch.ones(1, config['frames'], config['height'], config['width'], 
                               device=self.device)]
            
            try:
              
                _ = self.model.vace_encode_frames(frames, None, masks)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
               
                times = []
                for _ in range(3): 
                    start = time.time()
                    _ = self.model.vace_encode_frames(frames, None, masks)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                fps = config['frames'] / avg_time
                
                logger.info(f"  Encoding time: {avg_time:.3f}s (±{np.std(times):.3f}s)")
                logger.info(f"  FPS: {fps:.1f}")
                
                self.results.append({
                    "operation": "encode",
                    "config": config,
                    "time": avg_time,
                    "fps": fps
                })
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
        
        return self.results


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Vace Implementation')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to model checkpoints directory')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (skip generation)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    
    args = parser.parse_args()
   
    if not Path(args.checkpoint_dir).exists():
        logger.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
        return 1
    
    try:
       
        tester = VaceTester(
            checkpoint_dir=args.checkpoint_dir,
            device_id=args.device
        )
        
      
        success = tester.run_all_tests()
      
        if args.benchmark:
            benchmark = PerformanceBenchmark(tester)
            benchmark.benchmark_encoding_speed()
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())