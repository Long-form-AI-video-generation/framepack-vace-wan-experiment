"""
Enhanced Benchmark Suite for FramepackVace
Comprehensive performance testing and analysis
"""

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
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import csv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from vace.models.wan.framepack_vace import FramepackVace
from wan.text2video import WanT2V, T5EncoderModel, WanVAE
from vace.models.wan.modules.model import VaceWanModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Store individual benchmark result"""
    test_name: str
    operation: str
    frames: int
    resolution: Tuple[int, int]
    execution_time: float
    memory_used_gb: float
    peak_memory_gb: float
    fps: float
    throughput: float  # frames per second processed
    success: bool
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class SystemMetrics:
    """System metrics during benchmark"""
    cpu_usage_percent: float
    ram_usage_gb: float
    gpu_usage_percent: float
    gpu_memory_gb: float
    gpu_temp_celsius: float


class EnhancedPerformanceMonitor:
    """Advanced performance monitoring with detailed metrics"""
    
    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval
        self.gpu_memory_samples = []
        self.cpu_memory_samples = []
        self.gpu_usage_samples = []
        self.cpu_usage_samples = []
        self.timestamps = []
        self.monitoring = False
        
    @contextmanager
    def monitor(self, operation_name="Operation"):
        """Context manager for monitoring with detailed metrics"""
        self.reset()
        self.start_monitoring()
        start_time = time.time()
        
        try:
            yield self
        finally:
            execution_time = time.time() - start_time
            self.stop_monitoring()
            self.execution_time = execution_time
            self.operation_name = operation_name
            self.calculate_statistics()
    
    def reset(self):
        """Reset all metrics"""
        self.gpu_memory_samples = []
        self.cpu_memory_samples = []
        self.gpu_usage_samples = []
        self.cpu_usage_samples = []
        self.timestamps = []
        
    def start_monitoring(self):
        """Start collecting performance metrics"""
        self.monitoring = True
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3
        else:
            self.initial_gpu_memory = 0
        
        self.initial_cpu_memory = psutil.Process().memory_info().rss / 1024**3
        self.start_time = time.time()
    
    def sample_metrics(self):
        """Sample current system metrics"""
        if not self.monitoring:
            return
            
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)
        
        cpu_mem = psutil.Process().memory_info().rss / 1024**3
        cpu_usage = psutil.cpu_percent(interval=None)
        self.cpu_memory_samples.append(cpu_mem)
        self.cpu_usage_samples.append(cpu_usage)
      
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            self.gpu_memory_samples.append(gpu_mem)
            
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_usage_samples.append(gpu.load * 100)
            except:
                self.gpu_usage_samples.append(0)
        else:
            self.gpu_memory_samples.append(0)
            self.gpu_usage_samples.append(0)
    
    def stop_monitoring(self):
        """Stop collecting metrics and finalize"""
        self.monitoring = False
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.final_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            self.peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        else:
            self.final_gpu_memory = 0
            self.peak_gpu_memory = 0
        
        self.final_cpu_memory = psutil.Process().memory_info().rss / 1024**3
    
    def calculate_statistics(self):
        """Calculate statistical metrics"""
        if self.gpu_memory_samples:
            self.avg_gpu_memory = np.mean(self.gpu_memory_samples)
            self.std_gpu_memory = np.std(self.gpu_memory_samples)
            self.max_gpu_memory = np.max(self.gpu_memory_samples)
        else:
            self.avg_gpu_memory = 0
            self.std_gpu_memory = 0
            self.max_gpu_memory = 0
        
        if self.cpu_memory_samples:
            self.avg_cpu_memory = np.mean(self.cpu_memory_samples)
            self.peak_cpu_memory = np.max(self.cpu_memory_samples)
        else:
            self.avg_cpu_memory = 0
            self.peak_cpu_memory = 0
        
        if self.gpu_usage_samples:
            self.avg_gpu_usage = np.mean(self.gpu_usage_samples)
        else:
            self.avg_gpu_usage = 0
        
        if self.cpu_usage_samples:
            self.avg_cpu_usage = np.mean(self.cpu_usage_samples)
        else:
            self.avg_cpu_usage = 0
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        return {
            'operation': self.operation_name,
            'execution_time': self.execution_time,
            'gpu_memory': {
                'initial': self.initial_gpu_memory,
                'final': self.final_gpu_memory,
                'peak': self.peak_gpu_memory,
                'average': self.avg_gpu_memory,
                'std': self.std_gpu_memory
            },
            'cpu_memory': {
                'initial': self.initial_cpu_memory,
                'final': self.final_cpu_memory,
                'peak': self.peak_cpu_memory,
                'average': self.avg_cpu_memory
            },
            'utilization': {
                'avg_gpu_usage': self.avg_gpu_usage,
                'avg_cpu_usage': self.avg_cpu_usage
            }
        }

class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for FramepackVace"""
    
    def __init__(self, model, device, output_dir="benchmark_results"):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.monitor = EnhancedPerformanceMonitor()
        
    def benchmark_encoding_performance(self):
        """Benchmark encoding at various configurations"""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Encoding Performance")
        logger.info("="*60)
        
        configurations = [
          
            {"frames": 41, "height": 480, "width": 832, "name": "480p_standard"},
            {"frames": 81, "height": 480, "width": 832, "name": "480p_long"},
            {"frames": 122, "height": 480, "width": 832, "name": "480p_extended"},
           
            {"frames": 41, "height": 720, "width": 1280, "name": "720p_standard"},
            {"frames": 81, "height": 720, "width": 1280, "name": "720p_long"},
           
            {"frames": 41, "height": 576, "width": 1024, "name": "16:9_alternative"},
            {"frames": 41, "height": 512, "width": 512, "name": "square"},
        ]
        
        encoding_results = []
        
        for config in configurations:
            logger.info(f"\nTesting {config['name']}: {config['frames']}f @ {config['height']}x{config['width']}")
            
            try:
                # Create test data
                frames = [torch.randn(3, config['frames'], config['height'], 
                                     config['width'], device=self.device) * 0.5]
                masks = [torch.ones(1, config['frames'], config['height'], 
                                   config['width'], device=self.device)]
                
                
                _ = self.model.vace_encode_frames(frames, None, masks)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                gc.collect()
                
                times = []
                memory_peaks = []
                
                for run in range(3):
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    with self.monitor.monitor(f"Encoding_{config['name']}_run{run}"):
                        start = time.time()
                        encoded = self.model.vace_encode_frames(frames, None, masks)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        elapsed = time.time() - start
                        times.append(elapsed)
                      
                        for _ in range(5):
                            self.monitor.sample_metrics()
                            time.sleep(0.02)
                    
                    memory_peaks.append(self.monitor.peak_gpu_memory)
                
               
                avg_time = np.mean(times)
                std_time = np.std(times)
                fps = config['frames'] / avg_time
                pixels_per_second = (config['frames'] * config['height'] * config['width']) / avg_time / 1e6  # Megapixels/s
                
                result = BenchmarkResult(
                    test_name=config['name'],
                    operation="encoding",
                    frames=config['frames'],
                    resolution=(config['height'], config['width']),
                    execution_time=avg_time,
                    memory_used_gb=self.monitor.final_gpu_memory - self.monitor.initial_gpu_memory,
                    peak_memory_gb=np.mean(memory_peaks),
                    fps=fps,
                    throughput=pixels_per_second,
                    success=True
                )
                
                encoding_results.append(result)
                self.results.append(result)
                
                logger.info(f"  ✓ Time: {avg_time:.3f}±{std_time:.3f}s")
                logger.info(f"  ✓ FPS: {fps:.1f}")
                logger.info(f"  ✓ Throughput: {pixels_per_second:.2f} MP/s")
                logger.info(f"  ✓ Peak Memory: {result.peak_memory_gb:.2f} GB")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
                result = BenchmarkResult(
                    test_name=config['name'],
                    operation="encoding",
                    frames=config['frames'],
                    resolution=(config['height'], config['width']),
                    execution_time=0,
                    memory_used_gb=0,
                    peak_memory_gb=0,
                    fps=0,
                    throughput=0,
                    success=False,
                    error_message=str(e)
                )
                encoding_results.append(result)
                self.results.append(result)
        
        return encoding_results
    
    def benchmark_context_operations(self):
        """Benchmark context building and selection operations"""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Context Operations")
        logger.info("="*60)
        
        context_sizes = [30, 60, 90, 120, 150, 180]
        context_results = []
        
        for size in context_sizes:
            logger.info(f"\nTesting context size: {size} frames")
            
            try:
               
                num_chunks = size // 30
                accumulated = [torch.randn(8, 30, 6, 52, device=self.device) 
                             for _ in range(max(1, num_chunks))]
                
              
                with self.monitor.monitor(f"Context_Build_{size}"):
                    start = time.time()
                    context = self.model.build_hierarchical_context_latent(accumulated, section_id=1)
                    build_time = time.time() - start
                
               
                with self.monitor.monitor(f"Context_Select_{size}"):
                    start = time.time()
                    selected = self.model.pick_context_v2(context, section_id=1)
                    select_time = time.time() - start
                
                total_time = build_time + select_time
                
                result = BenchmarkResult(
                    test_name=f"context_{size}",
                    operation="context_ops",
                    frames=size,
                    resolution=(6, 52),
                    execution_time=total_time,
                    memory_used_gb=self.monitor.final_gpu_memory - self.monitor.initial_gpu_memory,
                    peak_memory_gb=self.monitor.peak_gpu_memory,
                    fps=size / total_time,
                    throughput=size / total_time,
                    success=True
                )
                
                context_results.append(result)
                self.results.append(result)
                
                logger.info(f"  ✓ Build time: {build_time:.3f}s")
                logger.info(f"  ✓ Select time: {select_time:.3f}s")
                logger.info(f"  ✓ Total: {total_time:.3f}s")
                logger.info(f"  ✓ Throughput: {size/total_time:.1f} frames/s")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
        
        return context_results
    
    def benchmark_mask_operations(self):
        """Benchmark mask generation and processing"""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Mask Operations")
        logger.info("="*60)
        
        mask_configs = [
            {"frames": 164, "height": 480, "width": 832, "name": "standard"},
            {"frames": 324, "height": 480, "width": 832, "name": "long"},
            {"frames": 164, "height": 720, "width": 1280, "name": "hd"},
        ]
        
        mask_results = []
        
        for config in mask_configs:
            logger.info(f"\nTesting mask {config['name']}: {config['frames']}f @ {config['height']}x{config['width']}")
            
            try:
                frame_shape = (3, config['frames'], config['height'], config['width'])
                
              
                with self.monitor.monitor(f"Mask_Initial_{config['name']}"):
                    start = time.time()
                    mask_init = self.model.create_temporal_blend_mask_v2(
                        frame_shape, section_id=0, initial=True
                    )
                    init_time = time.time() - start
                
            
                cont_times = []
                for section in range(1, 4):
                    start = time.time()
                    mask_cont = self.model.create_temporal_blend_mask_v2(
                        frame_shape, section_id=section
                    )
                    cont_times.append(time.time() - start)
                
                avg_cont_time = np.mean(cont_times)
                total_time = init_time + avg_cont_time
                
                result = BenchmarkResult(
                    test_name=f"mask_{config['name']}",
                    operation="mask_ops",
                    frames=config['frames'],
                    resolution=(config['height'], config['width']),
                    execution_time=total_time,
                    memory_used_gb=self.monitor.final_gpu_memory - self.monitor.initial_gpu_memory,
                    peak_memory_gb=self.monitor.peak_gpu_memory,
                    fps=config['frames'] / total_time,
                    throughput=config['frames'] / total_time,
                    success=True
                )
                
                mask_results.append(result)
                self.results.append(result)
                
                logger.info(f"  ✓ Initial mask: {init_time:.3f}s")
                logger.info(f"  ✓ Continuation mask (avg): {avg_cont_time:.3f}s")
                logger.info(f"  ✓ Throughput: {config['frames']/total_time:.1f} frames/s")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
        
        return mask_results
    
    def benchmark_memory_scaling(self):
        """Benchmark memory usage scaling with different parameters"""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Memory Scaling")
        logger.info("="*60)
        
        memory_configs = [
            {"frames": 20, "height": 480, "width": 832},
            {"frames": 41, "height": 480, "width": 832},
            {"frames": 82, "height": 480, "width": 832},
            {"frames": 123, "height": 480, "width": 832},
            {"frames": 41, "height": 360, "width": 640},
            {"frames": 41, "height": 576, "width": 1024},
            {"frames": 41, "height": 720, "width": 1280},
        ]
        
        memory_results = []
        
        for config in memory_configs:
            name = f"{config['frames']}f_{config['height']}x{config['width']}"
            logger.info(f"\nTesting memory for: {name}")
            
            try:
              
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
               
                frames = [torch.randn(3, config['frames'], config['height'], 
                                     config['width'], device=self.device) * 0.5]
                
                with self.monitor.monitor(f"Memory_{name}"):
                 
                    encoded = self.model.vace_encode_frames(frames, None)
                    
                 
                    _ = encoded[0].sum()
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    
                
                    for _ in range(10):
                        self.monitor.sample_metrics()
                        time.sleep(0.05)
                
                memory_data = {
                    'config': name,
                    'frames': config['frames'],
                    'resolution': f"{config['height']}x{config['width']}",
                    'pixels': config['height'] * config['width'],
                    'total_pixels': config['frames'] * config['height'] * config['width'],
                    'peak_memory_gb': self.monitor.peak_gpu_memory,
                    'avg_memory_gb': self.monitor.avg_gpu_memory,
                    'memory_per_frame_mb': (self.monitor.peak_gpu_memory * 1024) / config['frames'],
                    'memory_per_megapixel_mb': (self.monitor.peak_gpu_memory * 1024) / 
                                               (config['frames'] * config['height'] * config['width'] / 1e6)
                }
                
                memory_results.append(memory_data)
                
                logger.info(f"  ✓ Peak memory: {memory_data['peak_memory_gb']:.2f} GB")
                logger.info(f"  ✓ Per frame: {memory_data['memory_per_frame_mb']:.1f} MB")
                logger.info(f"  ✓ Per megapixel: {memory_data['memory_per_megapixel_mb']:.1f} MB")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
        
        return memory_results
    
    def benchmark_generation_pipeline(self):
        """Benchmark the full generation pipeline"""
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK: Generation Pipeline")
        logger.info("="*60)
        
        generation_configs = [
            {"frames": 41, "steps": 5, "name": "quick_test"},
            {"frames": 82, "steps": 5, "name": "medium_test"},
            {"frames": 41, "steps": 10, "name": "quality_test"},
        ]
        
        generation_results = []
        
        for config in generation_configs:
            logger.info(f"\nTesting generation {config['name']}: {config['frames']} frames, {config['steps']} steps")
            
            try:
               
                test_frames = [torch.randn(3, config['frames'], 480, 832, device=self.device) * 0.5]
                test_masks = [torch.ones(1, config['frames'], 480, 832, device=self.device)]
                
                with self.monitor.monitor(f"Generation_{config['name']}"):
                    start = time.time()
                    
                    result_video = self.model.generate_with_framepack(
                        input_prompt="A test video generation",
                        input_frames=test_frames,
                        input_masks=test_masks,
                        input_ref_images=[None],
                        size=(832, 480),
                        frame_num=config['frames'],
                        sampling_steps=config['steps'],
                        guide_scale=3.0,
                        seed=42,
                        offload_model=False
                    )
                    
                    gen_time = time.time() - start
                
                if result_video is not None:
                    fps = config['frames'] / gen_time
                    steps_per_second = config['steps'] / gen_time
                    
                    result = BenchmarkResult(
                        test_name=config['name'],
                        operation="generation",
                        frames=config['frames'],
                        resolution=(480, 832),
                        execution_time=gen_time,
                        memory_used_gb=self.monitor.final_gpu_memory - self.monitor.initial_gpu_memory,
                        peak_memory_gb=self.monitor.peak_gpu_memory,
                        fps=fps,
                        throughput=steps_per_second,
                        success=True
                    )
                    
                    generation_results.append(result)
                    self.results.append(result)
                    
                    logger.info(f"  ✓ Generation time: {gen_time:.2f}s")
                    logger.info(f"  ✓ FPS: {fps:.2f}")
                    logger.info(f"  ✓ Steps/s: {steps_per_second:.2f}")
                    logger.info(f"  ✓ Peak memory: {result.peak_memory_gb:.2f} GB")
                    
            except Exception as e:
                logger.error(f"  ✗ Failed: {str(e)}")
        
        return generation_results
    
    def run_stress_test(self, duration_seconds=60):
        """Run stress test for stability"""
        logger.info("\n" + "="*60)
        logger.info(f"STRESS TEST: Running for {duration_seconds} seconds")
        logger.info("="*60)
        
        start_time = time.time()
        iterations = 0
        errors = []
        success_count = 0
        
        while time.time() - start_time < duration_seconds:
            try:
              
                frames = np.random.choice([20, 41, 82])
                height = np.random.choice([480, 576])
                width = np.random.choice([832, 1024])
                
                test_frames = [torch.randn(3, frames, height, width, device=self.device) * 0.5]
                
                encoded = self.model.vace_encode_frames(test_frames, None)
                decoded = self.model.decode_latent(encoded)
                
                iterations += 1
                success_count += 1
                
                if iterations % 5 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                
                if iterations % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  Progress: {iterations} iterations, {elapsed:.1f}s elapsed")
                
            except Exception as e:
                errors.append({
                    'iteration': iterations,
                    'error': str(e),
                    'config': f"{frames}f_{height}x{width}"
                })
                logger.error(f"  Error at iteration {iterations}: {str(e)[:100]}")
        
        elapsed = time.time() - start_time
        
        stress_result = {
            'duration': elapsed,
            'iterations': iterations,
            'success_count': success_count,
            'error_count': len(errors),
            'success_rate': success_count / iterations if iterations > 0 else 0,
            'throughput': iterations / elapsed,
            'errors': errors[:5]  
        }
        
        logger.info(f"\nStress Test Results:")
        logger.info(f"  Duration: {elapsed:.1f}s")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  Success rate: {stress_result['success_rate']:.1%}")
        logger.info(f"  Throughput: {stress_result['throughput']:.2f} iter/s")
        logger.info(f"  Errors: {len(errors)}")
        
        return stress_result
    
    def generate_report(self):
        """Generate comprehensive benchmark report with visualizations"""
        logger.info("\n" + "="*60)
        logger.info("GENERATING BENCHMARK REPORT")
        logger.info("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
      
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved raw results to: {csv_path}")
        
      
        summary = {
            'timestamp': timestamp,
            'total_tests': len(self.results),
            'successful_tests': df['success'].sum() if 'success' in df else 0,
            'operations': {
                op: {
                    'count': len(df[df['operation'] == op]),
                    'avg_time': df[df['operation'] == op]['execution_time'].mean(),
                    'avg_memory': df[df['operation'] == op]['peak_memory_gb'].mean(),
                    'avg_fps': df[df['operation'] == op]['fps'].mean()
                }
                for op in df['operation'].unique()
            } if 'operation' in df else {},
            'system_info': self._get_system_info()
        }
        
     
        json_path = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"  Saved JSON report to: {json_path}")
        
     
        self._generate_plots(df, timestamp)
        
        return summary
    
    def _generate_plots(self, df, timestamp):
        """Generate visualization plots"""
        if df.empty:
            logger.warning("No data to visualize")
            return
        
       
        sns.set_style("whitegrid")
        
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'FramepackVace Benchmark Results - {timestamp}', fontsize=16)
        
      
        if 'operation' in df and 'execution_time' in df:
            ax = axes[0, 0]
            df_ops = df.groupby('operation')['execution_time'].mean().sort_values()
            df_ops.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_xlabel('Execution Time (s)')
            ax.set_title('Average Execution Time by Operation')
      
        if 'test_name' in df and 'peak_memory_gb' in df:
            ax = axes[0, 1]
            df_sorted = df.sort_values('peak_memory_gb', ascending=False).head(10)
            ax.barh(range(len(df_sorted)), df_sorted['peak_memory_gb'], color='coral')
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels(df_sorted['test_name'].values, fontsize=8)
            ax.set_xlabel('Peak Memory (GB)')
            ax.set_title('Top 10 Memory Usage by Test')
    
        if 'frames' in df and 'fps' in df:
            ax = axes[0, 2]
            for op in df['operation'].unique() if 'operation' in df else ['all']:
                df_op = df[df['operation'] == op] if op != 'all' else df
                ax.scatter(df_op['frames'], df_op['fps'], label=op, alpha=0.7)
            ax.set_xlabel('Number of Frames')
            ax.set_ylabel('FPS')
            ax.set_title('FPS vs Frame Count')
            ax.legend()
     
        if 'frames' in df and 'peak_memory_gb' in df:
            ax = axes[1, 0]
            ax.scatter(df['frames'], df['peak_memory_gb'], c=df['execution_time'], 
                      cmap='viridis', alpha=0.6)
            ax.set_xlabel('Number of Frames')
            ax.set_ylabel('Peak Memory (GB)')
            ax.set_title('Memory Scaling with Frame Count')
            cbar = plt.colorbar(ax.collections[0], ax=ax)
            cbar.set_label('Execution Time (s)')
    
        if 'throughput' in df and 'operation' in df:
            ax = axes[1, 1]
            df_throughput = df.groupby('operation')['throughput'].mean()
            df_throughput.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_ylabel('Throughput')
            ax.set_title('Average Throughput by Operation')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
        if 'success' in df:
            ax = axes[1, 2]
            success_counts = df['success'].value_counts()
            colors = ['#90EE90', '#FFB6C1']  # Light green and light red
            ax.pie(success_counts.values, labels=['Success', 'Failed'], 
                  autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Test Success Rate')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved plots to: {plot_path}")
        
        
        self._generate_detailed_plots(df, timestamp)
    
    def _generate_detailed_plots(self, df, timestamp):
        """Generate additional detailed analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Detailed Performance Analysis - {timestamp}', fontsize=14)
      
        if 'frames' in df and 'peak_memory_gb' in df and 'resolution' in df:
            ax = axes[0, 0]
            
            df['megapixels'] = df.apply(
                lambda row: row['frames'] * row['resolution'][0] * row['resolution'][1] / 1e6 
                if isinstance(row['resolution'], tuple) else 0, axis=1
            )
            df['memory_per_mp'] = df['peak_memory_gb'] * 1024 / df['megapixels']  # MB per megapixel
            
            valid_data = df[df['megapixels'] > 0]
            if not valid_data.empty:
                ax.scatter(valid_data['megapixels'], valid_data['memory_per_mp'], alpha=0.6)
                ax.set_xlabel('Total Megapixels')
                ax.set_ylabel('Memory per Megapixel (MB)')
                ax.set_title('Memory Efficiency')
        
        
        if 'frames' in df and 'execution_time' in df:
            ax = axes[0, 1]
            df['time_per_frame'] = df['execution_time'] / df['frames']
            
            
            if 'operation' in df:
                for op in df['operation'].unique():
                    df_op = df[df['operation'] == op]
                    ax.scatter(df_op['frames'], df_op['time_per_frame'], label=op, alpha=0.7)
                ax.legend()
            else:
                ax.scatter(df['frames'], df['time_per_frame'], alpha=0.7)
            
            ax.set_xlabel('Number of Frames')
            ax.set_ylabel('Time per Frame (s)')
            ax.set_title('Processing Time Efficiency')
      
        if 'resolution' in df and 'fps' in df:
            ax = axes[1, 0]
            
            df['resolution_area'] = df['resolution'].apply(
                lambda x: x[0] * x[1] if isinstance(x, tuple) else 0
            )
            
            valid_data = df[df['resolution_area'] > 0]
            if not valid_data.empty:
                ax.scatter(valid_data['resolution_area'], valid_data['fps'], alpha=0.6)
                ax.set_xlabel('Resolution (total pixels)')
                ax.set_ylabel('FPS')
                ax.set_title('Resolution Impact on FPS')
        
        if 'timestamp' in df and 'execution_time' in df:
            ax = axes[1, 1]
           
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            df['relative_time'] = (df['timestamp_dt'] - df['timestamp_dt'].min()).dt.total_seconds()
            
            ax.plot(df['relative_time'], df['execution_time'].cumsum(), marker='o', markersize=4)
            ax.set_xlabel('Time (seconds from start)')
            ax.set_ylabel('Cumulative Execution Time (s)')
            ax.set_title('Benchmark Timeline')
        
        plt.tight_layout()
       
        detailed_plot_path = self.output_dir / f"benchmark_detailed_{timestamp}.png"
        plt.savefig(detailed_plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved detailed plots to: {detailed_plot_path}")
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        info['cpu_count'] = psutil.cpu_count()
        info['ram_gb'] = psutil.virtual_memory().total / 1024**3
        
        return info
    
    def run_all_benchmarks(self, quick=False):
        """Run all benchmark tests"""
        logger.info("\n" + "="*70)
        logger.info("RUNNING COMPREHENSIVE BENCHMARK SUITE")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Run individual benchmarks
        self.benchmark_encoding_performance()
        self.benchmark_context_operations()
        self.benchmark_mask_operations()
        
        if not quick:
            self.benchmark_memory_scaling()
            self.benchmark_generation_pipeline()
            self.run_stress_test(duration_seconds=30)
        
        # Generate report
        report = self.generate_report()
        
        total_time = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK COMPLETE")
        logger.info("="*70)
        logger.info(f"Total benchmark time: {total_time:.1f} seconds")
        logger.info(f"Total tests run: {report['total_tests']}")
        logger.info(f"Successful tests: {report['successful_tests']}")
        logger.info(f"Success rate: {report['successful_tests']/report['total_tests']*100:.1f}%")
        
        return report


def main():
    """Main function for standalone benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive FramepackVace Benchmark')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to model checkpoints directory')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark (skip long tests)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--stress-duration', type=int, default=30,
                       help='Stress test duration in seconds')
    
    args = parser.parse_args()
    
 
    from framepack_test import FramepackTester, TestConfig
    
    try:
       
        logger.info("Initializing model for benchmarking...")
        tester = FramepackTester(
            checkpoint_dir=args.checkpoint_dir,
            device_id=args.device
        )
        
      
        benchmark = ComprehensiveBenchmark(
            model=tester.model,
            device=tester.device,
            output_dir=args.output_dir
        )
        
        
        if args.stress_duration != 30:
            logger.info(f"Using custom stress duration: {args.stress_duration}s")
        
        
        report = benchmark.run_all_benchmarks(quick=args.quick)
        
        # Save final summary
        summary_path = Path(args.output_dir) / "benchmark_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FRAMEPACK VACE BENCHMARK SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Total Tests: {report['total_tests']}\n")
            f.write(f"Successful: {report['successful_tests']}\n")
            f.write(f"Success Rate: {report['successful_tests']/report['total_tests']*100:.1f}%\n\n")
            
            f.write("Operation Summary:\n")
            for op, stats in report['operations'].items():
                f.write(f"\n{op.upper()}:\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Avg Time: {stats['avg_time']:.3f}s\n")
                f.write(f"  Avg Memory: {stats['avg_memory']:.2f} GB\n")
                f.write(f"  Avg FPS: {stats['avg_fps']:.1f}\n")
        
        logger.info(f"\nBenchmark summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())