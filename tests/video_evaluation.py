import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
import cv2
from tqdm import tqdm
import lpips
import decord
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ComparisonResults:
    """Results from comparing original vs frame-packed models"""
    
    
    fvd_score: float
    
    
    quality_retention: float
    lpips_difference: float
    ssim_difference: float
    
   
    temporal_consistency_original: float
    temporal_consistency_framepack: float
    temporal_improvement: float
    
    
    motion_smoothness_original: float
    motion_smoothness_framepack: float
    
    
    generation_time_original: float
    generation_time_framepack: float
    speedup: float
    
    
    max_frames_original: int
    max_frames_framepack: int
    length_improvement: float
    
    boundary_artifacts: float
    section_quality_variance: float
    
    def to_dict(self):
        return {
            'fvd_score': self.fvd_score,
            'quality_retention': self.quality_retention,
            'lpips_difference': self.lpips_difference,
            'ssim_difference': self.ssim_difference,
            'temporal_consistency': {
                'original': self.temporal_consistency_original,
                'framepack': self.temporal_consistency_framepack,
                'improvement': self.temporal_improvement
            },
            'motion_smoothness': {
                'original': self.motion_smoothness_original,
                'framepack': self.motion_smoothness_framepack
            },
            'efficiency': {
                'time_original': self.generation_time_original,
                'time_framepack': self.generation_time_framepack,
                'speedup': self.speedup
            },
            'length_capability': {
                'max_frames_original': self.max_frames_original,
                'max_frames_framepack': self.max_frames_framepack,
                'improvement_factor': self.length_improvement
            },
            'framepack_specific': {
                'boundary_artifacts': self.boundary_artifacts,
                'section_variance': self.section_quality_variance
            }
        }


class ComparativeVideoEvaluator:
    """Evaluator for comparing original and frame-packed video generation models"""
    
    def __init__(self, device='cuda', save_dir='evaluation_results'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
       
        import torchvision.models as models
        self.i3d_model = models.video.r3d_18(pretrained=True).to(self.device)
        self.i3d_model.fc = nn.Identity()
        self.i3d_model.eval()
        
        
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()
        
       
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """Load video and convert to tensor"""
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path)
        
        video = vr[:].float()  
        video = video.permute(3, 0, 1, 2)  
        video = video / 127.5 - 1.0  
        
        return video
    
    def compute_fvd(self, videos_original: List[torch.Tensor], 
                    videos_framepack: List[torch.Tensor]) -> float:
        """Compute Fréchet Video Distance between two sets of videos"""
        
        def extract_features(videos):
            features = []
            with torch.no_grad():
                for video in videos:
                   
                    C, T, H, W = video.shape
                    
                    
                    if T < 16:  
                        
                        padding = 16 - T
                        video = F.pad(video, (0, 0, 0, 0, 0, padding), mode='replicate')
                        T = 16
                    
                    
                    video_norm = (video + 1) / 2
                    
                    
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).to(video.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).to(video.device)
                    video_norm = (video_norm - mean) / std
                    
                    
                    for i in range(0, T - 15, 8):
                        
                        clip = video_norm[:, i:i+16, :, :]
                        
                        clip = clip.unsqueeze(0).to(self.device)
                        
                        
                        feat = self.i3d_model(clip).squeeze().cpu().numpy()
                        features.append(feat)
            
            return np.array(features)
        
        
        self.logger.info("Extracting I3D features from original videos...")
        features_orig = extract_features(videos_original)
        
        self.logger.info("Extracting I3D features from frame-packed videos...")
        features_fp = extract_features(videos_framepack)
        
        
        mu1, sigma1 = features_orig.mean(0), np.cov(features_orig, rowvar=False)
        mu2, sigma2 = features_fp.mean(0), np.cov(features_fp, rowvar=False)
        
        # Calculate Fréchet distance
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fvd)
    
    def compute_lpips_similarity(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        """Compute average LPIPS similarity between two videos"""
        min_frames = min(video1.shape[1], video2.shape[1])
        lpips_scores = []
        
        with torch.no_grad():
            for i in range(min_frames):
                frame1 = video1[:, i].unsqueeze(0).to(self.device)
                frame2 = video2[:, i].unsqueeze(0).to(self.device)
                
                
                frame1 = (frame1 + 1) / 2
                frame2 = (frame2 + 1) / 2
                
                score = self.lpips_fn(frame1, frame2).item()
                lpips_scores.append(score)
        
        return float(np.mean(lpips_scores))
    
    def compute_ssim(self, video1: torch.Tensor, video2: torch.Tensor) -> float:
        """Compute average SSIM between two videos"""
        from skimage.metrics import structural_similarity as ssim
        
        min_frames = min(video1.shape[1], video2.shape[1])
        ssim_scores = []
        
        video1_np = video1.cpu().numpy()
        video2_np = video2.cpu().numpy()
        
        for i in range(min_frames):
            frame1 = video1_np[:, i].transpose(1, 2, 0)
            frame2 = video2_np[:, i].transpose(1, 2, 0)
            
            
            frame1 = (frame1 + 1) / 2
            frame2 = (frame2 + 1) / 2
            
            score = ssim(frame1, frame2, channel_axis=2, data_range=1.0)
            ssim_scores.append(score)
        
        return float(np.mean(ssim_scores))
    
    def compute_temporal_consistency(self, video: torch.Tensor) -> float:
        """Compute temporal consistency using optical flow variance"""
        video_np = video.cpu().numpy()
        C, T, H, W = video_np.shape
        
       
        video_np = np.clip((video_np + 1) * 127.5, 0, 255).astype(np.uint8)
        video_np = video_np.transpose(1, 2, 3, 0)  
        
        flow_magnitudes = []
        
        for t in range(T - 1):
            frame1 = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(video_np[t + 1], cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(np.mean(magnitude))
        
        
        flow_variance = np.var(flow_magnitudes)
        consistency = 1.0 / (1.0 + flow_variance)
        
        return float(consistency)
    
    def compute_motion_smoothness(self, video: torch.Tensor) -> float:
        """Compute motion smoothness via second derivative"""
        T = video.shape[1]
        if T < 3:
            return 0.0
        
        
        first_diff = video[:, 1:] - video[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
       
        acceleration = torch.norm(second_diff, p=2, dim=(0, 2, 3))
        smoothness = 1.0 / (1.0 + float(torch.mean(acceleration).item()))
        
        return smoothness
    
    def detect_boundary_artifacts(self, video: torch.Tensor, 
                                 boundaries: List[int]) -> float:
        """Detect artifacts at frame-pack boundaries"""
        if not boundaries:
            return 0.0
        
        boundary_scores = []
        non_boundary_scores = []
        
        with torch.no_grad():
            
            for boundary in boundaries:
                if boundary >= video.shape[1] - 1 or boundary <= 0:
                    continue
                
                frame_before = video[:, boundary-1].unsqueeze(0).to(self.device)
                frame_after = video[:, boundary].unsqueeze(0).to(self.device)
                
                
                frame_before = (frame_before + 1) / 2
                frame_after = (frame_after + 1) / 2
                
               
                diff = self.lpips_fn(frame_before, frame_after).item()
                boundary_scores.append(diff)
            
            
            for _ in range(len(boundaries)):
                t = np.random.randint(1, video.shape[1] - 1)
                if t not in boundaries:
                    frame_before = video[:, t-1].unsqueeze(0).to(self.device)
                    frame_after = video[:, t].unsqueeze(0).to(self.device)
                    
                    
                    frame_before = (frame_before + 1) / 2
                    frame_after = (frame_after + 1) / 2
                    
                    diff = self.lpips_fn(frame_before, frame_after).item()
                    non_boundary_scores.append(diff)
        
        if not boundary_scores or not non_boundary_scores:
            return 0.0
        
        
        artifact_score = np.mean(boundary_scores) / (np.mean(non_boundary_scores) + 1e-6)
        
        return float(artifact_score)
    
    def compare_videos(self, 
                      original_path: str,
                      framepack_path: str,
                      boundaries: Optional[List[int]] = None,
                      generation_time_original: Optional[float] = None,
                      generation_time_framepack: Optional[float] = None) -> ComparisonResults:
        """Compare a single pair of videos"""
        
        self.logger.info(f"Comparing videos:")
        self.logger.info(f"  Original: {original_path}")
        self.logger.info(f"  Frame-packed: {framepack_path}")
        
        
        video_orig = self.load_video(original_path)
        video_fp = self.load_video(framepack_path)
        
        self.logger.info(f"Original shape: {video_orig.shape}")
        self.logger.info(f"Frame-packed shape: {video_fp.shape}")
        
       
        fvd = self.compute_fvd([video_orig], [video_fp])
        
        
        lpips_diff = self.compute_lpips_similarity(video_orig, video_fp)
        ssim_diff = self.compute_ssim(video_orig, video_fp)
        
        
        temporal_orig = self.compute_temporal_consistency(video_orig)
        temporal_fp = self.compute_temporal_consistency(video_fp)
        temporal_improvement = (temporal_fp - temporal_orig) / (temporal_orig + 1e-6)
        
        
        smooth_orig = self.compute_motion_smoothness(video_orig)
        smooth_fp = self.compute_motion_smoothness(video_fp)
        
       
        if boundaries:
            boundary_artifacts = self.detect_boundary_artifacts(video_fp, boundaries)
        else:
           
            boundaries = self.auto_detect_boundaries(video_fp.shape[1])
            boundary_artifacts = self.detect_boundary_artifacts(video_fp, boundaries)
        
        
        section_variance = self.compute_section_quality_variance(video_fp, boundaries)
        
       
        quality_retention = (1 - lpips_diff) * ssim_diff * (temporal_fp / temporal_orig)
        
       
        length_improvement = video_fp.shape[1] / video_orig.shape[1]
        
       
        if generation_time_original and generation_time_framepack:
            speedup = generation_time_original / generation_time_framepack
        else:
            speedup = 0.0
        
        results = ComparisonResults(
            fvd_score=fvd,
            quality_retention=quality_retention,
            lpips_difference=lpips_diff,
            ssim_difference=ssim_diff,
            temporal_consistency_original=temporal_orig,
            temporal_consistency_framepack=temporal_fp,
            temporal_improvement=temporal_improvement,
            motion_smoothness_original=smooth_orig,
            motion_smoothness_framepack=smooth_fp,
            generation_time_original=generation_time_original or 0.0,
            generation_time_framepack=generation_time_framepack or 0.0,
            speedup=speedup,
            max_frames_original=video_orig.shape[1],
            max_frames_framepack=video_fp.shape[1],
            length_improvement=length_improvement,
            boundary_artifacts=boundary_artifacts,
            section_quality_variance=section_variance
        )
        
        return results
    
    def auto_detect_boundaries(self, num_frames: int,
                              initial_frames: int = 41,
                              generation_frames: int = 30,
                              context_frames: int = 11) -> List[int]:
        """Auto-detect frame-pack boundaries based on typical parameters"""
        boundaries = []
        
        if num_frames <= initial_frames:
            return boundaries
        
        boundaries.append(initial_frames)
        
        advance_frames = generation_frames - context_frames
        current_pos = initial_frames
        
        while current_pos + advance_frames < num_frames:
            current_pos += advance_frames
            boundaries.append(current_pos)
        
        return boundaries
    
    def compute_section_quality_variance(self, video: torch.Tensor, 
                                        boundaries: List[int]) -> float:
        """Compute variance in quality across sections"""
        section_qualities = []
        
        boundaries_ext = [0] + boundaries + [video.shape[1]]
        
        for i in range(len(boundaries_ext) - 1):
            start = boundaries_ext[i]
            end = boundaries_ext[i + 1]
            
            if end - start < 3:
                continue
            
            section = video[:, start:end]
            
            
            smoothness = self.compute_motion_smoothness(section)
            temporal = self.compute_temporal_consistency(section)
            
            quality = (temporal + smoothness) / 2
            section_qualities.append(quality)
        
        if len(section_qualities) < 2:
            return 0.0
        
        return float(np.var(section_qualities))
    
    def batch_compare(self, 
                     original_videos: List[str],
                     framepack_videos: List[str],
                     save_results: bool = True) -> Dict:
        """Compare multiple pairs of videos"""
        
        assert len(original_videos) == len(framepack_videos), \
            "Must have same number of original and frame-packed videos"
        
        all_results = []
        
        
        all_orig_tensors = []
        all_fp_tensors = []
        
        for orig_path, fp_path in tqdm(zip(original_videos, framepack_videos), 
                                       desc="Comparing video pairs",
                                       total=len(original_videos)):
            try:
                
                video_orig = self.load_video(orig_path)
                video_fp = self.load_video(fp_path)
                all_orig_tensors.append(video_orig)
                all_fp_tensors.append(video_fp)
                
                
                result = self.compare_videos(orig_path, fp_path)
                all_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error comparing {orig_path} vs {fp_path}: {e}")
                continue
        
        if not all_results:
            return {'error': 'No videos successfully compared'}
        
       
        self.logger.info("Computing overall FVD across all videos...")
        overall_fvd = self.compute_fvd(all_orig_tensors, all_fp_tensors)
        
        
        aggregated = self._aggregate_comparison_results(all_results, overall_fvd)
        
        
        if save_results:
            output_file = self.save_dir / 'comparison_results.json'
            with open(output_file, 'w') as f:
                json.dump(aggregated, f, indent=2)
            self.logger.info(f"Results saved to: {output_file}")
            
            
            self._generate_comparison_plots(aggregated)
        
        return aggregated
    
    def _aggregate_comparison_results(self, results: List[ComparisonResults], 
                                     overall_fvd: float) -> Dict:
        """Aggregate comparison results"""
        
        result_dicts = [r.to_dict() for r in results]
        
        aggregated = {
            'num_comparisons': len(results),
            'overall_fvd': overall_fvd,
            'summary': {
                'quality': {
                    'mean_quality_retention': np.mean([r.quality_retention for r in results]),
                    'mean_lpips_difference': np.mean([r.lpips_difference for r in results]),
                    'mean_ssim_difference': np.mean([r.ssim_difference for r in results]),
                },
                'temporal': {
                    'mean_consistency_original': np.mean([r.temporal_consistency_original for r in results]),
                    'mean_consistency_framepack': np.mean([r.temporal_consistency_framepack for r in results]),
                    'mean_improvement': np.mean([r.temporal_improvement for r in results]),
                },
                'efficiency': {
                    'mean_speedup': np.mean([r.speedup for r in results if r.speedup > 0]),
                    'mean_length_improvement': np.mean([r.length_improvement for r in results]),
                },
                'artifacts': {
                    'mean_boundary_artifacts': np.mean([r.boundary_artifacts for r in results]),
                    'mean_section_variance': np.mean([r.section_quality_variance for r in results]),
                }
            },
            'individual_results': result_dicts,
            'verdict': self._generate_verdict(results, overall_fvd)
        }
        
        return aggregated
    
    def _generate_verdict(self, results: List[ComparisonResults], 
                         overall_fvd: float) -> Dict:
        """Generate overall assessment"""
        
        mean_quality_retention = np.mean([r.quality_retention for r in results])
        mean_speedup = np.mean([r.speedup for r in results if r.speedup > 0])
        mean_length_improvement = np.mean([r.length_improvement for r in results])
        mean_boundary_artifacts = np.mean([r.boundary_artifacts for r in results])
        
        verdict = {
            'overall_assessment': '',
            'strengths': [],
            'weaknesses': [],
            'recommendation': ''
        }
        
        
        if overall_fvd < 50:
            verdict['strengths'].append(f"Excellent FVD score ({overall_fvd:.2f}) - nearly identical distribution")
        elif overall_fvd < 100:
            verdict['strengths'].append(f"Good FVD score ({overall_fvd:.2f}) - similar distribution")
        elif overall_fvd < 200:
            verdict['weaknesses'].append(f"Moderate FVD score ({overall_fvd:.2f}) - some distribution shift")
        else:
            verdict['weaknesses'].append(f"High FVD score ({overall_fvd:.2f}) - significant distribution shift")
        
        
        if mean_quality_retention > 0.9:
            verdict['strengths'].append(f"Excellent quality retention ({mean_quality_retention:.2%})")
        elif mean_quality_retention > 0.8:
            verdict['strengths'].append(f"Good quality retention ({mean_quality_retention:.2%})")
        else:
            verdict['weaknesses'].append(f"Quality degradation ({mean_quality_retention:.2%})")
        
       
        if mean_speedup > 1.5:
            verdict['strengths'].append(f"Significant speedup ({mean_speedup:.1f}x faster)")
        if mean_length_improvement > 2:
            verdict['strengths'].append(f"Major length improvement ({mean_length_improvement:.1f}x longer videos)")
        
       
        if mean_boundary_artifacts < 1.2:
            verdict['strengths'].append("Minimal boundary artifacts")
        elif mean_boundary_artifacts > 1.5:
            verdict['weaknesses'].append("Noticeable boundary artifacts")
        
       
        if overall_fvd < 100 and mean_quality_retention > 0.85 and mean_speedup > 1.5:
            verdict['overall_assessment'] = "EXCELLENT - Frame-packing successfully improves efficiency with minimal quality loss"
            verdict['recommendation'] = "Ready for production deployment"
        elif overall_fvd < 150 and mean_quality_retention > 0.75:
            verdict['overall_assessment'] = "GOOD - Frame-packing provides clear benefits with acceptable trade-offs"
            verdict['recommendation'] = "Suitable for most use cases"
        elif overall_fvd < 200 and mean_quality_retention > 0.65:
            verdict['overall_assessment'] = "ACCEPTABLE - Frame-packing works but with noticeable quality impact"
            verdict['recommendation'] = "Consider tuning parameters to improve quality"
        else:
            verdict['overall_assessment'] = "NEEDS IMPROVEMENT - Significant quality degradation"
            verdict['recommendation'] = "Review frame-packing implementation and parameters"
        
        return verdict
    
    def _generate_comparison_plots(self, results: Dict):
        """Generate comprehensive visualization plots"""
        
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        
        self._create_comprehensive_visualization(results)
        
        
        self._create_individual_plots(results)
        
        self.logger.info(f"All visualizations saved to: {self.save_dir}")
    
    def _create_comprehensive_visualization(self, results: Dict):
        """Create comprehensive visualization dashboard"""
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as mpatches
        
        
        if 'overall_fvd' in results:
            
            fvd_score = results['overall_fvd']
            quality_retention = results['summary']['quality']['mean_quality_retention']
            lpips_diff = results['summary']['quality']['mean_lpips_difference']
            ssim_diff = results['summary']['quality']['mean_ssim_difference']
            temporal_orig = results['summary']['temporal']['mean_consistency_original']
            temporal_fp = results['summary']['temporal']['mean_consistency_framepack']
            temporal_improvement = results['summary']['temporal']['mean_improvement']
            length_improvement = results['summary']['efficiency']['mean_length_improvement']
            boundary_artifacts = results['summary']['artifacts']['mean_boundary_artifacts']
            section_variance = results['summary']['artifacts']['mean_section_variance']
            
            frames_orig = 121  
            frames_fp = int(frames_orig * length_improvement)
        else:
           
            fvd_score = results.get('fvd_score', 0)
            quality_retention = results.get('quality_retention', 0)
            lpips_diff = results.get('lpips_difference', 0)
            ssim_diff = results.get('ssim_difference', 0)
            temporal = results.get('temporal_consistency', {})
            temporal_orig = temporal.get('original', 0)
            temporal_fp = temporal.get('framepack', 0)
            temporal_improvement = temporal.get('improvement', 0)
            length = results.get('length_capability', {})
            frames_orig = length.get('max_frames_original', 0)
            frames_fp = length.get('max_frames_framepack', 0)
            length_improvement = length.get('improvement_factor', 0)
            framepack = results.get('framepack_specific', {})
            boundary_artifacts = framepack.get('boundary_artifacts', 0)
            section_variance = framepack.get('section_variance', 0)
        
        motion = results.get('motion_smoothness', {})
        motion_orig = motion.get('original', 0) if motion else 0
        motion_fp = motion.get('framepack', 0) if motion else 0
        
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        
        ax1 = fig.add_subplot(gs[0, 0:2])
        fvd_color = '#2ecc71' if fvd_score < 50 else '#f39c12' if fvd_score < 100 else '#e74c3c'
        bars = ax1.barh(['Your Model'], [fvd_score], color=fvd_color, height=0.3)
        
       
        ax1.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='Excellent (<50)')
        ax1.axvline(x=100, color='orange', linestyle='--', alpha=0.5, label='Good (<100)')
        ax1.axvline(x=200, color='red', linestyle='--', alpha=0.5, label='Acceptable (<200)')
        
        
        ax1.text(fvd_score + 2, 0, f'{fvd_score:.2f}', va='center', fontweight='bold', fontsize=14)
        
        ax1.set_xlabel('FVD Score', fontsize=12)
        ax1.set_title('Fréchet Video Distance (Lower is Better)', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, max(250, fvd_score + 50))
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = ['LPIPS\n(Lower Better)', 'SSIM\n(Higher Better)', 'Quality\nRetention']
        values = [lpips_diff, ssim_diff, quality_retention]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Quality Preservation Metrics', fontsize=14, fontweight='bold')
        
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2%}' if val < 1 else f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(True, axis='y', alpha=0.3)
        
        
        ax3 = fig.add_subplot(gs[0, 3])
        models = ['Original', 'Frame-Packed']
        temporal_values = [temporal_orig, temporal_fp]
        colors = ['#95a5a6', '#9b59b6']
        
        bars = ax3.bar(models, temporal_values, color=colors, alpha=0.8)
        
       
        if temporal_improvement > 0:
            ax3.annotate('', xy=(1, temporal_fp), xytext=(0, temporal_orig),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax3.text(0.5, (temporal_orig + temporal_fp)/2, f'+{temporal_improvement:.1%}',
                    ha='center', va='bottom', color='green', fontweight='bold', fontsize=12)
        
        ax3.set_ylabel('Consistency Score', fontsize=12)
        ax3.set_title('Temporal Consistency', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        
        
        for bar, val in zip(bars, temporal_values):
            ax3.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, axis='y', alpha=0.3)
        
        
        ax4 = fig.add_subplot(gs[1, 0])
        length_data = [frames_orig, frames_fp]
        models = ['Original', 'Frame-Packed']
        colors = ['#34495e', '#e67e22']
        
        bars = ax4.bar(models, length_data, color=colors, alpha=0.8)
        
        ax4.set_ylabel('Number of Frames', fontsize=12)
        ax4.set_title(f'Video Length Capability ({length_improvement:.1f}x Improvement)', 
                      fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, length_data):
            ax4.text(bar.get_x() + bar.get_width()/2., val + 2,
                    f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax4.grid(True, axis='y', alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        models = ['Original', 'Frame-Packed']
        motion_values = [motion_orig, motion_fp]
        colors = ['#16a085', '#8e44ad']
        
        bars = ax5.bar(models, motion_values, color=colors, alpha=0.8)
        
        ax5.set_ylabel('Smoothness Score', fontsize=12)
        ax5.set_title('Motion Smoothness (Higher is Better)', fontsize=14, fontweight='bold')
        ax5.set_ylim(0, max(motion_values) * 1.2 if max(motion_values) > 0 else 1)
        
        for bar, val in zip(bars, motion_values):
            ax5.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax5.grid(True, axis='y', alpha=0.3)
       
        ax6 = fig.add_subplot(gs[1, 2])
        issues = ['Boundary\nArtifacts', 'Section\nVariance']
        issue_values = [boundary_artifacts, section_variance * 100] 
        
        colors = []
        for i, val in enumerate([boundary_artifacts, section_variance]):
            if i == 0:  
                if val < 1.2:
                    colors.append('#2ecc71')  
                elif val < 1.5:
                    colors.append('#f39c12')  
                else:
                    colors.append('#e74c3c') 
            else:  
                colors.append('#3498db')  
        
        bars = ax6.bar(issues, issue_values, color=colors, alpha=0.8)
        
        ax6.set_ylabel('Score', fontsize=12)
        ax6.set_title('Frame-Pack Specific Metrics', fontsize=14, fontweight='bold')
        
        
        ax6.text(0, issue_values[0] + 0.05, f'{boundary_artifacts:.2f}',
                ha='center', va='bottom', fontweight='bold')
        ax6.text(1, issue_values[1] + 0.05, f'{section_variance:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        
        ax6.axhline(y=1.5, color='red', linestyle='--', alpha=0.3, label='Artifact Threshold')
        ax6.legend()
        ax6.grid(True, axis='y', alpha=0.3)
        
    
        
        fvd_norm = max(0, min(1, 1 - (fvd_score / 200)))  
        artifacts_norm = max(0, min(1, 1 - (boundary_artifacts - 1) / 2))  
        length_norm = min(1, length_improvement / 4)  
        
        values = [fvd_norm, quality_retention, temporal_fp, length_norm, artifacts_norm]
        values += values[:1] 
        

        ax8 = fig.add_subplot(gs[2, :2])
        ax8.axis('off')
        
        
        summary_text = f"""
    EXECUTIVE SUMMARY
    ═══════════════════════════════════════
    
    ✅ PRIMARY SUCCESS METRIC
    • FVD Score: {fvd_score:.2f} ({"Excellent" if fvd_score < 50 else "Good" if fvd_score < 100 else "Moderate"})
    
    📊 KEY IMPROVEMENTS
    • Video Length: {length_improvement:.1f}x longer ({frames_orig} → {frames_fp} frames)
    • Temporal Consistency: {temporal_improvement:+.1%} improvement
    • Motion Smoothness: {'✓ Maintained' if motion_fp >= motion_orig * 0.9 else '⚠ Degraded'}
    
    ⚠️ AREAS FOR OPTIMIZATION
    • Boundary Artifacts: {'Minimal' if boundary_artifacts < 1.2 else 'Moderate' if boundary_artifacts < 1.5 else 'Significant'} ({boundary_artifacts:.2f})
    • Frame Similarity: {quality_retention:.1%} retention (target: >70%)
    
    
    """
        
        ax8.text(0.5, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontfamily='monospace')
        
        
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis('off')
        
        
        grade_score = (fvd_norm * 0.4 + quality_retention * 0.2 + 
                      temporal_fp * 0.2 + length_norm * 0.1 + artifacts_norm * 0.1)
        
        
        breakdown_text = f"""
    Scoring Breakdown:
    ─────────────────
    FVD Performance:    {fvd_norm:.0%}
    Quality Retention:  {quality_retention:.0%}
    Temporal Quality:   {temporal_fp:.0%}
    Artifact Control:   {artifacts_norm:.0%}
   
    """
        
        ax9.text(0.65, 0, breakdown_text, transform=ax9.transAxes,
                fontsize=10, va='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        
        fig.suptitle('Frame-Pack Video Generation Evaluation Report', 
                    fontsize=18, fontweight='bold', y=0.98)
        
       
        output_path = self.save_dir / 'comprehensive_evaluation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comprehensive visualization saved to: {output_path}")
    
    def _create_individual_plots(self, results: Dict):
        """Create individual plots for modular use in presentations"""
        
        
        if 'overall_fvd' in results:
            
            fvd_score = results['overall_fvd']
            temporal_orig = results['summary']['temporal']['mean_consistency_original']
            temporal_fp = results['summary']['temporal']['mean_consistency_framepack']
            temporal_improvement = results['summary']['temporal']['mean_improvement']
            length_improvement = results['summary']['efficiency']['mean_length_improvement']
            frames_orig = 121 
            frames_fp = int(frames_orig * length_improvement)
        else:
            
            fvd_score = results.get('fvd_score', 0)
            temporal = results.get('temporal_consistency', {})
            temporal_orig = temporal.get('original', 0)
            temporal_fp = temporal.get('framepack', 0)
            temporal_improvement = temporal.get('improvement', 0)
            length = results.get('length_capability', {})
            frames_orig = length.get('max_frames_original', 0)
            frames_fp = length.get('max_frames_framepack', 0)
            length_improvement = length.get('improvement_factor', 0)
        
        motion = results.get('motion_smoothness', {})
        motion_orig = motion.get('original', 0) if motion else 0
        motion_fp = motion.get('framepack', 0) if motion else 0
        
      
        fig, ax = plt.subplots(figsize=(10, 6))
        
        
        benchmarks = {
            'Your Model': fvd_score,
            'Typical Good Model': 75,
            'Typical Acceptable': 150,
            'Poor Quality': 250
        }
        
        colors = ['#2ecc71' if v < 50 else '#f39c12' if v < 100 else '#e67e22' if v < 200 else '#e74c3c' 
                  for v in benchmarks.values()]
        
        bars = ax.bar(benchmarks.keys(), benchmarks.values(), color=colors, alpha=0.8)
        
        
        bars[0].set_edgecolor('black')
        bars[0].set_linewidth(3)
        
        ax.set_ylabel('FVD Score', fontsize=14)
        ax.set_title('Fréchet Video Distance Comparison\n(Lower is Better)', fontsize=16, fontweight='bold')
        ax.axhline(y=50, color='green', linestyle='--', alpha=0.3, label='Excellent Threshold')
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.3, label='Good Threshold')
        
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'fvd_comparison.png', dpi=150)
        plt.close()
        
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
       
        ax = axes[0]
        models = ['Original', 'Frame-Packed']
        values = [temporal_orig, temporal_fp]
        colors = ['#95a5a6', '#9b59b6']
        bars = ax.bar(models, values, color=colors, alpha=0.8)
        
        if temporal_improvement > 0:
            ax.text(0.5, max(values) * 1.1, f'+{temporal_improvement:.1%}', 
                   ha='center', fontsize=14, fontweight='bold', color='green')
        
        ax.set_title('Temporal Consistency', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.grid(True, axis='y', alpha=0.3)

        ax = axes[1]
        values = [frames_orig, frames_fp]
        bars = ax.bar(models, values, color=['#34495e', '#e67e22'], alpha=0.8)
        
        ax.text(0.5, max(values) * 1.05, f'{length_improvement:.1f}x longer', 
               ha='center', fontsize=14, fontweight='bold', color='green')
        
        ax.set_title('Video Length Capability', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frames', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        
        ax = axes[2]
        values = [motion_orig, motion_fp]
        bars = ax.bar(models, values, color=['#16a085', '#8e44ad'], alpha=0.8)
        
        ax.set_title('Motion Smoothness', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        fig.suptitle('Original vs Frame-Packed Model Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'before_after_comparison.png', dpi=150)
        plt.close()
        
        self.logger.info(f"Individual plots saved to: {self.save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Compare original vs frame-packed video generation')
    
   
    parser.add_argument('--original', type=str, nargs='+', required=True,
                       help='Path(s) to original model videos')
    parser.add_argument('--framepack', type=str, nargs='+', required=True,
                       help='Path(s) to frame-packed model videos')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--boundaries', type=int, nargs='*',
                       help='Frame-pack boundaries (if known)')
    parser.add_argument('--time_original', type=float,
                       help='Generation time for original model (seconds)')
    parser.add_argument('--time_framepack', type=float,
                       help='Generation time for frame-packed model (seconds)')
    
    args = parser.parse_args()
    
   
    if len(args.original) != len(args.framepack):
        raise ValueError("Must provide equal number of original and frame-packed videos")
    
    
    evaluator = ComparativeVideoEvaluator(
        device=args.device,
        save_dir=args.save_dir
    )
    
    
    if len(args.original) == 1:
        
        results = evaluator.compare_videos(
            args.original[0],
            args.framepack[0],
            boundaries=args.boundaries,
            generation_time_original=args.time_original,
            generation_time_framepack=args.time_framepack
        )
        
       
        print("\n" + "="*70)
        print("VIDEO COMPARISON RESULTS")
        print("="*70)
        print(f"\nOriginal Video: {args.original[0]}")
        print(f"Frame-Packed Video: {args.framepack[0]}")
        
        print("\n📊 PRIMARY METRIC:")
        print(f"  FVD Score: {results.fvd_score:.2f}")
        if results.fvd_score < 50:
            print("  ✅ Excellent - Nearly identical distribution")
        elif results.fvd_score < 100:
            print("  ✅ Good - Similar distribution")
        elif results.fvd_score < 200:
            print("  ⚠️  Moderate - Some distribution shift")
        else:
            print("  ❌ Poor - Significant distribution shift")
        
        print("\n🎨 QUALITY METRICS:")
        print(f"  Quality Retention: {results.quality_retention:.2%}")
        print(f"  LPIPS Difference: {results.lpips_difference:.4f}")
        print(f"  SSIM: {results.ssim_difference:.4f}")
        
        print("\n🎬 TEMPORAL METRICS:")
        print(f"  Original Consistency: {results.temporal_consistency_original:.4f}")
        print(f"  Frame-Pack Consistency: {results.temporal_consistency_framepack:.4f}")
        print(f"  Improvement: {results.temporal_improvement:+.2%}")
        
        print("\n⚡ EFFICIENCY METRICS:")
        print(f"  Max Frames - Original: {results.max_frames_original}")
        print(f"  Max Frames - Frame-Pack: {results.max_frames_framepack}")
        print(f"  Length Improvement: {results.length_improvement:.1f}x")
        if results.speedup > 0:
            print(f"  Speedup: {results.speedup:.1f}x")
        
        print("\n🔍 FRAME-PACK SPECIFIC:")
        print(f"  Boundary Artifacts: {results.boundary_artifacts:.3f}")
        if results.boundary_artifacts < 1.2:
            print("  ✅ Minimal boundary artifacts")
        elif results.boundary_artifacts < 1.5:
            print("  ⚠️  Some boundary artifacts")
        else:
            print("  ❌ Noticeable boundary artifacts")
        print(f"  Section Quality Variance: {results.section_quality_variance:.4f}")
        
        
        output_file = evaluator.save_dir / 'single_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
        
        evaluator._generate_comparison_plots(results.to_dict())
    
    else:
       
        results = evaluator.batch_compare(
            args.original,
            args.framepack,
            save_results=True
        )
        
       
        print("\n" + "="*70)
        print("BATCH COMPARISON SUMMARY")
        print("="*70)
        print(f"\nVideos Compared: {results['num_comparisons']}")
        
        print("\n📊 OVERALL FVD SCORE: {:.2f}".format(results['overall_fvd']))
        
        print("\n🎨 QUALITY SUMMARY:")
        print(f"  Mean Quality Retention: {results['summary']['quality']['mean_quality_retention']:.2%}")
        print(f"  Mean LPIPS Difference: {results['summary']['quality']['mean_lpips_difference']:.4f}")
        print(f"  Mean SSIM: {results['summary']['quality']['mean_ssim_difference']:.4f}")
        
        print("\n🎬 TEMPORAL SUMMARY:")
        print(f"  Original Consistency: {results['summary']['temporal']['mean_consistency_original']:.4f}")
        print(f"  Frame-Pack Consistency: {results['summary']['temporal']['mean_consistency_framepack']:.4f}")
        print(f"  Mean Improvement: {results['summary']['temporal']['mean_improvement']:+.2%}")
        
        print("\n⚡ EFFICIENCY SUMMARY:")
        print(f"  Mean Speedup: {results['summary']['efficiency']['mean_speedup']:.1f}x")
        print(f"  Mean Length Improvement: {results['summary']['efficiency']['mean_length_improvement']:.1f}x")
        
        print("\n🔍 ARTIFACTS SUMMARY:")
        print(f"  Mean Boundary Artifacts: {results['summary']['artifacts']['mean_boundary_artifacts']:.3f}")
        print(f"  Mean Section Variance: {results['summary']['artifacts']['mean_section_variance']:.4f}")
        
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        print(f"\n{results['verdict']['overall_assessment']}")
        
        if results['verdict']['strengths']:
            print("\n✅ Strengths:")
            for strength in results['verdict']['strengths']:
                print(f"  • {strength}")
        
        if results['verdict']['weaknesses']:
            print("\n❌ Weaknesses:")
            for weakness in results['verdict']['weaknesses']:
                print(f"  • {weakness}")
        
        print(f"\n💡 Recommendation: {results['verdict']['recommendation']}")
        
        print(f"\n💾 Full results saved to: {evaluator.save_dir}")
        print(f"   - comparison_results.json")
        print(f"   - comparison_plots.png")


if __name__ == "__main__":
    main()