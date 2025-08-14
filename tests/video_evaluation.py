
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


@dataclass
class EvaluationResults:
 
    fd_mean: float
    fd_std: float
    temporal_consistency: float
    motion_smoothness: float
    section_quality_variance: float
    
    def to_dict(self):
        return {
            'frame_discrepancy_mean': self.fd_mean,
            'frame_discrepancy_std': self.fd_std,
            'temporal_consistency': self.temporal_consistency,
            'motion_smoothness': self.motion_smoothness,
            'section_quality_variance': self.section_quality_variance
        }


class VideoEvaluator:
    
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
       
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
      
        import torchvision.models as models
        self.feature_extractor = models.resnet50(pretrained=True).to(self.device)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.eval()
       
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def compute_frame_discrepancy(self, video: torch.Tensor, 
                                  section_boundaries: List[int]) -> Tuple[float, float]:
       
        discrepancies = []
        
        with torch.no_grad():
            for boundary in section_boundaries:
                if boundary >= video.shape[1] - 1 or boundary <= 0:
                    continue
               
                frame_before = video[:, boundary-1].unsqueeze(0).to(self.device)
                frame_after = video[:, boundary].unsqueeze(0).to(self.device)
                
              
                frame_before = self.preprocess(frame_before)
                frame_after = self.preprocess(frame_after)
               
                feat_before = self.feature_extractor(frame_before)
                feat_after = self.feature_extractor(frame_after)
           
                discrepancy = torch.norm(feat_before - feat_after, p=2).item()
                discrepancies.append(discrepancy)
        
        if not discrepancies:
            return 0.0, 0.0
        
        return float(np.mean(discrepancies)), float(np.std(discrepancies))
    
    def compute_temporal_consistency(self, video: torch.Tensor) -> float:
        
        video_np = video.cpu().numpy()
        C, T, H, W = video_np.shape
        
       
        video_np = np.clip((video_np + 1) * 127.5, 0, 255).astype(np.uint8)
        video_np = video_np.transpose(1, 2, 3, 0)
        
        flow_magnitudes = []
        
        for t in range(min(T - 1, 100)):  
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
      
        T = video.shape[1]
        if T < 3:
            return 0.0
        
     
        first_diff = video[:, 1:] - video[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        
        acceleration = torch.norm(second_diff, p=2, dim=(0, 2, 3))
        smoothness = float(torch.mean(acceleration).item())
        
        return smoothness
    
    def compute_section_quality_variance(self, video: torch.Tensor, 
                                        section_boundaries: List[int]) -> float:
      
        section_qualities = []
        
       
        boundaries = [0] + section_boundaries + [video.shape[1]]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            if end - start < 3: 
                continue
            
            section = video[:, start:end]
            
            smoothness = self.compute_motion_smoothness(section)
            temporal = self.compute_temporal_consistency(section)
            
            quality = temporal / (1.0 + smoothness)
            section_qualities.append(quality)
        
        if len(section_qualities) < 2:
            return 0.0
        
        return float(np.var(section_qualities))


class FramepackVideoEvaluator:
   
    
    def __init__(self, device='cuda', save_dir='evaluation_results'):
        self.device = device
        self.evaluator = VideoEvaluator(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_section_boundaries(self, num_frames: int, 
                              initial_frames: int = 41,
                              generation_frames: int = 30,
                              context_frames: int = 11) -> List[int]:
        
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
    
    def load_video(self, video_path: str) -> torch.Tensor:
       
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path)
        
      
        video = vr[:].float()  
        
        video = video.permute(3, 0, 1, 2)
        
        video = video / 127.5 - 1.0
        
        return video
    
    def evaluate_video(self, 
                       video_path: str,
                       custom_boundaries: Optional[List[int]] = None,
                       save_results: bool = True) -> Dict:
        
        self.logger.info(f"Evaluating video: {video_path}")
        
       
        video = self.load_video(video_path)
        C, T, H, W = video.shape
        
        self.logger.info(f"Video shape: {video.shape}")
        
       
        if custom_boundaries is None:
            boundaries = self.get_section_boundaries(T)
        else:
            boundaries = custom_boundaries
        
        self.logger.info(f"Section boundaries: {boundaries}")
        
        
        self.logger.info("Computing Frame Discrepancy...")
        fd_mean, fd_std = self.evaluator.compute_frame_discrepancy(video, boundaries)
        
        self.logger.info("Computing Temporal Consistency...")
        temporal_consistency = self.evaluator.compute_temporal_consistency(video)
        
        self.logger.info("Computing Motion Smoothness...")
        motion_smoothness = self.evaluator.compute_motion_smoothness(video)
        
        self.logger.info("Computing Section Quality Variance...")
        section_variance = self.evaluator.compute_section_quality_variance(video, boundaries)
        
       
        self.logger.info("Analyzing individual sections...")
        section_metrics = []
        boundaries_ext = [0] + boundaries + [T]
        
        for i in range(len(boundaries_ext) - 1):
            start = boundaries_ext[i]
            end = boundaries_ext[i + 1]
            
            if end - start < 3:
                continue
            
            section = video[:, start:end]
            
            section_smooth = self.evaluator.compute_motion_smoothness(section)
            section_temporal = self.evaluator.compute_temporal_consistency(section)
            
            section_info = {
                'section_id': i,
                'start_frame': start,
                'end_frame': end,
                'num_frames': end - start,
                'smoothness': float(section_smooth),
                'temporal_consistency': float(section_temporal)
            }
            
           
            if i > 0:
                boundary_idx = boundaries_ext[i]
                section_info['boundary_frame'] = boundary_idx
                
                
                if boundary_idx in boundaries:
                    fd_single = self.evaluator.compute_frame_discrepancy(
                        video, [boundary_idx]
                    )[0]
                    section_info['boundary_discrepancy'] = float(fd_single)
            
            section_metrics.append(section_info)
        
       
        results = {
            'video_path': str(video_path),
            'video_info': {
                'num_frames': T,
                'height': H,
                'width': W,
                'fps': 24  
            },
            'boundaries': boundaries,
            'metrics': {
                'frame_discrepancy_mean': float(fd_mean),
                'frame_discrepancy_std': float(fd_std),
                'temporal_consistency': float(temporal_consistency),
                'motion_smoothness': float(motion_smoothness),
                'section_quality_variance': float(section_variance)
            },
            'per_section': section_metrics,
            'quality_assessment': self._assess_quality(fd_mean, temporal_consistency, motion_smoothness)
        }
        
       
        if save_results:
            output_file = self.save_dir / f"{Path(video_path).stem}_evaluation.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {output_file}")
        
        return results
    
    def _assess_quality(self, fd_mean: float, temporal: float, smoothness: float) -> Dict:
        
        assessment = {
            'overall': '',
            'continuity': '',
            'motion': '',
            'recommendations': []
        }
        
        
        if fd_mean < 0.3:
            assessment['continuity'] = 'Excellent - Seamless transitions'
        elif fd_mean < 0.6:
            assessment['continuity'] = 'Good - Minor discontinuities'
        elif fd_mean < 1.0:
            assessment['continuity'] = 'Fair - Noticeable boundaries'
        else:
            assessment['continuity'] = 'Poor - Strong discontinuities'
            assessment['recommendations'].append('Increase context_scale or context frames')
       
        if temporal > 0.8 and smoothness < 0.5:
            assessment['motion'] = 'Excellent - Smooth and consistent'
        elif temporal > 0.6 and smoothness < 1.0:
            assessment['motion'] = 'Good - Generally smooth'
        elif temporal > 0.4:
            assessment['motion'] = 'Fair - Some jerkiness'
            assessment['recommendations'].append('Consider increasing sampling steps')
        else:
            assessment['motion'] = 'Poor - Jerky motion'
            assessment['recommendations'].append('Review motion generation parameters')
        
        quality_score = (temporal * 2 + (1.0 / (1.0 + fd_mean)) + (1.0 / (1.0 + smoothness))) / 4
        
        if quality_score > 0.7:
            assessment['overall'] = 'High Quality'
        elif quality_score > 0.5:
            assessment['overall'] = 'Good Quality'
        elif quality_score > 0.3:
            assessment['overall'] = 'Acceptable Quality'
        else:
            assessment['overall'] = 'Low Quality'
        
        assessment['quality_score'] = float(quality_score)
        
        return assessment
    
    def batch_evaluate(self, video_paths: List[str], save_summary: bool = True) -> Dict:
        
        all_results = []
        
        for video_path in tqdm(video_paths, desc="Evaluating videos"):
            try:
                result = self.evaluate_video(video_path, save_results=True)
                all_results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating {video_path}: {e}")
                continue
        
        if not all_results:
            return {'error': 'No videos successfully evaluated'}
        
       
        aggregated = self._aggregate_results(all_results)
        
        
        if save_summary:
            summary_file = self.save_dir / 'batch_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(aggregated, f, indent=2)
            self.logger.info(f"Summary saved to: {summary_file}")
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
      
        metrics_list = [r['metrics'] for r in results]
        
        aggregated = {
            'num_videos': len(results),
            'video_files': [r['video_path'] for r in results],
            'aggregated_metrics': {}
        }
        
        # Aggregate each metric
        for metric in ['frame_discrepancy_mean', 'frame_discrepancy_std', 
                      'temporal_consistency', 'motion_smoothness', 
                      'section_quality_variance']:
            values = [m[metric] for m in metrics_list]
            aggregated['aggregated_metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        
        quality_scores = [r['quality_assessment']['quality_score'] for r in results]
        aggregated['quality_distribution'] = {
            'high_quality': sum(1 for s in quality_scores if s > 0.7),
            'good_quality': sum(1 for s in quality_scores if 0.5 < s <= 0.7),
            'acceptable_quality': sum(1 for s in quality_scores if 0.3 < s <= 0.5),
            'low_quality': sum(1 for s in quality_scores if s <= 0.3),
            'average_score': float(np.mean(quality_scores))
        }
        
        return aggregated


def main():
    parser = argparse.ArgumentParser(description='Evaluate FramepackVace generated videos')
    parser.add_argument('video_path', type=str, nargs='+', 
                       help='Path(s) to video file(s) to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--boundaries', type=int, nargs='*',
                       help='Custom section boundaries (frame indices)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--initial_frames', type=int, default=41,
                       help='Number of frames in initial section')
    parser.add_argument('--generation_frames', type=int, default=30,
                       help='Number of frames generated per section')
    parser.add_argument('--context_frames', type=int, default=11,
                       help='Number of context frames')
    
    args = parser.parse_args()
    
   
    evaluator = FramepackVideoEvaluator(
        device=args.device,
        save_dir=args.save_dir
    )
    
  
    if len(args.video_path) == 1:
       
        results = evaluator.evaluate_video(
            args.video_path[0],
            custom_boundaries=args.boundaries,
            save_results=not args.no_save
        )
        
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Video: {results['video_path']}")
        print(f"Frames: {results['video_info']['num_frames']}")
        print(f"Size: {results['video_info']['width']}x{results['video_info']['height']}")
        print(f"Section boundaries: {results['boundaries']}")
        print("\nMETRICS:")
        print(f"  Frame Discrepancy: {results['metrics']['frame_discrepancy_mean']:.4f} ± {results['metrics']['frame_discrepancy_std']:.4f}")
        print(f"  Temporal Consistency: {results['metrics']['temporal_consistency']:.4f}")
        print(f"  Motion Smoothness: {results['metrics']['motion_smoothness']:.4f}")
        print(f"  Section Quality Variance: {results['metrics']['section_quality_variance']:.4f}")
        print("\nQUALITY ASSESSMENT:")
        print(f"  Overall: {results['quality_assessment']['overall']}")
        print(f"  Continuity: {results['quality_assessment']['continuity']}")
        print(f"  Motion: {results['quality_assessment']['motion']}")
        print(f"  Quality Score: {results['quality_assessment']['quality_score']:.3f}")
        
        if results['quality_assessment']['recommendations']:
            print("\nRECOMMENDATIONS:")
            for rec in results['quality_assessment']['recommendations']:
                print(f"  - {rec}")
    
    else:
      
        results = evaluator.batch_evaluate(
            args.video_path,
            save_summary=not args.no_save
        )
        
       
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Videos evaluated: {results['num_videos']}")
        print("\nAGGREGATED METRICS:")
        for metric, stats in results['aggregated_metrics'].items():
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
        
        print("\nQUALITY DISTRIBUTION:")
        dist = results['quality_distribution']
        print(f"  High Quality: {dist['high_quality']} videos")
        print(f"  Good Quality: {dist['good_quality']} videos")
        print(f"  Acceptable: {dist['acceptable_quality']} videos")
        print(f"  Low Quality: {dist['low_quality']} videos")
        print(f"  Average Score: {dist['average_score']:.3f}")


if __name__ == "__main__":
    main()