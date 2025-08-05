#!/usr/bin/env python3
"""
Enhanced Video Processor with Custom Basketball YOLO
Integrates custom-trained YOLO model for better basketball analysis
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

class EnhancedBasketballProcessor:
    """Enhanced basketball video processor using custom YOLO model."""
    
    def __init__(self, custom_model_path: str = None):
        """Initialize with custom or default YOLO model."""
        self.custom_model_path = custom_model_path
        self.model = None
        self.load_model()
        
        # Enhanced class mapping for basketball objects
        self.basketball_classes = {
            0: "player",
            1: "ball", 
            2: "referee",
            3: "basket",
            4: "board"
        }
        
        # Colors for visualization
        self.class_colors = {
            0: (255, 0, 0),    # Red for players
            1: (255, 165, 0),  # Orange for ball
            2: (0, 0, 255),    # Blue for referee
            3: (0, 255, 0),    # Green for basket
            4: (128, 0, 128)   # Purple for board
        }
        
        # Court dimensions (NBA regulation in feet)
        self.court_dimensions = {
            'length': 94,  # feet
            'width': 50,   # feet
            'basket_height': 10,  # feet
            'three_point_distance': 23.75  # feet from center
        }
        
    def load_model(self):
        """Load custom YOLO model or fall back to default."""
        if self.custom_model_path and Path(self.custom_model_path).exists():
            print(f"🤖 Loading custom basketball YOLO model: {self.custom_model_path}")
            self.model = YOLO(self.custom_model_path)
            self.is_custom_model = True
        else:
            print("🤖 Loading default YOLO model (YOLOv8n)")
            self.model = YOLO('yolov8n.pt')
            self.is_custom_model = False
            # Map person class to player for default model
            self.default_class_mapping = {0: 0}  # person -> player
            
    def detect_objects(self, frame: np.ndarray, confidence: float = 0.25) -> Dict:
        """Detect basketball objects in frame with enhanced information."""
        results = self.model(frame, conf=confidence, verbose=False)
        
        detections = {
            'players': [],
            'ball': [],
            'referees': [], 
            'baskets': [],
            'boards': [],
            'frame_info': {
                'timestamp': datetime.now().isoformat(),
                'frame_shape': frame.shape,
                'model_type': 'custom' if self.is_custom_model else 'default'
            }
        }
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Map class ID to basketball object type
                object_type = self._map_class_to_object_type(class_id)
                
                if object_type:
                    detection_info = {
                        'id': i,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'area': (x2 - x1) * (y2 - y1),
                        'aspect_ratio': (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                    }
                    
                    detections[f"{object_type}s"].append(detection_info)
                    
        return detections
        
    def _map_class_to_object_type(self, class_id: int) -> Optional[str]:
        """Map YOLO class ID to basketball object type."""
        if self.is_custom_model:
            return self.basketball_classes.get(class_id)
        else:
            # For default YOLO, only map person class to player
            if class_id == 0:  # person class
                return "player"
            return None
            
    def analyze_court_context(self, detections: Dict, frame_shape: Tuple[int, int, int]) -> Dict:
        """Analyze spatial relationships and court context."""
        height, width = frame_shape[:2]
        
        analysis = {
            'court_regions': self._identify_court_regions(detections, width, height),
            'player_formations': self._analyze_player_formations(detections['players']),
            'ball_possession': self._analyze_ball_possession(detections),
            'basket_proximity': self._analyze_basket_proximity(detections),
            'referee_positioning': self._analyze_referee_positioning(detections)
        }
        
        return analysis
        
    def _identify_court_regions(self, detections: Dict, width: int, height: int) -> Dict:
        """Identify which court regions objects are in."""
        regions = {
            'left_court': {'x_range': (0, width//2), 'objects': []},
            'right_court': {'x_range': (width//2, width), 'objects': []},
            'center_court': {'x_range': (width//3, 2*width//3), 'objects': []},
            'top_half': {'y_range': (0, height//2), 'objects': []},
            'bottom_half': {'y_range': (height//2, height), 'objects': []}
        }
        
        for obj_type, obj_list in detections.items():
            if obj_type == 'frame_info':
                continue
                
            for obj in obj_list:
                x, y = obj['center']
                
                # Check horizontal regions
                if x < width // 2:
                    regions['left_court']['objects'].append(f"{obj_type}_{obj['id']}")
                else:
                    regions['right_court']['objects'].append(f"{obj_type}_{obj['id']}")
                    
                if width//3 <= x <= 2*width//3:
                    regions['center_court']['objects'].append(f"{obj_type}_{obj['id']}")
                    
                # Check vertical regions
                if y < height // 2:
                    regions['top_half']['objects'].append(f"{obj_type}_{obj['id']}")
                else:
                    regions['bottom_half']['objects'].append(f"{obj_type}_{obj['id']}")
                    
        return regions
        
    def _analyze_player_formations(self, players: List[Dict]) -> Dict:
        """Analyze player positioning and formations."""
        if len(players) < 2:
            return {'formation_type': 'insufficient_players', 'clusters': []}
            
        # Extract player positions
        positions = np.array([player['center'] for player in players])
        
        # Simple clustering to identify team formations
        from sklearn.cluster import KMeans
        
        formation_analysis = {
            'total_players': len(players),
            'average_position': positions.mean(axis=0).tolist(),
            'position_spread': {
                'x_std': float(np.std(positions[:, 0])),
                'y_std': float(np.std(positions[:, 1]))
            }
        }
        
        # Try to identify clusters (teams)
        if len(players) >= 4:
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(positions)
                
                formation_analysis['clusters'] = {
                    'team_1': [i for i, c in enumerate(clusters) if c == 0],
                    'team_2': [i for i, c in enumerate(clusters) if c == 1]
                }
                
                formation_analysis['cluster_centers'] = kmeans.cluster_centers_.tolist()
                
            except Exception as e:
                formation_analysis['clustering_error'] = str(e)
                
        return formation_analysis
        
    def _analyze_ball_possession(self, detections: Dict) -> Dict:
        """Analyze ball possession and proximity to players."""
        ball_analysis = {
            'ball_detected': len(detections['ball']) > 0,
            'ball_count': len(detections['ball']),
            'possession_info': None
        }
        
        if detections['ball'] and detections['players']:
            ball = detections['ball'][0]  # Take first ball detection
            ball_pos = ball['center']
            
            # Find closest player to ball
            min_distance = float('inf')
            closest_player = None
            
            for i, player in enumerate(detections['players']):
                player_pos = player['center']
                distance = np.sqrt((ball_pos[0] - player_pos[0])**2 + 
                                 (ball_pos[1] - player_pos[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_player = i
                    
            ball_analysis['possession_info'] = {
                'closest_player_id': closest_player,
                'distance_to_closest_player': float(min_distance),
                'ball_position': ball_pos,
                'possession_threshold': 50  # pixels
            }
            
            # Determine if ball is likely possessed
            ball_analysis['likely_possessed'] = min_distance < 50
            
        return ball_analysis
        
    def _analyze_basket_proximity(self, detections: Dict) -> Dict:
        """Analyze proximity of players and ball to baskets."""
        proximity_analysis = {
            'baskets_detected': len(detections['baskets']),
            'proximity_info': []
        }
        
        for basket in detections['baskets']:
            basket_pos = basket['center']
            basket_info = {
                'basket_id': basket['id'],
                'basket_position': basket_pos,
                'nearby_players': [],
                'ball_distance': None
            }
            
            # Check player proximity to basket
            for i, player in enumerate(detections['players']):
                player_pos = player['center']
                distance = np.sqrt((basket_pos[0] - player_pos[0])**2 + 
                                 (basket_pos[1] - player_pos[1])**2)
                
                if distance < 100:  # Within 100 pixels of basket
                    basket_info['nearby_players'].append({
                        'player_id': i,
                        'distance': float(distance)
                    })
                    
            # Check ball proximity to basket
            if detections['ball']:
                ball_pos = detections['ball'][0]['center']
                ball_distance = np.sqrt((basket_pos[0] - ball_pos[0])**2 + 
                                      (basket_pos[1] - ball_pos[1])**2)
                basket_info['ball_distance'] = float(ball_distance)
                
            proximity_analysis['proximity_info'].append(basket_info)
            
        return proximity_analysis
        
    def _analyze_referee_positioning(self, detections: Dict) -> Dict:
        """Analyze referee positioning relative to play."""
        ref_analysis = {
            'referees_detected': len(detections['referees']),
            'referee_positions': []
        }
        
        for ref in detections['referees']:
            ref_pos = ref['center']
            ref_info = {
                'referee_id': ref['id'],
                'position': ref_pos,
                'court_zone': self._determine_court_zone(ref_pos)
            }
            ref_analysis['referee_positions'].append(ref_info)
            
        return ref_analysis
        
    def _determine_court_zone(self, position: List[int]) -> str:
        """Determine which zone of the court a position is in."""
        # This is a simplified zone determination
        # In a real implementation, you might use court transformation
        x, y = position
        
        if x < 200:
            return "left_zone"
        elif x > 400:
            return "right_zone"
        else:
            return "center_zone"
            
    def process_video_enhanced(self, video_path: str, output_dir: str = None,
                             confidence: float = 0.25, max_frames: int = None) -> Dict:
        """Process entire video with enhanced basketball analysis."""
        video_path = Path(video_path)
        
        if output_dir is None:
            output_dir = f"enhanced_analysis_{video_path.stem}"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎥 Enhanced basketball video analysis: {video_path}")
        print(f"📁 Output directory: {output_dir}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            
        print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Initialize tracking data
        all_detections = []
        frame_analyses = []
        summary_stats = {
            'total_frames': 0,
            'frames_with_players': 0,
            'frames_with_ball': 0,
            'frames_with_referees': 0,
            'frames_with_baskets': 0,
            'average_players_per_frame': 0,
            'ball_possession_frames': 0
        }
        
        # Process frames
        frame_count = 0
        
        while cap.isOpened() and (max_frames is None or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            detections = self.detect_objects(frame, confidence)
            
            # Analyze court context
            court_analysis = self.analyze_court_context(detections, frame.shape)
            
            # Store frame data
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'detections': detections,
                'analysis': court_analysis
            }
            
            all_detections.append(frame_data)
            
            # Update summary statistics
            summary_stats['total_frames'] += 1
            if detections['players']:
                summary_stats['frames_with_players'] += 1
            if detections['ball']:
                summary_stats['frames_with_ball'] += 1
            if detections['referees']:
                summary_stats['frames_with_referees'] += 1
            if detections['baskets']:
                summary_stats['frames_with_baskets'] += 1
            if court_analysis['ball_possession']['likely_possessed']:
                summary_stats['ball_possession_frames'] += 1
                
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"   Processed {frame_count}/{total_frames} frames...")
                
        cap.release()
        
        # Calculate final statistics
        if summary_stats['total_frames'] > 0:
            player_counts = [len(f['detections']['players']) for f in all_detections]
            summary_stats['average_players_per_frame'] = sum(player_counts) / len(player_counts)
            
        # Save results
        results = {
            'video_info': {
                'path': str(video_path),
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration_seconds': total_frames / fps
            },
            'model_info': {
                'model_path': self.custom_model_path,
                'is_custom': self.is_custom_model,
                'confidence_threshold': confidence
            },
            'summary_statistics': summary_stats,
            'frame_detections': all_detections[:1000],  # Limit for file size
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        results_file = output_dir / "enhanced_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create summary CSV for easy analysis
        self._create_summary_csv(all_detections, output_dir / "frame_summary.csv")
        
        # Generate visualization report
        self._generate_visualization_report(results, output_dir)
        
        print(f"✅ Enhanced analysis complete!")
        print(f"📊 Summary statistics:")
        for key, value in summary_stats.items():
            print(f"   {key}: {value}")
            
        return results
        
    def _create_summary_csv(self, all_detections: List[Dict], output_path: Path):
        """Create CSV summary of detections."""
        rows = []
        
        for frame_data in all_detections:
            frame_num = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            detections = frame_data['detections']
            analysis = frame_data['analysis']
            
            row = {
                'frame': frame_num,
                'timestamp': timestamp,
                'players_count': len(detections['players']),
                'ball_detected': len(detections['ball']) > 0,
                'referees_count': len(detections['referees']),
                'baskets_count': len(detections['baskets']),
                'boards_count': len(detections['boards']),
                'ball_possessed': analysis['ball_possession']['likely_possessed'],
                'formation_players': analysis['player_formations']['total_players']
            }
            
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"📄 Summary CSV saved: {output_path}")
        
    def _generate_visualization_report(self, results: Dict, output_dir: Path):
        """Generate visualization report."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Basketball Analysis Report', fontsize=16)
        
        # Extract data for plotting
        frames = [f['frame_number'] for f in results['frame_detections']]
        player_counts = [len(f['detections']['players']) for f in results['frame_detections']]
        ball_detected = [len(f['detections']['ball']) > 0 for f in results['frame_detections']]
        
        # Plot 1: Player count over time
        axes[0, 0].plot(frames, player_counts)
        axes[0, 0].set_title('Players Detected Over Time')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].grid(True)
        
        # Plot 2: Ball detection over time
        axes[0, 1].plot(frames, ball_detected, 'o-', markersize=2)
        axes[0, 1].set_title('Ball Detection Over Time')
        axes[0, 1].set_xlabel('Frame Number')
        axes[0, 1].set_ylabel('Ball Detected')
        axes[0, 1].grid(True)
        
        # Plot 3: Object type distribution
        stats = results['summary_statistics']
        object_types = ['Players', 'Ball', 'Referees', 'Baskets']
        object_counts = [
            stats['frames_with_players'],
            stats['frames_with_ball'],
            stats['frames_with_referees'],
            stats['frames_with_baskets']
        ]
        
        axes[1, 0].bar(object_types, object_counts)
        axes[1, 0].set_title('Frames with Object Types')
        axes[1, 0].set_ylabel('Number of Frames')
        
        # Plot 4: Summary statistics
        axes[1, 1].text(0.1, 0.8, f"Total Frames: {stats['total_frames']}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Avg Players/Frame: {stats['average_players_per_frame']:.1f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Ball Detection Rate: {stats['frames_with_ball']/stats['total_frames']*100:.1f}%", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Possession Frames: {stats['ball_possession_frames']}", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        report_path = output_dir / "analysis_report.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization report saved: {report_path}")


if __name__ == "__main__":
    # Example usage
    processor = EnhancedBasketballProcessor()
    
    print("🏀 Enhanced Basketball Video Processor")
    print("=" * 40)
    
    video_path = input("Enter video path: ").strip()
    
    if Path(video_path).exists():
        confidence = input("Confidence threshold [0.25]: ").strip()
        confidence = float(confidence) if confidence else 0.25
        
        max_frames = input("Max frames to process [all]: ").strip()
        max_frames = int(max_frames) if max_frames else None
        
        results = processor.process_video_enhanced(
            video_path, 
            confidence=confidence,
            max_frames=max_frames
        )
        
        print("\n🎉 Analysis complete! Check the output directory for results.")
        
    else:
        print("❌ Video file not found!")
