"""
Video processing module for basketball GNN
Extracts player tracking data from video files using YOLOv8
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path
import json


class BasketballVideoProcessor:
    """
    Process basketball videos to extract player tracking data.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the video processor.
        
        Args:
            model_path: Path to custom YOLO model. If None, uses default YOLOv8
        """
        
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom YOLO model from {model_path}")
        else:
            # Use pre-trained YOLOv8 model
            self.model = YOLO('yolov8n.pt')  # nano version for speed
            print("Loaded default YOLOv8n model")
        
        # Basketball court dimensions (approximate)
        self.court_length = 28.65  # meters
        self.court_width = 15.24   # meters
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
    def process_video(self, 
                     video_path: str, 
                     output_dir: str = "video_output",
                     max_frames: int = None,
                     confidence_threshold: float = 0.5,
                     save_annotated: bool = True) -> str:
        """
        Process a basketball video and extract tracking data.
        
        Args:
            video_path: Path to the input video
            output_dir: Directory to save outputs
            max_frames: Maximum number of frames to process (None for all)
            confidence_threshold: Minimum confidence for detections
            save_annotated: Whether to save annotated video
            
        Returns:
            Path to the generated tracking data CSV
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Setup video writer for annotated output
        if save_annotated:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            annotated_path = os.path.join(output_dir, 'annotated_video.mp4')
            out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
        
        # Track data storage
        tracking_data = []
        frame_num = 0
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_num >= max_frames):
                break
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Extract detections
            detections = self._extract_detections(
                results[0], 
                frame_num, 
                confidence_threshold
            )
            
            # Add to tracking data
            tracking_data.extend(detections)
            
            # Annotate frame if requested
            if save_annotated:
                annotated_frame = self._annotate_frame(frame, results[0])
                out.write(annotated_frame)
            
            frame_num += 1
            
            # Progress update
            if frame_num % 30 == 0:
                print(f"Processed {frame_num}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if save_annotated:
            out.release()
            print(f"Annotated video saved to: {annotated_path}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(tracking_data)
        
        if len(df) > 0:
            # Add some basic processing
            df = self._post_process_tracks(df)
            
            # Save to CSV
            csv_path = os.path.join(output_dir, 'tracking_data.csv')
            df.to_csv(csv_path, index=False)
            print(f"Tracking data saved to: {csv_path}")
            
            # Save metadata
            metadata = {
                'video_path': video_path,
                'fps': fps,
                'resolution': [width, height],
                'total_frames': frame_num,
                'total_detections': len(df),
                'confidence_threshold': confidence_threshold
            }
            
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return csv_path
        else:
            print("No detections found in video!")
            return None
    
    def _extract_detections(self, 
                           result, 
                           frame_num: int, 
                           confidence_threshold: float) -> List[Dict]:
        """Extract detection data from YOLO results."""
        
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                # Only process person detections above threshold
                if cls == self.person_class_id and score >= confidence_threshold:
                    x1, y1, x2, y2 = box
                    
                    # Calculate center point
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detection = {
                        'frame': frame_num,
                        'player_id': i,  # Simple ID based on detection order
                        'x': center_x,
                        'y': center_y,
                        'bbox_x1': x1,
                        'bbox_y1': y1,
                        'bbox_x2': x2,
                        'bbox_y2': y2,
                        'confidence': score,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def _annotate_frame(self, frame: np.ndarray, result) -> np.ndarray:
        """Annotate frame with detections."""
        
        annotated = frame.copy()
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                if cls == self.person_class_id and score >= 0.5:
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence score
                    cv2.putText(annotated, f'{score:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
        
        return annotated
    
    def _post_process_tracks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process tracking data."""
        
        if len(df) == 0:
            return df
        
        # Sort by frame
        df = df.sort_values(['frame', 'player_id']).reset_index(drop=True)
        
        # Add normalized coordinates (0-1)
        if 'x' in df.columns and 'y' in df.columns:
            # Assume we know frame dimensions - you might want to pass these in
            df['x_norm'] = df['x'] / df['x'].max() if df['x'].max() > 0 else 0
            df['y_norm'] = df['y'] / df['y'].max() if df['y'].max() > 0 else 0
        
        # Calculate velocities (simple frame-to-frame difference)
        df['vx'] = df.groupby('player_id')['x'].diff().fillna(0)
        df['vy'] = df.groupby('player_id')['y'].diff().fillna(0)
        
        # Add team assignment (placeholder - random for now)
        np.random.seed(42)  # For reproducible results
        unique_players = df['player_id'].unique()
        team_assignments = np.random.choice([0, 1], size=len(unique_players))
        team_map = dict(zip(unique_players, team_assignments))
        df['team'] = df['player_id'].map(team_map)
        
        return df


def main():
    """Demo function for video processing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process basketball video')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--output_dir', default='video_output', 
                       help='Output directory')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Process video
    processor = BasketballVideoProcessor()
    csv_path = processor.process_video(
        args.video_path,
        args.output_dir,
        args.max_frames,
        args.confidence
    )
    
    if csv_path:
        print(f"\n✅ Video processing complete!")
        print(f"📊 Tracking data: {csv_path}")
        print(f"📁 All outputs in: {args.output_dir}")
        print(f"\n🚀 Next step: Run GNN analysis with:")
        print(f"python main.py --data_path {csv_path}")


if __name__ == "__main__":
    main()
