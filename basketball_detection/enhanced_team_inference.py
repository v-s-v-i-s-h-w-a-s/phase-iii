"""
Enhanced Basketball Detection with Team Classification
Real-time detection, tracking, and team classification on video files
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import time
import pandas as pd
from datetime import datetime
import json

from src.team_classifier import TeamClassifier

class EnhancedBasketballInference:
    def __init__(self, model_path=None):
        """
        Initialize enhanced inference engine with team classification
        Args:
            model_path: Path to trained model, if None uses default
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        # Default model path
        if model_path is None:
            model_path = "./models/basketball_yolo11n.pt"
        
        self.model_path = model_path
        
        # Detection settings
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4
        
        # Initialize team classifier
        self.team_classifier = TeamClassifier()
        
        # Default colors for non-player objects (BGR format)
        self.default_colors = {
            'ball': (0, 165, 255),      # Orange
            'hoop': (128, 0, 128)       # Purple
        }
        
        print("🏀 Enhanced Basketball Inference Engine initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        print("   Team Classification: Enabled")
        
    def load_model(self):
        """Load trained model"""
        if not Path(self.model_path).exists():
            print(f"❌ Model not found: {self.model_path}")
            print("   Please train a model first or use a valid model path")
            return False
            
        print("📥 Loading model...")
        self.model = YOLO(self.model_path)
        print("✅ Model loaded successfully!")
        return True
        
    def detect_frame(self, frame):
        """
        Detect objects in a single frame
        Returns: detections with boxes, classes, confidences
        """
        if self.model is None:
            return []
            
        # Run inference
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id
                    })
        
        return detections
    
    def process_detections_with_teams(self, frame, detections):
        """
        Process detections and add team classifications
        """
        # Separate players from other objects
        player_detections = [d for d in detections if d['class'] == 'player']
        other_detections = [d for d in detections if d['class'] != 'player']
        
        # Classify players into teams
        if player_detections:
            classified_players = self.team_classifier.classify_frame_players(frame, player_detections)
        else:
            classified_players = []
        
        # Combine all detections
        all_detections = classified_players + other_detections
        
        return all_detections
    
    def draw_enhanced_detections(self, frame, detections):
        """Draw detection boxes with team-based colors and enhanced visualization"""
        # Use team classifier's drawing method for better visualization
        annotated_frame = self.team_classifier.draw_team_detections(frame, detections)
        
        # Add additional information overlay
        self._add_detection_overlay(annotated_frame, detections)
        
        return annotated_frame
    
    def _add_detection_overlay(self, frame, detections):
        """Add detection statistics overlay"""
        # Count detections by type
        player_count = len([d for d in detections if d['class'] == 'player'])
        referee_count = len([d for d in detections if d['class'] == 'referee'])
        ball_count = len([d for d in detections if d['class'] == 'ball'])
        hoop_count = len([d for d in detections if d['class'] == 'hoop'])
        
        # Get team statistics
        team_stats = self.team_classifier.get_team_statistics(detections)
        
        # Create overlay text
        overlay_y = frame.shape[0] - 120
        
        # Background for overlay
        cv2.rectangle(frame, (10, overlay_y - 10), (300, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, overlay_y - 10), (300, frame.shape[0] - 10), (255, 255, 255), 2)
        
        # Detection counts
        cv2.putText(frame, "DETECTION STATS:", (15, overlay_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Players: {player_count}", (15, overlay_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Referees: {referee_count}", (15, overlay_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Balls: {ball_count}", (15, overlay_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Hoops: {hoop_count}", (15, overlay_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Team distribution
        if team_stats.get('team_counts'):
            cv2.putText(frame, "TEAMS:", (15, overlay_y + 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            team_y = overlay_y + 105
            for team, count in team_stats['team_counts'].items():
                if team != 'unknown':
                    cv2.putText(frame, f"{team}: {count}", (15, team_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    team_y += 12
    
    def process_video(self, video_path, output_path=None, save_results=True):
        """
        Process video file with enhanced detection and team classification
        """
        if not self.load_model():
            return None
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎥 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup output video
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./team_classified_analysis_{timestamp}.mp4"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Detection tracking
        all_detections = []
        frame_count = 0
        start_time = time.time()
        
        print("🔍 Processing frames with team classification...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            raw_detections = self.detect_frame(frame)
            
            # Process with team classification
            enhanced_detections = self.process_detections_with_teams(frame, raw_detections)
            
            # Draw enhanced visualizations
            annotated_frame = self.draw_enhanced_detections(frame.copy(), enhanced_detections)
            
            # Add frame info
            info_text = f"Frame: {frame_count+1}/{total_frames} | Objects: {len(enhanced_detections)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Store detections for analysis
            for detection in enhanced_detections:
                detection['frame'] = frame_count
                detection['timestamp'] = frame_count / fps
                all_detections.append(detection.copy())
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"   Progress: {progress:.1f}% | FPS: {fps_current:.1f}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save enhanced results
        if save_results and all_detections:
            # Save CSV
            results_path = output_path.replace('.mp4', '_detections.csv')
            df = pd.DataFrame(all_detections)
            df.to_csv(results_path, index=False)
            
            # Save team analysis JSON
            team_analysis_path = output_path.replace('.mp4', '_team_analysis.json')
            team_stats = self.team_classifier.get_team_statistics(all_detections)
            with open(team_analysis_path, 'w') as f:
                json.dump(team_stats, f, indent=2, default=str)
            
            print("📊 Results saved:")
            print(f"   Detections: {results_path}")
            print(f"   Team Analysis: {team_analysis_path}")
        
        # Print summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print("✅ Enhanced video processing complete!")
        print(f"   Output: {output_path}")
        print(f"   Processed: {frame_count} frames in {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total detections: {len(all_detections)}")
        
        # Print team analysis
        self._print_team_analysis(all_detections)
        
        # Get file size
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            print(f"📏 File size: {file_size:.1f} MB")
        
        return {
            'output_path': output_path,
            'detections': all_detections,
            'stats': {
                'total_frames': frame_count,
                'total_time': total_time,
                'avg_fps': avg_fps,
                'total_detections': len(all_detections)
            },
            'team_stats': self.team_classifier.get_team_statistics(all_detections)
        }
    
    def _print_team_analysis(self, detections):
        """Print detailed team analysis"""
        team_stats = self.team_classifier.get_team_statistics(detections)
        
        print("\n🏀 TEAM CLASSIFICATION ANALYSIS")
        print("=" * 40)
        
        if team_stats.get('team_counts'):
            print(f"Total Players Detected: {team_stats['total_players']}")
            print("\nTeam Distribution:")
            for team, count in team_stats['team_counts'].items():
                percentage = (count / team_stats['total_players']) * 100
                print(f"  {team.upper()}: {count} players ({percentage:.1f}%)")
        
        if team_stats.get('team_colors'):
            print("\nTeam Colors Identified:")
            for team, color in team_stats['team_colors'].items():
                print(f"  {team.upper()}: RGB{tuple(color[::-1])}")  # Convert BGR to RGB for display
        
        print()

def main():
    """Main function for enhanced basketball detection"""
    print("🏀 Enhanced Basketball Detection with Team Classification")
    print("=" * 60)
    
    # Initialize enhanced inference engine
    inference = EnhancedBasketballInference()
    
    # Check for video files
    current_dir = Path(".")
    video_files = (
        list(current_dir.glob("*.mp4")) + 
        list(current_dir.glob("*.avi")) +
        list(current_dir.glob("downloads/*.mp4"))
    )
    
    if video_files:
        print("📹 Found video files:")
        for i, video in enumerate(video_files):
            print(f"   {i+1}. {video}")
        
        # Process most recent video or downloads
        video_path = str(video_files[0])
        
        # Check for downloads directory
        downloads_dir = Path("downloads")
        if downloads_dir.exists():
            download_videos = list(downloads_dir.glob("*.mp4"))
            if download_videos:
                video_path = str(download_videos[0])  # Use most recent download
        
        print(f"\n🎯 Processing: {video_path}")
        
        results = inference.process_video(video_path)
        
        if results:
            print("\n✅ Processing complete!")
            print("   Enhanced video with team classification saved")
            print("   Teams automatically detected and color-coded")
    else:
        print("❌ No video files found!")
        print("   Please add video files (.mp4, .avi) to the current directory")
        print("   or run the video download script first")

if __name__ == "__main__":
    main()
