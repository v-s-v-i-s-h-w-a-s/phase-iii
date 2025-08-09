"""
Basketball Detection Inference Engine
Real-time detection and tracking on video files
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import time
import pandas as pd
from datetime import datetime

class BasketballInference:
    def __init__(self, model_path=None):
        """
        Initialize inference engine
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
        
        # Colors for different classes (BGR format)
        self.colors = {
            'player': (255, 0, 0),      # Blue
            'referee': (0, 255, 0),     # Green  
            'ball': (0, 165, 255),      # Orange
            'hoop': (128, 0, 128)       # Purple
        }
        
        print(f"🏀 Basketball Inference Engine initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        
    def load_model(self):
        """Load trained model"""
        if not Path(self.model_path).exists():
            print(f"❌ Model not found: {self.model_path}")
            print("   Please train a model first using train_model.py")
            return False
            
        print(f"📥 Loading model...")
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
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Get color for class
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None, save_results=True):
        """
        Process video file and detect basketball objects
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
            output_path = f"./outputs/detection_{timestamp}.mp4"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Detection tracking
        all_detections = []
        frame_count = 0
        start_time = time.time()
        
        print("🔍 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            detections = self.detect_frame(frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame.copy(), detections)
            
            # Add frame info
            info_text = f"Frame: {frame_count+1}/{total_frames} | Objects: {len(detections)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(annotated_frame)
            
            # Store detections for analysis
            for detection in detections:
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
        
        # Save detection results
        if save_results and all_detections:
            results_path = output_path.replace('.mp4', '_detections.csv')
            df = pd.DataFrame(all_detections)
            df.to_csv(results_path, index=False)
            print(f"📊 Detection results saved: {results_path}")
        
        # Print summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print("✅ Video processing complete!")
        print(f"   Output: {output_path}")
        print(f"   Processed: {frame_count} frames in {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total detections: {len(all_detections)}")
        
        return {
            'output_path': output_path,
            'detections': all_detections,
            'stats': {
                'total_frames': frame_count,
                'total_time': total_time,
                'avg_fps': avg_fps,
                'total_detections': len(all_detections)
            }
        }
    
    def analyze_detections(self, detections):
        """Analyze detection results"""
        if not detections:
            print("❌ No detections to analyze")
            return
            
        df = pd.DataFrame(detections)
        
        print("📊 Detection Analysis:")
        print(f"   Total detections: {len(df)}")
        
        # Class distribution
        class_counts = df['class'].value_counts()
        print("   Class distribution:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count}")
        
        # Confidence statistics
        print(f"   Average confidence: {df['confidence'].mean():.3f}")
        print(f"   Min confidence: {df['confidence'].min():.3f}")
        print(f"   Max confidence: {df['confidence'].max():.3f}")
        
        return df

def main():
    """Main inference function"""
    print("🏀 Basketball Detection Inference")
    print("=" * 40)
    
    # Initialize inference engine
    inference = BasketballInference()
    
    # Example usage - you can modify this section
    video_files = [
        # Add your video files here
        "./test_video.mp4",
        "./basketball_game.mp4"
    ]
    
    # Check for any video files in the current directory
    current_dir = Path(".")
    found_videos = list(current_dir.glob("*.mp4")) + list(current_dir.glob("*.avi"))
    
    if found_videos:
        print(f"📹 Found video files:")
        for i, video in enumerate(found_videos):
            print(f"   {i+1}. {video.name}")
        
        # Process first video
        video_path = str(found_videos[0])
        print(f"\n🎯 Processing: {video_path}")
        
        results = inference.process_video(video_path)
        
        if results:
            inference.analyze_detections(results['detections'])
    else:
        print("❌ No video files found!")
        print("   Please add video files (.mp4, .avi) to the current directory")

if __name__ == "__main__":
    main()
