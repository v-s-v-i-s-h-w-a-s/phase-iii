"""
Quick Preview - Hawks vs Knicks Sample Frames
Process first few frames to show immediate results
"""

from src.inference import BasketballInference
import cv2
import numpy as np
from pathlib import Path

def quick_preview():
    """Process first 10 frames for quick preview"""
    
    print("🏀 Hawks vs Knicks - Quick Preview")
    print("=" * 40)
    
    video_path = "hawks_vs_knicks.mp4"
    model_path = "./models/basketball_yolo11n.pt"
    
    # Initialize inference
    inference = BasketballInference(model_path)
    if not inference.load_model():
        return
    
    inference.conf_threshold = 0.3
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Process first 10 frames
    frame_count = 0
    sample_frames = [0, 100, 500, 1000, 2000, 5000, 10000]  # Sample different parts
    
    print("🔍 Processing sample frames...")
    
    for target_frame in sample_frames:
        if target_frame >= 17487:
            continue
            
        # Jump to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Detect objects
        detections = inference.detect_frame(frame)
        
        # Draw detections
        result_frame = inference.draw_detections(frame.copy(), detections)
        
        # Add frame info
        time_stamp = target_frame / 29  # 29 FPS
        info_text = f"Frame {target_frame} ({time_stamp:.1f}s) - Objects: {len(detections)}"
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save sample frame
        output_name = f"sample_frame_{target_frame:06d}.jpg"
        cv2.imwrite(output_name, result_frame)
        
        print(f"   Frame {target_frame:6d} ({time_stamp:5.1f}s): {len(detections)} objects detected -> {output_name}")
        
        # Print detected objects
        for detection in detections:
            print(f"      {detection['class']}: {detection['confidence']:.3f}")
    
    cap.release()
    print(f"\n✅ Sample frames saved! Check the .jpg files in current directory.")

if __name__ == "__main__":
    quick_preview()
