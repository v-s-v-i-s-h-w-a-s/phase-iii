#!/usr/bin/env python3
"""
Quick Improved Detection Test - Process just first 300 frames for faster testing
"""

import sys
import cv2
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")

try:
    from improved_basketball_intelligence import ImprovedBasketballIntelligence
    print("✅ Improved basketball intelligence imported successfully")
except ImportError as e:
    print(f"❌ Error importing improved basketball intelligence: {e}")
    exit(1)

def quick_improved_test():
    """Quick test with first 300 frames"""
    
    print("🏀 Quick Improved Detection Test (300 frames)")
    print("=" * 50)
    
    # Paths
    video_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4")
    model_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt")
    
    if not video_path.exists() or not model_path.exists():
        print("❌ Video or model not found")
        return
    
    print(f"✅ Using video: {video_path}")
    print(f"✅ Using model: {model_path}")
    
    # Initialize detector
    detector = ImprovedBasketballIntelligence(str(model_path))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III") / f"quick_improved_test_{timestamp}.mp4"
    
    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"🎯 Processing first 300 frames to: {output_path}")
    
    frame_count = 0
    max_frames = 300
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, trackers = detector.process_frame(frame)
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"   Processed {frame_count}/{max_frames} frames...")
                
        print(f"\n✅ Quick test completed!")
        print(f"📁 Output: {output_path}")
        print(f"🎯 Processed {frame_count} frames")
        
        # Get stats
        stats = detector.get_performance_stats()
        print(f"📊 Quick Stats:")
        print(f"   - Average FPS: {stats.get('average_fps', 0):.2f}")
        print(f"   - Total detections: {stats.get('total_detections', 0)}")
        print(f"   - Class distribution: {stats.get('class_distribution', {})}")
        
    finally:
        cap.release()
        out.release()
    
    return output_path

if __name__ == "__main__":
    output_file = quick_improved_test()
    if output_file:
        print(f"\n🎉 Quick improved test video created: {output_file}")
        print("🎯 This should show improved labeling accuracy!")
