"""
Hawks vs Knicks Basketball Detection and Tracking
Processes the Hawks vs Knicks video with basketball detection and tracking
"""

from src.inference import BasketballInference
import cv2
import time
from pathlib import Path

def process_hawks_vs_knicks():
    """Process the Hawks vs Knicks video with basketball detection"""
    
    print("🏀 Hawks vs Knicks - Basketball Detection & Tracking")
    print("=" * 60)
    
    video_path = "hawks_vs_knicks.mp4"
    
    # Check if video exists
    if not Path(video_path).exists():
        print("❌ Video file not found!")
        return
    
    # Check if model exists
    model_path = "./models/basketball_yolo11n.pt"
    if not Path(model_path).exists():
        print("❌ Trained model not found!")
        print("   Please run training first: python quick_test.py")
        return
    
    # Get video info first
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"🎥 Video Information:")
    print(f"   File: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Total frames: {total_frames}")
    print(f"   File size: {Path(video_path).stat().st_size / (1024*1024):.1f} MB")
    
    # Initialize inference
    print(f"\n🤖 Loading basketball detection model...")
    inference = BasketballInference(model_path)
    
    if not inference.load_model():
        print("❌ Failed to load model!")
        return
    
    # Configure for better basketball detection
    inference.conf_threshold = 0.3  # Lower threshold for better detection
    inference.iou_threshold = 0.5   # Higher threshold to reduce duplicates
    
    print(f"   Model loaded successfully!")
    print(f"   Confidence threshold: {inference.conf_threshold}")
    print(f"   IoU threshold: {inference.iou_threshold}")
    
    # Process the video
    print(f"\n🎯 Processing Hawks vs Knicks video...")
    
    output_path = "./outputs/hawks_vs_knicks_detected.mp4"
    
    start_time = time.time()
    
    results = inference.process_video(
        video_path=video_path,
        output_path=output_path,
        save_results=True
    )
    
    if results:
        print(f"\n📊 Detection Results Summary:")
        print(f"   Output video: {output_path}")
        print(f"   Processing time: {results['stats']['total_time']:.1f}s")
        print(f"   Average FPS: {results['stats']['avg_fps']:.1f}")
        print(f"   Total detections: {results['stats']['total_detections']}")
        
        # Analyze detections
        inference.analyze_detections(results['detections'])
        
        print(f"\n✅ Hawks vs Knicks analysis complete!")
        print(f"   📹 Annotated video: {output_path}")
        print(f"   📊 Detection data: {output_path.replace('.mp4', '_detections.csv')}")
        
        return results
    else:
        print("❌ Video processing failed!")
        return None

if __name__ == "__main__":
    process_hawks_vs_knicks()
