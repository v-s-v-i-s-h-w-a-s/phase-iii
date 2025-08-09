"""
Optimized Hawks vs Knicks Processing
Process video with better memory management and progress tracking
"""

from src.inference import BasketballInference
import cv2
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

def process_hawks_knicks_optimized():
    """Process Hawks vs Knicks video with optimizations"""
    
    print("🏀 Hawks vs Knicks - Optimized Processing")
    print("=" * 50)
    
    video_path = "hawks_vs_knicks.mp4"
    model_path = "./models/basketball_yolo11n.pt"
    
    # Check files
    if not Path(video_path).exists():
        print("❌ Video not found!")
        return
    if not Path(model_path).exists():
        print("❌ Model not found!")
        return
    
    # Initialize
    inference = BasketballInference(model_path)
    if not inference.load_model():
        return
    
    # Lower settings for speed
    inference.conf_threshold = 0.4  # Higher confidence for less noise
    inference.iou_threshold = 0.5
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps}FPS, {total_frames} frames")
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./outputs/hawks_knicks_{timestamp}.mp4"
    Path("./outputs").mkdir(exist_ok=True)
    
    # Process every Nth frame for speed (process every 2nd frame)
    skip_frames = 2
    actual_fps = fps // skip_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
    
    # Tracking
    all_detections = []
    processed_frames = 0
    start_time = time.time()
    
    print(f"🎯 Processing (every {skip_frames} frames for speed)...")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for speed
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue
        
        # Detect objects
        detections = inference.detect_frame(frame)
        
        # Draw detections
        result_frame = inference.draw_detections(frame.copy(), detections)
        
        # Add info overlay
        elapsed = time.time() - start_time
        current_fps = processed_frames / elapsed if elapsed > 0 else 0
        progress = (frame_idx / total_frames) * 100
        
        info_lines = [
            f"Hawks vs Knicks - Frame {frame_idx}/{total_frames}",
            f"Progress: {progress:.1f}% | FPS: {current_fps:.1f}",
            f"Objects: {len(detections)} | Time: {frame_idx/fps:.1f}s"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(result_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Write frame
        out.write(result_frame)
        
        # Store detections
        for detection in detections:
            detection['frame'] = frame_idx
            detection['timestamp'] = frame_idx / fps
            all_detections.append(detection.copy())
        
        processed_frames += 1
        
        # Progress update
        if processed_frames % 100 == 0:
            print(f"   Frame {frame_idx:6d} ({progress:5.1f}%) | "
                  f"FPS: {current_fps:4.1f} | Objects: {len(detections):2d}")
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Save results
    total_time = time.time() - start_time
    
    if all_detections:
        csv_path = output_path.replace('.mp4', '_detections.csv')
        df = pd.DataFrame(all_detections)
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Analysis Complete!")
        print(f"   📹 Output video: {output_path}")
        print(f"   📊 Detection data: {csv_path}")
        print(f"   ⏱️  Processing time: {total_time:.1f}s")
        print(f"   📈 Average FPS: {processed_frames/total_time:.1f}")
        print(f"   🎯 Total detections: {len(all_detections)}")
        
        # Quick analysis
        class_counts = df['class'].value_counts()
        print(f"   📋 Detection summary:")
        for class_name, count in class_counts.items():
            avg_conf = df[df['class'] == class_name]['confidence'].mean()
            print(f"      {class_name}: {count} detections (avg conf: {avg_conf:.3f})")
        
        return output_path, df
    else:
        print(f"❌ No detections found!")
        return None, None

if __name__ == "__main__":
    process_hawks_knicks_optimized()
