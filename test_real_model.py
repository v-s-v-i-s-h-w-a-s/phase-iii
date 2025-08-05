#!/usr/bin/env python3
"""
Test Enhanced Basketball YOLO Model with Consistent Team Tracking
Uses enhanced detection system with pre-initialized team colors
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import json
from datetime import datetime
import sys

# Import our enhanced detection system
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")
from enhanced_basketball_intelligence import EnhancedBasketballIntelligence, process_video_with_enhanced_detection

def test_enhanced_model_on_video():
    """Test the enhanced model with consistent team tracking"""
    
    # Model paths (try enhanced model first, fallback to real model)
    enhanced_model_patterns = [
        r"enhanced_basketball_training\enhanced_*\enhanced_basketball_*\weights\best.pt",
        r"basketball_type2_training\type2_dataset_*\basketball_type2_*\weights\best.pt",
        r"basketball_real_training\real_dataset_20250803_121502\weights\best.pt"
    ]
    
    model_path = None
    for pattern in enhanced_model_patterns:
        from glob import glob
        matches = glob(pattern)
        if matches:
            # Get the most recent model
            model_path = max(matches, key=lambda x: Path(x).stat().st_mtime)
            break
    
    if not model_path:
        model_path = r"basketball_real_training\real_dataset_20250803_121502\weights\best.pt"
    
    # Video path
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4"
    
    # Output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"enhanced_basketball_test_{timestamp}.mp4"
    
    print("🏀 Testing Enhanced Basketball Model with Consistent Teams")
    print("=" * 60)
    print(f"📁 Model: {model_path}")
    print(f"🎥 Video: {video_path}")
    print(f"💾 Output: {output_path}")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    # Process video with enhanced detection
    stats = process_video_with_enhanced_detection(model_path, video_path, output_path)
    
    print(f"\n🎉 Enhanced basketball test completed!")
    print(f"📁 Output video: {output_path}")
    
    # Save detailed statistics
    stats_path = output_path.replace('.mp4', '_detailed_stats.json')
    with open(stats_path, 'w') as f:
        # Convert to JSON-serializable format
        json_stats = {
            'total_frames': stats['total_frames'],
            'total_detections': sum(stats['class_detections'].values()),
            'class_detections': dict(stats['class_detections']),
            'team_assignments': dict(stats['team_assignments']),
            'average_detections_per_frame': np.mean(stats['detections_per_frame']) if stats['detections_per_frame'] else 0,
            'average_processing_time_ms': np.mean(stats['processing_times']) * 1000 if stats['processing_times'] else 0
        }
        json.dump(json_stats, f, indent=2)
    
    print(f"📊 Detailed statistics saved to: {stats_path}")
    
    return stats

def test_simple_model():
    """Simple test with basic YOLO detection"""
    
    # Model path (use the best trained model)
    model_path = r"basketball_real_training\real_dataset_20250803_121502\weights\best.pt"
    
    # Video path (use existing Hawks vs Knicks video)
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4"
    
    print("🏀 Testing Real Dataset YOLO Model on Hawks vs Knicks")
    print("=" * 60)
    print(f"📁 Model: {model_path}")
    print(f"🎥 Video: {video_path}")
    print(f"💾 Output: {output_path}")
    
    # Load the trained model
    print(f"\n📥 Loading trained model...")
    model = YOLO(model_path)
    
    # Class names from our training
    class_names = {
        0: 'ball',
        1: 'basket', 
        2: 'player',
        3: 'referee'
    }
    
    # Class colors for visualization
    colors = {
        0: (0, 255, 255),    # Yellow for ball
        1: (255, 0, 0),      # Blue for basket
        2: (0, 255, 0),      # Green for player
        3: (0, 0, 255)       # Red for referee
    }
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Detection statistics
    stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'class_detections': {name: 0 for name in class_names.values()},
        'processing_times': []
    }
    
    frame_count = 0
    start_time = time.time()
    
    print(f"\n🚀 Starting video processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        stats['total_frames'] = frame_count
        
        # Run detection
        frame_start = time.time()
        results = model(frame, verbose=False)
        frame_time = time.time() - frame_start
        stats['processing_times'].append(frame_time)
        
        # Process detections
        frame_detections = 0
        if results and len(results) > 0:
            detections = results[0]
            if hasattr(detections, 'boxes') and detections.boxes is not None:
                boxes = detections.boxes
                
                for box in boxes:
                    # Extract box information
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Filter by confidence
                    if conf > 0.3 and cls in class_names:  # Lower threshold for better detection
                        frame_detections += 1
                        stats['total_detections'] += 1
                        
                        class_name = class_names[cls]
                        stats['class_detections'][class_name] += 1
                        
                        # Draw bounding box
                        color = colors.get(cls, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if frame_detections > 0:
            stats['frames_with_detections'] += 1
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {frame_detections}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            print(f"   Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | ETA: {eta:.1f}s")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_frame_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    
    print(f"\n✅ Video processing completed!")
    print(f"📊 PROCESSING STATISTICS:")
    print(f"   Total Processing Time: {total_time:.1f} seconds")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Average Frame Processing: {avg_frame_time*1000:.1f}ms")
    
    print(f"\n🎯 DETECTION STATISTICS:")
    print(f"   Total Frames: {stats['total_frames']}")
    print(f"   Frames with Detections: {stats['frames_with_detections']} ({stats['frames_with_detections']/stats['total_frames']*100:.1f}%)")
    print(f"   Total Detections: {stats['total_detections']}")
    print(f"   Average Detections per Frame: {stats['total_detections']/stats['total_frames']:.1f}")
    
    print(f"\n📋 DETECTION BREAKDOWN:")
    for class_name, count in stats['class_detections'].items():
        percentage = (count / stats['total_detections']) * 100 if stats['total_detections'] > 0 else 0
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\n💾 Output saved to: {output_path}")
    
    # Save statistics to JSON
    stats_path = output_path.replace('.mp4', '_stats.json')
    with open(stats_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_stats = {
            'total_frames': int(stats['total_frames']),
            'frames_with_detections': int(stats['frames_with_detections']),
            'total_detections': int(stats['total_detections']),
            'class_detections': {k: int(v) for k, v in stats['class_detections'].items()},
            'processing_time_seconds': float(total_time),
            'average_fps': float(avg_fps),
            'average_frame_processing_ms': float(avg_frame_time * 1000)
        }
        json.dump(json_stats, f, indent=2)
    
    print(f"📊 Statistics saved to: {stats_path}")
    
    return stats

if __name__ == "__main__":
    test_enhanced_model_on_video()
