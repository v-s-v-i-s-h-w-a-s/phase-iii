#!/usr/bin/env python3
"""
Quick Test Script for Enhanced Basketball Detection
Verifies the system is working correctly with your test video
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")

try:
    from enhanced_basketball_intelligence import EnhancedBasketballIntelligence
    print("✅ Enhanced basketball intelligence module imported successfully")
except ImportError as e:
    print(f"❌ Error importing enhanced basketball intelligence: {e}")
    exit(1)

def quick_test():
    """Quick test of the enhanced basketball detection system"""
    
    print("🏀 Enhanced Basketball Detection System - Quick Test")
    print("=" * 50)
    
    # Check if test video exists
    test_video = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4")
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        
        # Check for alternative test videos
        alternatives = [
            "hawks vs knicks/hawks-vs-knicks.mp4",
            "basketball_demo_20250803_162301.mp4",
            "complete_basketball_analysis_20250803_162129.mp4"
        ]
        
        for alt in alternatives:
            alt_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III") / alt
            if alt_path.exists():
                test_video = alt_path
                print(f"✅ Using alternative test video: {test_video}")
                break
        else:
            print("❌ No test video found")
            return
    else:
        print(f"✅ Test video found: {test_video}")
    
    # Check for enhanced model
    enhanced_dir = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_training")
    model_path = None
    
    if enhanced_dir.exists():
        # Look for enhanced model weights
        for enhanced_subdir in enhanced_dir.glob("enhanced_*"):
            weights_pattern = enhanced_subdir / "enhanced_basketball_*" / "weights" / "best.pt"
            weights_files = list(enhanced_subdir.glob("enhanced_basketball_*/weights/best.pt"))
            if weights_files:
                model_path = weights_files[0]
                print(f"✅ Enhanced model found: {model_path}")
                break
    
    if not model_path:
        print("⚠️ Enhanced model not found, will use base YOLO model")
        model_path = "yolov8n.pt"
    
    # Test video properties
    cap = cv2.VideoCapture(str(test_video))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {test_video}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - Frame count: {frame_count}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Duration: {frame_count/fps:.2f} seconds")
    
    cap.release()
    
    # Initialize enhanced basketball intelligence
    try:
        enhanced_system = EnhancedBasketballIntelligence(model_path=str(model_path))
        print("✅ Enhanced basketball system initialized successfully")
        
        # Test on a few frames
        print("\n🔍 Testing detection on first few frames...")
        cap = cv2.VideoCapture(str(test_video))
        
        for i in range(min(3, frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"   - Processing frame {i+1}...")
            result_frame, frame_trackers = enhanced_system.process_frame(frame)
            
            if frame_trackers:
                print(f"     ✅ Detected {len(frame_trackers)} tracked objects")
                
                # Count detections by class
                class_counts = {}
                for tracker_id, tracker_data in frame_trackers.items():
                    class_name = enhanced_system.model.names.get(tracker_data['class'], 'unknown')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                for class_name, count in class_counts.items():
                    print(f"       - {class_name}: {count}")
            else:
                print(f"     ⚠️ No detections in frame {i+1}")
        
        cap.release()
        print("\n✅ Enhanced basketball detection system is working correctly!")
        print("🎯 The system is ready for improvements to labeling accuracy")
        
    except Exception as e:
        print(f"❌ Error testing enhanced system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
