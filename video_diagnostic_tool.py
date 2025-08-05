#!/usr/bin/env python3
"""
Video Diagnostic and Fix Tool
============================
Diagnose and fix video playback issues
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def diagnose_video(video_path):
    """Diagnose video file issues"""
    
    print(f"🔍 Diagnosing video: {video_path}")
    print("=" * 50)
    
    if not Path(video_path).exists():
        print("❌ Video file does not exist!")
        return False
    
    # Check file size
    file_size = Path(video_path).stat().st_size
    print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 1000:  # Less than 1KB
        print("❌ File is too small - likely corrupted or empty")
        return False
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Cannot open video with OpenCV")
        return False
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    print(f"📹 Video properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Frames: {frame_count}")
    print(f"   - Codec: {fourcc}")
    
    # Try to read first few frames
    frames_read = 0
    for i in range(min(10, frame_count)):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_read += 1
            if i == 0:
                print(f"✅ First frame: {frame.shape}")
        else:
            print(f"❌ Failed to read frame {i}")
            break
    
    cap.release()
    
    print(f"📊 Successfully read {frames_read} frames")
    
    if frames_read == 0:
        print("❌ No frames could be read - video is corrupted")
        return False
    
    print("✅ Video appears to be readable")
    return True

def create_test_video_with_compatible_codec(output_path):
    """Create a test video with compatible codec"""
    
    print(f"🎬 Creating test video: {output_path}")
    
    # Create test frames
    width, height = 800, 400
    fps = 30
    duration_seconds = 3
    total_frames = fps * duration_seconds
    
    # Use more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible than mp4v
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Cannot create video writer with XVID, trying H264...")
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Cannot create video writer with H264, trying default...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create colorful test frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * (x / width)),  # Red gradient
                    int(255 * (y / height)),  # Green gradient
                    int(255 * ((frame_num % fps) / fps))  # Blue animation
                ]
        
        # Add text
        cv2.putText(frame, f"TEST FRAME {frame_num}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_num/fps:.2f}s", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw some shapes
        cv2.circle(frame, (200 + frame_num * 2, 200), 30, (0, 255, 255), -1)
        cv2.rectangle(frame, (300, 150), (500, 250), (255, 0, 255), 3)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created successfully!")
    
    # Verify the test video
    return diagnose_video(output_path)

def fix_side_by_side_video_codec():
    """Create a fixed version of the side-by-side video with compatible codec"""
    
    # Find the latest side-by-side video
    side_by_side_videos = list(Path(r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks").glob("*side_by_side*.mp4"))
    
    if not side_by_side_videos:
        print("❌ No side-by-side video found to fix")
        return None
    
    latest_video = max(side_by_side_videos, key=lambda x: x.stat().st_mtime)
    print(f"🔧 Fixing video: {latest_video}")
    
    # Check original video
    if not diagnose_video(latest_video):
        print("❌ Original video has issues, cannot fix")
        return None
    
    # Create fixed version
    fixed_path = latest_video.parent / f"{latest_video.stem}_FIXED.avi"  # Use AVI for better compatibility
    
    cap = cv2.VideoCapture(str(latest_video))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use XVID codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(fixed_path), fourcc, fps, (width, height))
    
    frame_count = 0
    print("🔄 Converting video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ensure frame is valid
        if frame is not None and frame.size > 0:
            out.write(frame)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"   Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"✅ Fixed video created: {fixed_path}")
    print(f"📊 Processed {frame_count} frames")
    
    # Verify fixed video
    if diagnose_video(fixed_path):
        print("✅ Fixed video verified successfully!")
        return fixed_path
    else:
        print("❌ Fixed video still has issues")
        return None

def main():
    """Main diagnostic and fix function"""
    
    print("🏀 Basketball Video Diagnostic Tool")
    print("=" * 60)
    
    # Check if Hawks vs Knicks side-by-side video exists
    hawks_videos = list(Path(r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks").glob("*side_by_side*.mp4"))
    
    if hawks_videos:
        latest_video = max(hawks_videos, key=lambda x: x.stat().st_mtime)
        print(f"🎯 Found video: {latest_video.name}")
        
        # Diagnose the video
        is_healthy = diagnose_video(latest_video)
        
        if not is_healthy:
            print("\n🔧 Attempting to fix video...")
            fixed_video = fix_side_by_side_video_codec()
            
            if fixed_video:
                print(f"\n✅ SUCCESS! Fixed video: {fixed_video}")
                print("📺 Try playing this AVI file with VLC or Windows Media Player")
            else:
                print("\n❌ Could not fix video")
        else:
            print("\n✅ Video appears healthy, codec might be incompatible")
            print("🔧 Creating compatible version...")
            fixed_video = fix_side_by_side_video_codec()
            
            if fixed_video:
                print(f"\n✅ Compatible version created: {fixed_video}")
    else:
        print("❌ No side-by-side video found")
    
    # Create a test video to verify codec support
    print(f"\n🧪 Creating test video...")
    test_video_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III") / "codec_test.avi"
    
    if create_test_video_with_compatible_codec(test_video_path):
        print(f"✅ Test video created successfully: {test_video_path}")
        print("📺 Try playing this test video to verify codec support")
    
    print(f"\n💡 Recommendations:")
    print(f"   - Try VLC Media Player (supports most codecs)")
    print(f"   - Try Windows Media Player")
    print(f"   - Install K-Lite Codec Pack for additional codec support")
    print(f"   - Look for .avi files (more compatible than .mp4)")

if __name__ == "__main__":
    main()
