#!/usr/bin/env python3
"""
🏀 READY-TO-USE BASKETBALL VIDEO TESTER
=======================================
Simple script to test your basketball videos with enhanced team classification
"""

import os
import sys
sys.path.append('src')

from generalized_basketball_inference import GeneralizedBasketballInference
import json
import cv2

def test_basketball_video(video_path, max_frames=None):
    """Test basketball video with enhanced team classification"""
    
    print("🏀 BASKETBALL VIDEO ANALYSIS")
    print("=" * 40)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Initialize system
    print("✅ Loading enhanced basketball system...")
    inference_system = GeneralizedBasketballInference()
    
    print("🎯 System features:")
    print("   ✅ YOLO11 object detection")
    print("   ✅ Exactly 2 teams (basketball rules)")
    print("   ✅ Adaptive team classification")
    print("   ✅ Real-time processing")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    print(f"\n📹 Video: {os.path.basename(video_path)}")
    print(f"   📊 {total_frames} frames @ {fps:.1f} FPS ({duration:.1f}s)")
    
    if max_frames:
        print(f"   🎯 Processing first {max_frames} frames for quick test")
    
    # Process video
    try:
        print("\n🚀 Processing video...")
        output_path, results_path = inference_system.process_video(
            video_path, 
            max_frames=max_frames
        )
        
        print(f"✅ Processing complete!")
        print(f"📹 Output video: {output_path}")
        print(f"📊 Results: {results_path}")
        
        # Show results if available
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print(f"\n🎯 ANALYSIS RESULTS:")
            frames_processed = results.get('frames_processed', 0)
            total_detections = results.get('total_detections', 0)
            
            print(f"   📊 Frames processed: {frames_processed}")
            print(f"   👥 Player detections: {total_detections}")
            
            # Team statistics
            team_stats = results.get('team_stats', {})
            print(f"   🏀 Teams detected: {len(team_stats)}")
            
            for team, stats in team_stats.items():
                count = stats.get('count', 0)
                percentage = (count / total_detections * 100) if total_detections > 0 else 0
                print(f"      {team}: {count} detections ({percentage:.1f}%)")
            
            # Success rate
            classified = sum(stats.get('count', 0) for stats in team_stats.values())
            success_rate = (classified / total_detections * 100) if total_detections > 0 else 0
            print(f"   🎯 Classification success: {success_rate:.1f}%")
            
            # Basketball compliance
            if len(team_stats) == 2:
                print("   ✅ Basketball compliant: Exactly 2 teams detected!")
            else:
                print(f"   ⚠️ Detected {len(team_stats)} teams (basketball needs exactly 2)")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        print("Please check that the video file is valid and accessible")

def main():
    """Main function to handle command line arguments"""
    
    print("🏀 BASKETBALL VIDEO TESTER")
    print("=" * 30)
    
    if len(sys.argv) < 2:
        print("Usage: python basketball_tester.py <video_path> [max_frames]")
        print("\nExamples:")
        print("  python basketball_tester.py my_game.mp4")
        print("  python basketball_tester.py game.mp4 100")
        print("\n📹 Quick demo with sample video:")
        
        # Try with sample video if available
        sample_videos = ["hawks_vs_knicks.mp4", "enhanced_shot_detection.mp4"]
        
        for video in sample_videos:
            if os.path.exists(video):
                print(f"   Testing with {video}...")
                test_basketball_video(video, max_frames=50)
                break
        else:
            print("   No sample video found. Please provide a video path.")
        
        return
    
    video_path = sys.argv[1]
    max_frames = None
    
    if len(sys.argv) > 2:
        try:
            max_frames = int(sys.argv[2])
        except ValueError:
            print("⚠️ Max frames must be a number. Using unlimited frames.")
    
    test_basketball_video(video_path, max_frames)

if __name__ == "__main__":
    main()
