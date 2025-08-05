#!/usr/bin/env python3
"""
Quick Test: Basketball Video to 2D Conversion
Test the converter with a small sample before running full video
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")

try:
    from basketball_video_to_2d_converter import convert_basketball_video_to_2d
    print("✅ Basketball 2D converter imported successfully")
except ImportError as e:
    print(f"❌ Error importing converter: {e}")
    exit(1)

def quick_2d_test():
    """Quick test with first 100 frames"""
    
    print("🏀 Quick 2D Conversion Test")
    print("=" * 40)
    
    # Test video path
    video_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks-vs-knicks.mp4")
    
    if not video_path.exists():
        # Try alternative videos
        alternatives = [
            r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4",
            r"C:\Users\vish\Capstone PROJECT\Phase III\basketball_demo_20250803_162301.mp4"
        ]
        
        for alt in alternatives:
            alt_path = Path(alt)
            if alt_path.exists():
                video_path = alt_path
                break
        else:
            print("❌ No test video found")
            return
    
    print(f"✅ Using video: {video_path}")
    
    try:
        # Convert first 100 frames
        output_video, stats = convert_basketball_video_to_2d(
            video_path=str(video_path),
            max_frames=100  # Quick test with 100 frames
        )
        
        print(f"\n🎉 Quick test completed!")
        print(f"📁 2D Video: {output_video}")
        print(f"📊 Frames processed: {stats['processing_summary']['total_frames_processed']}")
        print(f"🏀 Average players: {stats['processing_summary']['average_players_per_frame']:.1f}")
        
        return output_video
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = quick_2d_test()
    if result:
        print(f"\n🎯 Success! Check your 2D basketball gameplay video: {result}")
    else:
        print("\n❌ Quick test failed")
