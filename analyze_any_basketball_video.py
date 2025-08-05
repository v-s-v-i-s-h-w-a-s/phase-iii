#!/usr/bin/env python3
"""
Quick Script to Run Side-by-Side Analysis on Any Basketball Video
================================================================
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")

try:
    from side_by_side_basketball_analyzer import create_side_by_side_analysis
    print("✅ Side-by-side analyzer imported successfully")
except ImportError as e:
    print(f"❌ Error importing analyzer: {e}")
    exit(1)

def analyze_any_basketball_video():
    """Run side-by-side analysis on any basketball video"""
    
    print("🏀 Universal Basketball Video Analyzer")
    print("=" * 50)
    
    # Available test videos
    video_options = [
        r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4",
        r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4",
        r"C:\Users\vish\Capstone PROJECT\Phase III\basketball_demo_20250803_162301.mp4",
        r"C:\Users\vish\Capstone PROJECT\Phase III\complete_basketball_analysis_20250803_162129.mp4"
    ]
    
    # Find available video
    selected_video = None
    for video_path in video_options:
        if Path(video_path).exists():
            selected_video = video_path
            break
    
    if not selected_video:
        print("❌ No basketball video found")
        return
    
    video_name = Path(selected_video).stem
    print(f"✅ Analyzing: {video_name}")
    print(f"📹 Video: {selected_video}")
    
    try:
        # For Hawks vs Knicks - process more frames (or remove max_frames for full game)
        if "hawks" in video_name.lower():
            max_frames_to_process = 600  # ~20 seconds at 30fps
            print(f"🏀 Processing Hawks vs Knicks game ({max_frames_to_process} frames)")
        else:
            max_frames_to_process = 300  # Standard test
            print(f"🏀 Processing {max_frames_to_process} frames")
        
        # Create side-by-side analysis
        output_video, stats = create_side_by_side_analysis(
            video_path=selected_video,
            max_frames=max_frames_to_process
        )
        
        print(f"\n🎉 Side-by-side analysis completed!")
        print(f"📁 Video created: {output_video}")
        print(f"\n📊 Key Statistics:")
        print(f"   - Frames: {stats['processing_summary']['total_frames_processed']}")
        print(f"   - Avg players/frame: {stats['processing_summary']['average_players_per_frame']:.1f}")
        print(f"   - Ball detection: {stats['processing_summary']['ball_detection_rate']:.1f}%")
        print(f"   - Home team: {stats['processing_summary']['total_home_detections']}")
        print(f"   - Away team: {stats['processing_summary']['total_away_detections']}")
        
        print(f"\n🎬 Your side-by-side video features:")
        print("   ✅ Original game footage (left)")
        print("   ✅ 2D tactical view (right)")
        print("   ✅ Real-time player tracking")
        print("   ✅ Team color identification")
        print("   ✅ Professional court visualization")
        print("   ✅ Statistics overlay")
        
        return output_video
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# Instructions for any video
def analyze_custom_video(video_path, max_frames=None):
    """
    Analyze any basketball video with side-by-side view
    
    Usage:
        analyze_custom_video("path/to/your/video.mp4", max_frames=500)
    """
    return create_side_by_side_analysis(
        video_path=video_path,
        max_frames=max_frames
    )

if __name__ == "__main__":
    # Run analysis
    result = analyze_any_basketball_video()
    
    if result:
        print(f"\n🎯 SUCCESS! Your side-by-side basketball analysis is ready!")
        print(f"📺 Open this file to watch: {result}")
    else:
        print("\n❌ Analysis failed")
    
    print(f"\n💡 To analyze ANY basketball video:")
    print(f"   from side_by_side_basketball_analyzer import create_side_by_side_analysis")
    print(f"   create_side_by_side_analysis('your_video.mp4')")
