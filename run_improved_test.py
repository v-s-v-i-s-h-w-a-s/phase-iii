#!/usr/bin/env python3
"""
Improved Basketball Detection Test
Generate new video with improved labeling accuracy
"""

import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")

try:
    from improved_basketball_intelligence import process_video_with_improved_detection
    print("✅ Improved basketball intelligence imported successfully")
except ImportError as e:
    print(f"❌ Error importing improved basketball intelligence: {e}")
    exit(1)

def run_improved_detection_test():
    """Run improved detection on the latest test video"""
    
    print("🏀 Improved Basketball Detection Test")
    print("=" * 50)
    
    # Use your latest test video
    video_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4")
    
    # Enhanced model path
    model_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt")
    
    # Check paths
    if not video_path.exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    if not model_path.exists():
        print(f"❌ Enhanced model not found: {model_path}")
        return
    
    print(f"✅ Test video: {video_path}")
    print(f"✅ Enhanced model: {model_path}")
    
    # Generate output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III") / f"improved_basketball_detection_{timestamp}.mp4"
    
    print(f"🎯 Output will be: {output_path}")
    print()
    
    try:
        # Process video with improved detection
        print("🚀 Starting improved basketball detection...")
        stats = process_video_with_improved_detection(
            str(video_path), 
            str(model_path), 
            str(output_path)
        )
        
        print(f"\n🎉 Improved detection completed successfully!")
        print(f"📁 Output video: {output_path}")
        print(f"📊 Performance Statistics:")
        print(f"   - Total frames processed: {stats.get('total_frames_processed', 0)}")
        print(f"   - Average FPS: {stats.get('average_fps', 0):.2f}")
        print(f"   - Total detections: {stats.get('total_detections', 0)}")
        print(f"   - Average detections per frame: {stats.get('average_detections_per_frame', 0):.2f}")
        
        if 'class_distribution' in stats:
            print(f"   - Class distribution:")
            for class_name, count in stats['class_distribution'].items():
                print(f"     * {class_name}: {count}")
        
        if 'team_assignments' in stats:
            print(f"   - Team assignments:")
            for team, count in stats['team_assignments'].items():
                print(f"     * {team}: {count}")
        
        # Save detailed statistics
        stats_path = str(output_path).replace('.mp4', '_detailed_stats.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📋 Detailed stats saved: {stats_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during improved detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_improved_detection_test()
    if success:
        print("\n✅ Improved basketball detection test completed successfully!")
        print("🎯 The new video should have much better labeling accuracy!")
    else:
        print("\n❌ Improved detection test failed!")
