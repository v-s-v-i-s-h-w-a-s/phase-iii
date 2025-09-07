"""
Run Basketball Inference on New Video
Process the downloaded YouTube video using the tracker inference system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference import BasketballInference

def run_inference_on_new_video():
    """Run inference on the newly downloaded video"""
    
    # Video paths
    input_video = "downloads/basketball_video.mp4"
    output_video = "new_video_analysis.mp4"
    
    print("🏀 Basketball Detection - New Video Analysis")
    print("=" * 55)
    
    if not Path(input_video).exists():
        print(f"❌ Video not found: {input_video}")
        return
    
    print(f"📹 Input video: {input_video}")
    print(f"📹 Output video: {output_video}")
    print()
    
    try:
        # Initialize inference engine
        print("🔄 Initializing Basketball Inference Engine...")
        inference = BasketballInference()
        
        # Load model
        print("📥 Loading detection model...")
        inference.load_model()
        
        # Process video
        print("🎯 Processing video with basketball detection...")
        print("This will detect and track:")
        print("  - Players (blue boxes)")
        print("  - Referees (green boxes)")  
        print("  - Basketball (orange boxes)")
        print("  - Hoops (purple boxes)")
        print()
        
        results = inference.process_video(input_video, output_video)
        
        print("✅ Video processing complete!")
        print(f"📊 Results: {results}")
        
        # Check output file
        if Path(output_video).exists():
            size = Path(output_video).stat().st_size
            print(f"📹 Output video created: {output_video}")
            print(f"📏 File size: {size/1024/1024:.1f} MB")
        
        return output_video
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return None

if __name__ == "__main__":
    run_inference_on_new_video()
