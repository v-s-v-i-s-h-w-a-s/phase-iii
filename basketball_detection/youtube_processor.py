"""
YouTube Basketball Video Processor for Tracker Branch
Download and process basketball video using the tracker branch system
"""

import yt_dlp
import cv2
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference import BasketballInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeBasketballProcessor:
    """Download and process YouTube basketball videos"""
    
    def __init__(self):
        self.output_dir = Path("./downloads")
        self.output_dir.mkdir(exist_ok=True)
        
    def download_video(self, url: str, max_duration: int = 300) -> str:
        """Download video from YouTube"""
        try:
            # yt-dlp options
            ydl_opts = {
                'format': 'best[height<=720][ext=mp4]',  # 720p MP4
                'outtmpl': str(self.output_dir / 'basketball_video.%(ext)s'),
                'writesubtitles': False,
                'writeautomaticsub': False,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }],
                # Limit duration to 5 minutes for processing
                'external_downloader_args': {
                    'ffmpeg': ['-t', str(max_duration)]
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video: {url}")
                ydl.download([url])
                
                # Find the downloaded file
                video_files = list(self.output_dir.glob("basketball_video.*"))
                if video_files:
                    return str(video_files[0])
                    
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return None
    
    def process_with_tracker_system(self, video_path: str):
        """Process video using the tracker branch inference system"""
        try:
            logger.info("Initializing Basketball Inference System...")
            
            # Initialize the inference engine
            inference = BasketballInference()
            
            # Load model
            inference.load_model()
            
            # Process video
            output_path = "tracker_basketball_analysis.mp4"
            logger.info(f"Processing video: {video_path}")
            
            # Run inference on video
            results = inference.process_video(video_path, output_path)
            
            logger.info(f"Processing complete! Output: {output_path}")
            return output_path, results
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None, None

def main():
    """Main function"""
    print("🏀 YouTube Basketball Video Processor - Tracker Branch")
    print("=" * 65)
    
    # YouTube URL provided by user
    video_url = "https://youtu.be/I7pTpMjqNRM?si=xjP21KxGEeVI8Pfn"
    
    processor = YouTubeBasketballProcessor()
    
    print(f"📹 Video URL: {video_url}")
    print("⬇️ Downloading video...")
    
    # Download video
    video_path = processor.download_video(video_url, max_duration=300)  # 5 minutes max
    
    if not video_path:
        print("❌ Failed to download video!")
        return
    
    print(f"✅ Video downloaded: {video_path}")
    print("🔄 Starting basketball detection analysis...")
    
    # Process with tracker system
    output_video, results = processor.process_with_tracker_system(video_path)
    
    if output_video and Path(output_video).exists():
        print(f"\n🎯 SUCCESS! Analysis complete!")
        print(f"📹 Output video: {output_video}")
        print(f"📊 The tracker system processed the video with:")
        print("   - Player tracking and detection")
        print("   - Ball movement analysis") 
        print("   - Court element recognition")
        print("   - Real-time inference annotations")
        
        if results:
            print(f"\n📈 Analysis Results:")
            print(f"   {results}")
    else:
        print("❌ Video processing failed!")
        print("Please check the logs for error details")

if __name__ == "__main__":
    main()
