#!/usr/bin/env python3
"""
Quick Video Analysis Script for Basketball GNN
Easy-to-use script for processing basketball videos
"""

import os
import sys
import argparse
from pathlib import Path

def process_basketball_video(video_path: str, 
                           max_frames: int = 300,
                           confidence: float = 0.6,
                           train_epochs: int = 20):
    """
    Process a basketball video and run GNN analysis.
    
    Args:
        video_path: Path to your basketball video
        max_frames: Maximum frames to process (to limit processing time)
        confidence: Detection confidence threshold (0.0-1.0)
        train_epochs: Number of training epochs for the GNN
    """
    
    print("🏀 Basketball Video Analysis with GNN")
    print("=" * 50)
    print()
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"📹 Input video: {video_path}")
    print(f"🎯 Processing {max_frames} frames (max)")
    print(f"📊 Detection confidence: {confidence}")
    print(f"🧠 Training epochs: {train_epochs}")
    print()
    
    try:
        # Step 1: Process video to extract player tracking
        print("🔍 Step 1: Extracting player positions from video...")
        from video_processor import BasketballVideoProcessor
        
        processor = BasketballVideoProcessor()
        video_output_dir = f"video_analysis_{Path(video_path).stem}"
        
        tracking_csv = processor.process_video(
            video_path,
            video_output_dir,
            max_frames=max_frames,
            confidence_threshold=confidence,
            save_annotated=True
        )
        
        if not tracking_csv:
            print("❌ Failed to extract tracking data from video")
            return False
        
        print(f"✅ Tracking data saved: {tracking_csv}")
        print(f"🎥 Annotated video saved: {video_output_dir}/annotated_video.mp4")
        print()
        
        # Step 2: Run GNN analysis
        print("🧠 Step 2: Running GNN tactical analysis...")
        
        # Import main pipeline
        from main import BasketballGNNPipeline
        
        # Configuration for the analysis
        config = {
            'graph_builder': {
                'proximity_threshold': 200.0,  # Adjusted for video analysis
                'min_players': 2,
                'max_players': 15
            },
            'model_type': 'gcn',
            'hidden_channels': 64,
            'out_channels': 32,
            'num_epochs': train_epochs,
            'learning_rate': 0.01,
            'batch_size': 8,  # Smaller batch for video data
            'plot_training': True,
            'save_graphs': False,
            'create_animation': True  # Create animation for video results
        }
        
        # Initialize and run pipeline
        pipeline = BasketballGNNPipeline(config)
        results = pipeline.run_complete_pipeline(
            tracking_path=tracking_csv,
            train_new_model=True
        )
        
        print("✅ GNN analysis completed!")
        print()
        
        # Step 3: Results summary
        print("📊 Analysis Results:")
        print("-" * 30)
        print(f"• Frames analyzed: {results['num_frames']}")
        print(f"• Average players per frame: {results['avg_players_per_frame']:.1f}")
        
        if results['formation_analysis']['formation_stability']:
            import numpy as np
            avg_stability = np.mean(results['formation_analysis']['formation_stability'])
            print(f"• Formation stability: {avg_stability:.3f}")
        
        print()
        print("📁 Generated Files:")
        print(f"• Original video: {video_path}")
        print(f"• Annotated video: {video_output_dir}/annotated_video.mp4")
        print(f"• Tracking data: {tracking_csv}")
        print(f"• GNN visualizations: results/")
        print(f"• Trained model: models/basketball_gnn_gcn.pth")
        print()
        
        print("🎉 Analysis complete! Check the output files for results.")
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your video contains clear basketball gameplay")
        print("2. Try adjusting the confidence threshold (--confidence)")
        print("3. Reduce max_frames if processing is slow (--max_frames)")
        print("4. Ensure you have enough disk space for outputs")
        return False


def main():
    """Command line interface for video analysis."""
    
    parser = argparse.ArgumentParser(
        description="Analyze basketball videos with Graph Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_video.py my_game.mp4
  python analyze_video.py my_game.mp4 --max_frames 200 --confidence 0.7
  python analyze_video.py my_game.mp4 --epochs 30
        """
    )
    
    parser.add_argument('video_path', 
                       help='Path to your basketball video file')
    parser.add_argument('--max_frames', type=int, default=300,
                       help='Maximum frames to process (default: 300)')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Player detection confidence (0.0-1.0, default: 0.6)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='GNN training epochs (default: 20)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence < 0.0 or args.confidence > 1.0:
        print("❌ Confidence must be between 0.0 and 1.0")
        return
    
    if args.max_frames < 10:
        print("❌ max_frames must be at least 10")
        return
    
    # Run analysis
    success = process_basketball_video(
        args.video_path,
        args.max_frames,
        args.confidence,
        args.epochs
    )
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Watch the annotated video to see player detections")
        print("2. Check results/ folder for tactical visualizations")
        print("3. Experiment with different confidence thresholds")
        print("4. Try training with more epochs for better results")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
