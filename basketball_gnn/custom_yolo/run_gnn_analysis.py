#!/usr/bin/env python3
"""
Run GNN Analysis with Custom YOLO Detections
Simple script to run basketball tactical analysis
"""

import sys
import os
from pathlib import Path
import json

def run_gnn_analysis():
    """Run GNN analysis with the custom YOLO detections."""
    
    print("Basketball GNN Analysis with Custom YOLO")
    print("=" * 50)
    
    # Check if we have the detection data
    detections_file = "custom_yolo_detections.json"
    tracks_file = "custom_yolo_player_tracks.csv"
    video_file = "../video_analysis_hawks_vs_knicks/annotated_video.mp4"
    
    if not Path(detections_file).exists():
        print(f"Detection data not found: {detections_file}")
        print("Please run the custom YOLO analysis first!")
        return
    
    # Load detection statistics
    with open(detections_file, 'r') as f:
        detection_data = json.load(f)
    
    print("Custom YOLO Detection Results:")
    print(f"  Video: {detection_data['video_info']['path']}")
    print(f"  Frames: {detection_data['video_info']['total_frames']}")
    print(f"  Resolution: {detection_data['video_info']['width']}x{detection_data['video_info']['height']}")
    
    print("\nDetection Statistics:")
    for class_name, count in detection_data['statistics'].items():
        print(f"  {class_name}: {count} detections")
    
    print(f"\nTotal detections: {sum(detection_data['statistics'].values())}")
    
    # Try to run the main analysis
    print("\nRunning Basketball Tactical Analysis...")
    
    try:
        # Change to parent directory
        os.chdir('..')
        
        # Import and use the video processor
        from video_processor import VideoProcessor
        from gnn_model.gnn_tactical_analyzer import GNNTacticalAnalyzer
        
        print("Initializing video processor...")
        processor = VideoProcessor()
        
        print("Processing video with enhanced basketball detection...")
        
        # Process the video
        frames_data, player_data = processor.process_video(
            video_file,
            max_frames=300,
            confidence_threshold=0.25,
            use_custom_detections=True,
            custom_detections_file=f"custom_yolo/{detections_file}"
        )
        
        if frames_data:
            print(f"Successfully processed {len(frames_data)} frames")
            
            # Run GNN analysis
            print("Running GNN tactical analysis...")
            
            analyzer = GNNTacticalAnalyzer()
            
            # Train and analyze
            results = analyzer.analyze_tactical_patterns(
                frames_data, 
                player_data,
                epochs=50
            )
            
            print("GNN Analysis Complete!")
            print("Results saved to results/ directory")
            
            # Show summary
            if 'tactical_insights' in results:
                print("\nTactical Insights:")
                for insight in results['tactical_insights'][:5]:  # Show first 5
                    print(f"  - {insight}")
            
        else:
            print("No frame data was processed")
            
    except Exception as e:
        print(f"Error during GNN analysis: {str(e)}")
        print("\nAlternative: Use the detection data for manual analysis")
        print(f"Detection file: {detections_file}")
        print(f"Tracks file: {tracks_file}")
        
    finally:
        # Return to custom_yolo directory
        if Path('custom_yolo').exists():
            os.chdir('custom_yolo')
    
    print("\nAnalysis Complete!")
    print("Check the results/ directory for output files")

if __name__ == "__main__":
    run_gnn_analysis()
