"""
Main script for Basketball GNN Analysis
End-to-end pipeline from tracking data to GNN insights
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add project modules to path
sys.path.append('.')

# Import our modules
from graph_builder.build_graph import BasketballGraphBuilder, create_dummy_tracking_data
from gnn_model.model import create_model
from gnn_model.train import BasketballGNNTrainer
from gnn_model.predict import BasketballGNNPredictor
from vis.visualize_graph import BasketballGraphVisualizer
from utils.yolo_tracking_parser import YOLOTrackingParser
from utils.pose_loader import PoseDataLoader


class BasketballGNNPipeline:
    """
    Complete pipeline for basketball tactical analysis using GNNs.
    """
    
    DEFAULT_MODEL_PATH = "models/basketball_gnn_gcn.pth"
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.graph_builder = BasketballGraphBuilder(**config.get('graph_builder', {}))
        self.visualizer = BasketballGraphVisualizer(**config.get('visualizer', {}))
        
        # Data storage
        self.tracking_data = None
        self.pose_data = None
        self.graphs = []
        self.model = None
        
        print(f"Initialized Basketball GNN Pipeline on {self.device}")
    
    def load_data(self, 
                  tracking_path: Optional[str] = None,
                  pose_path: Optional[str] = None,
                  data_format: str = "csv") -> Dict:
        """
        Load tracking and pose data.
        
        Args:
            tracking_path: Path to tracking data
            pose_path: Path to pose data
            data_format: Data format ("csv", "json", "yolo")
            
        Returns:
            Data loading summary
        """
        
        summary = {'tracking_loaded': False, 'pose_loaded': False}
        
        # Load tracking data
        if tracking_path and os.path.exists(tracking_path):
            try:
                if data_format == "csv":
                    self.tracking_data = pd.read_csv(tracking_path)
                elif data_format == "yolo":
                    parser = YOLOTrackingParser()
                    # Assume YOLO folder format
                    img_size = self.config.get('image_size', (1920, 1080))
                    self.tracking_data = parser.parse_detection_folder(
                        tracking_path, img_size[0], img_size[1]
                    )
                else:
                    raise ValueError(f"Unsupported format: {data_format}")
                
                print(f"Loaded tracking data: {self.tracking_data.shape}")
                summary['tracking_loaded'] = True
                
            except Exception as e:
                print(f"Error loading tracking data: {e}")
        else:
            # Create dummy data for testing
            print("Creating dummy tracking data for testing...")
            self.tracking_data = create_dummy_tracking_data(
                num_frames=self.config.get('num_test_frames', 100),
                num_players=self.config.get('num_test_players', 10)
            )
            summary['tracking_loaded'] = True
        
        # Load pose data
        if pose_path and os.path.exists(pose_path):
            try:
                pose_loader = PoseDataLoader(self.config.get('pose_format', 'mediapipe'))
                
                if data_format == "csv":
                    self.pose_data = pd.read_csv(pose_path)
                elif data_format == "json":
                    if os.path.isdir(pose_path):
                        self.pose_data = pose_loader.load_pose_folder(pose_path)
                    else:
                        if self.config.get('pose_format', 'mediapipe') == 'mediapipe':
                            self.pose_data = pose_loader.load_mediapipe_json(pose_path)
                        else:
                            self.pose_data = pose_loader.load_openpose_json(pose_path)
                
                print(f"Loaded pose data: {self.pose_data.shape}")
                summary['pose_loaded'] = True
                
            except Exception as e:
                print(f"Error loading pose data: {e}")
        
        return summary
    
    def build_graphs(self, frame_range: Optional[tuple] = None) -> List:
        """
        Build graphs from loaded data.
        
        Args:
            frame_range: (start_frame, end_frame) or None for all
            
        Returns:
            List of built graphs
        """
        
        if self.tracking_data is None:
            raise ValueError("No tracking data loaded")
        
        print("Building graphs from tracking data...")
        
        # Filter tracking data if needed
        parser = YOLOTrackingParser()
        filtered_tracking = parser.filter_tracking_data(
            self.tracking_data,
            min_track_length=self.config.get('min_track_length', 5),
            max_players=self.config.get('max_players', 12)
        )
        
        # Build graphs
        self.graphs = self.graph_builder.build_sequence_graphs(
            filtered_tracking,
            self.pose_data,
            frame_range
        )
        
        print(f"Built {len(self.graphs)} graphs")
        
        # Save graphs if requested
        if self.config.get('save_graphs', False):
            self.save_graphs()
        
        return self.graphs
    
    def save_graphs(self, output_dir: str = "data/frame_graphs"):
        """Save built graphs to disk."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, graph in enumerate(self.graphs):
            frame_id = getattr(graph, 'frame_id', i)
            output_path = os.path.join(output_dir, f"frame_{frame_id:06d}.pt")
            torch.save(graph, output_path)
        
        print(f"Saved {len(self.graphs)} graphs to {output_dir}")
    
    def load_graphs(self, input_dir: str = "data/frame_graphs") -> List:
        """Load graphs from disk."""
        
        graph_files = list(Path(input_dir).glob("*.pt"))
        self.graphs = []
        
        for graph_file in graph_files:
            graph = torch.load(graph_file, map_location=self.device)
            self.graphs.append(graph)
        
        print(f"Loaded {len(self.graphs)} graphs from {input_dir}")
        return self.graphs
    
    def train_model(self, 
                   model_type: str = "gcn",
                   num_epochs: int = 100,
                   save_model: bool = True) -> Dict:
        """
        Train GNN model on built graphs.
        
        Args:
            model_type: Type of model to train
            num_epochs: Number of training epochs
            save_model: Whether to save trained model
            
        Returns:
            Training results
        """
        
        if not self.graphs:
            raise ValueError("No graphs available for training")
        
        print(f"Training {model_type} model...")
        
        # Determine model parameters from first graph
        sample_graph = self.graphs[0]
        model_config = {
            'in_channels': sample_graph.x.shape[1],
            'hidden_channels': self.config.get('hidden_channels', 64),
            'out_channels': self.config.get('out_channels', 32)
        }
        
        # Initialize trainer
        trainer = BasketballGNNTrainer(model_type, model_config, device=str(self.device))
        
        # Train model
        history = trainer.train_unsupervised(
            self.graphs,
            num_epochs=num_epochs,
            learning_rate=self.config.get('learning_rate', 0.01),
            batch_size=self.config.get('batch_size', 16)
        )
        
        self.model = trainer.model
        
        # Save model
        if save_model:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/basketball_gnn_{model_type}.pth"
            trainer.save_model(model_path, metadata={'config': self.config})
            print(f"Model saved to {model_path}")
        
        # Plot training history
        if self.config.get('plot_training', True):
            trainer.plot_training_history()
        
        return history
    
    def load_model(self, model_path: str):
        """Load trained model."""
        
        predictor = BasketballGNNPredictor(model_path, device=str(self.device))
        self.model = predictor.model
        return predictor
    
    def analyze_tactics(self, 
                       predictor: Optional[Any] = None,
                       n_clusters: int = 2) -> Dict:
        """
        Perform tactical analysis using trained model.
        
        Args:
            predictor: Trained predictor (loads default if None)
            n_clusters: Number of clusters for team analysis
            
        Returns:
            Analysis results
        """
        
        if not self.graphs:
            raise ValueError("No graphs available for analysis")
        
        if predictor is None:
            # Try to load default model
            if os.path.exists(self.DEFAULT_MODEL_PATH):
                predictor = self.load_model(self.DEFAULT_MODEL_PATH)
            else:
                raise ValueError("No trained model available")
        
        print("Performing tactical analysis...")
        
        # Cluster players
        cluster_labels = predictor.cluster_players(self.graphs, n_clusters)
        
        # Analyze formations
        formation_analysis = predictor.analyze_formations(self.graphs, cluster_labels)
        
        # Detect patterns
        tactical_patterns = predictor.detect_tactical_patterns(self.graphs)
        
        # Combine results
        results = {
            'cluster_labels': cluster_labels,
            'formation_analysis': formation_analysis,
            'tactical_patterns': tactical_patterns,
            'num_frames': len(self.graphs),
            'avg_players_per_frame': np.mean([g.num_nodes for g in self.graphs])
        }
        
        print(f"Analysis completed for {len(self.graphs)} frames")
        
        return results
    
    def visualize_results(self, analysis_results: Dict, save_plots: bool = True):
        """
        Visualize analysis results.
        
        Args:
            analysis_results: Results from tactical analysis
            save_plots: Whether to save visualization plots
        """
        
        print("Creating visualizations...")
        
        # Graph sequence visualization
        if analysis_results['cluster_labels']:
            self.visualizer.visualize_sequence(
                self.graphs[:8],  # Show first 8 frames
                analysis_results['cluster_labels'][:8],
                save_path="results/sequence_visualization.png" if save_plots else None
            )
        
        # Formation analysis plots
        self.visualizer.plot_formation_analysis(
            analysis_results['formation_analysis'],
            save_path="results/formation_analysis.png" if save_plots else None
        )
        
        # Create animation if requested
        if self.config.get('create_animation', False):
            max_frames = min(len(self.graphs), 20)  # Limit for performance
            self.visualizer.create_animation(
                self.graphs[:max_frames],
                analysis_results['cluster_labels'][:max_frames] if analysis_results['cluster_labels'] else None,
                save_path="results/tactical_animation.gif" if save_plots else None
            )
    
    def run_complete_pipeline(self, 
                            tracking_path: Optional[str] = None,
                            pose_path: Optional[str] = None,
                            train_new_model: bool = True) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            tracking_path: Path to tracking data
            pose_path: Path to pose data
            train_new_model: Whether to train a new model
            
        Returns:
            Complete analysis results
        """
        
        print("=== Basketball GNN Tactical Analysis Pipeline ===")
        
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Step 1: Load data
        print("\n1. Loading data...")
        self.load_data(tracking_path, pose_path)
        
        # Step 2: Build graphs
        print("\n2. Building graphs...")
        self.build_graphs()
        
        if not self.graphs:
            raise ValueError("No graphs could be built from the data")
        
        # Step 3: Train or load model
        if train_new_model:
            print("\n3. Training GNN model...")
            self.train_model(
                model_type=self.config.get('model_type', 'gcn'),
                num_epochs=self.config.get('num_epochs', 50)
            )
        else:
            print("\n3. Loading existing model...")
            try:
                predictor = self.load_model(self.DEFAULT_MODEL_PATH)
            except Exception:
                print("No existing model found, training new one...")
                self.train_model()
                predictor = None
        
        # Step 4: Analyze tactics
        print("\n4. Analyzing tactics...")
        predictor = self.load_model(self.DEFAULT_MODEL_PATH)
        analysis_results = self.analyze_tactics(predictor)
        
        # Step 5: Visualize results
        print("\n5. Creating visualizations...")
        self.visualize_results(analysis_results)
        
        # Summary
        print("\n=== Analysis Summary ===")
        print(f"Frames analyzed: {analysis_results['num_frames']}")
        print(f"Average players per frame: {analysis_results['avg_players_per_frame']:.1f}")
        
        if analysis_results['formation_analysis']['formation_stability']:
            avg_stability = np.mean(analysis_results['formation_analysis']['formation_stability'])
            print(f"Average formation stability: {avg_stability:.3f}")
        
        if analysis_results['tactical_patterns']['formation_transitions']:
            avg_transitions = np.mean(analysis_results['tactical_patterns']['formation_transitions'])
            print(f"Average formation transition rate: {avg_transitions:.3f}")
        
        print("\nResults saved to 'results/' directory")
        
        return analysis_results


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Basketball GNN Tactical Analysis")
    parser.add_argument('--tracking', type=str, help="Path to tracking data")
    parser.add_argument('--video', type=str, help="Path to basketball video file")
    parser.add_argument('--pose', type=str, help="Path to pose data")
    parser.add_argument('--config', type=str, help="Path to config file")
    parser.add_argument('--demo', action='store_true', help="Run demo with dummy data")
    parser.add_argument('--train', action='store_true', help="Train new model")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--max_frames', type=int, default=None, help="Max frames to process from video")
    parser.add_argument('--confidence', type=float, default=0.5, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'graph_builder': {
            'proximity_threshold': 150.0,
            'min_players': 3,
            'max_players': 12
        },
        'model_type': 'gcn',
        'hidden_channels': 64,
        'out_channels': 32,
        'num_epochs': args.epochs,
        'learning_rate': 0.01,
        'batch_size': 16,
        'plot_training': True,
        'save_graphs': False,
        'create_animation': False
    }
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Initialize pipeline
    pipeline = BasketballGNNPipeline(config)
    
    # Handle video input
    tracking_path = args.tracking
    if args.video:
        print(f"Processing video: {args.video}")
        from video_processor import BasketballVideoProcessor
        
        # Process video to extract tracking data
        processor = BasketballVideoProcessor()
        video_output_dir = "video_output"
        
        tracking_path = processor.process_video(
            args.video,
            video_output_dir,
            args.max_frames,
            args.confidence,
            save_annotated=True
        )
        
        if tracking_path:
            print(f"✅ Video processed successfully!")
            print(f"📊 Tracking data: {tracking_path}")
            print(f"🎥 Annotated video: {video_output_dir}/annotated_video.mp4")
        else:
            print("❌ Video processing failed!")
            return
    
    if args.demo:
        print("Running demo with dummy data...")
        pipeline.run_complete_pipeline(train_new_model=True)
    else:
        # Run with real data
        pipeline.run_complete_pipeline(
            tracking_path=tracking_path,
            pose_path=args.pose,
            train_new_model=args.train
        )
    
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
