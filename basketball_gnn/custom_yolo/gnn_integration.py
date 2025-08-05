#!/usr/bin/env python3
"""
GNN Integration with Custom Basketball YOLO
Integrates custom-trained YOLO model with the existing GNN system
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from custom_yolo.enhanced_processor import EnhancedBasketballProcessor
from graph_builder.graph_builder import BasketballGraphBuilder
from gnn_model.model import BasketballGCN, BasketballGraphSAGE
from gnn_model.train import train_basketball_gnn
from gnn_model.predict import predict_tactical_patterns
from vis.visualizer import BasketballVisualizer

class CustomYOLO_GNN_Integration:
    """Integrates custom YOLO model with basketball GNN analysis."""
    
    def __init__(self, custom_yolo_path: str = None):
        """Initialize with custom YOLO model path."""
        self.custom_yolo_path = custom_yolo_path
        self.enhanced_processor = EnhancedBasketballProcessor(custom_yolo_path)
        self.graph_builder = BasketballGraphBuilder()
        self.visualizer = BasketballVisualizer()
        
        print("🔗 Basketball GNN + Custom YOLO Integration")
        print(f"   Custom YOLO: {'✅' if custom_yolo_path else '❌ Using default'}")
        print(f"   Model loaded: {'✅' if self.enhanced_processor.is_custom_model else '❌'}")
        
    def process_video_with_custom_yolo(self, video_path: str, output_dir: str = None,
                                     confidence: float = 0.25, max_frames: int = None) -> Dict:
        """Process video using custom YOLO and prepare for GNN analysis."""
        print("\n🎥 Step 1: Enhanced Video Processing with Custom YOLO")
        print("=" * 50)
        
        # Process video with enhanced detection
        enhanced_results = self.enhanced_processor.process_video_enhanced(
            video_path, output_dir, confidence, max_frames
        )
        
        # Convert enhanced detections to GNN-compatible format
        gnn_data = self._convert_to_gnn_format(enhanced_results)
        
        # Save GNN-compatible data
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(f"gnn_analysis_{Path(video_path).stem}")
        output_path.mkdir(exist_ok=True)
        
        gnn_data_file = output_path / "gnn_tracking_data.csv"
        pd.DataFrame(gnn_data).to_csv(gnn_data_file, index=False)
        
        print(f"✅ GNN-compatible data saved: {gnn_data_file}")
        
        return {
            'enhanced_results': enhanced_results,
            'gnn_data': gnn_data,
            'gnn_data_file': str(gnn_data_file),
            'output_dir': str(output_path)
        }
        
    def _convert_to_gnn_format(self, enhanced_results: Dict) -> List[Dict]:
        """Convert enhanced YOLO results to GNN-compatible format."""
        gnn_data = []
        
        for frame_data in enhanced_results['frame_detections']:
            frame_num = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            detections = frame_data['detections']
            analysis = frame_data['analysis']
            
            # Process players (main focus for GNN)
            for i, player in enumerate(detections['players']):
                x, y = player['center']
                bbox = player['bbox']
                
                # Determine team assignment from clustering if available
                team = 'unknown'
                if 'clusters' in analysis['player_formations']:
                    clusters = analysis['player_formations']['clusters']
                    if i in clusters.get('team_1', []):
                        team = 'team_1'
                    elif i in clusters.get('team_2', []):
                        team = 'team_2'
                        
                # Additional features from enhanced analysis
                features = {
                    'frame_id': frame_num,
                    'timestamp': timestamp,
                    'player_id': f"player_{i}",
                    'x': x,
                    'y': y,
                    'team': team,
                    'confidence': player['confidence'],
                    'bbox_area': player['area'],
                    'aspect_ratio': player['aspect_ratio'],
                    'court_region': self._get_player_court_region(player, analysis),
                    'near_ball': self._is_near_ball(player, detections['ball']),
                    'near_basket': self._is_near_basket(player, detections['baskets']),
                    'ball_possession': self._has_ball_possession(i, analysis['ball_possession'])
                }
                
                gnn_data.append(features)
                
            # Add ball information if detected
            for ball in detections['ball']:
                x, y = ball['center']
                features = {
                    'frame_id': frame_num,
                    'timestamp': timestamp,
                    'player_id': 'ball',
                    'x': x,
                    'y': y,
                    'team': 'ball',
                    'confidence': ball['confidence'],
                    'bbox_area': ball['area'],
                    'aspect_ratio': ball['aspect_ratio'],
                    'court_region': 'ball_region',
                    'near_ball': True,
                    'near_basket': len(detections['baskets']) > 0,
                    'ball_possession': False
                }
                gnn_data.append(features)
                
        return gnn_data
        
    def _get_player_court_region(self, player: Dict, analysis: Dict) -> str:
        """Determine court region for player."""
        player_id = f"players_{player['id']}"
        regions = analysis['court_regions']
        
        for region, info in regions.items():
            if player_id in info['objects']:
                return region
                
        return 'unknown_region'
        
    def _is_near_ball(self, player: Dict, balls: List[Dict]) -> bool:
        """Check if player is near the ball."""
        if not balls:
            return False
            
        player_pos = np.array(player['center'])
        ball_pos = np.array(balls[0]['center'])
        distance = np.linalg.norm(player_pos - ball_pos)
        
        return distance < 50  # Within 50 pixels
        
    def _is_near_basket(self, player: Dict, baskets: List[Dict]) -> bool:
        """Check if player is near any basket."""
        if not baskets:
            return False
            
        player_pos = np.array(player['center'])
        
        for basket in baskets:
            basket_pos = np.array(basket['center'])
            distance = np.linalg.norm(player_pos - basket_pos)
            if distance < 100:  # Within 100 pixels
                return True
                
        return False
        
    def _has_ball_possession(self, player_index: int, ball_possession: Dict) -> bool:
        """Check if player has ball possession."""
        if not ball_possession.get('possession_info'):
            return False
            
        return (ball_possession['possession_info']['closest_player_id'] == player_index and
                ball_possession['likely_possessed'])
                
    def build_enhanced_graphs(self, gnn_data_file: str, output_dir: str) -> str:
        """Build graphs using enhanced player data."""
        print("\n🕸️ Step 2: Enhanced Graph Construction")
        print("=" * 50)
        
        # Load GNN data
        df = pd.read_csv(gnn_data_file)
        
        # Build graphs with enhanced features
        graphs_file = self.graph_builder.build_graphs_from_tracking(
            gnn_data_file, 
            os.path.join(output_dir, "enhanced_graphs.pkl")
        )
        
        print(f"✅ Enhanced graphs built: {graphs_file}")
        
        # Generate enhanced visualizations
        self._create_enhanced_visualizations(df, output_dir)
        
        return graphs_file
        
    def _create_enhanced_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create visualizations with enhanced data."""
        output_path = Path(output_dir)
        
        # Enhanced court visualization with object types
        self.visualizer.plot_court_with_enhanced_objects(df, output_path / "enhanced_court_view.png")
        
        # Ball possession analysis
        self.visualizer.plot_ball_possession_timeline(df, output_path / "ball_possession_timeline.png")
        
        # Team formation analysis with custom YOLO data
        self.visualizer.plot_enhanced_formations(df, output_path / "enhanced_formations.png")
        
        print("✅ Enhanced visualizations created")
        
    def train_enhanced_gnn(self, graphs_file: str, output_dir: str, 
                         model_type: str = "gcn", epochs: int = 100) -> str:
        """Train GNN with enhanced features from custom YOLO."""
        print(f"\n🧠 Step 3: Enhanced GNN Training ({model_type.upper()})")
        print("=" * 50)
        
        # Train model with enhanced features
        model_path = train_basketball_gnn(
            graphs_file,
            model_type=model_type,
            epochs=epochs,
            output_dir=output_dir,
            enhanced_features=True  # Use enhanced features from custom YOLO
        )
        
        print(f"✅ Enhanced GNN model trained: {model_path}")
        return model_path
        
    def predict_enhanced_patterns(self, model_path: str, graphs_file: str, 
                                output_dir: str) -> Dict:
        """Predict tactical patterns using enhanced GNN."""
        print("\n🔮 Step 4: Enhanced Tactical Pattern Prediction")
        print("=" * 50)
        
        # Predict patterns with enhanced model
        predictions = predict_tactical_patterns(
            model_path,
            graphs_file,
            output_dir=output_dir,
            enhanced_mode=True
        )
        
        print("✅ Enhanced tactical patterns predicted")
        return predictions
        
    def run_complete_analysis(self, video_path: str, output_dir: str = None,
                            confidence: float = 0.25, epochs: int = 100,
                            model_type: str = "gcn", max_frames: int = None) -> Dict:
        """Run complete analysis pipeline with custom YOLO + GNN."""
        print("🚀 COMPLETE BASKETBALL ANALYSIS PIPELINE")
        print("   Custom YOLO → Enhanced Detection → GNN Analysis")
        print("=" * 60)
        
        if output_dir is None:
            output_dir = f"complete_analysis_{Path(video_path).stem}"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        try:
            # Step 1: Enhanced video processing
            video_results = self.process_video_with_custom_yolo(
                video_path, str(output_dir), confidence, max_frames
            )
            results['video_processing'] = video_results
            
            # Step 2: Enhanced graph building
            graphs_file = self.build_enhanced_graphs(
                video_results['gnn_data_file'], 
                str(output_dir)
            )
            results['graph_building'] = {'graphs_file': graphs_file}
            
            # Step 3: Enhanced GNN training
            model_path = self.train_enhanced_gnn(
                graphs_file, str(output_dir), model_type, epochs
            )
            results['gnn_training'] = {'model_path': model_path}
            
            # Step 4: Enhanced prediction
            predictions = self.predict_enhanced_patterns(
                model_path, graphs_file, str(output_dir)
            )
            results['predictions'] = predictions
            
            # Step 5: Generate comprehensive report
            self._generate_comprehensive_report(results, output_dir)
            
            print("\n🎉 COMPLETE ANALYSIS FINISHED!")
            print(f"📁 All results saved to: {output_dir}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            raise
            
    def _generate_comprehensive_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive analysis report."""
        report = {
            "analysis_type": "Custom YOLO + Basketball GNN",
            "timestamp": pd.Timestamp.now().isoformat(),
            "custom_yolo_used": self.enhanced_processor.is_custom_model,
            "custom_yolo_path": self.custom_yolo_path,
            "summary": {
                "video_processed": "video_processing" in results,
                "graphs_built": "graph_building" in results,
                "model_trained": "gnn_training" in results,
                "predictions_made": "predictions" in results
            }
        }
        
        # Add detailed statistics
        if "video_processing" in results:
            enhanced_stats = results["video_processing"]["enhanced_results"]["summary_statistics"]
            report["detection_statistics"] = enhanced_stats
            
        # Add model performance
        if "predictions" in results:
            report["prediction_results"] = results["predictions"]
            
        # Save report
        report_file = output_dir / "comprehensive_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Comprehensive report saved: {report_file}")
        
    def compare_models(self, video_path: str, custom_model_path: str = None,
                      output_dir: str = "model_comparison") -> Dict:
        """Compare custom YOLO vs default YOLO in GNN pipeline."""
        print("🔄 MODEL COMPARISON: Custom YOLO vs Default YOLO")
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        comparison_results = {}
        
        # Test with default YOLO
        print("\n1️⃣ Testing with Default YOLO...")
        default_integration = CustomYOLO_GNN_Integration(custom_yolo_path=None)
        default_results = default_integration.run_complete_analysis(
            video_path, 
            str(output_dir / "default_yolo_analysis"),
            max_frames=500  # Limit for comparison
        )
        comparison_results['default_yolo'] = default_results
        
        # Test with custom YOLO (if available)
        if custom_model_path and Path(custom_model_path).exists():
            print("\n2️⃣ Testing with Custom YOLO...")
            custom_integration = CustomYOLO_GNN_Integration(custom_yolo_path=custom_model_path)
            custom_results = custom_integration.run_complete_analysis(
                video_path,
                str(output_dir / "custom_yolo_analysis"),
                max_frames=500  # Limit for comparison
            )
            comparison_results['custom_yolo'] = custom_results
        else:
            print("⚠️ Custom YOLO model not found, skipping custom analysis")
            
        # Generate comparison report
        self._generate_comparison_report(comparison_results, output_dir)
        
        return comparison_results
        
    def _generate_comparison_report(self, comparison_results: Dict, output_dir: Path):
        """Generate model comparison report."""
        report = {
            "comparison_type": "Custom YOLO vs Default YOLO for Basketball GNN",
            "timestamp": pd.Timestamp.now().isoformat(),
            "models_compared": list(comparison_results.keys())
        }
        
        # Compare detection statistics
        detection_comparison = {}
        for model_name, results in comparison_results.items():
            if "video_processing" in results:
                stats = results["video_processing"]["enhanced_results"]["summary_statistics"]
                detection_comparison[model_name] = {
                    "total_frames": stats["total_frames"],
                    "frames_with_players": stats["frames_with_players"],
                    "frames_with_ball": stats["frames_with_ball"],
                    "average_players_per_frame": stats["average_players_per_frame"],
                    "ball_detection_rate": stats["frames_with_ball"] / stats["total_frames"] if stats["total_frames"] > 0 else 0
                }
                
        report["detection_comparison"] = detection_comparison
        
        # Add recommendations
        if len(detection_comparison) >= 2:
            custom_rate = detection_comparison.get("custom_yolo", {}).get("ball_detection_rate", 0)
            default_rate = detection_comparison.get("default_yolo", {}).get("ball_detection_rate", 0)
            
            if custom_rate > default_rate:
                report["recommendation"] = "Custom YOLO shows better ball detection performance"
            elif custom_rate < default_rate:
                report["recommendation"] = "Default YOLO performs better for this video"
            else:
                report["recommendation"] = "Both models show similar performance"
                
        # Save comparison report
        comparison_file = output_dir / "model_comparison_report.json"
        with open(comparison_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"📊 Model comparison report saved: {comparison_file}")


def main():
    """Main function for interactive custom YOLO + GNN analysis."""
    print("🏀 Custom Basketball YOLO + GNN Integration")
    print("=" * 50)
    
    print("\nSelect an option:")
    print("1. Run complete analysis with custom YOLO")
    print("2. Process video only (custom YOLO)")
    print("3. Compare custom vs default YOLO")
    print("4. Train custom YOLO model first")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        video_path = input("Enter video path: ").strip()
        custom_model = input("Enter custom YOLO model path (optional): ").strip() or None
        
        if Path(video_path).exists():
            integration = CustomYOLO_GNN_Integration(custom_model)
            integration.run_complete_analysis(video_path)
        else:
            print("❌ Video file not found!")
            
    elif choice == "2":
        video_path = input("Enter video path: ").strip()
        custom_model = input("Enter custom YOLO model path (optional): ").strip() or None
        
        if Path(video_path).exists():
            integration = CustomYOLO_GNN_Integration(custom_model)
            integration.process_video_with_custom_yolo(video_path)
        else:
            print("❌ Video file not found!")
            
    elif choice == "3":
        video_path = input("Enter video path: ").strip()
        custom_model = input("Enter custom YOLO model path: ").strip()
        
        if Path(video_path).exists():
            integration = CustomYOLO_GNN_Integration()
            integration.compare_models(video_path, custom_model)
        else:
            print("❌ Video file not found!")
            
    elif choice == "4":
        print("🔗 Please run the custom YOLO trainer first:")
        print("   python custom_yolo/yolo_trainer.py")
        print("   Then come back to run the integration!")
        
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
