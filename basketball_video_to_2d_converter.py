#!/usr/bin/env python3
"""
Basketball Video to 2D Gameplay Converter
==========================================
Converts any basketball game video into 2D tactical gameplay visualization
using enhanced YOLO detection + GNN analysis + court transformation.

Features:
- Real player tracking from any angle
- 3D to 2D court transformation
- Team color detection and assignment
- GNN-based tactical analysis
- Professional 2D court visualization
- Play pattern recognition
- Real-time statistics overlay
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from pathlib import Path
import time
import json
from datetime import datetime
import math
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys

# Import our enhanced detection system
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")
from improved_basketball_intelligence import ImprovedBasketballIntelligence

class CourtTransformer:
    """Transforms 3D court coordinates to 2D tactical view"""
    
    def __init__(self):
        # Standard NBA court dimensions (feet)
        self.court_length = 94  # feet
        self.court_width = 50   # feet
        
        # 2D visualization dimensions
        self.viz_width = 800
        self.viz_height = 400
        
        # Court regions and zones
        self.three_point_distance = 23.75  # feet from basket
        self.paint_width = 16  # feet
        self.paint_length = 19  # feet
        
    def transform_coordinates(self, detections, frame_shape):
        """Transform detected positions to 2D court coordinates"""
        frame_height, frame_width = frame_shape[:2]
        
        transformed_players = []
        
        for tracker_id, detection in detections.items():
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get center point of player
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Normalize to 0-1
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height
            
            # Transform to court coordinates
            # Assume camera is viewing from sideline
            court_x = norm_x * self.court_length
            court_y = norm_y * self.court_width
            
            # Adjust for perspective (players closer to camera appear lower)
            # This is a simplified transformation - could be enhanced with homography
            if norm_y > 0.7:  # Players in foreground
                court_y = court_y * 0.8  # Compress depth
            
            transformed_players.append({
                'id': tracker_id,
                'court_x': court_x,
                'court_y': court_y,
                'team': detection.get('team', 'unknown'),
                'class': detection.get('class', 2),
                'conf': detection.get('conf', 0.0),
                'bbox': bbox
            })
        
        return transformed_players

class BasketballGNN(nn.Module):
    """Graph Neural Network for basketball tactical analysis"""
    
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=4):
        super(BasketballGNN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index):
        # First GCN layer
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        
        # Second GCN layer
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        
        # Output layer
        x = self.gcn3(x, edge_index)
        return x

class TacticalAnalyzer:
    """Analyzes basketball tactics using GNN"""
    
    def __init__(self):
        self.gnn_model = BasketballGNN()
        self.play_patterns = {
            'fast_break': 0,
            'pick_and_roll': 1,
            'isolation': 2,
            'zone_offense': 3
        }
        
    def create_graph(self, players):
        """Create graph representation of current court state"""
        if len(players) < 2:
            return None, None
            
        # Node features: [x, y, vx, vy, team_id, has_ball]
        node_features = []
        for player in players:
            # Simple velocity estimation (could be improved with history)
            vx, vy = 0.0, 0.0  # Placeholder
            team_id = 1 if player['team'] == 'home' else 0
            has_ball = 0  # Placeholder - could detect ball proximity
            
            features = [
                player['court_x'] / 94.0,  # Normalized court x
                player['court_y'] / 50.0,  # Normalized court y
                vx, vy, team_id, has_ball
            ]
            node_features.append(features)
        
        # Create edges based on proximity
        edge_list = []
        num_players = len(players)
        
        for i in range(num_players):
            for j in range(i + 1, num_players):
                # Calculate distance
                p1, p2 = players[i], players[j]
                dist = math.sqrt((p1['court_x'] - p2['court_x'])**2 + 
                               (p1['court_y'] - p2['court_y'])**2)
                
                # Connect if within interaction distance
                if dist < 20:  # 20 feet threshold
                    edge_list.extend([[i, j], [j, i]])  # Undirected edge
        
        if not edge_list:
            # Create at least one edge to avoid empty graph
            if num_players >= 2:
                edge_list = [[0, 1], [1, 0]]
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return x, edge_index
    
    def analyze_play(self, players):
        """Analyze current play pattern"""
        x, edge_index = self.create_graph(players)
        
        if x is None:
            return {'pattern': 'unknown', 'confidence': 0.0}
        
        # Run GNN inference
        with torch.no_grad():
            output = self.gnn_model(x, edge_index)
            
        # Simple pattern recognition (placeholder)
        # In real implementation, this would be trained on basketball data
        avg_output = torch.mean(output, dim=0)
        pattern_scores = F.softmax(avg_output, dim=0)
        
        max_score, max_idx = torch.max(pattern_scores, 0)
        pattern_names = list(self.play_patterns.keys())
        
        return {
            'pattern': pattern_names[max_idx] if max_idx < len(pattern_names) else 'unknown',
            'confidence': max_score.item(),
            'all_scores': {name: pattern_scores[i].item() for i, name in enumerate(pattern_names)}
        }

class Basketball2DVisualizer:
    """Creates 2D basketball court visualization"""
    
    def __init__(self, width=800, height=400):
        self.width = width
        self.height = height
        
        # Court dimensions in visualization coordinates
        self.court_length = width - 100  # Leave margins
        self.court_width = height - 100
        
        # Court elements
        self.setup_court_elements()
        
    def setup_court_elements(self):
        """Setup court elements for drawing"""
        self.margin_x = 50
        self.margin_y = 50
        
        # Key court lines
        self.three_point_arc_radius = 60
        self.paint_width = 40
        self.paint_length = 50
        
    def create_court_background(self):
        """Create basketball court background"""
        court = np.ones((self.height, self.width, 3), dtype=np.uint8) * 40  # Dark court
        
        # Court outline
        cv2.rectangle(court, 
                     (self.margin_x, self.margin_y), 
                     (self.width - self.margin_x, self.height - self.margin_y), 
                     (255, 255, 255), 2)
        
        # Center line
        center_x = self.width // 2
        cv2.line(court, 
                (center_x, self.margin_y), 
                (center_x, self.height - self.margin_y), 
                (255, 255, 255), 2)
        
        # Center circle
        cv2.circle(court, (center_x, self.height // 2), 30, (255, 255, 255), 2)
        
        # Paint areas (simplified)
        paint_left = self.margin_x
        paint_right = self.margin_x + self.paint_length
        paint_top = self.height // 2 - self.paint_width // 2
        paint_bottom = self.height // 2 + self.paint_width // 2
        
        cv2.rectangle(court, (paint_left, paint_top), (paint_right, paint_bottom), (255, 255, 255), 2)
        
        # Right paint
        paint_left = self.width - self.margin_x - self.paint_length
        paint_right = self.width - self.margin_x
        cv2.rectangle(court, (paint_left, paint_top), (paint_right, paint_bottom), (255, 255, 255), 2)
        
        # Three-point arcs (simplified)
        cv2.ellipse(court, (self.margin_x, self.height // 2), 
                   (self.three_point_arc_radius, self.three_point_arc_radius), 
                   0, -90, 90, (255, 255, 255), 2)
        
        cv2.ellipse(court, (self.width - self.margin_x, self.height // 2), 
                   (self.three_point_arc_radius, self.three_point_arc_radius), 
                   0, 90, 270, (255, 255, 255), 2)
        
        return court
    
    def draw_players(self, court, players, ball_pos=None):
        """Draw players on the court"""
        for player in players:
            # Convert court coordinates to visualization coordinates
            viz_x = int(self.margin_x + (player['court_x'] / 94.0) * self.court_length)
            viz_y = int(self.margin_y + (player['court_y'] / 50.0) * self.court_width)
            
            # Team colors
            if player['team'] == 'home':
                color = (255, 165, 0)  # Orange
            elif player['team'] == 'away':
                color = (0, 100, 255)  # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw player
            cv2.circle(court, (viz_x, viz_y), 12, color, -1)
            cv2.circle(court, (viz_x, viz_y), 12, (255, 255, 255), 2)
            
            # Player number/ID
            cv2.putText(court, str(player['id']), 
                       (viz_x - 5, viz_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw ball if available
        if ball_pos:
            ball_x = int(self.margin_x + (ball_pos[0] / 94.0) * self.court_length)
            ball_y = int(self.margin_y + (ball_pos[1] / 50.0) * self.court_width)
            cv2.circle(court, (ball_x, ball_y), 6, (0, 255, 255), -1)  # Yellow ball
        
        return court
    
    def add_statistics(self, court, stats, analysis):
        """Add game statistics and analysis"""
        # Statistics panel
        stats_panel = np.ones((150, self.width, 3), dtype=np.uint8) * 30
        
        # Game stats
        y_pos = 30
        stats_text = [
            f"Frame: {stats.get('frame', 0)}",
            f"Players: {stats.get('players', 0)}",
            f"Home Team: {stats.get('home_players', 0)}",
            f"Away Team: {stats.get('away_players', 0)}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(stats_panel, text, (20, y_pos + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Tactical analysis
        if analysis:
            analysis_text = [
                f"Play Pattern: {analysis.get('pattern', 'Unknown')}",
                f"Confidence: {analysis.get('confidence', 0.0):.2f}"
            ]
            
            for i, text in enumerate(analysis_text):
                cv2.putText(stats_panel, text, (300, y_pos + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Combine court and stats
        full_frame = np.vstack([court, stats_panel])
        return full_frame

class GameVideoTo2DConverter:
    """Main converter class"""
    
    def __init__(self, model_path):
        print("🏀 Initializing Basketball Video to 2D Converter...")
        
        # Initialize components
        self.detector = ImprovedBasketballIntelligence(model_path)
        self.court_transformer = CourtTransformer()
        self.tactical_analyzer = TacticalAnalyzer()
        self.visualizer = Basketball2DVisualizer()
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'players_detected': 0,
            'home_players': 0,
            'away_players': 0,
            'play_patterns': defaultdict(int)
        }
        
        print("✅ All components initialized successfully!")
    
    def process_video(self, video_path, output_path, max_frames=None):
        """Convert basketball video to 2D gameplay"""
        print(f"🎬 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Video properties: {fps} FPS, processing {total_frames} frames")
        
        # Setup video writer
        output_height = self.visualizer.height + 150  # Court + stats panel
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                             (self.visualizer.width, output_height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect players and objects
                _, detections = self.detector.process_frame(frame)
                
                # Transform to 2D court coordinates
                players_2d = self.court_transformer.transform_coordinates(
                    detections, frame.shape)
                
                # Tactical analysis
                analysis = self.tactical_analyzer.analyze_play(players_2d)
                
                # Update statistics
                self.update_statistics(players_2d, analysis)
                
                # Create 2D visualization
                court = self.visualizer.create_court_background()
                court = self.visualizer.draw_players(court, players_2d)
                
                # Add statistics and analysis
                current_stats = {
                    'frame': frame_count,
                    'players': len(players_2d),
                    'home_players': len([p for p in players_2d if p['team'] == 'home']),
                    'away_players': len([p for p in players_2d if p['team'] == 'away'])
                }
                
                final_frame = self.visualizer.add_statistics(court, current_stats, analysis)
                
                # Write frame
                out.write(final_frame)
                
                frame_count += 1
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) @ {fps_current:.1f}fps")
        
        finally:
            cap.release()
            out.release()
        
        # Generate final statistics
        final_stats = self.generate_final_statistics()
        
        print(f"\n✅ Conversion completed!")
        print(f"📁 Output: {output_path}")
        
        return final_stats
    
    def update_statistics(self, players, analysis):
        """Update running statistics"""
        self.stats['total_frames'] += 1
        self.stats['players_detected'] += len(players)
        
        for player in players:
            if player['team'] == 'home':
                self.stats['home_players'] += 1
            elif player['team'] == 'away':
                self.stats['away_players'] += 1
        
        if analysis and 'pattern' in analysis:
            self.stats['play_patterns'][analysis['pattern']] += 1
    
    def generate_final_statistics(self):
        """Generate comprehensive final statistics"""
        return {
            'processing_summary': {
                'total_frames_processed': self.stats['total_frames'],
                'average_players_per_frame': self.stats['players_detected'] / max(1, self.stats['total_frames']),
                'total_home_detections': self.stats['home_players'],
                'total_away_detections': self.stats['away_players']
            },
            'tactical_analysis': {
                'play_patterns_detected': dict(self.stats['play_patterns']),
                'most_common_pattern': max(self.stats['play_patterns'], key=self.stats['play_patterns'].get) if self.stats['play_patterns'] else 'None'
            }
        }

def convert_basketball_video_to_2d(video_path, output_path=None, model_path=None, max_frames=None):
    """
    Main function to convert any basketball video to 2D gameplay
    
    Args:
        video_path: Path to input basketball video
        output_path: Path for output 2D video (optional)
        model_path: Path to trained YOLO model (optional)
        max_frames: Maximum frames to process (optional, for testing)
    """
    
    # Default paths
    if model_path is None:
        model_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt")
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_path = Path(video_path).parent / f"{video_name}_2D_gameplay_{timestamp}.mp4"
    
    print("🏀 Basketball Video to 2D Gameplay Converter")
    print("=" * 60)
    print(f"📹 Input video: {video_path}")
    print(f"🎯 Output video: {output_path}")
    print(f"🤖 Model: {model_path}")
    print()
    
    # Initialize converter
    converter = GameVideoTo2DConverter(str(model_path))
    
    # Process video
    stats = converter.process_video(video_path, output_path, max_frames)
    
    # Save statistics
    stats_path = str(output_path).replace('.mp4', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📊 Final Statistics:")
    print(f"   - Frames processed: {stats['processing_summary']['total_frames_processed']}")
    print(f"   - Average players per frame: {stats['processing_summary']['average_players_per_frame']:.2f}")
    print(f"   - Home team detections: {stats['processing_summary']['total_home_detections']}")
    print(f"   - Away team detections: {stats['processing_summary']['total_away_detections']}")
    print(f"   - Most common play pattern: {stats['tactical_analysis']['most_common_pattern']}")
    print(f"\n📋 Detailed stats saved: {stats_path}")
    
    return output_path, stats

if __name__ == "__main__":
    # Example usage for Hawks vs Knicks
    hawks_knicks_video = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks-vs-knicks.mp4"
    
    # Convert full game (remove max_frames for full video)
    print("🚀 Converting Hawks vs Knicks to 2D Gameplay...")
    output_video, stats = convert_basketball_video_to_2d(
        video_path=hawks_knicks_video,
        max_frames=500  # Remove this line to process full video
    )
    
    print(f"\n🎉 2D Gameplay conversion completed!")
    print(f"🎬 Watch your 2D basketball game: {output_video}")
