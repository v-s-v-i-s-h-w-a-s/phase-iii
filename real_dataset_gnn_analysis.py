#!/usr/bin/env python3
"""
Basketball GNN Analysis with Real Dataset Trained YOLO Model
Comprehensive basketball tactical analysis using custom trained YOLO + GNN
"""

import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import json
import torch
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime

class RealDatasetBasketballGNN:
    def __init__(self, model_path=None):
        """
        Initialize Basketball GNN with real dataset trained YOLO model
        """
        # Use the best real dataset trained model
        if model_path is None:
            model_path = r"basketball_real_training\real_dataset_20250803_121502\weights\best.pt"
        
        self.model_path = model_path
        self.yolo_model = None
        
        # Basketball field dimensions (in meters)
        self.field_length = 28.0  # NBA court length
        self.field_width = 15.0   # NBA court width
        
        # Load model
        self._load_model()
        
        # GNN parameters
        self.player_interaction_threshold = 5.0  # meters
        self.team_detection_enabled = True
        
        print(f"🏀 Real Dataset Basketball GNN Initialized")
        print(f"📁 Model: {model_path}")
        print(f"🏟️  Court: {self.field_length}m x {self.field_width}m")
    
    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            print(f"📥 Loading trained YOLO model...")
            self.yolo_model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect_objects(self, frame, confidence_threshold=0.3):
        """
        Detect basketball objects using trained YOLO model
        
        Returns:
            detections: List of detection dictionaries
        """
        if self.yolo_model is None:
            return []
        
        # Run detection
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            detection_result = results[0]
            if hasattr(detection_result, 'boxes') and detection_result.boxes is not None:
                boxes = detection_result.boxes
                
                for box in boxes:
                    # Extract box information
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Filter by confidence
                    if conf >= confidence_threshold:
                        # Map class ID to name
                        class_names = {0: 'ball', 1: 'basket', 2: 'player', 3: 'referee'}
                        class_name = class_names.get(cls, f'class_{cls}')
                        
                        # Calculate center point
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        detection = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
        
        return detections
    
    def pixel_to_field_coordinates(self, pixel_coords, frame_shape):
        """
        Convert pixel coordinates to field coordinates
        Simple linear mapping - in practice would use homography
        """
        height, width = frame_shape[:2]
        x_pixel, y_pixel = pixel_coords
        
        # Normalize to [0, 1]
        x_norm = x_pixel / width
        y_norm = y_pixel / height
        
        # Map to field coordinates
        field_x = x_norm * self.field_length
        field_y = y_norm * self.field_width
        
        return [field_x, field_y]
    
    def assign_teams(self, player_detections):
        """
        Assign players to teams based on position and heuristics
        In practice, would use jersey color detection
        """
        if len(player_detections) < 2:
            return player_detections
        
        # Simple heuristic: split by x-coordinate (left/right sides)
        players_with_teams = []
        for i, player in enumerate(player_detections):
            # Assign team based on position or alternating pattern
            team = 'Team_A' if i % 2 == 0 else 'Team_B'
            player['team'] = team
            player['player_id'] = f"{team}_Player_{i//2 + 1}"
            players_with_teams.append(player)
        
        return players_with_teams
    
    def calculate_player_distances(self, players):
        """Calculate distances between all players"""
        distances = {}
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players[i+1:], i+1):
                pos1 = player1['field_position']
                pos2 = player2['field_position']
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                key = f"{player1['player_id']}-{player2['player_id']}"
                distances[key] = distance
        
        return distances
    
    def build_interaction_graph(self, players, ball_position=None):
        """
        Build graph representing player interactions
        """
        G = nx.Graph()
        
        # Add player nodes
        for i, player in enumerate(players):
            # Ensure player has an ID
            player_id = player.get('player_id', f"Player_{i}")
            
            G.add_node(
                player_id,
                team=player.get('team', f'Team_{i%2}'),
                position=player['field_position'],
                bbox=player['bbox'],
                confidence=player['confidence']
            )
        
        # Add ball node if detected
        if ball_position:
            G.add_node('ball', position=ball_position, type='ball')
        
        # Calculate distances and add edges for close players
        player_list = []
        for i, player in enumerate(players):
            player_id = player.get('player_id', f"Player_{i}")
            player_list.append({'id': player_id, 'position': player['field_position']})
        
        # Add edges between close players
        for i, player1 in enumerate(player_list):
            for j, player2 in enumerate(player_list[i+1:], i+1):
                pos1 = player1['position']
                pos2 = player2['position']
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance <= self.player_interaction_threshold:
                    G.add_edge(player1['id'], player2['id'], distance=distance, interaction_strength=1.0/distance)
        
        # Add edges from players to ball if within threshold
        if ball_position:
            for i, player in enumerate(players):
                player_id = player.get('player_id', f"Player_{i}")
                pos = player['field_position']
                ball_distance = np.sqrt((pos[0] - ball_position[0])**2 + (pos[1] - ball_position[1])**2)
                
                if ball_distance <= self.player_interaction_threshold:
                    G.add_edge(player_id, 'ball', distance=ball_distance, type='ball_proximity')
        
        return G
    
    def analyze_tactical_formation(self, G):
        """
        Analyze tactical formation and patterns
        """
        analysis = {
            'total_players': 0,
            'teams': {},
            'formations': {},
            'interactions': {
                'total_edges': G.number_of_edges(),
                'avg_clustering': 0,
                'team_cohesion': {}
            },
            'ball_control': None
        }
        
        # Analyze by team
        teams = {}
        for node, data in G.nodes(data=True):
            if 'team' in data:
                team = data['team']
                if team not in teams:
                    teams[team] = []
                teams[team].append(node)
                analysis['total_players'] += 1
        
        analysis['teams'] = {team: len(players) for team, players in teams.items()}
        
        # Calculate clustering coefficient
        if G.number_of_nodes() > 2:
            analysis['interactions']['avg_clustering'] = nx.average_clustering(G)
        
        # Analyze team cohesion (average distance within teams)
        for team, players in teams.items():
            if len(players) > 1:
                team_distances = []
                for i, p1 in enumerate(players):
                    for p2 in players[i+1:]:
                        if G.has_edge(p1, p2):
                            team_distances.append(G[p1][p2]['distance'])
                
                analysis['interactions']['team_cohesion'][team] = {
                    'avg_distance': np.mean(team_distances) if team_distances else 0,
                    'connections': len(team_distances)
                }
        
        # Check ball control
        if 'ball' in G.nodes():
            ball_neighbors = list(G.neighbors('ball'))
            if ball_neighbors:
                # Find closest player to ball
                closest_player = None
                min_distance = float('inf')
                
                for player in ball_neighbors:
                    distance = G['ball'][player]['distance']
                    if distance < min_distance:
                        min_distance = distance
                        closest_player = player
                
                if closest_player and 'team' in G.nodes[closest_player]:
                    analysis['ball_control'] = G.nodes[closest_player]['team']
        
        return analysis
    
    def visualize_frame_analysis(self, frame, detections, G, analysis, output_path=None):
        """
        Visualize frame with detections and graph analysis
        """
        vis_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Colors for different classes
        colors = {
            'ball': (0, 255, 255),      # Yellow
            'basket': (255, 0, 0),      # Blue
            'player': (0, 255, 0),      # Green
            'referee': (0, 0, 255),     # Red
            'Team_A': (255, 0, 255),    # Magenta
            'Team_B': (0, 255, 255)     # Cyan
        }
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_name = detection['class']
            conf = detection['confidence']
            
            # Choose color
            if class_name == 'player' and 'team' in detection:
                color = colors.get(detection['team'], colors['player'])
                label = f"{detection.get('player_id', class_name)}: {conf:.2f}"
            else:
                color = colors.get(class_name, (255, 255, 255))
                label = f"{class_name}: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw connections between players
        for edge in G.edges():
            node1, node2 = edge
            if node1 != 'ball' and node2 != 'ball':
                # Get pixel positions
                node1_data = G.nodes[node1]
                node2_data = G.nodes[node2]
                
                if 'bbox' in node1_data and 'bbox' in node2_data:
                    # Calculate centers from bboxes
                    bbox1 = node1_data['bbox']
                    bbox2 = node2_data['bbox']
                    
                    center1 = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
                    center2 = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))
                    
                    # Draw line
                    cv2.line(vis_frame, center1, center2, (255, 255, 255), 1)
        
        # Add analysis text
        y_offset = 30
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        analysis_texts = [
            f"Players: {analysis['total_players']}",
            f"Teams: {', '.join([f'{team}: {count}' for team, count in analysis['teams'].items()])}",
            f"Interactions: {analysis['interactions']['total_edges']}",
            f"Clustering: {analysis['interactions']['avg_clustering']:.3f}",
            f"Ball Control: {analysis['ball_control'] or 'None'}"
        ]
        
        for text in analysis_texts:
            cv2.putText(vis_frame, text, (10, y_offset), font, 0.6, text_color, 2)
            y_offset += 25
        
        if output_path:
            cv2.imwrite(output_path, vis_frame)
        
        return vis_frame
    
    def analyze_video(self, video_path, output_path=None, max_frames=None):
        """
        Analyze entire video with GNN
        """
        print(f"\n🎥 Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Analysis storage
        video_analysis = {
            'frames': [],
            'summary': {
                'total_frames': 0,
                'avg_players_per_frame': 0,
                'ball_detection_rate': 0,
                'team_ball_control': {'Team_A': 0, 'Team_B': 0, 'None': 0},
                'tactical_patterns': []
            }
        }
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections = self.detect_objects(frame, confidence_threshold=0.3)
            
            # Separate by type
            players = [d for d in detections if d['class'] == 'player']
            balls = [d for d in detections if d['class'] == 'ball']
            
            # Convert to field coordinates and assign teams
            for detection in detections:
                detection['field_position'] = self.pixel_to_field_coordinates(
                    detection['center'], frame.shape
                )
            
            # Assign teams to players
            if players:
                players = self.assign_teams(players)
                # Update detections list with team assignments
                player_idx = 0
                for i, detection in enumerate(detections):
                    if detection['class'] == 'player':
                        detections[i].update(players[player_idx])
                        player_idx += 1
            
            # Build interaction graph
            ball_position = balls[0]['field_position'] if balls else None
            G = self.build_interaction_graph(players, ball_position)
            
            # Analyze tactics
            tactical_analysis = self.analyze_tactical_formation(G)
            
            # Store frame analysis
            frame_analysis = {
                'frame_number': frame_count,
                'detections': len(detections),
                'players': len(players),
                'ball_detected': len(balls) > 0,
                'tactical_analysis': tactical_analysis
            }
            video_analysis['frames'].append(frame_analysis)
            
            # Visualize frame
            vis_frame = self.visualize_frame_analysis(frame, detections, G, tactical_analysis)
            
            if output_path:
                out.write(vis_frame)
            
            # Progress update
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% | Frame {frame_count}/{total_frames}")
        
        # Calculate summary statistics
        if video_analysis['frames']:
            total_players = sum(f['players'] for f in video_analysis['frames'])
            total_ball_detections = sum(1 for f in video_analysis['frames'] if f['ball_detected'])
            
            video_analysis['summary']['total_frames'] = frame_count
            video_analysis['summary']['avg_players_per_frame'] = total_players / frame_count
            video_analysis['summary']['ball_detection_rate'] = total_ball_detections / frame_count
            
            # Ball control analysis
            for frame in video_analysis['frames']:
                ball_control = frame['tactical_analysis']['ball_control']
                if ball_control:
                    video_analysis['summary']['team_ball_control'][ball_control] += 1
                else:
                    video_analysis['summary']['team_ball_control']['None'] += 1
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        
        total_time = time.time() - start_time
        print(f"\n✅ Video analysis completed in {total_time:.1f} seconds")
        
        # Save analysis
        analysis_path = video_path.replace('.mp4', '_gnn_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(video_analysis, f, indent=2)
        print(f"📊 Analysis saved to: {analysis_path}")
        
        return video_analysis
    
    def print_analysis_summary(self, analysis):
        """Print formatted analysis summary"""
        summary = analysis['summary']
        
        print(f"\n📈 VIDEO ANALYSIS SUMMARY")
        print(f"=" * 50)
        print(f"Total Frames: {summary['total_frames']}")
        print(f"Average Players per Frame: {summary['avg_players_per_frame']:.1f}")
        print(f"Ball Detection Rate: {summary['ball_detection_rate']*100:.1f}%")
        
        print(f"\n🏀 BALL CONTROL ANALYSIS:")
        for team, frames in summary['team_ball_control'].items():
            percentage = (frames / summary['total_frames']) * 100
            print(f"   {team}: {frames} frames ({percentage:.1f}%)")
        
        print(f"\n🎯 KEY INSIGHTS:")
        if summary['avg_players_per_frame'] > 5:
            print(f"   • High player activity detected")
        if summary['ball_detection_rate'] > 0.3:
            print(f"   • Good ball tracking performance")
        
        team_a_control = summary['team_ball_control'].get('Team_A', 0)
        team_b_control = summary['team_ball_control'].get('Team_B', 0)
        
        if team_a_control > team_b_control:
            print(f"   • Team_A had more ball possession")
        elif team_b_control > team_a_control:
            print(f"   • Team_B had more ball possession")
        else:
            print(f"   • Ball possession was relatively balanced")

def main():
    """Main analysis function"""
    print("🏀 Real Dataset Basketball GNN Analysis")
    print("=" * 60)
    
    # Initialize GNN system
    gnn = RealDatasetBasketballGNN()
    
    # Analyze basketball video
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\basketball_gnn\video_analysis_hawks_vs_knicks\annotated_video.mp4"
    output_path = r"C:\Users\vish\Capstone PROJECT\Phase III\real_dataset_gnn_analysis.mp4"
    
    # Run analysis
    analysis = gnn.analyze_video(video_path, output_path, max_frames=300)
    
    if analysis:
        # Print summary
        gnn.print_analysis_summary(analysis)
        
        print(f"\n✅ Analysis completed!")
        print(f"📁 Output video: {output_path}")
        print(f"📊 Analysis data: {video_path.replace('.mp4', '_gnn_analysis.json')}")

if __name__ == "__main__":
    main()
