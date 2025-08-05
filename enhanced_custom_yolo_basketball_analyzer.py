#!/usr/bin/env python3
"""
Enhanced Basketball Tracking with Custom YOLO + 2D Mapping
==========================================================
Uses our best custom trained YOLO model to track:
- Players (with team assignment)
- Ball (with trajectory)
- Referee
- Basket/Rim

Then maps all tracking to professional 2D court visualization.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time
import json
from datetime import datetime
import math
from collections import defaultdict, deque
from sklearn.cluster import KMeans

class CustomYOLOBasketballTracker:
    """Enhanced tracker using our best custom trained YOLO model"""
    
    def __init__(self):
        print("🏀 Initializing Custom YOLO Basketball Tracker...")
        
        # Find our best custom trained models
        self.model_paths = [
            r"enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt",
            r"basketball_real_training\real_dataset_20250803_121502\weights\best.pt",
            r"basketball_gnn\custom_yolo_training_20250803_155623\weights\best.pt",
            r"yolov8n.pt"  # Fallback
        ]
        
        self.model = None
        for model_path in self.model_paths:
            if Path(model_path).exists():
                print(f"✅ Loading custom trained model: {model_path}")
                self.model = YOLO(model_path)
                break
        
        if not self.model:
            print("❌ No custom model found, using YOLOv8n")
            self.model = YOLO('yolov8n.pt')
        
        # Enhanced class mapping for basketball
        self.basketball_classes = {
            0: {'name': 'ball', 'color': (0, 255, 255), 'size': 6},      # Bright Yellow Ball
            1: {'name': 'basket', 'color': (255, 100, 0), 'size': 10},   # Orange Basket/Rim
            2: {'name': 'player', 'color': (0, 255, 0), 'size': 12},     # Green Player (will be team colored)
            3: {'name': 'referee', 'color': (128, 128, 128), 'size': 8}  # Gray Referee
        }
        
        # Enhanced detection thresholds for each class
        self.confidence_thresholds = {
            0: 0.2,  # Ball - lower threshold for better detection
            1: 0.3,  # Basket
            2: 0.25, # Player - optimized for our custom model
            3: 0.4   # Referee
        }
        
        # Team color tracking
        self.team_colors = {
            'home': {'color': (255, 165, 0), 'name': 'HOME'},    # Orange
            'away': {'color': (0, 100, 255), 'name': 'AWAY'},    # Blue
            'unknown': {'color': (128, 128, 128), 'name': 'UNK'} # Gray
        }
        
        # Player tracking with team assignment
        self.player_trackers = {}
        self.next_player_id = 1
        self.team_assignments = {}
        self.player_jersey_colors = {}
        
        # Ball tracking
        self.ball_trajectory = deque(maxlen=30)
        
        # Statistics
        self.detection_stats = {
            'total_frames': 0,
            'players_detected': 0,
            'balls_detected': 0,
            'referees_detected': 0,
            'baskets_detected': 0,
            'team_home': 0,
            'team_away': 0
        }
        
        print("✅ Custom YOLO tracker initialized with enhanced basketball detection!")
    
    def detect_objects(self, frame):
        """Run custom YOLO detection on frame"""
        results = self.model(frame, conf=0.2, iou=0.4, verbose=False)
        
        detections = {
            'players': [],
            'ball': None,
            'referees': [],
            'baskets': []
        }
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    
                    # Apply class-specific confidence thresholds
                    if cls in self.confidence_thresholds and conf >= self.confidence_thresholds[cls]:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': cls,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        }
                        
                        # Sort detections by class
                        if cls == 0:  # Ball
                            detections['ball'] = detection
                            self.detection_stats['balls_detected'] += 1
                        elif cls == 1:  # Basket
                            detections['baskets'].append(detection)
                            self.detection_stats['baskets_detected'] += 1
                        elif cls == 2:  # Player
                            detections['players'].append(detection)
                            self.detection_stats['players_detected'] += 1
                        elif cls == 3:  # Referee
                            detections['referees'].append(detection)
                            self.detection_stats['referees_detected'] += 1
        
        return detections
    
    def analyze_jersey_colors(self, frame, bbox):
        """Analyze player jersey colors for team assignment"""
        x1, y1, x2, y2 = bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return None
        
        # Focus on torso area (middle part)
        h, w = player_region.shape[:2]
        torso_y1 = int(h * 0.2)
        torso_y2 = int(h * 0.7)
        torso_x1 = int(w * 0.1)
        torso_x2 = int(w * 0.9)
        
        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return None
        
        torso = player_region[torso_y1:torso_y2, torso_x1:torso_x2]
        
        # Get dominant colors
        pixels = torso.reshape(-1, 3)
        
        # Remove very dark and very bright pixels
        pixel_brightness = np.mean(pixels, axis=1)
        valid_pixels = pixels[(pixel_brightness > 30) & (pixel_brightness < 225)]
        
        if len(valid_pixels) < 10:
            return None
        
        try:
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=min(3, len(valid_pixels)), random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert BGR to RGB for better color analysis
            colors = [(int(c[2]), int(c[1]), int(c[0])) for c in colors]
            return colors
        except:
            return None
    
    def assign_team(self, player_id, jersey_colors):
        """Assign player to team based on jersey colors"""
        if not jersey_colors or player_id in self.team_assignments:
            return self.team_assignments.get(player_id, 'unknown')
        
        # Simple team assignment based on color similarity
        # This can be enhanced with more sophisticated color clustering
        
        # Calculate color distances to known team colors
        home_distances = []
        away_distances = []
        
        for color in jersey_colors:
            # Distance to home team color (orange)
            home_dist = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color, (255, 165, 0))))
            home_distances.append(home_dist)
            
            # Distance to away team color (blue)
            away_dist = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color, (0, 100, 255))))
            away_distances.append(away_dist)
        
        avg_home_dist = np.mean(home_distances)
        avg_away_dist = np.mean(away_distances)
        
        # Assign to closer team
        if avg_home_dist < avg_away_dist:
            team = 'home'
        else:
            team = 'away'
        
        self.team_assignments[player_id] = team
        return team
    
    def track_players(self, frame, detections):
        """Enhanced player tracking with team assignment"""
        current_players = []
        
        for i, player_detection in enumerate(detections['players']):
            bbox = player_detection['bbox']
            center = player_detection['center']
            
            # Analyze jersey colors
            jersey_colors = self.analyze_jersey_colors(frame, bbox)
            
            # Assign player ID (simplified tracking)
            player_id = self.next_player_id
            self.next_player_id += 1
            
            # Assign team
            team = self.assign_team(player_id, jersey_colors)
            
            player_info = {
                'id': player_id,
                'bbox': bbox,
                'center': center,
                'team': team,
                'confidence': player_detection['confidence'],
                'jersey_colors': jersey_colors
            }
            
            current_players.append(player_info)
            
            # Update team statistics
            if team == 'home':
                self.detection_stats['team_home'] += 1
            elif team == 'away':
                self.detection_stats['team_away'] += 1
        
        return current_players
    
    def track_ball(self, detections):
        """Enhanced ball tracking with trajectory"""
        if detections['ball']:
            ball_center = detections['ball']['center']
            self.ball_trajectory.append(ball_center)
            return detections['ball']
        return None

class Basketball2DMapper:
    """Maps real court positions to 2D tactical view"""
    
    def __init__(self, court_width=600, court_height=400):
        self.court_width = court_width
        self.court_height = court_height
        self.margin = 30
        
        # Court dimensions
        self.court_viz_width = court_width - 2 * self.margin
        self.court_viz_height = court_height - 2 * self.margin
        
    def create_2d_court(self):
        """Create professional 2D basketball court"""
        court = np.ones((self.court_height, self.court_width, 3), dtype=np.uint8) * 34  # Dark court color
        
        # Court outline
        cv2.rectangle(court, 
                     (self.margin, self.margin), 
                     (self.court_width - self.margin, self.court_height - self.margin), 
                     (255, 255, 255), 2)
        
        # Center line
        center_x = self.court_width // 2
        cv2.line(court, 
                (center_x, self.margin), 
                (center_x, self.court_height - self.margin), 
                (255, 255, 255), 2)
        
        # Center circle
        center_y = self.court_height // 2
        cv2.circle(court, (center_x, center_y), 40, (255, 255, 255), 2)
        
        # Paint areas (key)
        paint_width = 60
        paint_height = int(self.court_viz_height * 0.6)
        paint_y1 = center_y - paint_height // 2
        paint_y2 = center_y + paint_height // 2
        
        # Left paint
        cv2.rectangle(court, (self.margin, paint_y1), (self.margin + paint_width, paint_y2), (255, 255, 255), 2)
        
        # Right paint
        cv2.rectangle(court, (self.court_width - self.margin - paint_width, paint_y1), 
                     (self.court_width - self.margin, paint_y2), (255, 255, 255), 2)
        
        # Three-point arcs
        arc_radius = 80
        cv2.ellipse(court, (self.margin + 20, center_y), (arc_radius, arc_radius), 
                   0, -90, 90, (255, 255, 255), 2)
        cv2.ellipse(court, (self.court_width - self.margin - 20, center_y), (arc_radius, arc_radius), 
                   0, 90, 270, (255, 255, 255), 2)
        
        # Baskets
        cv2.circle(court, (self.margin + 15, center_y), 8, (255, 165, 0), -1)  # Left basket
        cv2.circle(court, (self.court_width - self.margin - 15, center_y), 8, (255, 165, 0), -1)  # Right basket
        
        return court
    
    def map_coordinates(self, real_x, real_y, frame_width, frame_height):
        """Map real coordinates to 2D court position"""
        # Normalize coordinates
        norm_x = real_x / frame_width
        norm_y = real_y / frame_height
        
        # Map to court coordinates with perspective correction
        court_x = int(self.margin + norm_x * self.court_viz_width)
        court_y = int(self.margin + norm_y * self.court_viz_height)
        
        # Ensure within bounds
        court_x = max(self.margin, min(self.court_width - self.margin, court_x))
        court_y = max(self.margin, min(self.court_height - self.margin, court_y))
        
        return court_x, court_y
    
    def draw_players_2d(self, court, players, frame_shape):
        """Draw players on 2D court"""
        frame_height, frame_width = frame_shape[:2]
        
        for player in players:
            center_x, center_y = player['center']
            court_x, court_y = self.map_coordinates(center_x, center_y, frame_width, frame_height)
            
            # Team color
            if player['team'] == 'home':
                color = (255, 165, 0)  # Orange
            elif player['team'] == 'away':
                color = (0, 100, 255)  # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw player
            cv2.circle(court, (court_x, court_y), 8, color, -1)
            cv2.circle(court, (court_x, court_y), 8, (255, 255, 255), 1)
            
            # Player ID
            cv2.putText(court, str(player['id']), (court_x - 5, court_y + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return court
    
    def draw_ball_2d(self, court, ball, frame_shape, trajectory=None):
        """Draw ball on 2D court with trajectory"""
        if not ball:
            return court
        
        frame_height, frame_width = frame_shape[:2]
        center_x, center_y = ball['center']
        court_x, court_y = self.map_coordinates(center_x, center_y, frame_width, frame_height)
        
        # Draw ball trajectory
        if trajectory and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                prev_x, prev_y = trajectory[i-1]
                curr_x, curr_y = trajectory[i]
                
                prev_court_x, prev_court_y = self.map_coordinates(prev_x, prev_y, frame_width, frame_height)
                curr_court_x, curr_court_y = self.map_coordinates(curr_x, curr_y, frame_width, frame_height)
                
                # Fading trail effect
                alpha = i / len(trajectory)
                thickness = max(1, int(3 * alpha))
                cv2.line(court, (prev_court_x, prev_court_y), (curr_court_x, curr_court_y), 
                        (0, 255, 255), thickness)
        
        # Draw current ball position
        cv2.circle(court, (court_x, court_y), 5, (0, 255, 255), -1)  # Yellow ball
        cv2.circle(court, (court_x, court_y), 5, (255, 255, 255), 1)
        
        return court
    
    def add_game_info(self, court, frame_num, stats):
        """Add game information overlay"""
        info_y = 20
        
        # Frame info
        cv2.putText(court, f"Frame: {frame_num}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Player counts
        cv2.putText(court, f"Players: {stats.get('current_players', 0)}", (10, info_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Team counts
        home_count = stats.get('home_count', 0)
        away_count = stats.get('away_count', 0)
        
        cv2.putText(court, f"HOME: {home_count}", (10, info_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(court, f"AWAY: {away_count}", (10, info_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        
        # Ball status
        ball_status = "Ball: YES" if stats.get('ball_detected', False) else "Ball: NO"
        ball_color = (0, 255, 0) if stats.get('ball_detected', False) else (255, 0, 0)
        cv2.putText(court, ball_status, (10, info_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 1)
        
        return court

class EnhancedBasketballAnalyzer:
    """Main analyzer combining custom YOLO tracking with 2D mapping"""
    
    def __init__(self):
        print("🏀 Initializing Enhanced Basketball Analyzer...")
        self.tracker = CustomYOLOBasketballTracker()
        self.mapper = Basketball2DMapper()
        print("✅ Enhanced analyzer ready!")
    
    def process_video_enhanced(self, video_path, output_path=None, max_frames=None):
        """Process video with enhanced tracking and 2D mapping"""
        print(f"🎬 Processing: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
        
        # Output setup
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(video_path).stem
            output_path = f"{video_name}_enhanced_analysis_{timestamp}.mp4"
        
        # Side-by-side dimensions
        court_width = 600
        side_by_side_width = frame_width + court_width
        side_by_side_height = max(frame_height, 400)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (side_by_side_width, side_by_side_height))
        
        if not out.isOpened():
            print("❌ Failed to create output video")
            return None
        
        frame_count = 0
        start_time = time.time()
        
        print(f"🎯 Creating enhanced analysis: {output_path}")
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects using custom YOLO
                detections = self.tracker.detect_objects(frame)
                
                # Track players with team assignment
                players = self.tracker.track_players(frame, detections)
                
                # Track ball
                ball = self.tracker.track_ball(detections)
                
                # Draw detections on original frame
                annotated_frame = self.draw_detections_on_frame(frame, detections, players, ball)
                
                # Create 2D court view
                court_2d = self.mapper.create_2d_court()
                
                # Map players to 2D
                court_2d = self.mapper.draw_players_2d(court_2d, players, frame.shape)
                
                # Map ball to 2D
                court_2d = self.mapper.draw_ball_2d(court_2d, ball, frame.shape, self.tracker.ball_trajectory)
                
                # Add game info
                current_stats = {
                    'current_players': len(players),
                    'home_count': len([p for p in players if p['team'] == 'home']),
                    'away_count': len([p for p in players if p['team'] == 'away']),
                    'ball_detected': ball is not None
                }
                court_2d = self.mapper.add_game_info(court_2d, frame_count + 1, current_stats)
                
                # Create side-by-side frame
                combined_frame = self.create_side_by_side_frame(annotated_frame, court_2d, side_by_side_width, side_by_side_height)
                
                # Write frame
                out.write(combined_frame)
                
                # Update statistics
                self.tracker.detection_stats['total_frames'] += 1
                
                frame_count += 1
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) @ {fps_current:.1f}fps")
        
        finally:
            cap.release()
            out.release()
        
        # Generate final report
        final_stats = self.generate_final_stats()
        
        # Verify output
        output_size = Path(output_path).stat().st_size / (1024 * 1024)
        
        print(f"\n✅ Enhanced Analysis Complete!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Size: {output_size:.2f} MB")
        print(f"🎯 Final Statistics:")
        for key, value in final_stats.items():
            print(f"   - {key}: {value}")
        
        if output_size > 1:
            return output_path, final_stats
        else:
            print("❌ Output file too small - analysis may have failed")
            return None, None
    
    def draw_detections_on_frame(self, frame, detections, players, ball):
        """Draw all detections on original frame"""
        annotated = frame.copy()
        
        # Draw players with team colors
        for player in players:
            bbox = player['bbox']
            x1, y1, x2, y2 = bbox
            
            # Team color
            if player['team'] == 'home':
                color = (255, 165, 0)  # Orange
            elif player['team'] == 'away':
                color = (0, 100, 255)  # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{player['team'].upper()} P{player['id']} {player['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw ball
        if ball:
            bbox = ball['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"BALL {ball['confidence']:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw referees
        for referee in detections['referees']:
            bbox = referee['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(annotated, f"REF {referee['confidence']:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Draw baskets
        for basket in detections['baskets']:
            bbox = basket['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(annotated, f"BASKET {basket['confidence']:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        return annotated
    
    def create_side_by_side_frame(self, original_frame, court_2d, target_width, target_height):
        """Create side-by-side frame"""
        # Resize court to match target height
        court_height = target_height
        court_width = 600
        court_resized = cv2.resize(court_2d, (court_width, court_height))
        
        # Create combined frame
        combined = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Place original frame (left side)
        original_height, original_width = original_frame.shape[:2]
        if original_height <= target_height and original_width <= target_width - court_width:
            combined[:original_height, :original_width] = original_frame
        else:
            # Resize if needed
            scale = min(target_height / original_height, (target_width - court_width) / original_width)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized_original = cv2.resize(original_frame, (new_width, new_height))
            combined[:new_height, :new_width] = resized_original
        
        # Place 2D court (right side)
        combined[:court_height, target_width - court_width:] = court_resized
        
        # Add title
        cv2.putText(combined, "ENHANCED BASKETBALL ANALYSIS - ORIGINAL + 2D TACTICAL", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined
    
    def generate_final_stats(self):
        """Generate comprehensive final statistics"""
        stats = self.tracker.detection_stats
        total_frames = max(1, stats['total_frames'])
        
        return {
            'total_frames_processed': stats['total_frames'],
            'avg_players_per_frame': stats['players_detected'] / total_frames,
            'avg_balls_per_frame': stats['balls_detected'] / total_frames,
            'avg_referees_per_frame': stats['referees_detected'] / total_frames,
            'avg_baskets_per_frame': stats['baskets_detected'] / total_frames,
            'total_home_players': stats['team_home'],
            'total_away_players': stats['team_away'],
            'team_ratio': stats['team_home'] / max(1, stats['team_away'])
        }

def run_enhanced_basketball_analysis(video_path, max_frames=300):
    """
    Main function to run enhanced basketball analysis
    
    Args:
        video_path: Path to basketball video
        max_frames: Number of frames to process (None for full video)
    """
    print("🚀 Starting Enhanced Basketball Analysis...")
    print("Using Custom Trained YOLO + 2D Court Mapping")
    print("=" * 60)
    
    analyzer = EnhancedBasketballAnalyzer()
    result = analyzer.process_video_enhanced(video_path, max_frames=max_frames)
    
    if result and result[0]:
        output_path, stats = result
        print(f"\n🎉 SUCCESS! Enhanced analysis completed!")
        print(f"📺 Video: {output_path}")
        print(f"🎯 Features:")
        print("   ✅ Custom trained YOLO detection")
        print("   ✅ Player tracking with team assignment")
        print("   ✅ Ball trajectory mapping")
        print("   ✅ Professional 2D court visualization")
        print("   ✅ Real-time statistics")
        return output_path
    else:
        print("❌ Analysis failed")
        return None

if __name__ == "__main__":
    # Test with Hawks vs Knicks
    hawks_video = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4"
    
    if Path(hawks_video).exists():
        result = run_enhanced_basketball_analysis(hawks_video, max_frames=300)
        if result:
            print(f"\n🎬 Watch your enhanced analysis: {result}")
    else:
        print("❌ Hawks vs Knicks video not found")
        # Try alternative videos
        alternatives = [
            r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4",
            r"C:\Users\vish\Capstone PROJECT\Phase III\basketball_demo_20250803_162301.mp4"
        ]
        
        for alt_video in alternatives:
            if Path(alt_video).exists():
                print(f"✅ Using alternative: {alt_video}")
                result = run_enhanced_basketball_analysis(alt_video, max_frames=200)
                if result:
                    print(f"\n🎬 Watch your enhanced analysis: {result}")
                break
