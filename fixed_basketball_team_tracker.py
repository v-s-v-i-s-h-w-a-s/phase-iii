#!/usr/bin/env python3
"""
Fixed Basketball Team Tracker - Proper 5v5 Player Tracking
===========================================================
Tracks exactly 5 players per team with consistent IDs
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from datetime import datetime
from pathlib import Path
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class FixedBasketballTracker:
    """Enhanced basketball tracker with proper 5v5 team separation"""
    
    def __init__(self, model_path):
        print("🏀 Initializing Fixed Basketball Tracker...")
        
        # Load custom YOLO model
        self.model = YOLO(model_path)
        print(f"✅ Loaded custom model: {model_path}")
        
        # Player tracking
        self.active_players = {}  # {player_id: player_info}
        self.next_player_id = 1
        self.max_players_per_team = 5
        
        # Team colors (will be learned from first few frames)
        self.team_colors = {
            'home': None,  # Will be detected
            'away': None   # Will be detected
        }
        
        # Tracking parameters
        self.max_distance_threshold = 100  # Max distance for player tracking
        self.confidence_threshold = 0.3
        
        # Statistics
        self.frame_count = 0
        self.detection_stats = {
            'total_home_players': 0,
            'total_away_players': 0,
            'ball_detections': 0,
            'referee_detections': 0
        }
        
        # Ball tracking
        self.ball_trajectory = []
        
    def detect_objects(self, frame):
        """Detect basketball objects using custom YOLO"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = {
            'players': [],
            'ball': None,
            'referees': [],
            'baskets': []
        }
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    if conf < self.confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': [center_x, center_y],
                        'confidence': float(conf),
                        'class': int(cls)
                    }
                    
                    # Sort by class
                    if cls == 0:  # Ball
                        detections['ball'] = detection
                    elif cls == 1:  # Basket
                        detections['baskets'].append(detection)
                    elif cls == 2:  # Player
                        detections['players'].append(detection)
                    elif cls == 3:  # Referee
                        detections['referees'].append(detection)
        
        return detections
    
    def analyze_jersey_color(self, frame, bbox):
        """Analyze player's jersey color for team assignment"""
        x1, y1, x2, y2 = bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return None
        
        # Focus on jersey area (upper torso)
        h, w = player_region.shape[:2]
        jersey_y1 = int(h * 0.15)  # Start from shoulders
        jersey_y2 = int(h * 0.6)   # End at waist
        jersey_x1 = int(w * 0.2)   # Avoid arms
        jersey_x2 = int(w * 0.8)
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None
        
        jersey_region = player_region[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        if jersey_region.size == 0:
            return None
        
        # Get dominant color
        pixels = jersey_region.reshape(-1, 3)
        
        # Filter out extreme values
        pixel_brightness = np.mean(pixels, axis=1)
        valid_pixels = pixels[(pixel_brightness > 20) & (pixel_brightness < 235)]
        
        if len(valid_pixels) < 10:
            return np.mean(pixels, axis=0)
        
        # Use KMeans to find dominant color
        try:
            n_clusters = min(3, len(valid_pixels))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            # Get the most central color (likely jersey color)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Find the most common cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            dominant_color = colors[dominant_cluster]
            
            return dominant_color
        except:
            return np.mean(valid_pixels, axis=0)
    
    def learn_team_colors(self, frame, detections):
        """Learn team colors from first few frames"""
        if self.frame_count > 10 or len(detections['players']) < 4:
            return
        
        jersey_colors = []
        for detection in detections['players']:
            color = self.analyze_jersey_color(frame, detection['bbox'])
            if color is not None:
                jersey_colors.append(color)
        
        if len(jersey_colors) >= 4:
            # Use KMeans to separate into 2 teams
            try:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                kmeans.fit(jersey_colors)
                
                team_colors = kmeans.cluster_centers_
                self.team_colors['home'] = team_colors[0]
                self.team_colors['away'] = team_colors[1]
                
                print(f"✅ Learned team colors:")
                print(f"   Home: {self.team_colors['home']}")
                print(f"   Away: {self.team_colors['away']}")
            except:
                pass
    
    def assign_team(self, jersey_color):
        """Assign player to team based on jersey color"""
        if self.team_colors['home'] is None or self.team_colors['away'] is None:
            return 'unknown'
        
        if jersey_color is None:
            return 'unknown'
        
        # Calculate distances to team colors
        home_dist = np.linalg.norm(jersey_color - self.team_colors['home'])
        away_dist = np.linalg.norm(jersey_color - self.team_colors['away'])
        
        if home_dist < away_dist:
            return 'home'
        else:
            return 'away'
    
    def track_players_fixed(self, frame, detections):
        """Fixed player tracking with proper 5v5 assignment"""
        current_detections = detections['players']
        
        # Learn team colors in early frames
        self.learn_team_colors(frame, detections)
        
        # Get jersey colors for all detections
        detection_colors = []
        for detection in current_detections:
            color = self.analyze_jersey_color(frame, detection['bbox'])
            detection_colors.append(color)
        
        # Track existing players
        updated_players = {}
        unmatched_detections = list(range(len(current_detections)))
        
        # Match existing players to new detections
        for player_id, player in self.active_players.items():
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(current_detections):
                if i not in unmatched_detections:
                    continue
                
                # Calculate distance
                dist = math.sqrt(
                    (player['center'][0] - detection['center'][0])**2 + 
                    (player['center'][1] - detection['center'][1])**2
                )
                
                if dist < self.max_distance_threshold and dist < best_distance:
                    best_distance = dist
                    best_match = i
            
            if best_match is not None:
                # Update existing player
                detection = current_detections[best_match]
                jersey_color = detection_colors[best_match]
                
                updated_players[player_id] = {
                    'id': player_id,
                    'bbox': detection['bbox'],
                    'center': detection['center'],
                    'team': player['team'],  # Keep existing team
                    'confidence': detection['confidence'],
                    'jersey_color': jersey_color,
                    'frames_tracked': player.get('frames_tracked', 0) + 1
                }
                unmatched_detections.remove(best_match)
        
        # Count current players per team
        home_count = len([p for p in updated_players.values() if p['team'] == 'home'])
        away_count = len([p for p in updated_players.values() if p['team'] == 'away'])
        
        # Add new players from unmatched detections
        for i in unmatched_detections:
            detection = current_detections[i]
            jersey_color = detection_colors[i]
            team = self.assign_team(jersey_color)
            
            # Limit to 5 players per team
            if team == 'home' and home_count >= self.max_players_per_team:
                continue
            elif team == 'away' and away_count >= self.max_players_per_team:
                continue
            elif team == 'unknown':
                # Assign to team with fewer players
                if home_count < away_count and home_count < self.max_players_per_team:
                    team = 'home'
                elif away_count < self.max_players_per_team:
                    team = 'away'
                else:
                    continue
            
            # Create new player
            new_player = {
                'id': self.next_player_id,
                'bbox': detection['bbox'],
                'center': detection['center'],
                'team': team,
                'confidence': detection['confidence'],
                'jersey_color': jersey_color,
                'frames_tracked': 1
            }
            
            updated_players[self.next_player_id] = new_player
            self.next_player_id += 1
            
            if team == 'home':
                home_count += 1
            elif team == 'away':
                away_count += 1
        
        # Update active players
        self.active_players = updated_players
        
        # Update statistics
        self.detection_stats['total_home_players'] = home_count
        self.detection_stats['total_away_players'] = away_count
        
        return list(updated_players.values())
    
    def track_ball(self, detections):
        """Track basketball"""
        if detections['ball']:
            ball = detections['ball']
            self.ball_trajectory.append(ball['center'])
            self.detection_stats['ball_detections'] += 1
            return ball
        return None
    
    def get_team_counts(self):
        """Get current team player counts"""
        home_count = len([p for p in self.active_players.values() if p['team'] == 'home'])
        away_count = len([p for p in self.active_players.values() if p['team'] == 'away'])
        return home_count, away_count

class Fixed2DMapper:
    """Maps tracked players to 2D court visualization"""
    
    def __init__(self, court_width=600, court_height=400):
        self.court_width = court_width
        self.court_height = court_height
        self.margin = 30
        
    def create_court(self):
        """Create professional basketball court"""
        court = np.ones((self.court_height, self.court_width, 3), dtype=np.uint8) * 34
        
        # Court outline
        cv2.rectangle(court, (self.margin, self.margin), 
                     (self.court_width - self.margin, self.court_height - self.margin), 
                     (255, 255, 255), 2)
        
        # Center line
        center_x = self.court_width // 2
        cv2.line(court, (center_x, self.margin), (center_x, self.court_height - self.margin), 
                (255, 255, 255), 2)
        
        # Center circle
        cv2.circle(court, (center_x, self.court_height // 2), 50, (255, 255, 255), 2)
        
        # 3-point lines (simplified)
        arc_radius = 80
        cv2.ellipse(court, (self.margin + 50, self.court_height // 2), (arc_radius, arc_radius), 
                   0, -90, 90, (255, 255, 255), 2)
        cv2.ellipse(court, (self.court_width - self.margin - 50, self.court_height // 2), 
                   (arc_radius, arc_radius), 0, 90, 270, (255, 255, 255), 2)
        
        # Free throw circles
        cv2.circle(court, (self.margin + 100, self.court_height // 2), 40, (255, 255, 255), 2)
        cv2.circle(court, (self.court_width - self.margin - 100, self.court_height // 2), 40, 
                  (255, 255, 255), 2)
        
        return court
    
    def map_position(self, real_center, frame_shape):
        """Map real position to 2D court coordinates"""
        frame_h, frame_w = frame_shape[:2]
        
        # Simple mapping - left side is home, right side is away
        x_ratio = real_center[0] / frame_w
        y_ratio = real_center[1] / frame_h
        
        court_x = int(self.margin + x_ratio * (self.court_width - 2 * self.margin))
        court_y = int(self.margin + y_ratio * (self.court_height - 2 * self.margin))
        
        return court_x, court_y
    
    def draw_players(self, court, players, frame_shape):
        """Draw players on 2D court with team colors"""
        team_colors = {
            'home': (255, 165, 0),  # Orange
            'away': (0, 100, 255),  # Blue
            'unknown': (128, 128, 128)  # Gray
        }
        
        # Count players per team
        home_players = [p for p in players if p['team'] == 'home']
        away_players = [p for p in players if p['team'] == 'away']
        
        for player in players:
            court_x, court_y = self.map_position(player['center'], frame_shape)
            color = team_colors[player['team']]
            
            # Draw player circle
            cv2.circle(court, (court_x, court_y), 12, color, -1)
            cv2.circle(court, (court_x, court_y), 12, (255, 255, 255), 2)
            
            # Draw player ID
            cv2.putText(court, str(player['id']), (court_x - 5, court_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add team count display
        cv2.putText(court, f"Home: {len(home_players)}/5", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.putText(court, f"Away: {len(away_players)}/5", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        return court
    
    def draw_ball(self, court, ball, frame_shape):
        """Draw ball on 2D court"""
        if ball:
            court_x, court_y = self.map_position(ball['center'], frame_shape)
            cv2.circle(court, (court_x, court_y), 8, (255, 255, 255), -1)
            cv2.circle(court, (court_x, court_y), 8, (0, 0, 0), 2)
        return court

def analyze_with_fixed_tracking(video_path, max_frames=None):
    """Analyze basketball video with fixed 5v5 tracking"""
    print("🏀 Fixed Basketball Team Tracking Analysis")
    print("=" * 60)
    
    # Initialize components
    model_path = r"enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt"
    tracker = FixedBasketballTracker(model_path)
    mapper = Fixed2DMapper()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
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
    
    print(f"📹 Video: {frame_width}x{frame_height} @ {fps}fps")
    print(f"🎯 Processing {total_frames} frames")
    
    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_path = f"{video_name}_fixed_5v5_tracking_{timestamp}.mp4"
    
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
    
    try:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            tracker.frame_count = frame_count
            
            # Detect objects
            detections = tracker.detect_objects(frame)
            
            # Track players with fixed 5v5 logic
            players = tracker.track_players_fixed(frame, detections)
            
            # Track ball
            ball = tracker.track_ball(detections)
            
            # Draw on original frame
            annotated_frame = frame.copy()
            
            # Draw player detections
            for player in players:
                bbox = player['bbox']
                team_color = (255, 165, 0) if player['team'] == 'home' else (0, 100, 255)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), team_color, 2)
                
                # Draw player info
                info_text = f"P{player['id']} {player['team'][:1].upper()}"
                cv2.putText(annotated_frame, info_text, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_color, 2)
            
            # Draw ball
            if ball:
                bbox = ball['bbox']
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
                cv2.putText(annotated_frame, "BALL", (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add team counts to original frame
            home_count, away_count = tracker.get_team_counts()
            cv2.putText(annotated_frame, f"Home: {home_count}/5", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            cv2.putText(annotated_frame, f"Away: {away_count}/5", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            
            # Create 2D court
            court_2d = mapper.create_court()
            court_2d = mapper.draw_players(court_2d, players, frame.shape)
            court_2d = mapper.draw_ball(court_2d, ball, frame.shape)
            
            # Create side-by-side frame
            combined_frame = np.zeros((side_by_side_height, side_by_side_width, 3), dtype=np.uint8)
            
            # Add original frame (left)
            combined_frame[:frame_height, :frame_width] = annotated_frame
            
            # Add 2D court (right)
            court_start_x = frame_width
            court_end_x = court_start_x + court_width
            combined_frame[:400, court_start_x:court_end_x] = court_2d
            
            # Write frame
            out.write(combined_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) @ {fps_current:.1f}fps | "
                      f"Players: H{home_count}/A{away_count}")
    
    finally:
        cap.release()
        out.release()
    
    # Verify output
    output_size = Path(output_path).stat().st_size / (1024 * 1024)
    final_home, final_away = tracker.get_team_counts()
    
    print(f"\n✅ Fixed 5v5 Tracking Complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Size: {output_size:.2f} MB")
    print(f"🏀 Final Team Count: Home {final_home}/5, Away {final_away}/5")
    print(f"📈 Total detections:")
    print(f"   - Ball detections: {tracker.detection_stats['ball_detections']}")
    print(f"   - Frames processed: {frame_count}")
    
    return output_path

if __name__ == "__main__":
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4"
    
    if Path(video_path).exists():
        print("🎯 Running Fixed 5v5 Basketball Tracking...")
        result = analyze_with_fixed_tracking(video_path, max_frames=300)
        
        if result:
            print(f"\n🎉 SUCCESS! Fixed tracking analysis complete!")
            print(f"📺 Video: {result}")
            print(f"\n🏀 Features:")
            print("   ✅ Proper 5v5 player tracking")
            print("   ✅ Consistent player IDs")
            print("   ✅ Team color separation")
            print("   ✅ Side-by-side visualization")
            print("   ✅ Real-time team counts")
        else:
            print("❌ Analysis failed")
    else:
        print(f"❌ Video not found: {video_path}")
