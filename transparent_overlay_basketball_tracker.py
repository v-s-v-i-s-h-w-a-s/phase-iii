#!/usr/bin/env python3
"""
Enhanced Basketball Tracker with Transparent 2D Overlay
======================================================
Creates AVI video with transparent 2D tactical overlay on original footage
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

class OverlayBasketballTracker:
    """Basketball tracker with transparent 2D overlay"""
    
    def __init__(self, model_path):
        print("🏀 Initializing Overlay Basketball Tracker...")
        
        # Load custom YOLO model
        self.model = YOLO(model_path)
        print(f"✅ Loaded custom model: {model_path}")
        
        # Player tracking
        self.active_players = {}
        self.next_player_id = 1
        self.max_players_per_team = 5
        
        # Team colors (learned from video)
        self.team_colors = {'home': None, 'away': None}
        
        # Tracking parameters
        self.max_distance_threshold = 100
        self.confidence_threshold = 0.3
        
        # Statistics
        self.frame_count = 0
        self.detection_stats = {
            'total_home_players': 0,
            'total_away_players': 0,
            'ball_detections': 0,
            'referee_detections': 0,
            'rim_detections': 0
        }
        
        # Ball and object tracking
        self.ball_trajectory = []
        self.detected_rims = []
        
    def detect_objects(self, frame):
        """Detect basketball objects using custom YOLO"""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = {
            'players': [],
            'ball': None,
            'referees': [],
            'rims': []
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
                        self.detection_stats['ball_detections'] += 1
                    elif cls == 1:  # Rim/Basket
                        detections['rims'].append(detection)
                        self.detection_stats['rim_detections'] += 1
                    elif cls == 2:  # Player
                        detections['players'].append(detection)
                    elif cls == 3:  # Referee
                        detections['referees'].append(detection)
                        self.detection_stats['referee_detections'] += 1
        
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
        jersey_y1 = int(h * 0.15)
        jersey_y2 = int(h * 0.6)
        jersey_x1 = int(w * 0.2)
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
            
            # Get the most central color
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Find the most common cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique_labels[np.argmax(counts)]
            dominant_color = colors[dominant_cluster]
            
            return dominant_color
        except Exception:
            return np.mean(valid_pixels, axis=0)
    
    def learn_team_colors(self, frame, detections):
        """Learn team colors from first few frames"""
        if self.frame_count > 15 or len(detections['players']) < 4:
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
                
                print("✅ Learned team colors for better tracking")
            except Exception:
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
        
        return 'home' if home_dist < away_dist else 'away'
    
    def track_players_enhanced(self, frame, detections):
        """Enhanced player tracking with proper 5v5 assignment"""
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
                    'team': player['team'],
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
        """Track basketball with trajectory"""
        if detections['ball']:
            ball = detections['ball']
            self.ball_trajectory.append(ball['center'])
            # Keep only last 30 trajectory points
            if len(self.ball_trajectory) > 30:
                self.ball_trajectory.pop(0)
            return ball
        return None
    
    def track_rims(self, detections):
        """Track basketball rims"""
        self.detected_rims = detections['rims']
        return detections['rims']

class TransparentOverlayMapper:
    """Creates transparent 2D tactical overlay"""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.margin = 40
        
        # Scale factor for overlay
        self.scale_factor = 0.8
        self.overlay_width = int(frame_width * self.scale_factor)
        self.overlay_height = int(frame_height * self.scale_factor)
        
        # Position overlay (bottom-right corner)
        self.overlay_x = frame_width - self.overlay_width - 20
        self.overlay_y = frame_height - self.overlay_height - 20
    
    def create_transparent_court(self):
        """Create transparent basketball court overlay"""
        # Create transparent overlay
        overlay = np.zeros((self.overlay_height, self.overlay_width, 4), dtype=np.uint8)
        
        # Court background (semi-transparent dark)
        overlay[:, :, 3] = 120  # Alpha channel
        overlay[:, :, 0:3] = [34, 34, 34]  # Dark court color
        
        # Court outline (white)
        cv2.rectangle(overlay, (self.margin, self.margin), 
                     (self.overlay_width - self.margin, self.overlay_height - self.margin), 
                     (255, 255, 255, 255), 3)
        
        # Center line
        center_x = self.overlay_width // 2
        cv2.line(overlay, (center_x, self.margin), (center_x, self.overlay_height - self.margin), 
                (255, 255, 255, 255), 2)
        
        # Center circle
        cv2.circle(overlay, (center_x, self.overlay_height // 2), 40, (255, 255, 255, 255), 2)
        
        # 3-point arcs
        arc_radius = 60
        cv2.ellipse(overlay, (self.margin + 40, self.overlay_height // 2), (arc_radius, arc_radius), 
                   0, -90, 90, (255, 255, 255, 255), 2)
        cv2.ellipse(overlay, (self.overlay_width - self.margin - 40, self.overlay_height // 2), 
                   (arc_radius, arc_radius), 0, 90, 270, (255, 255, 255, 255), 2)
        
        # Free throw circles
        cv2.circle(overlay, (self.margin + 80, self.overlay_height // 2), 30, (255, 255, 255, 255), 2)
        cv2.circle(overlay, (self.overlay_width - self.margin - 80, self.overlay_height // 2), 30, 
                  (255, 255, 255, 255), 2)
        
        return overlay
    
    def map_position_to_overlay(self, real_center):
        """Map real position to overlay coordinates"""
        x_ratio = real_center[0] / self.frame_width
        y_ratio = real_center[1] / self.frame_height
        
        overlay_x = int(self.margin + x_ratio * (self.overlay_width - 2 * self.margin))
        overlay_y = int(self.margin + y_ratio * (self.overlay_height - 2 * self.margin))
        
        return overlay_x, overlay_y
    
    def add_players_to_overlay(self, overlay, players):
        """Add players to transparent overlay"""
        team_colors = {
            'home': (255, 165, 0, 255),  # Orange with full alpha
            'away': (0, 100, 255, 255),  # Blue with full alpha
            'unknown': (128, 128, 128, 255)  # Gray with full alpha
        }
        
        for player in players:
            overlay_x, overlay_y = self.map_position_to_overlay(player['center'])
            color = team_colors[player['team']]
            
            # Draw player circle
            cv2.circle(overlay, (overlay_x, overlay_y), 10, color, -1)
            cv2.circle(overlay, (overlay_x, overlay_y), 10, (255, 255, 255, 255), 2)
            
            # Draw player ID
            cv2.putText(overlay, str(player['id']), (overlay_x - 5, overlay_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 255), 1)
        
        return overlay
    
    def add_ball_to_overlay(self, overlay, ball, ball_trajectory):
        """Add ball and trajectory to overlay"""
        if ball:
            overlay_x, overlay_y = self.map_position_to_overlay(ball['center'])
            
            # Draw ball trajectory
            if len(ball_trajectory) > 1:
                overlay_points = [self.map_position_to_overlay(point) for point in ball_trajectory]
                for i in range(1, len(overlay_points)):
                    alpha = int(255 * (i / len(overlay_points)))  # Fade effect
                    cv2.line(overlay, overlay_points[i-1], overlay_points[i], 
                            (255, 255, 255, alpha), 2)
            
            # Draw ball
            cv2.circle(overlay, (overlay_x, overlay_y), 6, (255, 255, 255, 255), -1)
            cv2.circle(overlay, (overlay_x, overlay_y), 6, (0, 0, 0, 255), 2)
        
        return overlay
    
    def add_rims_to_overlay(self, overlay, rims):
        """Add basketball rims to overlay"""
        for rim in rims:
            overlay_x, overlay_y = self.map_position_to_overlay(rim['center'])
            # Draw rim as rectangle
            cv2.rectangle(overlay, (overlay_x - 8, overlay_y - 3), (overlay_x + 8, overlay_y + 3), 
                         (255, 100, 100, 255), -1)
        
        return overlay
    
    def add_referees_to_overlay(self, overlay, referees):
        """Add referees to overlay"""
        for referee in referees:
            overlay_x, overlay_y = self.map_position_to_overlay(referee['center'])
            # Draw referee as triangle
            points = np.array([[overlay_x, overlay_y - 8], 
                              [overlay_x - 6, overlay_y + 6], 
                              [overlay_x + 6, overlay_y + 6]], np.int32)
            cv2.fillPoly(overlay, [points], (255, 255, 0, 255))  # Yellow referee
        
        return overlay
    
    def blend_overlay_with_frame(self, frame, overlay):
        """Blend transparent overlay with original frame"""
        # Convert overlay to BGR and extract alpha
        overlay_bgr = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0
        
        # Get the region of the frame where overlay will be placed
        y1, y2 = self.overlay_y, self.overlay_y + self.overlay_height
        x1, x2 = self.overlay_x, self.overlay_x + self.overlay_width
        
        # Ensure we don't go out of bounds
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        
        # Get actual dimensions after bounds checking
        actual_h = y2 - y1
        actual_w = x2 - x1
        
        if actual_h > 0 and actual_w > 0:
            # Resize overlay if needed
            if actual_h != self.overlay_height or actual_w != self.overlay_width:
                overlay_bgr = cv2.resize(overlay_bgr, (actual_w, actual_h))
                alpha = cv2.resize(alpha, (actual_w, actual_h))
            
            # Extract the region from the frame
            roi = frame[y1:y2, x1:x2]
            
            # Blend using alpha
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_bgr[:, :, c] * alpha
            
            # Put the blended region back
            frame[y1:y2, x1:x2] = roi
        
        return frame

def create_overlay_analysis_avi(video_path, output_minutes=2):
    """Create basketball analysis with transparent 2D overlay saved as AVI"""
    print("🏀 Enhanced Basketball Analysis with Transparent Overlay")
    print("=" * 65)
    
    # Initialize components
    model_path = r"enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt"
    tracker = OverlayBasketballTracker(model_path)
    
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
    
    # Calculate frames for desired duration
    max_frames = min(total_frames, fps * 60 * output_minutes)  # 2 minutes
    
    print(f"📹 Video: {frame_width}x{frame_height} @ {fps}fps")
    print(f"🎯 Creating {output_minutes}-minute analysis ({max_frames} frames)")
    
    # Initialize overlay mapper
    mapper = TransparentOverlayMapper(frame_width, frame_height)
    
    # Output setup - AVI format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_path = f"{video_name}_transparent_overlay_{timestamp}.avi"
    
    # AVI codec (XVID works well)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("❌ Failed to create AVI output")
        return None
    
    frame_count = 0
    start_time = time.time()
    
    print(f"🎯 Creating transparent overlay analysis: {output_path}")
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            tracker.frame_count = frame_count
            
            # Detect all objects
            detections = tracker.detect_objects(frame)
            
            # Track players with 5v5 logic
            players = tracker.track_players_enhanced(frame, detections)
            
            # Track other objects
            ball = tracker.track_ball(detections)
            rims = tracker.track_rims(detections)
            referees = detections['referees']
            
            # Draw basic detections on original frame
            annotated_frame = frame.copy()
            
            # Draw player detections with team colors
            for player in players:
                bbox = player['bbox']
                team_color = (255, 165, 0) if player['team'] == 'home' else (0, 100, 255)
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), team_color, 2)
                
                # Player info
                info_text = f"P{player['id']} {player['team'][:1].upper()}"
                cv2.putText(annotated_frame, info_text, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
            
            # Draw ball
            if ball:
                bbox = ball['bbox']
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
                cv2.putText(annotated_frame, "BALL", (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw referees
            for referee in referees:
                bbox = referee['bbox']
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
                cv2.putText(annotated_frame, "REF", (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw rims
            for rim in rims:
                bbox = rim['bbox']
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 100, 100), 2)
                cv2.putText(annotated_frame, "RIM", (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
            
            # Create transparent 2D overlay
            court_overlay = mapper.create_transparent_court()
            court_overlay = mapper.add_players_to_overlay(court_overlay, players)
            court_overlay = mapper.add_ball_to_overlay(court_overlay, ball, tracker.ball_trajectory)
            court_overlay = mapper.add_rims_to_overlay(court_overlay, rims)
            court_overlay = mapper.add_referees_to_overlay(court_overlay, referees)
            
            # Blend overlay with frame
            final_frame = mapper.blend_overlay_with_frame(annotated_frame, court_overlay)
            
            # Add team counts and info
            home_count = len([p for p in players if p['team'] == 'home'])
            away_count = len([p for p in players if p['team'] == 'away'])
            
            cv2.putText(final_frame, f"Home: {home_count}/5", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            cv2.putText(final_frame, f"Away: {away_count}/5", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            
            # Add timestamp
            time_text = f"Time: {frame_count//fps//60:02d}:{(frame_count//fps)%60:02d}"
            cv2.putText(final_frame, time_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to AVI
            out.write(final_frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / max_frames) * 100
                time_remaining = (max_frames - frame_count) / fps_current if fps_current > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{max_frames}) @ {fps_current:.1f}fps | "
                      f"ETA: {time_remaining:.1f}s | Players: H{home_count}/A{away_count}")
    
    finally:
        cap.release()
        out.release()
    
    # Verify output
    output_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Transparent Overlay Analysis Complete!")
    print(f"📁 Output: {output_path}")
    print(f"📊 Size: {output_size:.2f} MB")
    print(f"🎯 Duration: {output_minutes} minutes")
    print(f"🏀 Final Detection Stats:")
    for key, value in tracker.detection_stats.items():
        print(f"   - {key}: {value}")
    
    return output_path

if __name__ == "__main__":
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4"
    
    if Path(video_path).exists():
        print("🎯 Creating Enhanced Basketball Analysis with Transparent 2D Overlay...")
        result = create_overlay_analysis_avi(video_path, output_minutes=2)
        
        if result:
            print(f"\n🎉 SUCCESS! Transparent overlay analysis complete!")
            print(f"📺 AVI Video: {result}")
            print(f"\n🏀 Features:")
            print("   ✅ Transparent 2D tactical overlay on original footage")
            print("   ✅ 5v5 player tracking with consistent IDs")
            print("   ✅ Ball trajectory visualization")
            print("   ✅ Referee and rim detection")
            print("   ✅ AVI format for better compatibility")
            print("   ✅ 2-minute duration")
        else:
            print("❌ Analysis failed")
    else:
        print(f"❌ Video not found: {video_path}")
