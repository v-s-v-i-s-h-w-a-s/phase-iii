#!/usr/bin/env python3
"""
Enhanced Basketball Intelligence System with Consistent Team Colors
Advanced tracking with pre-initialized team colors and superior ball detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import json
from datetime import datetime
import colorsys
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import math

class ConsistentTeamColorTracker:
    """Manages consistent team color assignment and tracking"""
    
    def __init__(self):
        self.team_colors = {
            'home': {
                'primary': (255, 165, 0),    # Orange (Hawks-like)
                'secondary': (255, 215, 0),   # Gold
                'name': 'HOME',
                'bbox_color': (255, 165, 0),
                'text_color': (255, 255, 255)
            },
            'away': {
                'primary': (0, 100, 255),     # Blue (Knicks-like)  
                'secondary': (255, 69, 0),    # Red-Orange
                'name': 'AWAY',
                'bbox_color': (0, 100, 255),
                'text_color': (255, 255, 255)
            },
            'referee': {
                'primary': (128, 128, 128),   # Gray
                'secondary': (0, 0, 0),       # Black
                'name': 'REF',
                'bbox_color': (128, 128, 128),
                'text_color': (255, 255, 255)
            }
        }
        
        self.player_team_assignments = {}
        self.team_color_samples = {'home': [], 'away': []}
        self.color_initialized = False
        self.frame_count = 0
        
    def extract_dominant_colors(self, image_region, k=3):
        """Extract dominant colors from image region"""
        try:
            # Reshape image to be a list of pixels
            pixels = image_region.reshape(-1, 3)
            
            # Remove very dark and very bright pixels (shadows/highlights)
            mask = np.all(pixels > 30, axis=1) & np.all(pixels < 225, axis=1)
            if np.sum(mask) < 10:  # Not enough valid pixels
                return []
            
            filtered_pixels = pixels[mask]
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(k, len(filtered_pixels)), random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate color frequencies
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Sort colors by frequency
            color_freq = list(zip(colors[unique_labels], counts))
            color_freq.sort(key=lambda x: x[1], reverse=True)
            
            return [color for color, freq in color_freq]
            
        except Exception as e:
            print(f"Color extraction error: {e}")
            return []
    
    def color_distance(self, color1, color2):
        """Calculate Euclidean distance between two colors"""
        return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))
    
    def is_team_color(self, color, team_colors, threshold=80):
        """Check if color matches team colors"""
        for team_color in team_colors:
            if self.color_distance(color, team_color) < threshold:
                return True
        return False
    
    def assign_player_to_team(self, player_id, jersey_colors):
        """Assign player to team based on jersey colors"""
        if not jersey_colors or not self.color_initialized:
            return 'unknown'
        
        home_score = 0
        away_score = 0
        
        for color in jersey_colors[:2]:  # Check top 2 dominant colors
            # Check against home team colors
            if self.is_team_color(color, [self.team_colors['home']['primary'], 
                                        self.team_colors['home']['secondary']]):
                home_score += 1
            
            # Check against away team colors
            if self.is_team_color(color, [self.team_colors['away']['primary'], 
                                        self.team_colors['away']['secondary']]):
                away_score += 1
        
        if home_score > away_score:
            return 'home'
        elif away_score > home_score:
            return 'away'
        else:
            return 'unknown'
    
    def initialize_team_colors(self, players_data):
        """Initialize team colors from first few frames"""
        if self.color_initialized or self.frame_count < 30:
            return
        
        all_colors = {'home': [], 'away': []}
        
        # Collect colors from all players
        for player_id, data in players_data.items():
            if 'jersey_colors' in data and data['jersey_colors']:
                # Use simple heuristic: alternate assignment for initialization
                team = 'home' if len(all_colors['home']) <= len(all_colors['away']) else 'away'
                all_colors[team].extend(data['jersey_colors'][:2])
        
        # Update team colors based on collected samples
        for team in ['home', 'away']:
            if all_colors[team]:
                # Use most common colors for team
                color_counts = defaultdict(int)
                for color in all_colors[team]:
                    color_tuple = tuple(color)
                    color_counts[color_tuple] += 1
                
                if color_counts:
                    most_common = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
                    if len(most_common) >= 1:
                        self.team_colors[team]['primary'] = most_common[0][0]
                    if len(most_common) >= 2:
                        self.team_colors[team]['secondary'] = most_common[1][0]
        
        self.color_initialized = True
        print(f"🎨 Team colors initialized:")
        print(f"   Home: {self.team_colors['home']['primary']}")
        print(f"   Away: {self.team_colors['away']['primary']}")

class EnhancedBasketballIntelligence:
    """Enhanced basketball intelligence with consistent tracking"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.team_tracker = ConsistentTeamColorTracker()
        
        # Enhanced class configuration
        self.classes = {
            0: {'name': 'ball', 'color': (0, 255, 255), 'track': True},      # Bright Yellow
            1: {'name': 'basket', 'color': (255, 0, 0), 'track': True},     # Blue  
            2: {'name': 'player', 'color': (0, 255, 0), 'track': True},     # Green
            3: {'name': 'referee', 'color': (128, 128, 128), 'track': True} # Gray
        }
        
        # Tracking system
        self.trackers = {}
        self.next_id = 0
        self.tracking_history = defaultdict(lambda: deque(maxlen=30))
        
        # Enhanced detection parameters
        self.conf_thresholds = {
            0: 0.3,  # ball - lower threshold for better detection
            1: 0.4,  # basket
            2: 0.3,  # player
            3: 0.4   # referee
        }
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'detections_per_frame': [],
            'class_detections': defaultdict(int),
            'team_assignments': defaultdict(int),
            'ball_tracking_accuracy': [],
            'processing_times': []
        }
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracking(self, detections):
        """Update object tracking with consistent IDs"""
        current_frame_trackers = {}
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls_id = detection
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find best matching tracker
            best_match = None
            best_iou = 0.0
            
            for tracker_id, tracker_data in self.trackers.items():
                if tracker_data['class'] == cls_id:
                    iou = self.calculate_iou((x1, y1, x2, y2), tracker_data['bbox'])
                    if iou > best_iou and iou > 0.3:  # IoU threshold for matching
                        best_iou = iou
                        best_match = tracker_id
            
            if best_match:
                # Update existing tracker
                self.trackers[best_match]['bbox'] = (x1, y1, x2, y2)
                self.trackers[best_match]['center'] = center
                self.trackers[best_match]['conf'] = conf
                self.trackers[best_match]['last_seen'] = self.stats['total_frames']
                current_frame_trackers[best_match] = self.trackers[best_match]
            else:
                # Create new tracker
                tracker_id = self.next_id
                self.next_id += 1
                
                self.trackers[tracker_id] = {
                    'class': cls_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'conf': conf,
                    'last_seen': self.stats['total_frames'],
                    'team': 'unknown' if cls_id == 2 else None,  # Only players have teams
                    'jersey_colors': []
                }
                current_frame_trackers[tracker_id] = self.trackers[tracker_id]
        
        # Remove old trackers
        to_remove = []
        for tracker_id, tracker_data in self.trackers.items():
            if self.stats['total_frames'] - tracker_data['last_seen'] > 30:  # 30 frames timeout
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            del self.trackers[tracker_id]
        
        return current_frame_trackers
    
    def analyze_jersey_colors(self, frame, bbox):
        """Analyze jersey colors for team assignment"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract jersey region (upper part of player)
        height = y2 - y1
        jersey_y2 = y1 + int(height * 0.6)  # Upper 60% for jersey
        
        jersey_region = frame[y1:jersey_y2, x1:x2]
        if jersey_region.size > 0:
            dominant_colors = self.team_tracker.extract_dominant_colors(jersey_region)
            return dominant_colors[:3]  # Top 3 colors
        return []
    
    def process_frame(self, frame):
        """Process single frame with enhanced detection and tracking"""
        frame_start = time.time()
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Apply class-specific confidence thresholds
                    if cls_id in self.conf_thresholds and conf >= self.conf_thresholds[cls_id]:
                        detections.append((x1, y1, x2, y2, conf, cls_id))
                        self.stats['class_detections'][cls_id] += 1
        
        # Update tracking
        frame_trackers = self.update_tracking(detections)
        
        # Analyze jersey colors for players
        players_data = {}
        for tracker_id, tracker_data in frame_trackers.items():
            if tracker_data['class'] == 2:  # Player
                jersey_colors = self.analyze_jersey_colors(frame, tracker_data['bbox'])
                tracker_data['jersey_colors'] = jersey_colors
                players_data[tracker_id] = tracker_data
        
        # Initialize or update team colors
        self.team_tracker.frame_count = self.stats['total_frames']
        if not self.team_tracker.color_initialized:
            self.team_tracker.initialize_team_colors(players_data)
        
        # Assign teams to players
        for tracker_id, tracker_data in frame_trackers.items():
            if tracker_data['class'] == 2 and 'jersey_colors' in tracker_data:
                team = self.team_tracker.assign_player_to_team(tracker_id, tracker_data['jersey_colors'])
                tracker_data['team'] = team
                if team != 'unknown':
                    self.stats['team_assignments'][team] += 1
        
        # Draw enhanced annotations
        annotated_frame = self.draw_enhanced_annotations(frame, frame_trackers)
        
        # Update statistics
        self.stats['total_frames'] += 1
        self.stats['detections_per_frame'].append(len(detections))
        processing_time = time.time() - frame_start
        self.stats['processing_times'].append(processing_time)
        
        return annotated_frame, frame_trackers
    
    def draw_enhanced_annotations(self, frame, trackers):
        """Draw enhanced annotations with consistent team colors"""
        annotated_frame = frame.copy()
        
        for tracker_id, tracker_data in trackers.items():
            x1, y1, x2, y2 = map(int, tracker_data['bbox'])
            cls_id = tracker_data['class']
            conf = tracker_data['conf']
            
            # Determine colors
            if cls_id == 2:  # Player
                team = tracker_data.get('team', 'unknown')
                if team in self.team_tracker.team_colors:
                    color = self.team_tracker.team_colors[team]['bbox_color']
                    text_color = self.team_tracker.team_colors[team]['text_color']
                    team_name = self.team_tracker.team_colors[team]['name']
                else:
                    color = (128, 128, 128)  # Gray for unknown
                    text_color = (255, 255, 255)
                    team_name = 'UNK'
            elif cls_id == 3:  # Referee
                color = self.team_tracker.team_colors['referee']['bbox_color']
                text_color = self.team_tracker.team_colors['referee']['text_color']
                team_name = self.team_tracker.team_colors['referee']['name']
            else:
                color = self.classes[cls_id]['color']
                text_color = (255, 255, 255)
                team_name = None
            
            # Draw bounding box with thicker lines for better visibility
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            class_name = self.classes[cls_id]['name']
            if cls_id in [2, 3] and team_name:  # Player or referee
                label = f"{team_name}-{class_name.upper()}"
            else:
                label = class_name.upper()
            
            label += f" {conf:.2f} ID:{tracker_id}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw tracking ID for players
            if cls_id == 2:
                cv2.circle(annotated_frame, tracker_data['center'], 5, color, -1)
        
        # Add frame information
        info_text = f"Frame: {self.stats['total_frames']} | Detections: {len(trackers)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add team count
        team_counts = defaultdict(int)
        for tracker_data in trackers.values():
            if tracker_data['class'] == 2:  # Player
                team = tracker_data.get('team', 'unknown')
                team_counts[team] += 1
        
        team_info = f"Teams - HOME: {team_counts['home']}, AWAY: {team_counts['away']}, UNK: {team_counts['unknown']}"
        cv2.putText(annotated_frame, team_info, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame

def process_video_with_enhanced_detection(model_path, video_path, output_path):
    """Process video with enhanced detection and consistent team tracking"""
    
    print("🏀 ENHANCED BASKETBALL INTELLIGENCE")
    print("=" * 50)
    print(f"📁 Model: {model_path}")
    print(f"🎥 Video: {video_path}")
    print(f"💾 Output: {output_path}")
    
    # Initialize enhanced intelligence system
    intelligence = EnhancedBasketballIntelligence(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n🚀 Starting enhanced processing...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with enhanced intelligence
        annotated_frame, trackers = intelligence.process_frame(frame)
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress update
        if intelligence.stats['total_frames'] % 100 == 0:
            elapsed = time.time() - start_time
            progress = (intelligence.stats['total_frames'] / total_frames) * 100
            eta = (elapsed / intelligence.stats['total_frames']) * (total_frames - intelligence.stats['total_frames'])
            avg_fps = intelligence.stats['total_frames'] / elapsed
            print(f"   Progress: {progress:.1f}% | Frame {intelligence.stats['total_frames']}/{total_frames} | FPS: {avg_fps:.1f} | ETA: {eta:.1f}s")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n✅ Enhanced processing completed!")
    print(f"📊 FINAL STATISTICS:")
    print(f"   Total Time: {total_time:.1f}s")
    print(f"   Average FPS: {intelligence.stats['total_frames'] / total_time:.1f}")
    print(f"   Total Detections: {sum(intelligence.stats['class_detections'].values())}")
    
    for cls_id, count in intelligence.stats['class_detections'].items():
        class_name = intelligence.classes[cls_id]['name']
        print(f"   {class_name}: {count}")
    
    print(f"   Team Assignments:")
    for team, count in intelligence.stats['team_assignments'].items():
        print(f"     {team}: {count}")
    
    return intelligence.stats

if __name__ == "__main__":
    # Use the best available model
    model_path = r"basketball_real_training\real_dataset_20250803_121502\weights\best.pt"
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks_vs_knicks.mp4"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"enhanced_basketball_intelligence_{timestamp}.mp4"
    
    process_video_with_enhanced_detection(model_path, video_path, output_path)
