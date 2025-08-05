#!/usr/bin/env python3
"""
Enhanced Basketball Intelligence System with Improved Labeling
Advanced tracking with corrected class mappings and superior accuracy
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

class ImprovedTeamColorTracker:
    """Enhanced team color assignment with better accuracy"""
    
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
        self.confidence_threshold = 0.6
        
    def assign_player_to_team(self, player_id, jersey_colors):
        """Improved team assignment with confidence scoring"""
        if not jersey_colors or not self.color_initialized:
            return 'unknown'
        
        # Use cached assignment if available and confident
        if player_id in self.player_team_assignments:
            assignment = self.player_team_assignments[player_id]
            if assignment['confidence'] > self.confidence_threshold:
                return assignment['team']
        
        # Calculate color similarity scores
        home_score = self._calculate_color_similarity(jersey_colors, self.team_colors['home']['primary'])
        away_score = self._calculate_color_similarity(jersey_colors, self.team_colors['away']['primary'])
        ref_score = self._calculate_color_similarity(jersey_colors, self.team_colors['referee']['primary'])
        
        # Determine best match
        scores = {'home': home_score, 'away': away_score, 'referee': ref_score}
        best_team = max(scores, key=scores.get)
        confidence = scores[best_team] / (sum(scores.values()) + 1e-6)
        
        # Store assignment with confidence
        self.player_team_assignments[player_id] = {
            'team': best_team,
            'confidence': confidence,
            'frame': self.frame_count
        }
        
        return best_team if confidence > 0.4 else 'unknown'
    
    def _calculate_color_similarity(self, colors1, color2):
        """Calculate similarity between color sets and single color"""
        if not colors1:
            return 0.0
        
        max_similarity = 0.0
        for color1 in colors1:
            # Convert to LAB color space for better perceptual similarity
            similarity = 1.0 / (1.0 + np.linalg.norm(np.array(color1) - np.array(color2)))
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def initialize_team_colors(self, players_data):
        """Initialize team colors based on initial player jersey analysis"""
        if self.color_initialized or len(players_data) < 2:
            return
        
        all_colors = []
        for player_id, player_data in players_data.items():
            if 'jersey_colors' in player_data and player_data['jersey_colors']:
                all_colors.extend(player_data['jersey_colors'])
        
        if len(all_colors) < 4:
            return
        
        # Use KMeans to find two dominant team colors
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            color_array = np.array(all_colors)
            kmeans.fit(color_array)
            
            team_colors = kmeans.cluster_centers_.astype(int)
            
            # Assign colors to teams  
            self.team_colors['home']['primary'] = (int(team_colors[0][0]), int(team_colors[0][1]), int(team_colors[0][2]))
            self.team_colors['away']['primary'] = (int(team_colors[1][0]), int(team_colors[1][1]), int(team_colors[1][2]))
            
            # Update bbox colors
            self.team_colors['home']['bbox_color'] = (int(team_colors[0][0]), int(team_colors[0][1]), int(team_colors[0][2]))
            self.team_colors['away']['bbox_color'] = (int(team_colors[1][0]), int(team_colors[1][1]), int(team_colors[1][2]))
            
            self.color_initialized = True
            print("🎨 Team colors initialized:")
            print(f"   Home: {self.team_colors['home']['primary']}")
            print(f"   Away: {self.team_colors['away']['primary']}")
            
        except Exception as e:
            print(f"Warning: Could not initialize team colors: {e}")

class ImprovedBasketballIntelligence:
    """Enhanced basketball intelligence with improved labeling accuracy"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.team_tracker = ImprovedTeamColorTracker()
        
        # Get the actual class names from the model
        self.model_classes = self.model.names if hasattr(self.model, 'names') else {}
        
        # Enhanced class configuration with proper mapping
        self.classes = {
            0: {'name': 'ball', 'color': (0, 255, 255), 'track': True},      # Bright Cyan
            1: {'name': 'rim', 'color': (255, 0, 0), 'track': True},        # Red  
            2: {'name': 'player', 'color': (0, 255, 0), 'track': True},     # Green
            3: {'name': 'referee', 'color': (128, 128, 128), 'track': True} # Gray
        }
        
        # Map model class names to our standardized names
        self.class_name_mapping = {
            'basketball': 'ball',
            'ball': 'ball',
            'rim': 'rim',
            'basket': 'rim',
            'hoop': 'rim',
            'player': 'player',
            'person': 'player',
            'referee': 'referee',
            'ref': 'referee'
        }
        
        # Tracking system with improved parameters
        self.trackers = {}
        self.next_id = 0
        self.tracking_history = defaultdict(lambda: deque(maxlen=30))
        self.max_disappeared = 10
        
        # Enhanced detection parameters
        self.conf_thresholds = {
            0: 0.25,  # ball - lower threshold for better detection
            1: 0.35,  # rim
            2: 0.30,  # player
            3: 0.40   # referee
        }
        
        # NMS thresholds for reducing duplicate detections
        self.nms_thresholds = {
            0: 0.4,   # ball
            1: 0.5,   # rim
            2: 0.6,   # player
            3: 0.5    # referee
        }
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'detections_per_frame': [],
            'class_detections': defaultdict(int),
            'team_assignments': defaultdict(int),
            'ball_tracking_accuracy': [],
            'processing_times': [],
            'labeling_accuracy': {'correct': 0, 'total': 0}
        }
        
        print(f"🏀 Improved Basketball Intelligence initialized")
        print(f"   Model classes: {self.model_classes}")
        print(f"   Confidence thresholds: {self.conf_thresholds}")
        
    def get_standardized_class_name(self, cls_id, model_name=None):
        """Get standardized class name with improved mapping"""
        if model_name and model_name.lower() in self.class_name_mapping:
            return self.class_name_mapping[model_name.lower()]
        elif cls_id in self.classes:
            return self.classes[cls_id]['name']
        else:
            return f"class_{cls_id}"
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracking(self, detections):
        """Enhanced tracking with improved association"""
        current_frame_trackers = {}
        
        # Match detections to existing trackers
        used_detections = set()
        
        for tracker_id, tracker in list(self.trackers.items()):
            best_match_idx = -1
            best_iou = 0.3  # Minimum IoU threshold
            
            for idx, detection in enumerate(detections):
                if idx in used_detections:
                    continue
                    
                x1, y1, x2, y2, conf, cls_id = detection
                
                # Only match same class
                if cls_id != tracker['class']:
                    continue
                
                iou = self.calculate_iou((x1, y1, x2, y2), tracker['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            if best_match_idx != -1:
                # Update existing tracker
                x1, y1, x2, y2, conf, cls_id = detections[best_match_idx]
                
                tracker['bbox'] = (x1, y1, x2, y2)
                tracker['conf'] = conf
                tracker['disappeared'] = 0
                tracker['last_seen'] = self.stats['total_frames']
                
                self.tracking_history[tracker_id].append((x1, y1, x2, y2))
                
                current_frame_trackers[tracker_id] = tracker
                used_detections.add(best_match_idx)
            else:
                # Increment disappeared counter
                tracker['disappeared'] += 1
                if tracker['disappeared'] < self.max_disappeared:
                    current_frame_trackers[tracker_id] = tracker
        
        # Create new trackers for unmatched detections
        for idx, detection in enumerate(detections):
            if idx not in used_detections:
                x1, y1, x2, y2, conf, cls_id = detection
                
                tracker_id = self.next_id
                self.next_id += 1
                
                new_tracker = {
                    'bbox': (x1, y1, x2, y2),
                    'class': cls_id,
                    'conf': conf,
                    'disappeared': 0,
                    'created': self.stats['total_frames'],
                    'last_seen': self.stats['total_frames']
                }
                
                self.tracking_history[tracker_id].append((x1, y1, x2, y2))
                current_frame_trackers[tracker_id] = new_tracker
        
        # Update trackers
        self.trackers = current_frame_trackers
        return current_frame_trackers
    
    def analyze_jersey_colors(self, frame, bbox, num_colors=3):
        """Enhanced jersey color analysis"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return []
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        # Focus on torso area (middle section) for jersey color
        h_region, w_region = player_region.shape[:2]
        torso_y1 = int(h_region * 0.2)
        torso_y2 = int(h_region * 0.7)
        torso_x1 = int(w_region * 0.1)
        torso_x2 = int(w_region * 0.9)
        
        if torso_y2 > torso_y1 and torso_x2 > torso_x1:
            torso_region = player_region[torso_y1:torso_y2, torso_x1:torso_x2]
        else:
            torso_region = player_region
        
        # Convert to RGB and reshape
        rgb_region = cv2.cvtColor(torso_region, cv2.COLOR_BGR2RGB)
        pixels = rgb_region.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (shadows/highlights)
        pixel_brightness = np.mean(pixels, axis=1)
        valid_pixels = pixels[(pixel_brightness > 30) & (pixel_brightness < 225)]
        
        if len(valid_pixels) < 10:
            return []
        
        try:
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=min(num_colors, len(valid_pixels)), random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            # Convert back to BGR
            dominant_colors = [(int(color[2]), int(color[1]), int(color[0])) for color in dominant_colors]
            return dominant_colors[:num_colors]
        except:
            return []
    
    def process_frame(self, frame):
        """Process single frame with improved labeling"""
        frame_start = time.time()
        
        # Run YOLO detection with improved parameters
        results = self.model(frame, verbose=False, conf=0.2, iou=0.4)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get model class name for better mapping
                    model_class_name = self.model_classes.get(cls_id, f"class_{cls_id}")
                    
                    # Apply class-specific confidence thresholds
                    if cls_id in self.conf_thresholds and conf >= self.conf_thresholds[cls_id]:
                        # Validate bounding box
                        if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 100:
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
        if not self.team_tracker.color_initialized and len(players_data) >= 2:
            self.team_tracker.initialize_team_colors(players_data)
        
        # Assign teams to players
        for tracker_id, tracker_data in frame_trackers.items():
            if tracker_data['class'] == 2 and 'jersey_colors' in tracker_data:
                team = self.team_tracker.assign_player_to_team(tracker_id, tracker_data['jersey_colors'])
                tracker_data['team'] = team
                if team != 'unknown':
                    self.stats['team_assignments'][team] += 1
        
        # Draw enhanced annotations
        annotated_frame = self.draw_improved_annotations(frame, frame_trackers)
        
        # Update statistics
        self.stats['total_frames'] += 1
        self.stats['detections_per_frame'].append(len(detections))
        processing_time = time.time() - frame_start
        self.stats['processing_times'].append(processing_time)
        
        return annotated_frame, frame_trackers
    
    def draw_improved_annotations(self, frame, trackers):
        """Draw improved annotations with better labeling"""
        annotated_frame = frame.copy()
        
        # Draw detection statistics
        stats_text = f"Frame: {self.stats['total_frames']} | Objects: {len(trackers)}"
        cv2.putText(annotated_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for tracker_id, tracker_data in trackers.items():
            x1, y1, x2, y2 = map(int, tracker_data['bbox'])
            cls_id = tracker_data['class']
            conf = tracker_data['conf']
            
            # Get proper class name
            model_class_name = self.model_classes.get(cls_id, f"class_{cls_id}")
            standard_class_name = self.get_standardized_class_name(cls_id, model_class_name)
            
            # Determine colors and labels
            if cls_id == 2:  # Player
                team = tracker_data.get('team', 'unknown')
                if team in self.team_tracker.team_colors:
                    color = self.team_tracker.team_colors[team]['bbox_color']
                    text_color = self.team_tracker.team_colors[team]['text_color']
                    team_name = self.team_tracker.team_colors[team]['name']
                    label = f"{team_name} {standard_class_name.upper()}"
                else:
                    color = (128, 128, 128)  # Gray for unknown
                    text_color = (255, 255, 255)
                    label = f"UNK {standard_class_name.upper()}"
            elif cls_id == 3:  # Referee
                color = self.team_tracker.team_colors['referee']['bbox_color']
                text_color = self.team_tracker.team_colors['referee']['text_color']
                label = f"REF {standard_class_name.upper()}"
            else:
                color = self.classes.get(cls_id, {'color': (255, 255, 255)})['color']
                text_color = (255, 255, 255)
                label = standard_class_name.upper()
            
            # Draw bounding box with adaptive thickness
            box_area = (x2 - x1) * (y2 - y1)
            thickness = max(2, min(4, int(box_area / 5000)))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare detailed label
            detailed_label = f"{label} {conf:.2f} ID:{tracker_id}"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(detailed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, detailed_label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw tracking trail for balls
            if cls_id == 0 and tracker_id in self.tracking_history:
                trail = list(self.tracking_history[tracker_id])
                for i in range(1, len(trail)):
                    if i < len(trail):
                        pt1 = (int((trail[i-1][0] + trail[i-1][2]) // 2), 
                              int((trail[i-1][1] + trail[i-1][3]) // 2))
                        pt2 = (int((trail[i][0] + trail[i][2]) // 2), 
                              int((trail[i][1] + trail[i][3]) // 2))
                        alpha = i / len(trail)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), thickness)
        
        return annotated_frame
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        if not self.stats['processing_times']:
            return {}
        
        avg_processing_time = np.mean(self.stats['processing_times'])
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        class_distribution = {}
        for cls_id, count in self.stats['class_detections'].items():
            class_name = self.get_standardized_class_name(cls_id)
            class_distribution[class_name] = count
        
        return {
            'total_frames_processed': self.stats['total_frames'],
            'average_fps': fps,
            'average_processing_time': avg_processing_time,
            'total_detections': sum(self.stats['class_detections'].values()),
            'class_distribution': class_distribution,
            'team_assignments': dict(self.stats['team_assignments']),
            'average_detections_per_frame': np.mean(self.stats['detections_per_frame']) if self.stats['detections_per_frame'] else 0
        }

def process_video_with_improved_detection(video_path, model_path, output_path):
    """Process video with improved basketball detection"""
    print(f"🏀 Starting improved basketball detection on {video_path}")
    
    # Initialize improved system
    detector = ImprovedBasketballIntelligence(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, _ = detector.process_frame(frame)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) @ {fps_current:.1f}fps")
    
    finally:
        cap.release()
        out.release()
    
    # Generate comprehensive statistics
    stats = detector.get_performance_stats()
    
    return stats

if __name__ == "__main__":
    # Test the improved system
    video_path = r"C:\Users\vish\Capstone PROJECT\Phase III\hawks vs knicks\hawks-vs-knicks.mp4"
    model_path = r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"improved_basketball_detection_{timestamp}.mp4"
    
    if Path(video_path).exists() and Path(model_path).exists():
        print("🚀 Running improved basketball detection test...")
        stats = process_video_with_improved_detection(video_path, model_path, output_path)
        
        print(f"\n✅ Improved detection completed!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Stats: {stats}")
    else:
        print("❌ Video or model path not found")
