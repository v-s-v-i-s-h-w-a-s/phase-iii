"""
Team Classification Module
Classifies basketball players into teams based on jersey colors
Handles occlusion and provides clear team visualization
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import colorsys

class TeamClassifier:
    def __init__(self):
        """Initialize team classifier"""
        self.team_colors = {}  # Will store dominant colors for each team
        self.team_assignments = {}  # Track player assignments over time
        self.frame_history = []  # Store recent classifications for stability
        self.history_size = 10  # Number of frames to consider for stable classification
        
        # Team visualization colors (BGR format)
        self.team_viz_colors = {
            'team_1': (0, 0, 255),      # Red
            'team_2': (255, 0, 0),      # Blue  
            'team_3': (0, 255, 0),      # Green
            'referee': (0, 255, 255),   # Yellow
            'unknown': (128, 128, 128)  # Gray
        }
        
        print("🏀 Team Classifier initialized")
        
    def extract_jersey_region(self, frame, bbox):
        """
        Extract jersey region from player bounding box
        Focus on upper torso area where jersey is most visible
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate jersey region (upper 60% of bounding box, middle 80% width)
        height = y2 - y1
        width = x2 - x1
        
        # Jersey region coordinates
        jersey_y1 = y1 + int(height * 0.1)  # Start slightly below top
        jersey_y2 = y1 + int(height * 0.7)  # Upper 60% of height
        jersey_x1 = x1 + int(width * 0.1)   # Middle 80% of width
        jersey_x2 = x2 - int(width * 0.1)
        
        # Ensure coordinates are within frame bounds
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)
        
        # Extract jersey region
        if jersey_y2 > jersey_y1 and jersey_x2 > jersey_x1:
            jersey_region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
            return jersey_region
        else:
            return None
    
    def get_dominant_colors(self, image, k=3):
        """
        Get dominant colors from image using K-means clustering
        Returns colors sorted by dominance
        """
        if image is None or image.size == 0:
            return []
        
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (likely shadows/highlights)
        brightness = np.mean(pixels, axis=1)
        valid_pixels = pixels[(brightness > 30) & (brightness < 225)]
        
        if len(valid_pixels) < 10:  # Not enough valid pixels
            return []
        
        # Apply K-means clustering
        try:
            kmeans = KMeans(n_clusters=min(k, len(valid_pixels)), random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            # Get colors and their frequencies
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Count frequency of each color
            color_counts = Counter(labels)
            
            # Sort colors by frequency
            sorted_colors = []
            for label in sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True):
                color = colors[label].astype(int)
                frequency = color_counts[label]
                sorted_colors.append((color, frequency))
            
            return sorted_colors
        except Exception as e:
            print(f"Warning: K-means clustering failed: {e}")
            return []
    
    def color_distance(self, color1, color2):
        """
        Calculate perceptual distance between two colors
        Uses Delta E CIE76 formula for better color matching
        """
        # Convert BGR to LAB color space for better perceptual distance
        try:
            # Convert to RGB first, then to LAB
            rgb1 = color1[::-1]  # BGR to RGB
            rgb2 = color2[::-1]  # BGR to RGB
            
            # Simple RGB distance for now (can be improved with LAB conversion)
            return np.sqrt(np.sum((rgb1 - rgb2) ** 2))
        except Exception as e:
            print(f"Warning: Color distance calculation failed: {e}")
            return float('inf')
    
    def classify_player_team(self, frame, bbox, player_id=None):
        """
        Classify a player into a team based on jersey color
        """
        # Extract jersey region
        jersey_region = self.extract_jersey_region(frame, bbox)
        if jersey_region is None:
            return 'unknown'
        
        # Get dominant colors from jersey
        dominant_colors = self.get_dominant_colors(jersey_region, k=3)
        if not dominant_colors:
            return 'unknown'
        
        # If we don't have team colors established yet, store this as potential team color
        if len(self.team_colors) < 2:
            return self._establish_team_colors(dominant_colors)
        
        # Match against established team colors
        best_team = 'unknown'
        min_distance = float('inf')
        
        for team_name, team_color in self.team_colors.items():
            for color, frequency in dominant_colors:
                distance = self.color_distance(color, team_color)
                # Weight by frequency of the color in the jersey
                weighted_distance = distance / (frequency + 1)
                
                if weighted_distance < min_distance:
                    min_distance = weighted_distance
                    best_team = team_name
        
        # Apply stability check using frame history
        if player_id is not None:
            return self._apply_temporal_stability(player_id, best_team)
        
        return best_team
    
    def _establish_team_colors(self, dominant_colors):
        """
        Establish team colors when we don't have them yet
        """
        if not dominant_colors:
            return 'unknown'
        
        primary_color = dominant_colors[0][0]  # Most dominant color
        
        # Check if this color is significantly different from existing team colors
        min_distance = float('inf')
        for existing_color in self.team_colors.values():
            distance = self.color_distance(primary_color, existing_color)
            min_distance = min(min_distance, distance)
        
        # If color is different enough, establish as new team
        if min_distance > 80 or len(self.team_colors) == 0:  # Threshold for color difference
            team_name = f'team_{len(self.team_colors) + 1}'
            self.team_colors[team_name] = primary_color
            print(f"🎨 Established {team_name} with color: {primary_color}")
            return team_name
        
        # Otherwise, assign to closest existing team
        closest_team = min(self.team_colors.keys(), 
                          key=lambda t: self.color_distance(primary_color, self.team_colors[t]))
        return closest_team
    
    def _apply_temporal_stability(self, player_id, current_classification):
        """
        Apply temporal stability to reduce classification noise
        """
        # Initialize player history if needed
        if player_id not in self.team_assignments:
            self.team_assignments[player_id] = []
        
        # Add current classification
        self.team_assignments[player_id].append(current_classification)
        
        # Keep only recent history
        if len(self.team_assignments[player_id]) > self.history_size:
            self.team_assignments[player_id] = self.team_assignments[player_id][-self.history_size:]
        
        # Return most common classification in recent history
        recent_classifications = self.team_assignments[player_id]
        if len(recent_classifications) >= 3:  # Need at least 3 frames for stability
            most_common = Counter(recent_classifications).most_common(1)[0][0]
            return most_common
        
        return current_classification
    
    def classify_frame_players(self, frame, player_detections):
        """
        Classify all players in a frame into teams
        """
        classified_players = []
        
        for i, detection in enumerate(player_detections):
            if detection['class'] == 'player':
                # Use detection index as temporary player ID
                player_id = f"p_{detection['bbox'][0]}_{detection['bbox'][1]}"
                team = self.classify_player_team(frame, detection['bbox'], player_id)
                
                # Update detection with team information
                detection['team'] = team
                detection['player_id'] = player_id
                
            classified_players.append(detection)
        
        return classified_players
    
    def get_team_color(self, team_name):
        """Get visualization color for a team"""
        return self.team_viz_colors.get(team_name, self.team_viz_colors['unknown'])
    
    def draw_team_detections(self, frame, detections):
        """
        Draw detection boxes with team-based colors
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Determine color based on class and team
            if class_name == 'player' and 'team' in detection:
                team = detection['team']
                color = self.get_team_color(team)
                label = f"{team}: {confidence:.2f}"
            elif class_name == 'referee':
                color = self.get_team_color('referee')
                label = f"referee: {confidence:.2f}"
            else:
                # Default colors for ball, hoop, etc.
                color_map = {
                    'ball': (0, 165, 255),      # Orange
                    'hoop': (128, 0, 128)       # Purple
                }
                color = color_map.get(class_name, (255, 255, 255))
                label = f"{class_name}: {confidence:.2f}"
            
            # Draw bounding box with thicker line for better visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw team legend
        self._draw_team_legend(frame)
        
        return frame
    
    def _draw_team_legend(self, frame):
        """Draw team color legend on frame"""
        if not self.team_colors:
            return
        
        y_offset = 50
        for i, (team_name, _) in enumerate(self.team_colors.items()):
            color = self.get_team_color(team_name)
            y_pos = y_offset + (i * 30)
            
            # Draw color box
            cv2.rectangle(frame, (10, y_pos - 10), (30, y_pos + 10), color, -1)
            cv2.rectangle(frame, (10, y_pos - 10), (30, y_pos + 10), (255, 255, 255), 1)
            
            # Draw team name
            cv2.putText(frame, team_name.upper(), (35, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def get_team_statistics(self, detections):
        """Get statistics about team classifications"""
        if not detections:
            return {}
        
        team_counts = defaultdict(int)
        total_players = 0
        
        for detection in detections:
            if detection['class'] == 'player':
                total_players += 1
                team = detection.get('team', 'unknown')
                team_counts[team] += 1
        
        stats = {
            'total_players': total_players,
            'team_counts': dict(team_counts),
            'team_colors': self.team_colors.copy()
        }
        
        return stats
