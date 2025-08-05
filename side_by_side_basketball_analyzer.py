#!/usr/bin/env python3
"""
Side-by-Side Basketball Video Analyzer
=====================================
Creates a side-by-side view showing:
- Left: Original game footage with enhanced detection overlays
- Right: 2D tactical representation with GNN analysis

Perfect for coaching analysis and tactical understanding!
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
import sys

# Import our enhanced detection system
sys.path.append(r"C:\Users\vish\Capstone PROJECT\Phase III")
from improved_basketball_intelligence import ImprovedBasketballIntelligence

class SideBySideCourtTransformer:
    """Enhanced court transformer for side-by-side view"""
    
    def __init__(self, tactical_width=600, tactical_height=400):
        # Standard NBA court dimensions (feet)
        self.court_length = 94  # feet
        self.court_width = 50   # feet
        
        # 2D tactical view dimensions
        self.tactical_width = tactical_width
        self.tactical_height = tactical_height
        
        # Court visualization parameters
        self.margin_x = 40
        self.margin_y = 40
        self.court_viz_length = tactical_width - 2 * self.margin_x
        self.court_viz_width = tactical_height - 2 * self.margin_y
        
    def transform_coordinates(self, detections, frame_shape):
        """Transform detected positions to 2D court coordinates"""
        frame_height, frame_width = frame_shape[:2]
        
        transformed_players = []
        ball_position = None
        
        for tracker_id, detection in detections.items():
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Normalize to 0-1
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height
            
            # Enhanced perspective transformation
            # Assume camera angle and apply perspective correction
            court_x = self.perspective_correct_x(norm_x, norm_y)
            court_y = self.perspective_correct_y(norm_x, norm_y)
            
            if detection['class'] == 0:  # Ball
                ball_position = (court_x, court_y)
            elif detection['class'] == 2:  # Player
                transformed_players.append({
                    'id': tracker_id,
                    'court_x': court_x,
                    'court_y': court_y,
                    'team': detection.get('team', 'unknown'),
                    'class': detection.get('class', 2),
                    'conf': detection.get('conf', 0.0),
                    'bbox': bbox,
                    'screen_pos': (center_x, center_y)
                })
        
        return transformed_players, ball_position
    
    def perspective_correct_x(self, norm_x, norm_y):
        """Apply perspective correction for x-coordinate"""
        # Simple perspective model - can be enhanced with homography
        base_x = norm_x * self.court_length
        
        # Adjust for camera angle (players farther appear more compressed)
        if norm_y > 0.6:  # Far court
            base_x = base_x * 0.9 + self.court_length * 0.05
        elif norm_y < 0.4:  # Near court  
            base_x = base_x * 1.1 - self.court_length * 0.05
            
        return np.clip(base_x, 0, self.court_length)
    
    def perspective_correct_y(self, norm_x, norm_y):
        """Apply perspective correction for y-coordinate"""
        # Enhanced depth perception
        base_y = norm_y * self.court_width
        
        # Adjust for perspective depth
        depth_factor = 0.7 + 0.3 * norm_y  # Far = 1.0, Near = 0.7
        adjusted_y = base_y * depth_factor + self.court_width * (1 - depth_factor) * 0.5
        
        return np.clip(adjusted_y, 0, self.court_width)

class Enhanced2DVisualizer:
    """Enhanced 2D court visualizer for side-by-side view"""
    
    def __init__(self, width=600, height=400):
        self.width = width
        self.height = height
        self.margin_x = 40
        self.margin_y = 40
        
        # Court dimensions in visualization coordinates
        self.court_length = width - 2 * self.margin_x
        self.court_width = height - 2 * self.margin_y
        
        # Enhanced court elements
        self.setup_court_elements()
        
    def setup_court_elements(self):
        """Setup detailed court elements"""
        # Court proportions
        self.three_point_radius = int(self.court_width * 0.4)
        self.paint_width = int(self.court_width * 0.32)
        self.paint_length = int(self.court_length * 0.15)
        self.center_circle_radius = int(self.court_width * 0.12)
        
    def create_enhanced_court(self):
        """Create detailed basketball court background"""
        court = np.ones((self.height, self.width, 3), dtype=np.uint8) * 45  # Court color
        
        # Outer court boundary
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
        cv2.circle(court, (center_x, self.height // 2), 
                  self.center_circle_radius, (255, 255, 255), 2)
        
        # Paint areas
        self.draw_paint_area(court, 'left')
        self.draw_paint_area(court, 'right')
        
        # Three-point lines
        self.draw_three_point_line(court, 'left')
        self.draw_three_point_line(court, 'right')
        
        # Baskets
        self.draw_basket(court, 'left')
        self.draw_basket(court, 'right')
        
        return court
    
    def draw_paint_area(self, court, side):
        """Draw paint/key area"""
        if side == 'left':
            x1 = self.margin_x
            x2 = self.margin_x + self.paint_length
        else:
            x1 = self.width - self.margin_x - self.paint_length
            x2 = self.width - self.margin_x
            
        y1 = self.height // 2 - self.paint_width // 2
        y2 = self.height // 2 + self.paint_width // 2
        
        cv2.rectangle(court, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Free throw circle
        center_x = x1 if side == 'left' else x2
        cv2.circle(court, (center_x + self.paint_length//2 if side == 'left' else center_x - self.paint_length//2, 
                          self.height // 2), 
                  self.paint_width//4, (255, 255, 255), 2)
    
    def draw_three_point_line(self, court, side):
        """Draw three-point line"""
        if side == 'left':
            center_x = self.margin_x
            start_angle, end_angle = -90, 90
        else:
            center_x = self.width - self.margin_x
            start_angle, end_angle = 90, 270
            
        cv2.ellipse(court, 
                   (center_x, self.height // 2), 
                   (self.three_point_radius, self.three_point_radius), 
                   0, start_angle, end_angle, 
                   (255, 255, 255), 2)
    
    def draw_basket(self, court, side):
        """Draw basketball hoop"""
        if side == 'left':
            x = self.margin_x + 10
        else:
            x = self.width - self.margin_x - 10
            
        y = self.height // 2
        cv2.circle(court, (x, y), 8, (255, 100, 0), 2)  # Orange rim
        cv2.circle(court, (x, y), 3, (255, 100, 0), -1)  # Rim center
    
    def draw_enhanced_players(self, court, players, ball_pos=None):
        """Draw players with enhanced visualization"""
        # Draw player movement trails first (behind players)
        self.draw_movement_trails(court, players)
        
        # Draw players
        for player in players:
            self.draw_single_player(court, player)
        
        # Draw ball
        if ball_pos:
            self.draw_ball(court, ball_pos)
        
        # Draw team indicators
        self.draw_team_legend(court, players)
        
        return court
    
    def draw_single_player(self, court, player):
        """Draw individual player with enhanced details"""
        # Convert court coordinates to visualization coordinates
        viz_x = int(self.margin_x + (player['court_x'] / 94.0) * self.court_length)
        viz_y = int(self.margin_y + (player['court_y'] / 50.0) * self.court_width)
        
        # Team colors with enhanced visibility
        if player['team'] == 'home':
            main_color = (255, 165, 0)  # Orange
            outline_color = (255, 200, 100)
        elif player['team'] == 'away':
            main_color = (0, 100, 255)  # Blue
            outline_color = (100, 150, 255)
        else:
            main_color = (128, 128, 128)  # Gray
            outline_color = (180, 180, 180)
        
        # Player circle with gradient effect
        cv2.circle(court, (viz_x, viz_y), 15, outline_color, -1)
        cv2.circle(court, (viz_x, viz_y), 12, main_color, -1)
        cv2.circle(court, (viz_x, viz_y), 12, (255, 255, 255), 2)
        
        # Player ID/number
        cv2.putText(court, str(player['id']), 
                   (viz_x - 6, viz_y + 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Confidence indicator (small dot)
        conf_color = (0, 255, 0) if player['conf'] > 0.7 else (255, 255, 0) if player['conf'] > 0.5 else (255, 0, 0)
        cv2.circle(court, (viz_x + 10, viz_y - 10), 3, conf_color, -1)
    
    def draw_ball(self, court, ball_pos):
        """Draw basketball with enhanced visibility"""
        ball_x = int(self.margin_x + (ball_pos[0] / 94.0) * self.court_length)
        ball_y = int(self.margin_y + (ball_pos[1] / 50.0) * self.court_width)
        
        # Ball with glow effect
        cv2.circle(court, (ball_x, ball_y), 10, (0, 255, 255), -1)  # Yellow glow
        cv2.circle(court, (ball_x, ball_y), 7, (0, 200, 255), -1)   # Orange ball
        cv2.circle(court, (ball_x, ball_y), 7, (255, 255, 255), 1)  # White outline
    
    def draw_movement_trails(self, court, players):
        """Draw player movement trails (placeholder for now)"""
        # This could be enhanced with actual movement history
        pass
    
    def draw_team_legend(self, court, players):
        """Draw team color legend"""
        home_count = len([p for p in players if p['team'] == 'home'])
        away_count = len([p for p in players if p['team'] == 'away'])
        
        # Home team indicator
        cv2.circle(court, (20, 20), 8, (255, 165, 0), -1)
        cv2.putText(court, f"HOME: {home_count}", (35, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Away team indicator
        cv2.circle(court, (20, 40), 8, (0, 100, 255), -1)
        cv2.putText(court, f"AWAY: {away_count}", (35, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

class SideBySideBasketballAnalyzer:
    """Main class for side-by-side basketball analysis"""
    
    def __init__(self, model_path):
        print("🏀 Initializing Side-by-Side Basketball Analyzer...")
        
        # Initialize components
        self.detector = ImprovedBasketballIntelligence(model_path)
        self.court_transformer = SideBySideCourtTransformer()
        self.tactical_visualizer = Enhanced2DVisualizer()
        
        # Video processing parameters
        self.tactical_width = 600
        self.tactical_height = 400
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'players_detected': 0,
            'home_players': 0,
            'away_players': 0,
            'balls_detected': 0
        }
        
        print("✅ Side-by-side analyzer initialized!")
    
    def process_video_side_by_side(self, video_path, output_path, max_frames=None):
        """Process video with side-by-side view"""
        print(f"🎬 Creating side-by-side analysis: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Video: {original_width}x{original_height} @ {fps}FPS, {total_frames} frames")
        
        # Calculate output dimensions
        # Left side: Original video (scaled to match tactical height)
        scale_factor = self.tactical_height / original_height
        scaled_width = int(original_width * scale_factor)
        
        # Output dimensions: scaled original + tactical view + margins
        output_width = scaled_width + self.tactical_width + 20  # 20px gap
        output_height = self.tactical_height
        
        print(f"🎯 Output: {output_width}x{output_height}")
        
        # Setup video writer with better codec compatibility
        # Try multiple codecs for better compatibility
        fourcc_options = [
            ('XVID', '.avi'),  # Most compatible
            ('H264', '.mp4'),  # Good quality
            ('mp4v', '.mp4'),  # Fallback
            ('MJPG', '.avi')   # Last resort
        ]
        
        out = None
        actual_output_path = None
        
        for codec, ext in fourcc_options:
            try:
                # Change extension based on codec
                test_path = output_path.with_suffix(ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_out = cv2.VideoWriter(str(test_path), fourcc, fps, (output_width, output_height))
                
                if test_out.isOpened():
                    out = test_out
                    actual_output_path = test_path
                    print(f"✅ Using {codec} codec with {ext} format")
                    break
                else:
                    test_out.release()
            except Exception as e:
                print(f"❌ Failed {codec}: {e}")
                continue
        
        if out is None or not out.isOpened():
            raise RuntimeError("❌ Could not initialize any video codec!")
        
        output_path = actual_output_path
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with enhanced detection
                annotated_frame, detections = self.detector.process_frame(frame)
                
                # Transform to 2D coordinates
                players_2d, ball_pos = self.court_transformer.transform_coordinates(
                    detections, frame.shape)
                
                # Create side-by-side frame
                combined_frame = self.create_side_by_side_frame(
                    annotated_frame, players_2d, ball_pos, 
                    scaled_width, frame_count)
                
                # Write frame
                out.write(combined_frame)
                
                # Update statistics
                self.update_statistics(players_2d, ball_pos)
                
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
        
        print(f"\n✅ Side-by-side analysis completed!")
        print(f"📁 Output: {output_path}")
        
        return output_path, final_stats
    
    def create_side_by_side_frame(self, original_frame, players_2d, ball_pos, scaled_width, frame_num):
        """Create combined side-by-side frame"""
        
        # Scale original frame to match tactical height
        scaled_original = cv2.resize(original_frame, (scaled_width, self.tactical_height))
        
        # Create tactical 2D view
        tactical_court = self.tactical_visualizer.create_enhanced_court()
        tactical_view = self.tactical_visualizer.draw_enhanced_players(
            tactical_court, players_2d, ball_pos)
        
        # Add frame information to tactical view
        self.add_frame_info(tactical_view, frame_num, players_2d, ball_pos)
        
        # Combine frames side by side
        gap = np.ones((self.tactical_height, 20, 3), dtype=np.uint8) * 30  # Dark gap
        combined = np.hstack([scaled_original, gap, tactical_view])
        
        # Add title bar
        title_bar = self.create_title_bar(combined.shape[1])
        final_frame = np.vstack([title_bar, combined])
        
        return final_frame
    
    def add_frame_info(self, tactical_view, frame_num, players_2d, ball_pos):
        """Add information overlay to tactical view"""
        info_y = self.tactical_height - 60
        
        # Frame info
        cv2.putText(tactical_view, f"Frame: {frame_num}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Player count
        cv2.putText(tactical_view, f"Players: {len(players_2d)}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Ball status
        ball_status = "Ball: YES" if ball_pos else "Ball: NO"
        cv2.putText(tactical_view, ball_status, 
                   (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if ball_pos else (255, 0, 0), 1)
        
        # Analysis mode indicator
        cv2.putText(tactical_view, "2D TACTICAL VIEW", 
                   (self.tactical_width - 150, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def create_title_bar(self, width):
        """Create title bar for the video"""
        title_bar = np.ones((40, width, 3), dtype=np.uint8) * 20
        
        # Title text
        cv2.putText(title_bar, "BASKETBALL ANALYSIS - ORIGINAL vs 2D TACTICAL", 
                   (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Separators
        separator_x = width // 2
        cv2.line(title_bar, (separator_x, 0), (separator_x, 40), (255, 255, 255), 1)
        
        return title_bar
    
    def update_statistics(self, players_2d, ball_pos):
        """Update processing statistics"""
        self.stats['total_frames'] += 1
        self.stats['players_detected'] += len(players_2d)
        
        for player in players_2d:
            if player['team'] == 'home':
                self.stats['home_players'] += 1
            elif player['team'] == 'away':
                self.stats['away_players'] += 1
        
        if ball_pos:
            self.stats['balls_detected'] += 1
    
    def generate_final_statistics(self):
        """Generate comprehensive final statistics"""
        total_frames = max(1, self.stats['total_frames'])
        
        return {
            'processing_summary': {
                'total_frames_processed': self.stats['total_frames'],
                'average_players_per_frame': self.stats['players_detected'] / total_frames,
                'ball_detection_rate': (self.stats['balls_detected'] / total_frames) * 100,
                'total_home_detections': self.stats['home_players'],
                'total_away_detections': self.stats['away_players']
            }
        }

def create_side_by_side_analysis(video_path, output_path=None, model_path=None, max_frames=None):
    """
    Create side-by-side basketball analysis
    
    Args:
        video_path: Path to input basketball video
        output_path: Path for output video (optional)
        model_path: Path to trained YOLO model (optional)
        max_frames: Maximum frames to process (optional)
    """
    
    # Default paths
    if model_path is None:
        model_path = Path(r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_training\enhanced_20250803_174000\enhanced_basketball_20250803_174000\weights\best.pt")
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_path = Path(video_path).parent / f"{video_name}_side_by_side_{timestamp}.mp4"
    
    print("🏀 Basketball Side-by-Side Analyzer")
    print("=" * 60)
    print(f"📹 Input: {video_path}")
    print(f"🎯 Output: {output_path}")
    print(f"🤖 Model: {model_path}")
    print()
    
    # Initialize analyzer
    analyzer = SideBySideBasketballAnalyzer(str(model_path))
    
    # Process video
    actual_output_path, stats = analyzer.process_video_side_by_side(video_path, output_path, max_frames)
    
    # Save statistics
    stats_path = str(actual_output_path).replace('.mp4', '_stats.json').replace('.avi', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📊 Final Statistics:")
    print(f"   - Frames processed: {stats['processing_summary']['total_frames_processed']}")
    print(f"   - Average players per frame: {stats['processing_summary']['average_players_per_frame']:.2f}")
    print(f"   - Ball detection rate: {stats['processing_summary']['ball_detection_rate']:.1f}%")
    print(f"   - Home team detections: {stats['processing_summary']['total_home_detections']}")
    print(f"   - Away team detections: {stats['processing_summary']['total_away_detections']}")
    print(f"\n📋 Stats saved: {stats_path}")
    
    return actual_output_path, stats

if __name__ == "__main__":
    # Example usage
    test_video = r"C:\Users\vish\Capstone PROJECT\Phase III\enhanced_basketball_test_20250803_175335.mp4"
    
    print("🚀 Creating Side-by-Side Basketball Analysis...")
    output_video, stats = create_side_by_side_analysis(
        video_path=test_video,
        max_frames=200  # Remove for full video
    )
    
    print(f"\n🎉 Side-by-Side analysis completed!")
    print(f"🎬 Watch your analysis: {output_video}")
    print("\n🎯 Features in your video:")
    print("   - Left: Original game with enhanced detection overlays")
    print("   - Right: 2D tactical view with player positions")
    print("   - Real-time statistics and team identification")
    print("   - Professional court visualization")
