"""
Feature Extraction Utilities for Basketball Player Analysis
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Tuple, Optional


class PlayerFeatureExtractor:
    """
    Extracts meaningful features from player tracking and pose data.
    """
    
    def __init__(self, court_width: float = 940.0, court_height: float = 500.0):
        self.court_width = court_width
        self.court_height = court_height
        
    def extract_position_features(self, x: float, y: float) -> Dict[str, float]:
        """
        Extract position-based features.
        
        Args:
            x, y: Player coordinates
            
        Returns:
            Dictionary of position features
        """
        features = {}
        
        # Normalized coordinates
        features['x_norm'] = x / self.court_width
        features['y_norm'] = y / self.court_height
        
        # Distance to court center
        center_x, center_y = self.court_width / 2, self.court_height / 2
        dist_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        features['dist_to_center'] = dist_to_center / math.sqrt(center_x**2 + center_y**2)
        
        # Court zone features
        features.update(self._get_court_zone_features(x, y))
        
        # Distance to baskets (assuming baskets at x=0 and x=court_width)
        features['dist_to_left_basket'] = x / self.court_width
        features['dist_to_right_basket'] = (self.court_width - x) / self.court_width
        
        return features
    
    def _get_court_zone_features(self, x: float, y: float) -> Dict[str, float]:
        """Get court zone one-hot encoding."""
        zones = {
            'left_court': 1.0 if x < self.court_width / 3 else 0.0,
            'center_court': 1.0 if self.court_width / 3 <= x <= 2 * self.court_width / 3 else 0.0,
            'right_court': 1.0 if x > 2 * self.court_width / 3 else 0.0,
            'top_half': 1.0 if y < self.court_height / 2 else 0.0,
            'bottom_half': 1.0 if y >= self.court_height / 2 else 0.0,
        }
        return zones
    
    def extract_velocity_features(self, 
                                curr_pos: Tuple[float, float],
                                prev_pos: Optional[Tuple[float, float]] = None,
                                time_delta: float = 1.0) -> Dict[str, float]:
        """
        Extract velocity and movement features.
        
        Args:
            curr_pos: Current (x, y) position
            prev_pos: Previous (x, y) position
            time_delta: Time between frames
            
        Returns:
            Dictionary of velocity features
        """
        features = {}
        
        if prev_pos is None:
            features.update({
                'velocity_x': 0.0,
                'velocity_y': 0.0,
                'speed': 0.0,
                'direction_angle': 0.0,
                'is_moving': 0.0
            })
            return features
        
        # Calculate velocity
        vx = (curr_pos[0] - prev_pos[0]) / time_delta
        vy = (curr_pos[1] - prev_pos[1]) / time_delta
        
        # Normalize by court dimensions
        vx_norm = vx / self.court_width
        vy_norm = vy / self.court_height
        
        # Speed and direction
        speed = math.sqrt(vx**2 + vy**2)
        speed_norm = speed / math.sqrt(self.court_width**2 + self.court_height**2)
        
        direction_angle = math.atan2(vy, vx)
        
        features.update({
            'velocity_x': vx_norm,
            'velocity_y': vy_norm,
            'speed': speed_norm,
            'direction_angle_cos': math.cos(direction_angle),
            'direction_angle_sin': math.sin(direction_angle),
            'is_moving': 1.0 if speed > 5.0 else 0.0  # threshold for "moving"
        })
        
        return features
    
    def extract_pose_features(self, pose_keypoints: Dict[str, float]) -> Dict[str, float]:
        """
        Extract features from pose keypoints (MediaPipe format).
        
        Args:
            pose_keypoints: Dictionary of pose landmarks
            
        Returns:
            Dictionary of pose features
        """
        features = {}
        
        # Body orientation
        if all(k in pose_keypoints for k in ['left_shoulder_x', 'right_shoulder_x', 
                                           'left_shoulder_y', 'right_shoulder_y']):
            shoulder_angle = math.atan2(
                pose_keypoints['left_shoulder_y'] - pose_keypoints['right_shoulder_y'],
                pose_keypoints['left_shoulder_x'] - pose_keypoints['right_shoulder_x']
            )
            features['body_orientation_cos'] = math.cos(shoulder_angle)
            features['body_orientation_sin'] = math.sin(shoulder_angle)
        else:
            features['body_orientation_cos'] = 0.0
            features['body_orientation_sin'] = 0.0
        
        # Arm positions (shooting stance detection)
        if all(k in pose_keypoints for k in ['left_wrist_y', 'right_wrist_y', 
                                           'left_shoulder_y', 'right_shoulder_y']):
            left_arm_raised = pose_keypoints['left_wrist_y'] < pose_keypoints['left_shoulder_y']
            right_arm_raised = pose_keypoints['right_wrist_y'] < pose_keypoints['right_shoulder_y']
            
            features['left_arm_raised'] = 1.0 if left_arm_raised else 0.0
            features['right_arm_raised'] = 1.0 if right_arm_raised else 0.0
            features['both_arms_raised'] = 1.0 if left_arm_raised and right_arm_raised else 0.0
        else:
            features.update({
                'left_arm_raised': 0.0,
                'right_arm_raised': 0.0,
                'both_arms_raised': 0.0
            })
        
        # Leg stance (running/jumping detection)
        if all(k in pose_keypoints for k in ['left_hip_y', 'right_hip_y', 
                                           'left_ankle_y', 'right_ankle_y']):
            avg_hip_y = (pose_keypoints['left_hip_y'] + pose_keypoints['right_hip_y']) / 2
            avg_ankle_y = (pose_keypoints['left_ankle_y'] + pose_keypoints['right_ankle_y']) / 2
            leg_bend = avg_hip_y - avg_ankle_y
            
            features['leg_bend'] = leg_bend / self.court_height
        else:
            features['leg_bend'] = 0.0
        
        return features
    
    def extract_team_context_features(self, 
                                    player_pos: Tuple[float, float],
                                    teammate_positions: List[Tuple[float, float]],
                                    opponent_positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Extract features based on team context and player relationships.
        
        Args:
            player_pos: Current player position
            teammate_positions: List of teammate positions
            opponent_positions: List of opponent positions
            
        Returns:
            Dictionary of team context features
        """
        features = {}
        
        x, y = player_pos
        
        # Distance to nearest teammate
        if teammate_positions:
            teammate_distances = [
                math.sqrt((x - tx)**2 + (y - ty)**2) 
                for tx, ty in teammate_positions
            ]
            features['dist_to_nearest_teammate'] = min(teammate_distances) / math.sqrt(self.court_width**2 + self.court_height**2)
            features['avg_dist_to_teammates'] = np.mean(teammate_distances) / math.sqrt(self.court_width**2 + self.court_height**2)
        else:
            features['dist_to_nearest_teammate'] = 1.0
            features['avg_dist_to_teammates'] = 1.0
        
        # Distance to nearest opponent
        if opponent_positions:
            opponent_distances = [
                math.sqrt((x - ox)**2 + (y - oy)**2) 
                for ox, oy in opponent_positions
            ]
            features['dist_to_nearest_opponent'] = min(opponent_distances) / math.sqrt(self.court_width**2 + self.court_height**2)
            features['avg_dist_to_opponents'] = np.mean(opponent_distances) / math.sqrt(self.court_width**2 + self.court_height**2)
        else:
            features['dist_to_nearest_opponent'] = 1.0
            features['avg_dist_to_opponents'] = 1.0
        
        # Team formation features
        if len(teammate_positions) >= 2:
            features.update(self._calculate_formation_features(player_pos, teammate_positions))
        
        return features
    
    def _calculate_formation_features(self, 
                                    player_pos: Tuple[float, float],
                                    teammate_positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate team formation-related features."""
        features = {}
        
        all_positions = [player_pos] + teammate_positions
        
        # Team centroid
        centroid_x = np.mean([pos[0] for pos in all_positions])
        centroid_y = np.mean([pos[1] for pos in all_positions])
        
        # Distance to team centroid
        dist_to_centroid = math.sqrt((player_pos[0] - centroid_x)**2 + (player_pos[1] - centroid_y)**2)
        features['dist_to_team_centroid'] = dist_to_centroid / math.sqrt(self.court_width**2 + self.court_height**2)
        
        # Team spread (how spread out the team is)
        distances_from_centroid = [
            math.sqrt((pos[0] - centroid_x)**2 + (pos[1] - centroid_y)**2)
            for pos in all_positions
        ]
        features['team_spread'] = np.std(distances_from_centroid) / math.sqrt(self.court_width**2 + self.court_height**2)
        
        # Player's relative position in team formation
        x_positions = [pos[0] for pos in all_positions]
        y_positions = [pos[1] for pos in all_positions]
        
        if len(set(x_positions)) > 1:
            features['x_rank_in_team'] = (sorted(x_positions).index(player_pos[0]) + 1) / len(x_positions)
        else:
            features['x_rank_in_team'] = 0.5
        
        if len(set(y_positions)) > 1:
            features['y_rank_in_team'] = (sorted(y_positions).index(player_pos[1]) + 1) / len(y_positions)
        else:
            features['y_rank_in_team'] = 0.5
        
        return features
    
    def combine_all_features(self, 
                           player_data: Dict,
                           prev_player_data: Optional[Dict] = None,
                           pose_data: Optional[Dict] = None,
                           team_context: Optional[Dict] = None) -> List[float]:
        """
        Combine all feature types into a single feature vector.
        
        Args:
            player_data: Current player data with x, y coordinates
            prev_player_data: Previous frame player data
            pose_data: Pose keypoints data
            team_context: Team context information
            
        Returns:
            Combined feature vector
        """
        all_features = {}
        
        # Position features
        pos_features = self.extract_position_features(player_data['x'], player_data['y'])
        all_features.update(pos_features)
        
        # Velocity features
        curr_pos = (player_data['x'], player_data['y'])
        prev_pos = None
        if prev_player_data:
            prev_pos = (prev_player_data['x'], prev_player_data['y'])
        
        vel_features = self.extract_velocity_features(curr_pos, prev_pos)
        all_features.update(vel_features)
        
        # Pose features
        if pose_data:
            pose_features = self.extract_pose_features(pose_data)
            all_features.update(pose_features)
        
        # Team context features
        if team_context:
            context_features = self.extract_team_context_features(
                curr_pos, 
                team_context.get('teammate_positions', []),
                team_context.get('opponent_positions', [])
            )
            all_features.update(context_features)
        
        # Convert to ordered list
        feature_vector = list(all_features.values())
        
        return feature_vector


def create_feature_names() -> List[str]:
    """Return list of feature names in order."""
    names = [
        # Position features
        'x_norm', 'y_norm', 'dist_to_center',
        'left_court', 'center_court', 'right_court', 'top_half', 'bottom_half',
        'dist_to_left_basket', 'dist_to_right_basket',
        
        # Velocity features
        'velocity_x', 'velocity_y', 'speed', 'direction_angle_cos', 'direction_angle_sin', 'is_moving',
        
        # Pose features (optional)
        'body_orientation_cos', 'body_orientation_sin',
        'left_arm_raised', 'right_arm_raised', 'both_arms_raised', 'leg_bend',
        
        # Team context features (optional)
        'dist_to_nearest_teammate', 'avg_dist_to_teammates',
        'dist_to_nearest_opponent', 'avg_dist_to_opponents',
        'dist_to_team_centroid', 'team_spread', 'x_rank_in_team', 'y_rank_in_team'
    ]
    
    return names


if __name__ == "__main__":
    print("Testing Player Feature Extractor...")
    
    extractor = PlayerFeatureExtractor()
    
    # Test position features
    pos_features = extractor.extract_position_features(200, 150)
    print(f"Position features: {len(pos_features)} features")
    
    # Test velocity features
    vel_features = extractor.extract_velocity_features((200, 150), (190, 140))
    print(f"Velocity features: {len(vel_features)} features")
    
    # Test combined features
    player_data = {'x': 200, 'y': 150}
    prev_data = {'x': 190, 'y': 140}
    
    combined = extractor.combine_all_features(player_data, prev_data)
    print(f"Combined features: {len(combined)} total features")
    
    feature_names = create_feature_names()
    print(f"Feature names: {len(feature_names)} names defined")
    
    print("Feature extraction test completed!")
