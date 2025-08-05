"""
Pose Loader: Load and process MediaPipe/OpenPose keypoint data
"""

import pandas as pd
import numpy as np
import json
import cv2
from typing import List, Dict, Tuple, Optional, Union
import os
from pathlib import Path


class PoseDataLoader:
    """
    Loader for pose estimation data from MediaPipe or OpenPose.
    Processes keypoint data for basketball player analysis.
    """
    
    def __init__(self, pose_format: str = "mediapipe"):
        """
        Args:
            pose_format: Format of pose data ("mediapipe" or "openpose")
        """
        self.pose_format = pose_format.lower()
        
        # MediaPipe pose landmarks (33 points)
        self.mediapipe_landmarks = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # OpenPose landmarks (25 points)
        self.openpose_landmarks = [
            'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist', 'mid_hip',
            'right_hip', 'right_knee', 'right_ankle', 'left_hip',
            'left_knee', 'left_ankle', 'right_eye', 'left_eye',
            'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe',
            'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel'
        ]
    
    def load_mediapipe_json(self, json_file_path: str) -> pd.DataFrame:
        """
        Load MediaPipe pose data from JSON file.
        
        Expected format:
        {
            "frames": [
                {
                    "frame_id": 1,
                    "detections": [
                        {
                            "person_id": 1,
                            "landmarks": [
                                {"x": 0.5, "y": 0.6, "z": 0.1, "visibility": 0.9},
                                ...
                            ]
                        }
                    ]
                }
            ]
        }
        
        Args:
            json_file_path: Path to MediaPipe JSON file
            
        Returns:
            DataFrame with pose data
        """
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            pose_data = []
            
            for frame_data in data.get('frames', []):
                frame_id = frame_data['frame_id']
                
                for detection in frame_data.get('detections', []):
                    person_id = detection['person_id']
                    landmarks = detection['landmarks']
                    
                    # Convert landmarks to flat dictionary
                    pose_dict = {
                        'frame_id': frame_id,
                        'player_id': person_id
                    }
                    
                    for i, landmark in enumerate(landmarks):
                        if i < len(self.mediapipe_landmarks):
                            name = self.mediapipe_landmarks[i]
                            pose_dict[f'{name}_x'] = landmark['x']
                            pose_dict[f'{name}_y'] = landmark['y']
                            pose_dict[f'{name}_z'] = landmark.get('z', 0.0)
                            pose_dict[f'{name}_visibility'] = landmark.get('visibility', 1.0)
                    
                    pose_data.append(pose_dict)
            
            return pd.DataFrame(pose_data)
            
        except Exception as e:
            print(f"Error loading MediaPipe JSON: {e}")
            return pd.DataFrame()
    
    def load_openpose_json(self, json_file_path: str) -> pd.DataFrame:
        """
        Load OpenPose data from JSON file.
        
        Expected format:
        {
            "people": [
                {
                    "person_id": 1,
                    "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...]
                }
            ]
        }
        
        Args:
            json_file_path: Path to OpenPose JSON file
            
        Returns:
            DataFrame with pose data
        """
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            pose_data = []
            
            # Extract frame ID from filename
            frame_id = self._extract_frame_id(Path(json_file_path).stem)
            
            for person in data.get('people', []):
                person_id = person.get('person_id', 0)
                keypoints = person.get('pose_keypoints_2d', [])
                
                # OpenPose format: [x1, y1, c1, x2, y2, c2, ...]
                pose_dict = {
                    'frame_id': frame_id,
                    'player_id': person_id
                }
                
                for i in range(0, len(keypoints), 3):
                    if i // 3 < len(self.openpose_landmarks):
                        name = self.openpose_landmarks[i // 3]
                        pose_dict[f'{name}_x'] = keypoints[i] if i < len(keypoints) else 0.0
                        pose_dict[f'{name}_y'] = keypoints[i + 1] if i + 1 < len(keypoints) else 0.0
                        pose_dict[f'{name}_confidence'] = keypoints[i + 2] if i + 2 < len(keypoints) else 0.0
                
                pose_data.append(pose_dict)
            
            return pd.DataFrame(pose_data)
            
        except Exception as e:
            print(f"Error loading OpenPose JSON: {e}")
            return pd.DataFrame()
    
    def load_pose_folder(self, folder_path: str) -> pd.DataFrame:
        """
        Load pose data from a folder of JSON files.
        
        Args:
            folder_path: Path to folder containing pose JSON files
            
        Returns:
            Combined DataFrame with all pose data
        """
        
        all_pose_data = []
        
        folder = Path(folder_path)
        json_files = list(folder.glob("*.json"))
        
        for json_file in json_files:
            if self.pose_format == "mediapipe":
                frame_data = self.load_mediapipe_json(str(json_file))
            else:  # openpose
                frame_data = self.load_openpose_json(str(json_file))
            
            if not frame_data.empty:
                all_pose_data.append(frame_data)
        
        if all_pose_data:
            return pd.concat(all_pose_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _extract_frame_id(self, filename: str) -> int:
        """Extract frame ID from filename."""
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0
    
    def extract_key_points(self, pose_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract key pose points for basketball analysis.
        
        Args:
            pose_df: Full pose DataFrame
            
        Returns:
            DataFrame with key points only
        """
        
        if pose_df.empty:
            return pose_df
        
        # Define key points for basketball analysis
        if self.pose_format == "mediapipe":
            key_points = [
                'nose', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle'
            ]
        else:  # openpose
            key_points = [
                'nose', 'neck', 'left_shoulder', 'right_shoulder',
                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle'
            ]
        
        # Select key point columns
        base_cols = ['frame_id', 'player_id']
        key_cols = base_cols.copy()
        
        for point in key_points:
            for suffix in ['_x', '_y']:
                col_name = f'{point}{suffix}'
                if col_name in pose_df.columns:
                    key_cols.append(col_name)
        
        return pose_df[key_cols].copy()
    
    def calculate_pose_features(self, pose_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived pose features for basketball analysis.
        
        Args:
            pose_df: Pose DataFrame
            
        Returns:
            DataFrame with calculated features
        """
        
        features_list = []
        
        for _, row in pose_df.iterrows():
            features = {
                'frame_id': row['frame_id'],
                'player_id': row['player_id']
            }
            
            # Body orientation (shoulder angle)
            if all(col in row and pd.notna(row[col]) for col in 
                   ['left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y']):
                
                shoulder_vector = np.array([
                    row['right_shoulder_x'] - row['left_shoulder_x'],
                    row['right_shoulder_y'] - row['left_shoulder_y']
                ])
                shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
                features['shoulder_angle'] = shoulder_angle
                features['body_orientation_cos'] = np.cos(shoulder_angle)
                features['body_orientation_sin'] = np.sin(shoulder_angle)
            else:
                features.update({
                    'shoulder_angle': 0.0,
                    'body_orientation_cos': 1.0,
                    'body_orientation_sin': 0.0
                })
            
            # Arm positions (for shooting detection)
            left_arm_raised = self._is_arm_raised(row, 'left')
            right_arm_raised = self._is_arm_raised(row, 'right')
            
            features.update({
                'left_arm_raised': 1.0 if left_arm_raised else 0.0,
                'right_arm_raised': 1.0 if right_arm_raised else 0.0,
                'both_arms_raised': 1.0 if left_arm_raised and right_arm_raised else 0.0
            })
            
            # Leg stance (crouch detection)
            features['is_crouching'] = 1.0 if self._is_crouching(row) else 0.0
            
            # Movement readiness (based on pose stability)
            features['pose_stability'] = self._calculate_pose_stability(row)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _is_arm_raised(self, row: pd.Series, side: str) -> bool:
        """Check if arm is raised (e.g., for shooting)."""
        
        shoulder_col = f'{side}_shoulder_y'
        wrist_col = f'{side}_wrist_y'
        
        if shoulder_col in row and wrist_col in row:
            if pd.notna(row[shoulder_col]) and pd.notna(row[wrist_col]):
                return row[wrist_col] < row[shoulder_col]  # Y decreases upward
        
        return False
    
    def _is_crouching(self, row: pd.Series) -> bool:
        """Check if player is in crouching position."""
        
        hip_cols = ['left_hip_y', 'right_hip_y']
        knee_cols = ['left_knee_y', 'right_knee_y']
        
        hip_y = None
        knee_y = None
        
        # Get average hip position
        valid_hip_y = [row[col] for col in hip_cols if col in row and pd.notna(row[col])]
        if valid_hip_y:
            hip_y = np.mean(valid_hip_y)
        
        # Get average knee position
        valid_knee_y = [row[col] for col in knee_cols if col in row and pd.notna(row[col])]
        if valid_knee_y:
            knee_y = np.mean(valid_knee_y)
        
        if hip_y is not None and knee_y is not None:
            # If knees are close to hips, likely crouching
            return abs(knee_y - hip_y) < 50  # Threshold in pixels
        
        return False
    
    def _calculate_pose_stability(self, row: pd.Series) -> float:
        """Calculate pose stability score (0-1)."""
        
        # Simple stability based on limb symmetry
        stability_score = 1.0
        
        # Check shoulder symmetry
        if all(col in row and pd.notna(row[col]) for col in 
               ['left_shoulder_y', 'right_shoulder_y']):
            shoulder_diff = abs(row['left_shoulder_y'] - row['right_shoulder_y'])
            stability_score *= max(0.0, 1.0 - shoulder_diff / 100.0)
        
        # Check hip symmetry
        if all(col in row and pd.notna(row[col]) for col in 
               ['left_hip_y', 'right_hip_y']):
            hip_diff = abs(row['left_hip_y'] - row['right_hip_y'])
            stability_score *= max(0.0, 1.0 - hip_diff / 100.0)
        
        return stability_score
    
    def align_with_tracking(self, 
                          pose_df: pd.DataFrame, 
                          tracking_df: pd.DataFrame,
                          max_distance: float = 100.0) -> pd.DataFrame:
        """
        Align pose data with tracking data based on spatial proximity.
        
        Args:
            pose_df: Pose DataFrame
            tracking_df: Tracking DataFrame
            max_distance: Maximum distance for pose-track matching
            
        Returns:
            Aligned pose DataFrame with corrected player IDs
        """
        
        aligned_data = []
        
        for frame_id in pose_df['frame_id'].unique():
            frame_poses = pose_df[pose_df['frame_id'] == frame_id]
            frame_tracks = tracking_df[tracking_df['frame_id'] == frame_id]
            
            if frame_tracks.empty:
                continue
            
            # Extract pose positions (use nose or neck as reference)
            pose_positions = []
            for _, pose_row in frame_poses.iterrows():
                if 'nose_x' in pose_row and pd.notna(pose_row['nose_x']):
                    pos = (pose_row['nose_x'], pose_row['nose_y'])
                elif 'neck_x' in pose_row and pd.notna(pose_row['neck_x']):
                    pos = (pose_row['neck_x'], pose_row['neck_y'])
                else:
                    pos = None
                pose_positions.append(pos)
            
            # Match poses to tracks
            for i, (_, pose_row) in enumerate(frame_poses.iterrows()):
                pose_pos = pose_positions[i]
                if pose_pos is None:
                    continue
                
                # Find closest track
                best_match = None
                best_distance = float('inf')
                
                for _, track_row in frame_tracks.iterrows():
                    track_pos = (track_row['x'], track_row['y'])
                    distance = np.sqrt((pose_pos[0] - track_pos[0])**2 + 
                                     (pose_pos[1] - track_pos[1])**2)
                    
                    if distance < best_distance and distance <= max_distance:
                        best_distance = distance
                        best_match = track_row['player_id']
                
                if best_match is not None:
                    aligned_row = pose_row.copy()
                    aligned_row['player_id'] = best_match
                    aligned_data.append(aligned_row)
        
        return pd.DataFrame(aligned_data)


def create_dummy_pose_data(output_path: str, num_frames: int = 50, num_players: int = 8) -> pd.DataFrame:
    """Create dummy pose data for testing."""
    
    rng = np.random.default_rng(42)
    pose_data = []
    
    # Key points for simplified pose
    key_points = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    for frame_id in range(1, num_frames + 1):
        for player_id in range(1, num_players + 1):
            pose_dict = {
                'frame_id': frame_id,
                'player_id': player_id
            }
            
            # Generate realistic pose coordinates
            center_x = 100 + (player_id % 5) * 150
            center_y = 100 + (player_id // 5) * 100
            
            for point in key_points:
                # Add some noise to create realistic pose variations
                x = center_x + rng.normal(0, 20)
                y = center_y + rng.normal(0, 30)
                
                pose_dict[f'{point}_x'] = x
                pose_dict[f'{point}_y'] = y
                pose_dict[f'{point}_visibility'] = rng.uniform(0.7, 1.0)
    
            pose_data.append(pose_dict)
    
    df = pd.DataFrame(pose_data)
    df.to_csv(output_path, index=False)
    print(f"Dummy pose data created: {output_path}")
    
    return df


if __name__ == "__main__":
    print("Testing Pose Data Loader...")
    
    # Create dummy data
    dummy_file = "dummy_pose.csv"
    dummy_df = create_dummy_pose_data(dummy_file, num_frames=20, num_players=6)
    print(f"Dummy pose data shape: {dummy_df.shape}")
    
    # Test loader
    loader = PoseDataLoader("mediapipe")
    
    # Extract key points
    key_points = loader.extract_key_points(dummy_df)
    print(f"Key points shape: {key_points.shape}")
    
    # Calculate features
    features = loader.calculate_pose_features(dummy_df)
    print(f"Pose features shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()}")
    
    print("Pose loader testing completed!")
    
    # Clean up
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
