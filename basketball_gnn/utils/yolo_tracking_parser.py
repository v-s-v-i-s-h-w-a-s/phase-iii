"""
YOLO Tracking Parser: Extract player tracking data from YOLO + DeepSORT output
"""

import pandas as pd
import numpy as np
import json
import cv2
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path


class YOLOTrackingParser:
    """
    Parser for YOLO + DeepSORT tracking results.
    Converts tracking output to standardized format for GNN processing.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
    def parse_yolo_txt(self, txt_file_path: str, frame_id: int) -> List[Dict]:
        """
        Parse YOLO format text file for a single frame.
        
        Format: class_id x_center y_center width height confidence track_id
        
        Args:
            txt_file_path: Path to YOLO txt file
            frame_id: Frame identifier
            
        Returns:
            List of detection dictionaries
        """
        
        detections = []
        
        if not os.path.exists(txt_file_path):
            return detections
        
        with open(txt_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5])
                        track_id = int(parts[6]) if len(parts) > 6 else -1
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'frame_id': frame_id,
                                'class_id': class_id,
                                'track_id': track_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height,
                                'confidence': confidence
                            })
                    except ValueError:
                        continue
        
        return detections
    
    def parse_detection_folder(self, folder_path: str, img_width: int, img_height: int) -> pd.DataFrame:
        """
        Parse a folder of YOLO detection files.
        
        Args:
            folder_path: Path to folder containing txt files
            img_width: Image width for coordinate conversion
            img_height: Image height for coordinate conversion
            
        Returns:
            DataFrame with tracking data
        """
        
        all_detections = []
        
        # Find all txt files
        folder = Path(folder_path)
        txt_files = list(folder.glob("*.txt"))
        
        for txt_file in txt_files:
            # Extract frame number from filename
            frame_id = self._extract_frame_id(txt_file.stem)
            
            # Parse detections
            frame_detections = self.parse_yolo_txt(str(txt_file), frame_id)
            all_detections.extend(frame_detections)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_detections)
        
        if not df.empty:
            # Convert normalized coordinates to pixel coordinates
            df['x'] = df['x_center'] * img_width
            df['y'] = df['y_center'] * img_height
            df['bbox_width'] = df['width'] * img_width
            df['bbox_height'] = df['height'] * img_height
            
            # Filter for person class (typically class_id = 0)
            df = df[df['class_id'] == 0].copy()
            
            # Rename track_id to player_id for consistency
            df['player_id'] = df['track_id']
            
        return df
    
    def _extract_frame_id(self, filename: str) -> int:
        """Extract frame ID from filename."""
        # Common patterns: frame_001.txt, 001.txt, img_001.txt
        import re
        
        # Look for numbers in filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])  # Take the last number found
        else:
            return 0
    
    def parse_deepsort_csv(self, csv_file_path: str) -> pd.DataFrame:
        """
        Parse DeepSORT CSV output.
        
        Expected format: frame_id, track_id, x, y, w, h, confidence
        
        Args:
            csv_file_path: Path to DeepSORT CSV file
            
        Returns:
            DataFrame with tracking data
        """
        
        try:
            df = pd.read_csv(csv_file_path)
            
            # Standardize column names
            column_mapping = {
                'frame': 'frame_id',
                'id': 'track_id',
                'track': 'track_id',
                'player': 'track_id',
                'bbox_left': 'x',
                'bbox_top': 'y',
                'left': 'x',
                'top': 'y'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['frame_id', 'track_id', 'x', 'y']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Convert bounding box to center coordinates
            if 'w' in df.columns and 'h' in df.columns:
                df['x'] = df['x'] + df['w'] / 2
                df['y'] = df['y'] + df['h'] / 2
            
            # Rename for consistency
            df['player_id'] = df['track_id']
            
            # Filter valid tracks
            df = df[df['player_id'] > 0].copy()
            
            return df
            
        except Exception as e:
            print(f"Error parsing CSV file: {e}")
            return pd.DataFrame()
    
    def parse_json_tracks(self, json_file_path: str) -> pd.DataFrame:
        """
        Parse JSON format tracking data.
        
        Expected format:
        {
            "frames": [
                {
                    "frame_id": 1,
                    "detections": [
                        {
                            "track_id": 1,
                            "bbox": [x, y, w, h],
                            "confidence": 0.9
                        }
                    ]
                }
            ]
        }
        
        Args:
            json_file_path: Path to JSON file
            
        Returns:
            DataFrame with tracking data
        """
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            all_detections = []
            
            for frame_data in data.get('frames', []):
                frame_id = frame_data['frame_id']
                
                for detection in frame_data.get('detections', []):
                    track_id = detection['track_id']
                    bbox = detection['bbox']
                    confidence = detection.get('confidence', 1.0)
                    
                    # Convert bbox to center coordinates
                    x = bbox[0] + bbox[2] / 2
                    y = bbox[1] + bbox[3] / 2
                    
                    all_detections.append({
                        'frame_id': frame_id,
                        'player_id': track_id,
                        'x': x,
                        'y': y,
                        'bbox_width': bbox[2],
                        'bbox_height': bbox[3],
                        'confidence': confidence
                    })
            
            return pd.DataFrame(all_detections)
            
        except Exception as e:
            print(f"Error parsing JSON file: {e}")
            return pd.DataFrame()
    
    def filter_tracking_data(self, 
                           df: pd.DataFrame,
                           min_track_length: int = 5,
                           max_players: int = 15) -> pd.DataFrame:
        """
        Filter and clean tracking data.
        
        Args:
            df: Input tracking DataFrame
            min_track_length: Minimum number of frames for a valid track
            max_players: Maximum number of players to keep
            
        Returns:
            Filtered DataFrame
        """
        
        if df.empty:
            return df
        
        # Handle both 'frame_id' and 'frame' column names
        frame_col = 'frame_id' if 'frame_id' in df.columns else 'frame'
        
        # Remove tracks that are too short
        track_lengths = df.groupby('player_id')[frame_col].count()
        valid_tracks = track_lengths[track_lengths >= min_track_length].index
        df = df[df['player_id'].isin(valid_tracks)].copy()
        
        # Keep only the most frequent players
        if len(valid_tracks) > max_players:
            top_players = track_lengths.nlargest(max_players).index
            df = df[df['player_id'].isin(top_players)].copy()
        
        # Sort by frame and player
        df = df.sort_values([frame_col, 'player_id']).reset_index(drop=True)
        
        return df
    
    def interpolate_missing_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing positions for tracks.
        
        Args:
            df: Input tracking DataFrame
            
        Returns:
            DataFrame with interpolated positions
        """
        
        if df.empty:
            return df
        
        # Handle both 'frame_id' and 'frame' column names
        frame_col = 'frame_id' if 'frame_id' in df.columns else 'frame'
        
        # Create complete frame x player grid
        all_frames = range(df[frame_col].min(), df[frame_col].max() + 1)
        all_players = df['player_id'].unique()
        
        # Create full index
        full_index = pd.MultiIndex.from_product([all_frames, all_players], 
                                              names=[frame_col, 'player_id'])
        
        # Reindex and interpolate
        df_full = df.set_index([frame_col, 'player_id']).reindex(full_index)
        
        # Interpolate positions for each player
        for player_id in all_players:
            player_mask = df_full.index.get_level_values('player_id') == player_id
            player_data = df_full.loc[player_mask]
            
            # Linear interpolation
            player_data['x'] = player_data['x'].interpolate(method='linear')
            player_data['y'] = player_data['y'].interpolate(method='linear')
            
            df_full.loc[player_mask] = player_data
        
        # Remove rows with NaN (beginning and end of tracks)
        df_full = df_full.dropna(subset=['x', 'y']).reset_index()
        
        return df_full
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """Save tracking data to CSV format."""
        
        # Select essential columns
        output_cols = ['frame_id', 'player_id', 'x', 'y']
        if 'confidence' in df.columns:
            output_cols.append('confidence')
        
        df[output_cols].to_csv(output_path, index=False)
        print(f"Tracking data saved to {output_path}")


def create_sample_tracking_data(output_path: str, num_frames: int = 100, num_players: int = 10):
    """Create sample tracking data for testing."""
    
    rng = np.random.default_rng(42)
    data = []
    
    for frame_id in range(1, num_frames + 1):
        for player_id in range(1, num_players + 1):
            # Simulate realistic player movement
            base_x = 100 + (player_id % 5) * 150
            base_y = 100 + (player_id // 5) * 100
            
            # Add movement over time
            x = base_x + rng.normal(0, 20) + frame_id * rng.normal(0, 2)
            y = base_y + rng.normal(0, 15) + frame_id * rng.normal(0, 1.5)
            
            # Clamp to court boundaries
            x = np.clip(x, 0, 940)
            y = np.clip(y, 0, 500)
            
            data.append({
                'frame_id': frame_id,
                'player_id': player_id,
                'x': x,
                'y': y,
                'confidence': rng.uniform(0.7, 1.0)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample tracking data created: {output_path}")
    
    return df


if __name__ == "__main__":
    print("Testing YOLO Tracking Parser...")
    
    # Create sample data
    sample_file = "sample_tracking.csv"
    sample_df = create_sample_tracking_data(sample_file, num_frames=50, num_players=8)
    print(f"Sample data shape: {sample_df.shape}")
    
    # Test parser
    parser = YOLOTrackingParser()
    
    # Filter data
    filtered_df = parser.filter_tracking_data(sample_df, min_track_length=10)
    print(f"Filtered data shape: {filtered_df.shape}")
    
    # Test interpolation
    interpolated_df = parser.interpolate_missing_positions(filtered_df)
    print(f"Interpolated data shape: {interpolated_df.shape}")
    
    print("Parser testing completed!")
    
    # Clean up
    if os.path.exists(sample_file):
        os.remove(sample_file)
