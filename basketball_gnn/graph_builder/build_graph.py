"""
Graph Builder Module: Convert frame-wise player data into PyTorch Geometric graphs
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import math


class BasketballGraphBuilder:
    """
    Builds graphs from basketball player tracking data.
    Each frame becomes a graph where nodes are players and edges represent interactions.
    """
    
    def __init__(self, 
                 proximity_threshold: float = 150.0,
                 min_players: int = 3,
                 max_players: int = 15,
                 court_width: float = 940.0,
                 court_height: float = 500.0):
        """
        Args:
            proximity_threshold: Distance threshold for creating edges (pixels)
            min_players: Minimum players required to create a graph
            max_players: Maximum players to include
            court_width: Basketball court width in pixels
            court_height: Basketball court height in pixels
        """
        self.proximity_threshold = proximity_threshold
        self.min_players = min_players
        self.max_players = max_players
        self.court_width = court_width
        self.court_height = court_height
        
    def build_proximity_edges(self, positions: np.ndarray) -> torch.Tensor:
        """
        Create edges based on player proximity.
        
        Args:
            positions: Player positions [num_players, 2] (x, y)
            
        Returns:
            edge_index: [2, num_edges] edge connectivity
        """
        num_players = positions.shape[0]
        edges = []
        
        for i in range(num_players):
            for j in range(i + 1, num_players):
                # Calculate Euclidean distance
                dist = np.linalg.norm(positions[i] - positions[j])
                
                if dist <= self.proximity_threshold:
                    # Add bidirectional edges
                    edges.extend([[i, j], [j, i]])
        
        if not edges:
            # If no proximity edges, create a minimal connected graph
            edges = [[0, 1], [1, 0]] if num_players >= 2 else []
            
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    
    def build_k_nearest_edges(self, positions: np.ndarray, k: int = 3) -> torch.Tensor:
        """
        Create edges to k-nearest neighbors for each player.
        
        Args:
            positions: Player positions [num_players, 2]
            k: Number of nearest neighbors
            
        Returns:
            edge_index: [2, num_edges] edge connectivity
        """
        num_players = positions.shape[0]
        edges = []
        
        for i in range(num_players):
            # Calculate distances to all other players
            distances = []
            for j in range(num_players):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append((dist, j))
            
            # Sort by distance and take k nearest
            distances.sort()
            k_nearest = min(k, len(distances))
            
            for _, j in distances[:k_nearest]:
                edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).T if edges else torch.empty((2, 0), dtype=torch.long)
    
    def extract_node_features(self, 
                            frame_data: pd.DataFrame,
                            prev_frame_data: Optional[pd.DataFrame] = None,
                            pose_data: Optional[pd.DataFrame] = None) -> torch.Tensor:
        """
        Extract node features for each player.
        
        Args:
            frame_data: Current frame player data
            prev_frame_data: Previous frame data for velocity calculation
            pose_data: Pose keypoints data
            
        Returns:
            features: [num_players, feature_dim] node features
        """
        features = []
        
        for _, player in frame_data.iterrows():
            player_features = []
            
            # Basic position features (normalized)
            x_norm = player['x'] / self.court_width
            y_norm = player['y'] / self.court_height
            player_features.extend([x_norm, y_norm])
            
            # Velocity features
            if prev_frame_data is not None:
                prev_player = prev_frame_data[prev_frame_data['player_id'] == player['player_id']]
                if not prev_player.empty:
                    vx = (player['x'] - prev_player.iloc[0]['x']) / self.court_width
                    vy = (player['y'] - prev_player.iloc[0]['y']) / self.court_height
                    player_features.extend([vx, vy])
                else:
                    player_features.extend([0.0, 0.0])
            else:
                player_features.extend([0.0, 0.0])
            
            # Distance to court center
            center_x, center_y = self.court_width / 2, self.court_height / 2
            dist_to_center = math.sqrt((player['x'] - center_x)**2 + (player['y'] - center_y)**2)
            dist_to_center_norm = dist_to_center / (math.sqrt(center_x**2 + center_y**2))
            player_features.append(dist_to_center_norm)
            
            # Player role encoding (if available)
            if 'role' in player:
                role_encoding = self._encode_player_role(player['role'])
                player_features.extend(role_encoding)
            
            # Pose features (simplified)
            if pose_data is not None:
                pose_player = pose_data[pose_data['player_id'] == player['player_id']]
                if not pose_player.empty:
                    # Use simplified pose features (e.g., body orientation)
                    pose_features = self._extract_pose_features(pose_player.iloc[0])
                    player_features.extend(pose_features)
            
            features.append(player_features)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _encode_player_role(self, role: str) -> List[float]:
        """Encode player role as one-hot vector."""
        roles = ['PG', 'SG', 'SF', 'PF', 'C', 'UNKNOWN']
        encoding = [0.0] * len(roles)
        
        if role in roles:
            encoding[roles.index(role)] = 1.0
        else:
            encoding[-1] = 1.0  # UNKNOWN
            
        return encoding
    
    def _extract_pose_features(self, pose_row: pd.Series) -> List[float]:
        """Extract simplified pose features."""
        # Simplified: just use a few key pose points
        # In practice, you'd process the full 33 MediaPipe landmarks
        features = []
        
        # Body orientation (simplified)
        if 'shoulder_left_x' in pose_row and 'shoulder_right_x' in pose_row:
            shoulder_angle = math.atan2(
                pose_row['shoulder_left_y'] - pose_row['shoulder_right_y'],
                pose_row['shoulder_left_x'] - pose_row['shoulder_right_x']
            )
            features.extend([math.cos(shoulder_angle), math.sin(shoulder_angle)])
        else:
            features.extend([0.0, 0.0])
            
        return features
    
    def build_frame_graph(self, 
                         frame_data: pd.DataFrame,
                         frame_id: int,
                         prev_frame_data: Optional[pd.DataFrame] = None,
                         pose_data: Optional[pd.DataFrame] = None,
                         edge_method: str = "proximity") -> Optional[Data]:
        """
        Build a complete graph for a single frame.
        
        Args:
            frame_data: Player data for current frame
            frame_id: Frame identifier
            prev_frame_data: Previous frame data
            pose_data: Pose data for current frame
            edge_method: "proximity" or "k_nearest"
            
        Returns:
            PyTorch Geometric Data object or None if insufficient players
        """
        
        # Filter valid players
        valid_players = frame_data.dropna(subset=['x', 'y'])
        
        if len(valid_players) < self.min_players:
            return None
        
        # Limit to max players
        if len(valid_players) > self.max_players:
            valid_players = valid_players.head(self.max_players)
        
        # Extract positions
        positions = valid_players[['x', 'y']].values
        
        # Build edges
        if edge_method == "proximity":
            edge_index = self.build_proximity_edges(positions)
        elif edge_method == "k_nearest":
            edge_index = self.build_k_nearest_edges(positions, k=3)
        else:
            raise ValueError(f"Unknown edge method: {edge_method}")
        
        # Extract node features
        node_features = self.extract_node_features(valid_players, prev_frame_data, pose_data)
        
        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            pos=torch.tensor(positions, dtype=torch.float),
            frame_id=frame_id,
            num_nodes=len(valid_players)
        )
        
        return graph
    
    def build_sequence_graphs(self, 
                            tracking_data: pd.DataFrame,
                            pose_data: Optional[pd.DataFrame] = None,
                            frame_range: Optional[Tuple[int, int]] = None) -> List[Data]:
        """
        Build graphs for a sequence of frames.
        
        Args:
            tracking_data: Full tracking dataset
            pose_data: Full pose dataset
            frame_range: (start_frame, end_frame) or None for all frames
            
        Returns:
            List of graph objects
        """
        
        graphs = []
        # Handle both 'frame_id' and 'frame' column names
        frame_col = 'frame_id' if 'frame_id' in tracking_data.columns else 'frame'
        frame_ids = sorted(tracking_data[frame_col].unique())
        
        if frame_range:
            start_frame, end_frame = frame_range
            frame_ids = [f for f in frame_ids if start_frame <= f <= end_frame]
        
        prev_frame_data = None
        
        for frame_id in frame_ids:
            # Get current frame data
            frame_data = tracking_data[tracking_data[frame_col] == frame_id]
            
            # Get pose data for this frame
            frame_pose_data = None
            if pose_data is not None:
                pose_frame_col = 'frame_id' if 'frame_id' in pose_data.columns else 'frame'
                frame_pose_data = pose_data[pose_data[pose_frame_col] == frame_id]
            
            # Build graph
            graph = self.build_frame_graph(
                frame_data, 
                frame_id, 
                prev_frame_data, 
                frame_pose_data
            )
            
            if graph is not None:
                graphs.append(graph)
            
            prev_frame_data = frame_data
        
        return graphs


def create_dummy_tracking_data(num_frames: int = 50, num_players: int = 10) -> pd.DataFrame:
    """Create dummy tracking data for testing."""
    
    data = []
    
    for frame_id in range(1, num_frames + 1):
        for player_id in range(1, num_players + 1):
            # Simulate player movement
            base_x = 100 + (player_id % 5) * 150
            base_y = 100 + (player_id // 5) * 100
            
            # Add some random movement
            x = base_x + np.random.normal(0, 20) + frame_id * np.random.normal(0, 2)
            y = base_y + np.random.normal(0, 15) + frame_id * np.random.normal(0, 1.5)
            
            data.append({
                'frame_id': frame_id,
                'player_id': player_id,
                'x': max(0, min(940, x)),
                'y': max(0, min(500, y))
            })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Testing Basketball Graph Builder...")
    
    # Create dummy data
    tracking_data = create_dummy_tracking_data(num_frames=10, num_players=8)
    print(f"Created dummy tracking data: {len(tracking_data)} records")
    
    # Initialize graph builder
    builder = BasketballGraphBuilder()
    
    # Build graphs for sequence
    graphs = builder.build_sequence_graphs(tracking_data)
    print(f"Built {len(graphs)} graphs")
    
    if graphs:
        sample_graph = graphs[0]
        print(f"Sample graph - Nodes: {sample_graph.num_nodes}, Edges: {sample_graph.edge_index.shape[1]}")
        print(f"Node features shape: {sample_graph.x.shape}")
        print(f"Position shape: {sample_graph.pos.shape}")
    
    print("Graph builder test completed!")
