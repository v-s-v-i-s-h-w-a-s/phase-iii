"""
Inference script for basketball GNN models
"""

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os

# Import our modules
import sys
sys.path.append('..')
from gnn_model.model import create_model
from graph_builder.build_graph import BasketballGraphBuilder


class BasketballGNNPredictor:
    """
    Inference class for basketball GNN models.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ("cuda", "cpu", or "auto")
        """
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load model from checkpoint."""
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        model_type = checkpoint.get('model_type', 'gcn')
        metadata = checkpoint.get('metadata', {})
        
        print(f"Debug: checkpoint keys: {checkpoint.keys()}")
        print(f"Debug: metadata: {metadata}")
        
        # Get model architecture from checkpoint metadata
        if 'metadata' in metadata and 'model_config' in metadata['metadata']:
            # Use new format with properly saved config
            model_config = metadata['metadata']['model_config']
            print(f"Using saved model config: {model_config}")
        elif 'model_config' in metadata:
            # Direct model config in metadata
            model_config = metadata['model_config']
            print(f"Using model config from metadata: {model_config}")
        elif 'config' in metadata and 'hidden_channels' in metadata['config']:
            model_config = {
                'in_channels': metadata['config'].get('hidden_channels', 64) // 2,  # Estimate from saved config
                'hidden_channels': metadata['config'].get('hidden_channels', 64),
                'out_channels': metadata['config'].get('out_channels', 32)
            }
        else:
            # Try to infer from the saved state dict
            state_dict = checkpoint['model_state_dict']
            
            # Get dimensions from the first layer
            if 'convs.0.lin.weight' in state_dict:
                first_layer_weight = state_dict['convs.0.lin.weight']
                hidden_channels = first_layer_weight.shape[0]
                in_channels = first_layer_weight.shape[1]
                print(f"Inferred from state dict: in_channels={in_channels}, hidden_channels={hidden_channels}")
            else:
                hidden_channels = 64
                in_channels = 5  # Default for basketball features (x, y, vx, vy, team)
            
            # Get output dimensions from the last layer
            if 'convs.1.lin.weight' in state_dict:
                last_layer_weight = state_dict['convs.1.lin.weight']
                out_channels = last_layer_weight.shape[0]
                print(f"Inferred output channels: {out_channels}")
            else:
                out_channels = 32
            
            model_config = {
                'in_channels': in_channels,
                'hidden_channels': hidden_channels,
                'out_channels': out_channels
            }
        
        print(f"Final model config: {model_config}")
        
        # Create and load model
        self.model = create_model(model_type, **model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {model_type} model from {model_path}")
        print(f"Model config: {model_config}")
        
    def predict_embeddings(self, graphs: List) -> List[np.ndarray]:
        """
        Generate embeddings for a list of graphs.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            
        Returns:
            List of embedding arrays for each graph
        """
        
        embeddings_list = []
        
        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(self.device)
                embeddings = self.model(graph.x, graph.edge_index)
                embeddings_list.append(embeddings.cpu().numpy())
        
        return embeddings_list
    
    def cluster_players(self, 
                       graphs: List,
                       n_clusters: int = 2,
                       method: str = "kmeans") -> List[np.ndarray]:
        """
        Cluster players based on learned embeddings.
        
        Args:
            graphs: List of graphs
            n_clusters: Number of clusters
            method: Clustering method ("kmeans", "spectral")
            
        Returns:
            List of cluster labels for each graph
        """
        
        embeddings_list = self.predict_embeddings(graphs)
        cluster_labels_list = []
        
        for embeddings in embeddings_list:
            if method == "kmeans":
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            cluster_labels_list.append(labels)
        
        return cluster_labels_list
    
    def analyze_formations(self, 
                          graphs: List,
                          cluster_labels_list: List[np.ndarray]) -> Dict:
        """
        Analyze team formations based on clustering results.
        
        Args:
            graphs: List of graphs
            cluster_labels_list: List of cluster labels for each graph
            
        Returns:
            Formation analysis results
        """
        
        analysis = {
            'formation_stability': [],
            'cluster_compactness': [],
            'spatial_distribution': [],
            'team_centroids': []
        }
        
        for graph, labels in zip(graphs, cluster_labels_list):
            positions = graph.pos.cpu().numpy()
            
            # Formation stability (silhouette score)
            if len(set(labels)) > 1:
                silhouette = silhouette_score(positions, labels, random_state=42)
                analysis['formation_stability'].append(silhouette)
            else:
                analysis['formation_stability'].append(-1.0)
            
            # Cluster compactness
            compactness_scores = []
            for cluster_id in set(labels):
                cluster_positions = positions[labels == cluster_id]
                if len(cluster_positions) > 1:
                    # Calculate intra-cluster distances
                    centroid = np.mean(cluster_positions, axis=0)
                    distances = np.linalg.norm(cluster_positions - centroid, axis=1)
                    compactness_scores.append(np.mean(distances))
            
            analysis['cluster_compactness'].append(np.mean(compactness_scores) if compactness_scores else 0.0)
            
            # Spatial distribution
            x_spread = np.std(positions[:, 0])
            y_spread = np.std(positions[:, 1])
            analysis['spatial_distribution'].append((x_spread, y_spread))
            
            # Team centroids for each cluster
            centroids = []
            for cluster_id in set(labels):
                cluster_positions = positions[labels == cluster_id]
                centroid = np.mean(cluster_positions, axis=0)
                centroids.append(centroid)
            analysis['team_centroids'].append(centroids)
        
        return analysis
    
    def detect_tactical_patterns(self, 
                                graphs: List,
                                window_size: int = 5) -> Dict:
        """
        Detect tactical patterns over time.
        
        Args:
            graphs: Sequential list of graphs
            window_size: Window size for pattern detection
            
        Returns:
            Detected patterns and transitions
        """
        
        # Get embeddings for all frames
        embeddings_list = self.predict_embeddings(graphs)
        
        patterns = {
            'formation_transitions': [],
            'player_role_changes': [],
            'team_coordination': []
        }
        
        for i in range(len(embeddings_list) - window_size + 1):
            window_embeddings = embeddings_list[i:i + window_size]
            window_graphs = graphs[i:i + window_size]
            
            # Analyze each pattern type
            formation_change = self._analyze_formation_transitions(window_embeddings)
            role_change = self._analyze_role_changes(window_graphs)
            coordination = self._analyze_team_coordination(window_graphs)
            
            patterns['formation_transitions'].append(formation_change)
            patterns['player_role_changes'].append(role_change)
            patterns['team_coordination'].append(coordination)
        
        return patterns
    
    def _analyze_formation_transitions(self, window_embeddings: List) -> float:
        """Analyze formation stability within window."""
        formation_changes = []
        
        for j in range(len(window_embeddings) - 1):
            emb1, emb2 = window_embeddings[j], window_embeddings[j + 1]
            
            if emb1.shape == emb2.shape and emb1.shape[0] == emb2.shape[0]:
                similarities = []
                for k in range(emb1.shape[0]):
                    sim = np.dot(emb1[k], emb2[k]) / (np.linalg.norm(emb1[k]) * np.linalg.norm(emb2[k]))
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                formation_changes.append(1.0 - avg_similarity)
        
        return np.mean(formation_changes) if formation_changes else 0.0
    
    def _analyze_role_changes(self, window_graphs: List) -> float:
        """Analyze player role consistency."""
        role_changes = []
        
        for j in range(len(window_graphs) - 1):
            pos1 = window_graphs[j].pos.cpu().numpy()
            pos2 = window_graphs[j + 1].pos.cpu().numpy()
            
            if pos1.shape == pos2.shape:
                position_changes = np.linalg.norm(pos2 - pos1, axis=1)
                role_changes.append(np.mean(position_changes))
        
        return np.mean(role_changes) if role_changes else 0.0
    
    def _analyze_team_coordination(self, window_graphs: List) -> float:
        """Analyze team coordination."""
        coordination_scores = []
        
        for j in range(len(window_graphs) - 1):
            pos1 = window_graphs[j].pos.cpu().numpy()
            pos2 = window_graphs[j + 1].pos.cpu().numpy()
            
            if pos1.shape == pos2.shape:
                movements = pos2 - pos1
                coordination = 1.0 / (1.0 + np.var(movements))
                coordination_scores.append(coordination)
        
        return np.mean(coordination_scores) if coordination_scores else 0.0
    
    def predict_next_positions(self, 
                             recent_graphs: List,
                             prediction_horizon: int = 1) -> np.ndarray:
        """
        Predict future player positions based on recent frames.
        
        Args:
            recent_graphs: Recent graph sequence
            prediction_horizon: How many frames ahead to predict
            
        Returns:
            Predicted positions [num_players, 2]
        """
        
        if len(recent_graphs) < 2:
            raise ValueError("Need at least 2 frames for prediction")
        
        # Extract recent positions
        recent_positions = [graph.pos.cpu().numpy() for graph in recent_graphs]
        
        # Simple linear extrapolation for now
        # In practice, you'd use a temporal GNN or RNN
        
        last_positions = recent_positions[-1]
        second_last_positions = recent_positions[-2]
        
        # Calculate velocity
        velocity = last_positions - second_last_positions
        
        # Predict next positions
        predicted_positions = last_positions + velocity * prediction_horizon
        
        return predicted_positions
    
    def visualize_predictions(self, 
                            graphs: List,
                            cluster_labels_list: List[np.ndarray],
                            save_path: Optional[str] = None,
                            max_frames: int = 10):
        """
        Visualize clustering predictions.
        
        Args:
            graphs: List of graphs
            cluster_labels_list: Cluster labels for each graph
            save_path: Path to save visualization
            max_frames: Maximum number of frames to plot
        """
        
        num_frames = min(len(graphs), max_frames)
        cols = min(5, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        _, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for idx in range(num_frames):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            graph = graphs[idx]
            labels = cluster_labels_list[idx]
            positions = graph.pos.cpu().numpy()
            
            # Plot players with cluster colors
            for cluster_id in set(labels):
                cluster_mask = labels == cluster_id
                cluster_positions = positions[cluster_mask]
                
                color = colors[cluster_id % len(colors)]
                ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], 
                          c=color, s=100, alpha=0.7, label=f'Team {cluster_id + 1}')
            
            # Draw court outline (simplified)
            court_width, court_height = 940, 500
            ax.plot([0, court_width, court_width, 0, 0], 
                   [0, 0, court_height, court_height, 0], 'k-', linewidth=2)
            
            # Center line
            ax.plot([court_width/2, court_width/2], [0, court_height], 'k--', alpha=0.5)
            
            ax.set_xlim(-50, court_width + 50)
            ax.set_ylim(-50, court_height + 50)
            ax.set_aspect('equal')
            ax.set_title(f'Frame {graph.frame_id}')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(num_frames, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def run_inference_demo():
    """Run inference demonstration."""
    
    print("Running basketball GNN inference demo...")
    
    # Check if model exists
    model_path = "../models/basketball_gnn_dummy.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run training first: python gnn_model/train.py")
        return
    
    # Create test data
    builder = BasketballGraphBuilder()
    from graph_builder.build_graph import create_dummy_tracking_data
    
    test_data = create_dummy_tracking_data(num_frames=20, num_players=8)
    test_graphs = builder.build_sequence_graphs(test_data)
    
    print(f"Created {len(test_graphs)} test graphs")
    
    # Initialize predictor
    predictor = BasketballGNNPredictor(model_path)
    
    # Predict clusters
    cluster_labels = predictor.cluster_players(test_graphs, n_clusters=2)
    print(f"Generated clusters for {len(cluster_labels)} frames")
    
    # Analyze formations
    formation_analysis = predictor.analyze_formations(test_graphs, cluster_labels)
    print(f"Average formation stability: {np.mean(formation_analysis['formation_stability']):.3f}")
    
    # Detect patterns
    patterns = predictor.detect_tactical_patterns(test_graphs)
    print(f"Average formation transition rate: {np.mean(patterns['formation_transitions']):.3f}")
    
    # Visualize results
    predictor.visualize_predictions(test_graphs[:8], cluster_labels[:8])
    
    # Predict next positions
    if len(test_graphs) >= 3:
        predicted_pos = predictor.predict_next_positions(test_graphs[-3:])
        print(f"Predicted positions shape: {predicted_pos.shape}")
    
    print("Inference demo completed!")


if __name__ == "__main__":
    run_inference_demo()
