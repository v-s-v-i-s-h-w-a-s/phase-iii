"""
Training script for basketball GNN models
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from typing import List, Dict, Optional, Tuple

# Import our modules
import sys
sys.path.append('..')
from gnn_model.model import create_model
from graph_builder.build_graph import BasketballGraphBuilder


class BasketballGNNTrainer:
    """
    Trainer class for basketball GNN models with unsupervised learning objectives.
    """
    
    def __init__(self, 
                 model_type: str = "gcn",
                 model_kwargs: Optional[Dict] = None,
                 device: str = "auto"):
        """
        Args:
            model_type: Type of GNN model ("gcn", "sage", "formation_classifier")
            model_kwargs: Model-specific arguments
            device: Device to use ("cuda", "cpu", or "auto")
        """
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize model
        if model_kwargs is None:
            model_kwargs = {}
        
        self.model = create_model(model_type, **model_kwargs)
        self.model.to(self.device)
        
        self.model_type = model_type
        self.training_history = {'loss': [], 'silhouette': [], 'cluster_quality': []}
        
    def contrastive_loss(self, embeddings: torch.Tensor, 
                        edge_index: torch.Tensor, 
                        temperature: float = 0.1) -> torch.Tensor:
        """
        Contrastive loss for learning player relationships.
        Connected players should have similar embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge connectivity [2, num_edges]
            temperature: Temperature parameter for contrastive learning
            
        Returns:
            Contrastive loss value
        """
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / temperature
        
        # Create positive and negative pairs
        num_nodes = embeddings.shape[0]
        
        # Positive pairs (connected nodes)
        pos_pairs = edge_index.t()  # [num_edges, 2]
        
        # For each positive pair, compute loss
        total_loss = 0.0
        
        for i, (node1, node2) in enumerate(pos_pairs):
            # Positive similarity
            pos_sim = similarity_matrix[node1, node2]
            
            # Negative similarities (all other nodes)
            neg_mask = torch.ones(num_nodes, dtype=torch.bool)
            neg_mask[node2] = False  # Exclude the positive pair
            neg_similarities = similarity_matrix[node1, neg_mask]
            
            # Compute contrastive loss for this pair
            denominator = torch.exp(pos_sim) + torch.sum(torch.exp(neg_similarities))
            loss = -torch.log(torch.exp(pos_sim) / denominator)
            total_loss += loss
        
        return total_loss / len(pos_pairs) if len(pos_pairs) > 0 else torch.tensor(0.0)
    
    def formation_coherence_loss(self, embeddings: torch.Tensor, 
                                positions: torch.Tensor) -> torch.Tensor:
        """
        Loss to encourage formation coherence.
        Players in similar spatial regions should have similar embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            positions: Player positions [num_nodes, 2]
            
        Returns:
            Formation coherence loss
        """
        
        num_nodes = embeddings.shape[0]
        
        if num_nodes < 2:
            return torch.tensor(0.0)
        
        # Compute spatial distances
        spatial_dists = torch.cdist(positions, positions)  # [num_nodes, num_nodes]
        
        # Compute embedding distances
        embedding_dists = torch.cdist(embeddings, embeddings)  # [num_nodes, num_nodes]
        
        # We want spatial similarity to correlate with embedding similarity
        # Use negative correlation since smaller distances = more similarity
        spatial_similarities = torch.exp(-spatial_dists / 100.0)  # Convert to similarities
        embedding_similarities = torch.exp(-embedding_dists)
        
        # Mean squared error between similarity matrices
        loss = F.mse_loss(embedding_similarities, spatial_similarities)
        
        return loss
    
    def train_unsupervised(self, 
                          graphs: List[Data],
                          num_epochs: int = 100,
                          learning_rate: float = 0.01,
                          batch_size: int = 32,
                          contrastive_weight: float = 1.0,
                          coherence_weight: float = 0.5) -> Dict:
        """
        Train the model using unsupervised objectives.
        
        Args:
            graphs: List of graph data objects
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            contrastive_weight: Weight for contrastive loss
            coherence_weight: Weight for formation coherence loss
            
        Returns:
            Training history dictionary
        """
        
        # Create data loader
        dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"Training for {num_epochs} epochs on {len(graphs)} graphs...")
        
        self.model.train()
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            epoch_loss = 0.0
            epoch_contrastive_loss = 0.0
            epoch_coherence_loss = 0.0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Contrastive loss
                contrastive_loss = self.contrastive_loss(embeddings, batch.edge_index)
                
                # Formation coherence loss
                coherence_loss = self.formation_coherence_loss(embeddings, batch.pos)
                
                # Combined loss
                total_loss = (contrastive_weight * contrastive_loss + 
                            coherence_weight * coherence_loss)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_contrastive_loss += contrastive_loss.item()
                epoch_coherence_loss += coherence_loss.item()
            
            # Average losses
            avg_loss = epoch_loss / len(dataloader)
            avg_contrastive = epoch_contrastive_loss / len(dataloader)
            avg_coherence = epoch_coherence_loss / len(dataloader)
            
            # Update scheduler
            scheduler.step(avg_loss)
            
            # Evaluate clustering quality periodically
            if (epoch + 1) % 10 == 0:
                cluster_metrics = self.evaluate_clustering(graphs[:50])  # Use subset for speed
                self.training_history['silhouette'].append(cluster_metrics['silhouette'])
                self.training_history['cluster_quality'].append(cluster_metrics['inertia'])
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Contrastive={avg_contrastive:.4f}, "
                      f"Coherence={avg_coherence:.4f}, "
                      f"Silhouette={cluster_metrics['silhouette']:.3f}")
            
            self.training_history['loss'].append(avg_loss)
        
        return self.training_history
    
    def evaluate_clustering(self, graphs: List[Data], n_clusters: int = 2) -> Dict:
        """
        Evaluate clustering quality of learned embeddings.
        
        Args:
            graphs: List of graphs to evaluate
            n_clusters: Number of clusters for K-means
            
        Returns:
            Dictionary of clustering metrics
        """
        
        self.model.eval()
        
        all_embeddings = []
        all_positions = []
        
        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(self.device)
                embeddings = self.model(graph.x, graph.edge_index)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_positions.append(graph.pos.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings_concat = np.vstack(all_embeddings)
        positions_concat = np.vstack(all_positions)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_concat)
        
        # Calculate metrics
        metrics = {}
        
        if len(set(cluster_labels)) > 1:
            metrics['silhouette'] = silhouette_score(embeddings_concat, cluster_labels, random_state=42)
            metrics['inertia'] = kmeans.inertia_
        else:
            metrics['silhouette'] = -1.0
            metrics['inertia'] = float('inf')
        
        # Spatial clustering for comparison
        spatial_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        spatial_labels = spatial_kmeans.fit_predict(positions_concat)
        
        if len(set(spatial_labels)) > 1 and len(set(cluster_labels)) > 1:
            metrics['spatial_agreement'] = adjusted_rand_score(cluster_labels, spatial_labels)
        else:
            metrics['spatial_agreement'] = 0.0
        
        metrics['num_clusters_found'] = len(set(cluster_labels))
        
        return metrics
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model and training history."""
        
        # Get model configuration
        model_config = {
            'in_channels': None,
            'hidden_channels': None,
            'out_channels': None
        }
        
        # Try to extract config from model state dict
        state_dict = self.model.state_dict()
        if 'convs.0.lin.weight' in state_dict:
            first_layer = state_dict['convs.0.lin.weight']
            model_config['hidden_channels'] = first_layer.shape[0]
            model_config['in_channels'] = first_layer.shape[1]
        
        if 'convs.1.lin.weight' in state_dict:
            last_layer = state_dict['convs.1.lin.weight']
            model_config['out_channels'] = last_layer.shape[0]
        
        # Merge with provided metadata
        if metadata is None:
            metadata = {}
        metadata['model_config'] = model_config
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'training_history': self.training_history,
            'metadata': metadata
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
        print(f"Model config: {model_config}")
    
    def load_model(self, filepath: str):
        """Load model from checkpoint."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        print(f"Model loaded from {filepath}")
        return checkpoint.get('metadata', {})
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training curves."""
        
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curve
        if self.training_history['loss']:
            axes[0].plot(self.training_history['loss'])
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True)
        
        # Silhouette score
        if self.training_history['silhouette']:
            epochs = list(range(10, len(self.training_history['loss']) + 1, 10))
            axes[1].plot(epochs, self.training_history['silhouette'])
            axes[1].set_title('Clustering Quality (Silhouette)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].grid(True)
        
        # Cluster inertia
        if self.training_history['cluster_quality']:
            epochs = list(range(10, len(self.training_history['loss']) + 1, 10))
            axes[2].plot(epochs, self.training_history['cluster_quality'])
            axes[2].set_title('Cluster Inertia')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Inertia')
            axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


def train_on_dummy_data():
    """Train model on dummy data for testing."""
    
    print("Creating dummy training data...")
    
    # Create graph builder
    builder = BasketballGraphBuilder()
    
    # Generate dummy tracking data
    from graph_builder.build_graph import create_dummy_tracking_data
    tracking_data = create_dummy_tracking_data(num_frames=100, num_players=10)
    
    # Build graphs
    graphs = builder.build_sequence_graphs(tracking_data)
    print(f"Created {len(graphs)} training graphs")
    
    if len(graphs) == 0:
        print("No graphs created! Check your data.")
        return
    
    # Initialize trainer
    trainer = BasketballGNNTrainer(
        model_type="gcn",
        model_kwargs={'in_channels': graphs[0].x.shape[1], 'hidden_channels': 32, 'out_channels': 16}
    )
    
    # Train model
    trainer.train_unsupervised(
        graphs, 
        num_epochs=50, 
        learning_rate=0.01,
        batch_size=16
    )
    
    # Evaluate final performance
    final_metrics = trainer.evaluate_clustering(graphs[:20])
    print(f"Final clustering metrics: {final_metrics}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save model
    os.makedirs("../models", exist_ok=True)
    trainer.save_model("../models/basketball_gnn_dummy.pth", metadata={'training_data': 'dummy'})
    
    return trainer


if __name__ == "__main__":
    print("Starting basketball GNN training...")
    trainer = train_on_dummy_data()
    print("Training completed!")
