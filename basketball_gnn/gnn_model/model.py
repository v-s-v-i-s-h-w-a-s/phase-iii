"""
Graph Neural Network Model for Basketball Player Relationship Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, global_mean_pool
from torch_geometric.data import Data


class PlayerInteractionGCN(torch.nn.Module):
    """
    Basic GCN model for player interaction analysis.
    
    Args:
        in_channels: Input feature dimension (x, y, velocity, pose features)
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of GCN layers
    """
    
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=32, num_layers=2):
        super(PlayerInteractionGCN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector for graph-level tasks
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        
        return x


class PlayerGraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for larger graphs and better scalability.
    """
    
    def __init__(self, in_channels=4, hidden_channels=64, out_channels=32, num_layers=2):
        super(PlayerGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Build layers
        layer_sizes = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        
        for i in range(num_layers):
            self.convs.append(GraphSAGE(
                in_channels=layer_sizes[i],
                hidden_channels=layer_sizes[i+1],
                num_layers=1,
                out_channels=layer_sizes[i+1],
                dropout=0.2
            ))
            
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            
        x = self.convs[-1](x, edge_index)
        return x


class TacticalFormationClassifier(torch.nn.Module):
    """
    Graph-level classifier for tactical formation recognition.
    """
    
    def __init__(self, node_embedding_dim=32, num_formations=5):
        super(TacticalFormationClassifier, self).__init__()
        
        self.node_encoder = PlayerInteractionGCN(out_channels=node_embedding_dim)
        
        # Graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(node_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_formations)
        )
        
    def forward(self, x, edge_index, batch):
        # Get node embeddings
        node_emb = self.node_encoder(x, edge_index)
        
        # Pool to graph-level representation
        graph_emb = global_mean_pool(node_emb, batch)
        
        # Classify formation
        formation_logits = self.classifier(graph_emb)
        
        return formation_logits, node_emb


def create_model(model_type="gcn", **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        model_type: "gcn", "sage", or "formation_classifier"
        **kwargs: Model-specific arguments
        
    Returns:
        Initialized model
    """
    
    if model_type == "gcn":
        return PlayerInteractionGCN(**kwargs)
    elif model_type == "sage":
        return PlayerGraphSAGE(**kwargs)
    elif model_type == "formation_classifier":
        return TacticalFormationClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("Testing GNN models...")
    
    # Create dummy data
    num_players = 10
    x = torch.randn(num_players, 4)  # (x, y, vx, vy)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    # Test GCN
    model = create_model("gcn", in_channels=4, hidden_channels=32, out_channels=16)
    embeddings = model(x, edge_index)
    print(f"GCN output shape: {embeddings.shape}")
    
    # Test GraphSAGE
    model_sage = create_model("sage", in_channels=4, hidden_channels=32, out_channels=16)
    embeddings_sage = model_sage(x, edge_index)
    print(f"GraphSAGE output shape: {embeddings_sage.shape}")
    
    print("Models created successfully!")
