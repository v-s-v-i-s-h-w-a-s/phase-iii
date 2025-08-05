# Sample GNN-Based Basketball Play Simulation Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Example Graph: Nodes = players + ball
# 5v5 = 10 players + 1 ball = 11 nodes
# Edges = interaction/proximity

class PlayGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PlayGNN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)  # Predict next position or action class

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        return self.lin(x)

# --- Graph Input Example ---
# Each node has (x, y, vx, vy, role_id, has_ball)
# For simplicity: input features = 6 per node

# 11 nodes, 6 features each
x = torch.tensor([
    [10, 25, 1, 0, 0, 0],  # PG
    [20, 30, 0, 1, 1, 0],  # SG
    [30, 35, 0, 0, 2, 0],  # SF
    [40, 20, -1, 0, 3, 0], # PF
    [15, 40, 0, -1, 4, 0], # C
    [60, 25, 0, 0, 0, 0],  # PG (def)
    [70, 30, 0, 0, 1, 0],
    [80, 35, 0, 0, 2, 0],
    [90, 20, 0, 0, 3, 0],
    [65, 40, 0, 0, 4, 0],
    [25, 25, 0, 0, 5, 1]   # Ball
], dtype=torch.float)

# Define edges: undirected (i.e., bidirectional)
# Edge_index = [2 x num_edges]
edge_index = torch.tensor([
    [0, 1, 0, 2, 4, 10, 1, 6, 4, 9],
    [1, 0, 2, 0, 9, 4, 6, 1, 10, 4]
], dtype=torch.long)  # Shape: [2, num_edges]

# Create a graph data object
data = Data(x=x, edge_index=edge_index)

# --- Model Init ---
model = PlayGNN(in_channels=6, hidden_channels=32, out_channels=2)  # Output = (next_x, next_y)

# --- Forward Pass ---
output = model(data.x, data.edge_index)

# Output: predicted next positions
print("🏀 GNN Basketball Play Simulation Results:")
print("=" * 50)
print()

position_names = [
    "Point Guard (Offense)",
    "Shooting Guard (Offense)", 
    "Small Forward (Offense)",
    "Power Forward (Offense)",
    "Center (Offense)",
    "Point Guard (Defense)",
    "Shooting Guard (Defense)",
    "Small Forward (Defense)", 
    "Power Forward (Defense)",
    "Center (Defense)",
    "Basketball"
]

for i, pos in enumerate(output):
    current_x, current_y = x[i][0].item(), x[i][1].item()
    next_x, next_y = pos[0].item(), pos[1].item()
    
    print(f"Node {i:2d} ({position_names[i]:25s}): "
          f"Current ({current_x:5.1f}, {current_y:5.1f}) → "
          f"Predicted ({next_x:6.2f}, {next_y:6.2f})")

print()
print("🎯 Analysis:")
print(f"- Total nodes (players + ball): {len(x)}")
print(f"- Total edges (interactions): {edge_index.shape[1]}")
print(f"- Input features per node: {x.shape[1]}")
print(f"- Output predictions per node: {output.shape[1]}")
print()
print("✅ GNN successfully predicted next positions for all players and ball!")
