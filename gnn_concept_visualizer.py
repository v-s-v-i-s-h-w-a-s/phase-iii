import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import cv2
from matplotlib.patches import FancyBboxPatch

class GNNConceptVisualizer:
    def __init__(self):
        self.court_width = 800
        self.court_height = 400
        
        # Sample game state
        self.players = [
            {"id": "A1", "team": "A", "x": 300, "y": 200, "role": "PG", "has_ball": True},
            {"id": "A2", "team": "A", "x": 350, "y": 150, "role": "SG", "has_ball": False},
            {"id": "A3", "team": "A", "x": 280, "y": 250, "role": "SF", "has_ball": False},
            {"id": "A4", "team": "A", "x": 200, "y": 180, "role": "PF", "has_ball": False},
            {"id": "A5", "team": "A", "x": 150, "y": 220, "role": "C", "has_ball": False},
            {"id": "B1", "team": "B", "x": 320, "y": 210, "role": "PG", "has_ball": False},
            {"id": "B2", "team": "B", "x": 370, "y": 160, "role": "SG", "has_ball": False},
            {"id": "B3", "team": "B", "x": 290, "y": 260, "role": "SF", "has_ball": False},
            {"id": "B4", "team": "B", "x": 450, "y": 190, "role": "PF", "has_ball": False},
            {"id": "B5", "team": "B", "x": 500, "y": 200, "role": "C", "has_ball": False}
        ]
        
        self.ball = {"x": 300, "y": 200}
        self.baskets = [{"x": 50, "y": 200}, {"x": 750, "y": 200}]
        
    def draw_court(self, ax):
        """Draw basketball court with detailed markings"""
        ax.set_xlim(0, self.court_width)
        ax.set_ylim(0, self.court_height)
        ax.set_aspect('equal')
        
        # Court outline
        court = patches.Rectangle((25, 25), 750, 350, linewidth=3, 
                                edgecolor='black', facecolor='#90EE90', alpha=0.3)
        ax.add_patch(court)
        
        # Center line
        ax.plot([400, 400], [25, 375], 'k-', linewidth=2)
        
        # Center circle
        center_circle = patches.Circle((400, 200), 50, linewidth=2, 
                                     edgecolor='black', facecolor='none')
        ax.add_patch(center_circle)
        
        # Paint areas
        left_paint = patches.Rectangle((25, 150), 100, 100, linewidth=2,
                                     edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(left_paint)
        
        right_paint = patches.Rectangle((675, 150), 100, 100, linewidth=2,
                                      edgecolor='red', facecolor='lightcoral', alpha=0.3)
        ax.add_patch(right_paint)
        
        # Free throw circles
        left_ft = patches.Circle((125, 200), 40, linewidth=2, 
                               edgecolor='blue', facecolor='none')
        ax.add_patch(left_ft)
        
        right_ft = patches.Circle((675, 200), 40, linewidth=2, 
                                edgecolor='red', facecolor='none')
        ax.add_patch(right_ft)
        
        # Baskets
        for i, basket in enumerate(self.baskets):
            color = 'blue' if i == 0 else 'red'
            basket_circle = patches.Circle((basket["x"], basket["y"]), 12, 
                                         facecolor='orange', edgecolor=color, linewidth=3)
            ax.add_patch(basket_circle)
        
        # Three-point arcs (simplified)
        theta = np.linspace(-np.pi/3, np.pi/3, 50)
        
        # Left three-point arc
        left_3pt_x = 50 + 180 * np.cos(theta)
        left_3pt_y = 200 + 180 * np.sin(theta)
        ax.plot(left_3pt_x, left_3pt_y, 'b-', linewidth=2)
        
        # Right three-point arc  
        right_3pt_x = 750 - 180 * np.cos(theta)
        right_3pt_y = 200 + 180 * np.sin(theta)
        ax.plot(right_3pt_x, right_3pt_y, 'r-', linewidth=2)
    
    def create_gnn_graph(self):
        """Create the GNN interaction graph"""
        G = nx.Graph()
        
        # Add player nodes with attributes
        for player in self.players:
            G.add_node(player["id"], 
                      team=player["team"],
                      position=(player["x"], player["y"]),
                      role=player["role"],
                      has_ball=player["has_ball"])
        
        # Add edges based on proximity and game context
        for i, p1 in enumerate(self.players):
            for j, p2 in enumerate(self.players[i+1:], i+1):
                distance = np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
                
                # Teammate connections (closer teammates have stronger connections)
                if p1["team"] == p2["team"] and distance < 100:
                    weight = 1.0 - (distance / 100)
                    G.add_edge(p1["id"], p2["id"], 
                              weight=weight, 
                              edge_type="teammate",
                              distance=distance)
                
                # Defensive pressure (opponent proximity)
                elif p1["team"] != p2["team"] and distance < 80:
                    weight = 1.0 - (distance / 80)
                    G.add_edge(p1["id"], p2["id"], 
                              weight=weight, 
                              edge_type="defense",
                              distance=distance)
        
        return G
    
    def visualize_gnn_concepts(self):
        """Create comprehensive GNN concept visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[2, 1, 1])
        
        # Main court view with GNN overlay
        ax1 = fig.add_subplot(gs[:, 0])
        self.draw_court(ax1)
        self.draw_gnn_overlay(ax1)
        ax1.set_title("Basketball Court with GNN Analysis", fontsize=16, fontweight='bold')
        
        # Node features visualization
        ax2 = fig.add_subplot(gs[0, 1])
        self.draw_node_features(ax2)
        ax2.set_title("Player Node Features", fontsize=14, fontweight='bold')
        
        # Edge relationships
        ax3 = fig.add_subplot(gs[0, 2])
        self.draw_edge_analysis(ax3)
        ax3.set_title("Edge Relationships", fontsize=14, fontweight='bold')
        
        # Tactical analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self.draw_tactical_analysis(ax4)
        ax4.set_title("Tactical Insights", fontsize=14, fontweight='bold')
        
        # GNN architecture
        ax5 = fig.add_subplot(gs[1, 2])
        self.draw_gnn_architecture(ax5)
        ax5.set_title("GNN Architecture", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def draw_gnn_overlay(self, ax):
        """Draw players, ball, and GNN graph overlay"""
        G = self.create_gnn_graph()
        
        # Draw players
        for player in self.players:
            color = 'blue' if player["team"] == "A" else 'red'
            size = 200 if player["has_ball"] else 120
            alpha = 0.9 if player["has_ball"] else 0.7
            
            # Player circle
            ax.scatter(player["x"], player["y"], c=color, s=size, 
                      alpha=alpha, edgecolors='black', linewidth=2, zorder=5)
            
            # Player ID
            ax.text(player["x"], player["y"]-30, f"{player['id']}\n{player['role']}", 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Draw ball
        ax.scatter(self.ball["x"], self.ball["y"], c='orange', s=100, 
                  marker='o', edgecolors='black', linewidth=2, zorder=6)
        ax.text(self.ball["x"]+15, self.ball["y"]+15, "BALL", fontsize=8, fontweight='bold')
        
        # Draw GNN edges
        for edge in G.edges(data=True):
            node1, node2, data = edge
            pos1 = G.nodes[node1]["position"]
            pos2 = G.nodes[node2]["position"]
            
            if data["edge_type"] == "teammate":
                # Green lines for teammate connections
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'g-', alpha=0.6, linewidth=3, zorder=1)
                # Add edge weight text
                mid_x, mid_y = (pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2
                ax.text(mid_x, mid_y, f"{data['weight']:.2f}", 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='lightgreen', alpha=0.7))
            
            elif data["edge_type"] == "defense":
                # Red dashed lines for defensive pressure
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'r--', alpha=0.5, linewidth=2, zorder=1)
                # Add pressure indicator
                mid_x, mid_y = (pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2
                ax.text(mid_x, mid_y, "DEF", 
                       fontsize=6, ha='center', va='center', color='red',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='pink', alpha=0.7))
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', linewidth=3, label='Teammate Connection'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Defensive Pressure'),
            plt.scatter([], [], c='blue', s=100, label='Team A'),
            plt.scatter([], [], c='red', s=100, label='Team B'),
            plt.scatter([], [], c='orange', s=100, label='Ball')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    def draw_node_features(self, ax):
        """Visualize node features for GNN"""
        ax.axis('off')
        
        # Sample player for feature visualization
        sample_player = self.players[0]  # A1 with ball
        
        features_text = f"""Player Node Features:

ID: {sample_player['id']}
Team: {sample_player['team']}
Position: ({sample_player['x']}, {sample_player['y']})
Role: {sample_player['role']}
Has Ball: {sample_player['has_ball']}

Derived Features:
• Distance to basket: {np.sqrt((sample_player['x'] - 50)**2 + (sample_player['y'] - 200)**2):.1f}
• Distance to center: {np.sqrt((sample_player['x'] - 400)**2 + (sample_player['y'] - 200)**2):.1f}
• Court zone: {"Backcourt" if sample_player['x'] < 400 else "Frontcourt"}
• Threat level: {"High" if sample_player['has_ball'] else "Medium"}

Node Vector:
[x_pos, y_pos, has_ball, 
 team_id, role_id, 
 dist_basket, dist_center,
 zone_id, threat_level]
"""
        
        ax.text(0.05, 0.95, features_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    def draw_edge_analysis(self, ax):
        """Visualize edge relationships and weights"""
        ax.axis('off')
        
        G = self.create_gnn_graph()
        
        # Analyze edges
        teammate_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['edge_type'] == 'teammate']
        defense_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['edge_type'] == 'defense']
        
        edge_text = f"""Edge Analysis:

Teammate Connections: {len(teammate_edges)}
Defensive Pressure: {len(defense_edges)}
Total Edges: {G.number_of_edges()}

Sample Teammate Edge:
{teammate_edges[0][0]} ↔ {teammate_edges[0][1]}
Weight: {teammate_edges[0][2]['weight']:.3f}
Distance: {teammate_edges[0][2]['distance']:.1f}

Sample Defense Edge:
{defense_edges[0][0]} ↔ {defense_edges[0][1]}
Weight: {defense_edges[0][2]['weight']:.3f}
Distance: {defense_edges[0][2]['distance']:.1f}

Edge Types:
• Teammate: Collaboration
• Defense: Opposition
• Passing: Ball movement
• Screening: Pick plays
"""
        
        ax.text(0.05, 0.95, edge_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
    
    def draw_tactical_analysis(self, ax):
        """Draw tactical analysis results"""
        ax.axis('off')
        
        G = self.create_gnn_graph()
        
        # Calculate tactical metrics
        team_a_players = [p for p in self.players if p["team"] == "A"]
        team_b_players = [p for p in self.players if p["team"] == "B"]
        
        # Team compactness
        def team_compactness(team_players):
            if len(team_players) < 2:
                return 0
            center_x = sum(p["x"] for p in team_players) / len(team_players)
            center_y = sum(p["y"] for p in team_players) / len(team_players)
            distances = [np.sqrt((p["x"] - center_x)**2 + (p["y"] - center_y)**2) for p in team_players]
            return sum(distances) / len(distances)
        
        # Ball position analysis
        ball_x = self.ball["x"]
        threat_level = min(1.0, (750 - ball_x) / 700)  # Closer to Team B basket = higher threat
        
        # Network metrics (handle disconnected graphs)
        try:
            avg_path_length = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            # Graph is not connected, calculate for largest component
            if G.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph) if subgraph.number_of_edges() > 0 else 0
            else:
                avg_path_length = 0
        
        tactical_text = f"""GNN Tactical Analysis:

Ball Possession: Team A
Ball Position: ({ball_x}, {self.ball['y']})
Threat Level: {threat_level:.2f}

Team Formation:
Team A Compactness: {team_compactness(team_a_players):.1f}
Team B Compactness: {team_compactness(team_b_players):.1f}

Network Metrics:
Graph Density: {nx.density(G):.3f}
Clustering Coeff: {nx.average_clustering(G):.3f}
Shortest Path: {avg_path_length:.2f}

Predicted Actions:
1. Pass to A2 (85% confidence)
2. Drive to basket (65% confidence)  
3. Screen by A4 (45% confidence)

Defensive Response:
• B1 should pressure ball
• B2 should deny passing lane
• B5 should protect rim
"""
        
        ax.text(0.05, 0.95, tactical_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    def draw_gnn_architecture(self, ax):
        """Draw GNN architecture diagram"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Input layer
        input_box = FancyBboxPatch((0.5, 8), 3, 1, boxstyle="round,pad=0.1", 
                                  facecolor='lightblue', edgecolor='black')
        ax.add_patch(input_box)
        ax.text(2, 8.5, "Input Layer\n(Player Features)", ha='center', va='center', fontsize=9)
        
        # GNN layers
        gnn1_box = FancyBboxPatch((0.5, 6), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black')
        ax.add_patch(gnn1_box)
        ax.text(2, 6.5, "GNN Layer 1\n(Message Passing)", ha='center', va='center', fontsize=9)
        
        gnn2_box = FancyBboxPatch((0.5, 4), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black')
        ax.add_patch(gnn2_box)
        ax.text(2, 4.5, "GNN Layer 2\n(Aggregation)", ha='center', va='center', fontsize=9)
        
        # Output layer
        output_box = FancyBboxPatch((0.5, 2), 3, 1, boxstyle="round,pad=0.1",
                                   facecolor='lightcoral', edgecolor='black')
        ax.add_patch(output_box)
        ax.text(2, 2.5, "Output Layer\n(Predictions)", ha='center', va='center', fontsize=9)
        
        # Arrows
        ax.arrow(2, 7.8, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(2, 5.8, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(2, 3.8, 0, -0.6, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Side annotations
        ax.text(5, 8.5, "• Player positions\n• Team assignments\n• Ball possession", 
               fontsize=8, va='center')
        ax.text(5, 6.5, "• Neighbor aggregation\n• Edge weight updates\n• Feature propagation", 
               fontsize=8, va='center')
        ax.text(5, 4.5, "• Graph-level features\n• Team formations\n• Tactical patterns", 
               fontsize=8, va='center')
        ax.text(5, 2.5, "• Next action prediction\n• Player movement\n• Game strategy", 
               fontsize=8, va='center')
        
        # Title
        ax.text(5, 9.5, "GNN Architecture for Basketball", ha='center', va='center', 
               fontsize=12, fontweight='bold')

def main():
    """Main function to create GNN concept visualization"""
    print("Creating Basketball GNN Concept Visualization...")
    
    visualizer = GNNConceptVisualizer()
    fig = visualizer.visualize_gnn_concepts()
    
    # Save the visualization
    output_path = "basketball_gnn_concepts.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"GNN concept visualization saved as: {output_path}")
    
    # Also create a simple diagram showing the complete system
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # System flow diagram
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Boxes for system components
    boxes = [
        {"pos": (1, 6), "size": (2, 1), "text": "Video Input\n(Basketball Game)", "color": "lightblue"},
        {"pos": (4, 6), "size": (2, 1), "text": "YOLO Detection\n(Custom Trained)", "color": "lightgreen"},
        {"pos": (7, 6), "size": (2, 1), "text": "Object Tracking\n(Players, Ball)", "color": "lightyellow"},
        {"pos": (10, 6), "size": (2, 1), "text": "GNN Analysis\n(Tactical Insights)", "color": "lightcoral"},
        {"pos": (4, 3), "size": (2, 1), "text": "Graph Construction\n(Nodes & Edges)", "color": "lightgray"},
        {"pos": (7, 3), "size": (2, 1), "text": "Message Passing\n(Feature Updates)", "color": "lavender"},
        {"pos": (10, 3), "size": (2, 1), "text": "Tactical Prediction\n(Next Actions)", "color": "lightpink"}
    ]
    
    for box in boxes:
        rect = FancyBboxPatch(box["pos"], box["size"][0], box["size"][1], 
                             boxstyle="round,pad=0.1", 
                             facecolor=box["color"], edgecolor='black')
        ax.add_patch(rect)
        ax.text(box["pos"][0] + box["size"][0]/2, box["pos"][1] + box["size"][1]/2, 
               box["text"], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrow_pairs = [
        ((3, 6.5), (4, 6.5)),    # Video → YOLO
        ((6, 6.5), (7, 6.5)),    # YOLO → Tracking
        ((9, 6.5), (10, 6.5)),   # Tracking → GNN
        ((5, 6), (5, 4)),        # YOLO → Graph Construction
        ((8, 6), (8, 4)),        # Tracking → Message Passing
        ((11, 6), (11, 4)),      # GNN → Prediction
        ((6, 3.5), (7, 3.5)),    # Graph → Message
        ((9, 3.5), (10, 3.5))    # Message → Prediction
    ]
    
    for start, end in arrow_pairs:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Title and description
    ax.text(7, 7.5, "Basketball GNN Analysis System Pipeline", 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.text(7, 1, "Complete end-to-end system: Video → Object Detection → Graph Neural Network → Tactical Analysis", 
           ha='center', va='center', fontsize=12, style='italic')
    
    # Save system diagram
    system_path = "basketball_gnn_system.png" 
    fig.savefig(system_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"System architecture diagram saved as: {system_path}")
    print("GNN visualization complete!")

if __name__ == "__main__":
    main()
