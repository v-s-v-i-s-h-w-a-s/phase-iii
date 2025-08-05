"""
Visualization module for basketball GNN analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import torch


class BasketballGraphVisualizer:
    """
    Visualizer for basketball player graphs and GNN analysis results.
    """
    
    def __init__(self, court_width: float = 940.0, court_height: float = 500.0):
        """
        Args:
            court_width: Basketball court width in pixels
            court_height: Basketball court height in pixels
        """
        self.court_width = court_width
        self.court_height = court_height
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def draw_court(self, ax, show_details: bool = True):
        """
        Draw basketball court outline and features.
        
        Args:
            ax: Matplotlib axis
            show_details: Whether to show court details (center circle, etc.)
        """
        
        # Court outline
        court_rect = Rectangle((0, 0), self.court_width, self.court_height, 
                              linewidth=3, edgecolor='black', facecolor='lightgreen', alpha=0.3)
        ax.add_patch(court_rect)
        
        if show_details:
            # Center line
            ax.plot([self.court_width/2, self.court_width/2], [0, self.court_height], 
                   'k--', linewidth=2, alpha=0.7)
            
            # Center circle
            center_circle = Circle((self.court_width/2, self.court_height/2), 50, 
                                 linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(center_circle)
            
            # Three-point lines (simplified)
            # Left side
            left_arc = Circle((0, self.court_height/2), 150, 
                            linewidth=2, edgecolor='blue', facecolor='none', alpha=0.6)
            ax.add_patch(left_arc)
            
            # Right side
            right_arc = Circle((self.court_width, self.court_height/2), 150, 
                             linewidth=2, edgecolor='blue', facecolor='none', alpha=0.6)
            ax.add_patch(right_arc)
            
            # Free throw circles
            left_ft = Circle((94, self.court_height/2), 40, 
                           linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
            right_ft = Circle((self.court_width - 94, self.court_height/2), 40, 
                            linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
            ax.add_patch(left_ft)
            ax.add_patch(right_ft)
        
        ax.set_xlim(-20, self.court_width + 20)
        ax.set_ylim(-20, self.court_height + 20)
        ax.set_aspect('equal')
        ax.set_xlabel('Court X (pixels)')
        ax.set_ylabel('Court Y (pixels)')
    
    def visualize_graph(self, 
                       graph_data,
                       cluster_labels: Optional[np.ndarray] = None,
                       show_edges: bool = True,
                       show_court: bool = True,
                       title: str = "Basketball Player Graph",
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualize a single graph with players and connections.
        
        Args:
            graph_data: PyTorch Geometric Data object
            cluster_labels: Cluster assignments for nodes
            show_edges: Whether to show graph edges
            show_court: Whether to show court outline
            title: Plot title
            ax: Matplotlib axis (creates new if None)
            
        Returns:
            Matplotlib axis object
        """
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Draw court
        if show_court:
            self.draw_court(ax)
        
        # Extract data
        positions = graph_data.pos.cpu().numpy()
        edge_index = graph_data.edge_index.cpu().numpy()
        
        # Set up colors
        if cluster_labels is not None:
            colors = plt.cm.Set1(cluster_labels / max(cluster_labels) if max(cluster_labels) > 0 else 0)
        else:
            colors = ['red'] * len(positions)
        
        # Draw edges
        if show_edges and edge_index.shape[1] > 0:
            for i in range(edge_index.shape[1]):
                node1, node2 = edge_index[0, i], edge_index[1, i]
                if node1 < len(positions) and node2 < len(positions):
                    x_coords = [positions[node1, 0], positions[node2, 0]]
                    y_coords = [positions[node1, 1], positions[node2, 1]]
                    ax.plot(x_coords, y_coords, 'gray', alpha=0.5, linewidth=1)
        
        # Draw players
        for i, (x, y) in enumerate(positions):
            color = colors[i] if cluster_labels is not None else 'red'
            ax.scatter(x, y, c=[color], s=200, alpha=0.8, edgecolor='black', linewidth=2)
            ax.text(x, y + 15, str(i), ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add cluster legend
        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            legend_elements = []
            for label in unique_labels:
                color = plt.cm.Set1(label / max(cluster_labels) if max(cluster_labels) > 0 else 0)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=f'Team {label + 1}'))
            ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return ax
    
    def visualize_sequence(self, 
                          graphs: List,
                          cluster_labels_list: Optional[List[np.ndarray]] = None,
                          max_frames: int = 8,
                          save_path: Optional[str] = None):
        """
        Visualize a sequence of graphs.
        
        Args:
            graphs: List of graph data objects
            cluster_labels_list: List of cluster labels for each graph
            max_frames: Maximum number of frames to show
            save_path: Path to save the plot
        """
        
        num_frames = min(len(graphs), max_frames)
        cols = 4
        rows = (num_frames + cols - 1) // cols
        
        _, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx in range(num_frames):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            graph = graphs[idx]
            labels = cluster_labels_list[idx] if cluster_labels_list else None
            
            frame_id = getattr(graph, 'frame_id', idx)
            title = f'Frame {frame_id}'
            
            self.visualize_graph(graph, labels, show_edges=True, title=title, ax=ax)
        
        # Hide empty subplots
        for idx in range(num_frames, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sequence visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_embeddings(self, 
                           embeddings: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           method: str = "pca",
                           title: str = "Player Embeddings",
                           save_path: Optional[str] = None):
        """
        Visualize high-dimensional embeddings in 2D.
        
        Args:
            embeddings: Embedding vectors [num_players, embedding_dim]
            labels: Cluster labels
            method: Dimensionality reduction method ("pca", "tsne")
            title: Plot title
            save_path: Path to save the plot
        """
        
        # Dimensionality reduction
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[colors[i]], label=f'Team {label + 1}', s=100, alpha=0.7)
            
            ax.legend()
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Embedding visualization saved to {save_path}")
        
        plt.show()
    
    def plot_formation_analysis(self, 
                              analysis_results: Dict,
                              save_path: Optional[str] = None):
        """
        Plot formation analysis results over time.
        
        Args:
            analysis_results: Results from formation analysis
            save_path: Path to save the plot
        """
        
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Formation stability
        if 'formation_stability' in analysis_results:
            axes[0, 0].plot(analysis_results['formation_stability'], 'b-', linewidth=2)
            axes[0, 0].set_title('Formation Stability (Silhouette Score)')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('Silhouette Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Cluster compactness
        if 'cluster_compactness' in analysis_results:
            axes[0, 1].plot(analysis_results['cluster_compactness'], 'r-', linewidth=2)
            axes[0, 1].set_title('Cluster Compactness')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Average Intra-cluster Distance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Spatial distribution
        if 'spatial_distribution' in analysis_results:
            spatial_data = analysis_results['spatial_distribution']
            x_spreads = [x for x, y in spatial_data]
            y_spreads = [y for x, y in spatial_data]
            
            axes[1, 0].plot(x_spreads, 'g-', linewidth=2, label='X spread')
            axes[1, 0].plot(y_spreads, 'm-', linewidth=2, label='Y spread')
            axes[1, 0].set_title('Team Spatial Distribution')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Standard Deviation')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Team centroid movement
        if 'team_centroids' in analysis_results:
            centroid_data = analysis_results['team_centroids']
            
            # Plot centroid trajectories for each team
            if centroid_data and len(centroid_data[0]) > 0:
                num_teams = len(centroid_data[0])
                colors = plt.cm.Set1(np.linspace(0, 1, num_teams))
                
                for team_id in range(num_teams):
                    x_coords = []
                    y_coords = []
                    
                    for frame_centroids in centroid_data:
                        if team_id < len(frame_centroids):
                            x_coords.append(frame_centroids[team_id][0])
                            y_coords.append(frame_centroids[team_id][1])
                    
                    if x_coords and y_coords:
                        axes[1, 1].plot(x_coords, y_coords, 'o-', color=colors[team_id], 
                                      linewidth=2, markersize=4, label=f'Team {team_id + 1}')
                
                axes[1, 1].set_title('Team Centroid Trajectories')
                axes[1, 1].set_xlabel('X Coordinate')
                axes[1, 1].set_ylabel('Y Coordinate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Formation analysis plot saved to {save_path}")
        
        plt.show()
    
    def create_animation(self, 
                        graphs: List,
                        cluster_labels_list: Optional[List[np.ndarray]] = None,
                        save_path: Optional[str] = None,
                        interval: int = 500,
                        show_trails: bool = True) -> FuncAnimation:
        """
        Create animated visualization of player movement and clustering.
        
        Args:
            graphs: List of graph data objects
            cluster_labels_list: List of cluster labels for each graph
            save_path: Path to save animation (e.g., 'animation.gif')
            interval: Time between frames in milliseconds
            show_trails: Whether to show player movement trails
            
        Returns:
            FuncAnimation object
        """
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Initialize empty plots
        
        def init():
            self.draw_court(ax)
            return []
        
        def animate(frame_idx):
            ax.clear()
            self.draw_court(ax)
            
            if frame_idx >= len(graphs):
                return []
            
            graph = graphs[frame_idx]
            positions = graph.pos.cpu().numpy()
            labels = cluster_labels_list[frame_idx] if cluster_labels_list else None
            
            # Set up colors
            if labels is not None:
                colors = plt.cm.Set1(labels / max(labels) if max(labels) > 0 else 0)
            else:
                colors = ['red'] * len(positions)
            
            # Draw trails
            if show_trails and frame_idx > 0:
                trail_length = min(5, frame_idx)
                for i in range(len(positions)):
                    trail_x = []
                    trail_y = []
                    
                    for j in range(max(0, frame_idx - trail_length), frame_idx + 1):
                        if j < len(graphs):
                            past_pos = graphs[j].pos.cpu().numpy()
                            if i < len(past_pos):
                                trail_x.append(past_pos[i, 0])
                                trail_y.append(past_pos[i, 1])
                    
                    if len(trail_x) > 1:
                        ax.plot(trail_x, trail_y, 'gray', alpha=0.5, linewidth=1)
            
            # Draw current positions
            for i, (x, y) in enumerate(positions):
                color = colors[i] if labels is not None else 'red'
                ax.scatter(x, y, c=[color], s=200, alpha=0.8, edgecolor='black', linewidth=2)
                ax.text(x, y + 15, str(i), ha='center', va='center', fontsize=10, fontweight='bold')
            
            frame_id = getattr(graph, 'frame_id', frame_idx)
            ax.set_title(f'Basketball Player Movement - Frame {frame_id}', fontsize=14, fontweight='bold')
            
            return []
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(graphs), 
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
            else:
                print("Unsupported format. Use .gif or .mp4")
            
            print(f"Animation saved to {save_path}")
        
        return anim


def demo_visualization():
    """Demonstrate visualization capabilities."""
    
    print("Running visualization demo...")
    
    # Create dummy data
    import sys
    sys.path.append('..')
    from graph_builder.build_graph import BasketballGraphBuilder, create_dummy_tracking_data
    
    # Generate data
    tracking_data = create_dummy_tracking_data(num_frames=10, num_players=8)
    builder = BasketballGraphBuilder()
    graphs = builder.build_sequence_graphs(tracking_data)
    
    # Create dummy cluster labels
    rng = np.random.default_rng(42)
    cluster_labels_list = []
    for graph in graphs:
        num_players = graph.pos.shape[0]
        labels = rng.integers(0, 2, num_players)
        cluster_labels_list.append(labels)
    
    # Initialize visualizer
    visualizer = BasketballGraphVisualizer()
    
    # Single graph visualization
    if graphs:
        print("Visualizing single graph...")
        visualizer.visualize_graph(graphs[0], cluster_labels_list[0], title="Demo Graph")
    
    # Sequence visualization
    print("Visualizing sequence...")
    visualizer.visualize_sequence(graphs[:6], cluster_labels_list[:6])
    
    # Embedding visualization (dummy)
    if graphs:
        print("Visualizing embeddings...")
        rng = np.random.default_rng(42)
        dummy_embeddings = rng.standard_normal((graphs[0].pos.shape[0], 16))
        visualizer.visualize_embeddings(dummy_embeddings, cluster_labels_list[0])
    
    print("Visualization demo completed!")


if __name__ == "__main__":
    demo_visualization()
