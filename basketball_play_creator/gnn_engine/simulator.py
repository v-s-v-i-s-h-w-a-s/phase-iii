"""
Graph Neural Network Engine for Basketball Play Simulation
Uses PyTorch Geometric to model player interactions and predict play outcomes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class PlayerRole(Enum):
    """Player roles in the play"""
    BALL_HANDLER = "ball_handler"
    SCREENER = "screener" 
    CUTTER = "cutter"
    SHOOTER = "shooter"
    DEFENDER = "defender"
    SUPPORT = "support"

@dataclass
class PlayerState:
    """Player state representation"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    role: PlayerRole
    has_ball: bool
    fatigue: float
    skill_level: float
    
@dataclass
class PlayState:
    """Complete play state"""
    players: List[PlayerState]
    ball_position: Tuple[float, float]
    timestamp: float
    score_probability: float
    play_success: bool
    
@dataclass
class SimulationResult:
    """Play simulation results"""
    states: List[PlayState]
    final_score_probability: float
    success_probability: float
    key_interactions: List[str]
    tactical_analysis: Dict[str, float]
    optimization_suggestions: List[str]

class BasketballGNN(nn.Module):
    """Graph Neural Network for basketball play modeling"""
    
    def __init__(self, node_features: int = 16, edge_features: int = 8, hidden_dim: int = 64, num_layers: int = 3):
        super(BasketballGNN, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Player encoding layers
        self.player_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        # Output layers
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # (x, y) coordinates
        )
        
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),  # 5 players
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.role_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(PlayerRole))
        )
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode player features
        x = self.player_encoder(x)
        
        # Apply graph convolutions with residual connections
        for conv in self.conv_layers:
            x_new = conv(x, edge_index)
            x = x + x_new  # Residual connection
            x = F.relu(x)
        
        # Predict positions
        positions = self.position_predictor(x)
        
        # Predict roles
        roles = self.role_predictor(x)
        
        # Global prediction (aggregate all players for success probability)
        global_features = global_mean_pool(x, batch)
        if global_features.size(1) < self.hidden_dim * 5:
            # Pad if we have fewer than 5 players
            padding = torch.zeros(global_features.size(0), self.hidden_dim * 5 - global_features.size(1))
            global_features = torch.cat([global_features, padding], dim=1)
        elif global_features.size(1) > self.hidden_dim * 5:
            # Truncate if we have more than 5 players
            global_features = global_features[:, :self.hidden_dim * 5]
            
        success_prob = self.success_predictor(global_features)
        
        return {
            'positions': positions,
            'roles': roles,
            'success_probability': success_prob
        }

class PlaySimulator:
    """Basketball play simulator using GNN"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the simulator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BasketballGNN().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        else:
            logger.info("Initialized simulator with untrained model")
        
        # Court dimensions (in pixels, assuming 940x500 court)
        self.court_width = 940
        self.court_height = 500
        
        # Basketball knowledge base
        self.position_weights = self._create_position_weights()
        self.interaction_rules = self._create_interaction_rules()
        
    def _create_position_weights(self) -> Dict[str, float]:
        """Create weights for different court positions"""
        return {
            'paint': 0.8,       # High-value area
            'free_throw': 0.6,   # Medium-value area
            'three_point': 0.7,  # High-value area for shooters
            'corner': 0.75,      # Corner three is valuable
            'baseline': 0.4,     # Lower value
            'perimeter': 0.5     # Medium value
        }
    
    def _create_interaction_rules(self) -> Dict[str, Dict[str, float]]:
        """Create rules for player interactions"""
        return {
            'screen_effectiveness': {
                'center_on_guard': 0.9,
                'forward_on_guard': 0.8,
                'guard_on_guard': 0.6,
                'guard_on_forward': 0.4
            },
            'passing_lanes': {
                'open': 0.9,
                'contested': 0.5,
                'blocked': 0.1
            },
            'shot_quality': {
                'wide_open': 0.8,
                'open': 0.6,
                'contested': 0.3,
                'heavily_contested': 0.1
            }
        }
    
    def simulate_play(self, players: List[PlayerState], actions: List[Any], duration: float = 10.0) -> SimulationResult:
        """
        Simulate a basketball play using GNN
        
        Args:
            players: List of player states
            actions: List of actions from parsed play
            duration: Play duration in seconds
            
        Returns:
            SimulationResult with complete simulation data
        """
        logger.info(f"Simulating play with {len(players)} players for {duration}s")
        
        # Initialize simulation
        states = []
        current_time = 0.0
        time_step = 0.5  # Simulate in 0.5 second intervals
        
        # Convert initial state to graph data
        graph_data = self._create_graph_data(players)
        
        while current_time < duration:
            # Run GNN prediction
            with torch.no_grad():
                predictions = self.model(graph_data)
            
            # Update player positions and roles
            new_players = self._update_players_from_predictions(players, predictions)
            
            # Calculate score probability
            score_prob = self._calculate_score_probability(new_players, current_time)
            
            # Create play state
            play_state = PlayState(
                players=new_players.copy(),
                ball_position=self._get_ball_position(new_players),
                timestamp=current_time,
                score_probability=score_prob,
                play_success=score_prob > 0.6
            )
            
            states.append(play_state)
            
            # Update for next iteration
            players = new_players
            graph_data = self._create_graph_data(players)
            current_time += time_step
        
        # Analyze simulation results
        final_score_prob = states[-1].score_probability if states else 0.0
        success_prob = self._calculate_success_probability(states)
        key_interactions = self._identify_key_interactions(states)
        tactical_analysis = self._analyze_tactics(states)
        optimization_suggestions = self._generate_optimizations(states)
        
        result = SimulationResult(
            states=states,
            final_score_probability=final_score_prob,
            success_probability=success_prob,
            key_interactions=key_interactions,
            tactical_analysis=tactical_analysis,
            optimization_suggestions=optimization_suggestions
        )
        
        logger.info(f"Simulation complete. Success probability: {success_prob:.2f}")
        return result
    
    def _create_graph_data(self, players: List[PlayerState]) -> Data:
        """Convert player states to PyTorch Geometric Data object"""
        # Node features: [x, y, vx, vy, has_ball, fatigue, skill, role_encoding...]
        node_features = []
        
        for player in players:
            # Position (normalized to 0-1)
            x = player.position[0] / self.court_width
            y = player.position[1] / self.court_height
            
            # Velocity (normalized)
            vx = player.velocity[0] / 10.0  # Assume max velocity of 10 units/s
            vy = player.velocity[1] / 10.0
            
            # Other features
            has_ball = 1.0 if player.has_ball else 0.0
            fatigue = player.fatigue
            skill = player.skill_level
            
            # One-hot encode role
            role_encoding = [0.0] * len(PlayerRole)
            for i, role in enumerate(PlayerRole):
                if role == player.role:
                    role_encoding[i] = 1.0
                    break
            
            # Combine features
            features = [x, y, vx, vy, has_ball, fatigue, skill] + role_encoding
            node_features.append(features)
        
        # Pad or truncate to ensure consistent feature size
        target_size = 16
        for i, features in enumerate(node_features):
            if len(features) < target_size:
                node_features[i] = features + [0.0] * (target_size - len(features))
            elif len(features) > target_size:
                node_features[i] = features[:target_size]
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edges (fully connected graph)
        num_nodes = len(players)
        edge_indices = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_indices.append([i, j])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Batch (single graph)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, batch=batch).to(self.device)
    
    def _update_players_from_predictions(self, players: List[PlayerState], predictions: Dict[str, torch.Tensor]) -> List[PlayerState]:
        """Update player states based on GNN predictions"""
        new_players = []
        
        # Get predictions
        positions = predictions['positions'].cpu().numpy()
        roles = torch.argmax(predictions['roles'], dim=1).cpu().numpy()
        
        for i, player in enumerate(players):
            # Update position based on prediction and physics
            pred_x = positions[i][0] * self.court_width
            pred_y = positions[i][1] * self.court_height
            
            # Apply physics constraints and smoothing
            new_x = self._apply_movement_constraints(player.position[0], pred_x)
            new_y = self._apply_movement_constraints(player.position[1], pred_y)
            
            # Update velocity
            new_vx = (new_x - player.position[0]) / 0.5
            new_vy = (new_y - player.position[1]) / 0.5
            
            # Update role based on prediction
            new_role = list(PlayerRole)[roles[i]] if i < len(roles) else player.role
            
            # Update fatigue (simple model)
            movement_distance = np.sqrt((new_x - player.position[0])**2 + (new_y - player.position[1])**2)
            fatigue_increase = movement_distance / 1000.0  # Fatigue increases with movement
            new_fatigue = min(player.fatigue + fatigue_increase, 1.0)
            
            # Determine ball possession (simple rule: closest player to ball)
            ball_pos = self._get_ball_position(players)
            distance_to_ball = np.sqrt((new_x - ball_pos[0])**2 + (new_y - ball_pos[1])**2)
            has_ball = distance_to_ball < 30.0  # Within 30 pixels
            
            new_player = PlayerState(
                position=(new_x, new_y),
                velocity=(new_vx, new_vy),
                role=new_role,
                has_ball=has_ball,
                fatigue=new_fatigue,
                skill_level=player.skill_level
            )
            
            new_players.append(new_player)
        
        return new_players
    
    def _apply_movement_constraints(self, current_pos: float, predicted_pos: float, max_speed: float = 50.0) -> float:
        """Apply realistic movement constraints"""
        max_change = max_speed * 0.5
        change = predicted_pos - current_pos
        
        # Limit change to max_speed
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
        
        return current_pos + change
    
    def _get_ball_position(self, players: List[PlayerState]) -> Tuple[float, float]:
        """Get current ball position based on player with ball"""
        for player in players:
            if player.has_ball:
                return player.position
        
        # If no player has ball, return center court
        return (self.court_width / 2, self.court_height / 2)
    
    def _calculate_score_probability(self, players: List[PlayerState], current_time: float) -> float:
        """Calculate probability of scoring based on current state"""
        ball_pos = self._get_ball_position(players)
        
        # Distance to basket (assume basket at (470, 50))
        basket_pos = (470, 50)
        distance_to_basket = np.sqrt((ball_pos[0] - basket_pos[0])**2 + (ball_pos[1] - basket_pos[1])**2)
        
        # Base probability based on distance
        max_distance = 400  # Max reasonable shooting distance
        distance_factor = max(0, (max_distance - distance_to_basket) / max_distance)
        
        # Position value factor
        position_factor = self._get_position_value(ball_pos)
        
        # Player skill factor
        ball_handler = next((p for p in players if p.has_ball), None)
        skill_factor = ball_handler.skill_level if ball_handler else 0.5
        
        # Time pressure factor (higher pressure near end)
        time_factor = 1.0 - (current_time / 10.0) * 0.2  # Slight decrease over time
        
        # Combine factors
        score_prob = distance_factor * position_factor * skill_factor * time_factor
        return min(max(score_prob, 0.0), 1.0)
    
    def _get_position_value(self, position: Tuple[float, float]) -> float:
        """Get value of a court position for scoring"""
        x, y = position
        
        # Paint area (high value)
        if 390 < x < 550 and y < 200:
            return self.position_weights['paint']
        
        # Three-point corner (high value)
        if (x < 100 or x > 840) and y < 150:
            return self.position_weights['corner']
        
        # Three-point arc (good value)
        center_distance = np.sqrt((x - 470)**2 + (y - 50)**2)
        if center_distance > 237:  # Three-point line distance
            return self.position_weights['three_point']
        
        # Free throw area
        if 420 < x < 520 and 150 < y < 250:
            return self.position_weights['free_throw']
        
        return self.position_weights['perimeter']
    
    def _calculate_success_probability(self, states: List[PlayState]) -> float:
        """Calculate overall play success probability"""
        if not states:
            return 0.0
        
        # Average score probability over time
        avg_score_prob = np.mean([state.score_probability for state in states])
        
        # Final state weight (more important)
        final_weight = 0.6
        avg_weight = 0.4
        
        final_score_prob = states[-1].score_probability
        
        return final_weight * final_score_prob + avg_weight * avg_score_prob
    
    def _identify_key_interactions(self, states: List[PlayState]) -> List[str]:
        """Identify key interactions during the play"""
        interactions = []
        
        # Screen interactions
        for i, state in enumerate(states[1:], 1):
            # Check for screens (players getting close)
            for j, player1 in enumerate(state.players):
                for k, player2 in enumerate(state.players[j+1:], j+1):
                    distance = np.sqrt(
                        (player1.position[0] - player2.position[0])**2 + 
                        (player1.position[1] - player2.position[1])**2
                    )
                    if distance < 40 and player1.role == PlayerRole.SCREENER:
                        interactions.append(f"Screen set at {state.timestamp:.1f}s")
                        break
        
        # Ball movement
        ball_movements = 0
        for i, state in enumerate(states[1:], 1):
            prev_ball_pos = self._get_ball_position(states[i-1].players)
            curr_ball_pos = self._get_ball_position(state.players)
            
            ball_distance = np.sqrt(
                (curr_ball_pos[0] - prev_ball_pos[0])**2 + 
                (curr_ball_pos[1] - prev_ball_pos[1])**2
            )
            
            if ball_distance > 50:  # Significant ball movement
                ball_movements += 1
        
        if ball_movements > 2:
            interactions.append(f"Good ball movement ({ball_movements} passes)")
        
        return interactions[:5]  # Return top 5 interactions
    
    def _analyze_tactics(self, states: List[PlayState]) -> Dict[str, float]:
        """Analyze tactical aspects of the play"""
        if not states:
            return {}
        
        analysis = {}
        
        # Ball movement analysis
        ball_positions = [self._get_ball_position(state.players) for state in states]
        ball_movement = sum(
            np.sqrt((ball_positions[i][0] - ball_positions[i-1][0])**2 + 
                   (ball_positions[i][1] - ball_positions[i-1][1])**2)
            for i in range(1, len(ball_positions))
        )
        analysis['ball_movement'] = min(ball_movement / 500.0, 1.0)  # Normalize
        
        # Player spacing analysis
        avg_spacing = []
        for state in states:
            distances = []
            for i, p1 in enumerate(state.players):
                for p2 in state.players[i+1:]:
                    dist = np.sqrt((p1.position[0] - p2.position[0])**2 + 
                                 (p1.position[1] - p2.position[1])**2)
                    distances.append(dist)
            if distances:
                avg_spacing.append(np.mean(distances))
        
        analysis['spacing'] = min(np.mean(avg_spacing) / 200.0, 1.0) if avg_spacing else 0.5
        
        # Tempo analysis
        total_movement = sum(
            sum(np.sqrt(p.velocity[0]**2 + p.velocity[1]**2) for p in state.players)
            for state in states
        )
        analysis['tempo'] = min(total_movement / (len(states) * len(states[0].players) * 10), 1.0)
        
        return analysis
    
    def _generate_optimizations(self, states: List[PlayState]) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        if not states:
            return suggestions
        
        # Analyze final success
        final_success = states[-1].score_probability
        
        if final_success < 0.4:
            suggestions.append("Consider better spacing to create open shots")
            suggestions.append("Add more ball movement to find better opportunities")
        
        # Analyze player roles
        role_distribution = {}
        for state in states[-3:]:  # Look at final 3 states
            for player in state.players:
                role_distribution[player.role] = role_distribution.get(player.role, 0) + 1
        
        if PlayerRole.SCREENER not in role_distribution:
            suggestions.append("Add screening action to create better scoring opportunities")
        
        if PlayerRole.CUTTER not in role_distribution:
            suggestions.append("Include cutting movement to stress the defense")
        
        # Analyze timing
        peak_probability = max(state.score_probability for state in states)
        peak_time = next(state.timestamp for state in states if state.score_probability == peak_probability)
        
        if peak_time < len(states) * 0.3:  # Peak too early
            suggestions.append("Extend the play to create better late-game opportunities")
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def save_model(self, model_path: str):
        """Save current model"""
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

# Example usage
if __name__ == "__main__":
    # Test the GNN simulator
    simulator = PlaySimulator()
    
    # Create test players
    test_players = [
        PlayerState(
            position=(470, 400),
            velocity=(0, 0),
            role=PlayerRole.BALL_HANDLER,
            has_ball=True,
            fatigue=0.0,
            skill_level=0.8
        ),
        PlayerState(
            position=(600, 350),
            velocity=(0, 0),
            role=PlayerRole.SHOOTER,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.7
        ),
        PlayerState(
            position=(340, 350),
            velocity=(0, 0),
            role=PlayerRole.CUTTER,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.6
        ),
        PlayerState(
            position=(470, 200),
            velocity=(0, 0),
            role=PlayerRole.SCREENER,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.7
        ),
        PlayerState(
            position=(550, 300),
            velocity=(0, 0),
            role=PlayerRole.SUPPORT,
            has_ball=False,
            fatigue=0.0,
            skill_level=0.6
        )
    ]
    
    # Run simulation
    result = simulator.simulate_play(test_players, [], duration=8.0)
    
    print("Simulation Results:")
    print(f"- Final Score Probability: {result.final_score_probability:.2f}")
    print(f"- Success Probability: {result.success_probability:.2f}")
    print(f"- States Generated: {len(result.states)}")
    print(f"- Key Interactions: {result.key_interactions}")
    print(f"- Tactical Analysis: {result.tactical_analysis}")
    print(f"- Optimization Suggestions: {result.optimization_suggestions}")
