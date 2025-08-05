import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import math
import random
from collections import deque

class BasketballGNNDemo:
    def __init__(self):
        # Court dimensions (scaled for visualization)
        self.court_width = 800
        self.court_height = 400
        self.fps = 10
        
        # Initialize players (10 players - 5 per team)
        self.players = self.initialize_players()
        self.referees = self.initialize_referees()
        self.ball = {"x": 400, "y": 200, "with_player": None}
        self.baskets = [{"x": 50, "y": 200}, {"x": 750, "y": 200}]
        
        # Game state
        self.current_play = 0
        self.frame_count = 0
        self.play_frames = []
        self.score = {"team_a": 0, "team_b": 0}
        
        # GNN visualization data
        self.interaction_graph = nx.Graph()
        self.tactical_analysis = {
            "formation": "Transition",
            "ball_possession": "Team A",
            "threat_level": 0.0,
            "passing_network": [],
            "defensive_pressure": 0.0
        }
        
        # Generate play sequences
        self.generate_play_sequences()
        
    def initialize_players(self):
        """Initialize 10 players (5 per team) with starting positions"""
        players = []
        
        # Team A (Blue) - Left side
        team_a_positions = [
            (150, 180), (120, 150), (120, 250), (180, 120), (180, 280)
        ]
        for i, (x, y) in enumerate(team_a_positions):
            players.append({
                "id": f"A{i+1}",
                "team": "A",
                "x": x, "y": y,
                "target_x": x, "target_y": y,
                "role": ["PG", "SG", "SF", "PF", "C"][i],
                "color": "blue",
                "has_ball": i == 0
            })
        
        # Team B (Red) - Right side
        team_b_positions = [
            (650, 220), (680, 150), (680, 250), (620, 120), (620, 280)
        ]
        for i, (x, y) in enumerate(team_b_positions):
            players.append({
                "id": f"B{i+1}",
                "team": "B",
                "x": x, "y": y,
                "target_x": x, "target_y": y,
                "role": ["PG", "SG", "SF", "PF", "C"][i],
                "color": "red",
                "has_ball": False
            })
        
        return players
    
    def initialize_referees(self):
        """Initialize 3 referees"""
        return [
            {"id": "R1", "x": 200, "y": 100, "target_x": 200, "target_y": 100},
            {"id": "R2", "x": 400, "y": 350, "target_x": 400, "target_y": 350},
            {"id": "R3", "x": 600, "y": 100, "target_x": 600, "target_y": 100}
        ]
    
    def generate_play_sequences(self):
        """Generate 4-5 basketball plays with transitions"""
        plays = [
            self.generate_fast_break_play(),
            self.generate_pick_and_roll_play(),
            self.generate_three_point_play(),
            self.generate_defensive_stop_play(),
            self.generate_transition_play()
        ]
        
        # Each play lasts about 10 seconds at 10 FPS = 100 frames
        total_frames = 0
        for play in plays:
            play_duration = len(play)
            self.play_frames.extend(play)
            total_frames += play_duration
        
        print(f"Generated {len(plays)} plays with {total_frames} total frames")
    
    def generate_fast_break_play(self):
        """Generate a fast break play sequence"""
        frames = []
        
        # Setup: Team A gets ball, runs fast break
        for frame in range(80):
            frame_data = {
                "players": [],
                "ball": {"x": 400, "y": 200, "with_player": "A1"},
                "play_type": "Fast Break",
                "description": f"Fast break in progress (frame {frame}/80)"
            }
            
            # Team A players run forward
            for i, player in enumerate(self.players[:5]):
                if i == 0:  # Point guard with ball
                    progress = frame / 80.0
                    x = 150 + (600 * progress)
                    y = 200 + 30 * math.sin(progress * math.pi)
                else:
                    progress = frame / 80.0
                    x = player["x"] + (500 * progress)
                    y = player["y"] + random.randint(-20, 20)
                
                frame_data["players"].append({
                    **player,
                    "x": max(50, min(750, x)),
                    "y": max(50, min(350, y))
                })
            
            # Team B players run back on defense
            for i, player in enumerate(self.players[5:]):
                progress = frame / 80.0
                x = player["x"] - (300 * progress)
                y = player["y"] + random.randint(-15, 15)
                
                frame_data["players"].append({
                    **player,
                    "x": max(50, min(750, x)),
                    "y": max(50, min(350, y))
                })
            
            # Update ball position
            progress = frame / 80.0
            frame_data["ball"]["x"] = 150 + (600 * progress)
            frame_data["ball"]["y"] = 200 + 30 * math.sin(progress * math.pi)
            
            frames.append(frame_data)
        
        # Add shot attempt
        for frame in range(20):
            frame_data = frames[-1].copy()
            frame_data["description"] = f"Shot attempt (frame {frame}/20)"
            frame_data["ball"]["with_player"] = None
            frame_data["ball"]["x"] = 700 + (frame * 2)
            frame_data["ball"]["y"] = 200 - (frame * 3)
            frames.append(frame_data)
        
        return frames
    
    def generate_pick_and_roll_play(self):
        """Generate a pick and roll play"""
        frames = []
        
        for frame in range(100):
            frame_data = {
                "players": [],
                "ball": {"x": 300, "y": 200, "with_player": "A1"},
                "play_type": "Pick and Roll",
                "description": f"Pick and roll execution (frame {frame}/100)"
            }
            
            # Simulate pick and roll movement
            progress = frame / 100.0
            
            for i, player in enumerate(self.players):
                if player["team"] == "A":
                    if i == 0:  # Ball handler
                        x = 300 + (100 * math.sin(progress * math.pi))
                        y = 200 + (50 * math.cos(progress * math.pi))
                    elif i == 4:  # Screen setter
                        x = 320 + (80 * progress)
                        y = 180 + (40 * progress)
                    else:
                        x = player["x"] + random.randint(-10, 10)
                        y = player["y"] + random.randint(-10, 10)
                else:
                    # Defense adjusts
                    x = player["x"] + random.randint(-15, 15)
                    y = player["y"] + random.randint(-15, 15)
                
                frame_data["players"].append({
                    **player,
                    "x": max(50, min(750, x)),
                    "y": max(50, min(350, y))
                })
            
            frames.append(frame_data)
        
        return frames
    
    def generate_three_point_play(self):
        """Generate a three-point shot play"""
        frames = []
        
        for frame in range(60):
            frame_data = {
                "players": [],
                "ball": {"x": 550, "y": 150, "with_player": "A2"},
                "play_type": "3-Point Shot",
                "description": f"Three-point attempt (frame {frame}/60)"
            }
            
            # Ball movement for 3-point shot
            if frame < 40:
                frame_data["ball"]["with_player"] = "A2"
                frame_data["ball"]["x"] = 550
                frame_data["ball"]["y"] = 150
            else:
                # Shot in air
                shot_progress = (frame - 40) / 20.0
                frame_data["ball"]["with_player"] = None
                frame_data["ball"]["x"] = 550 + (200 * shot_progress)
                frame_data["ball"]["y"] = 150 + (50 * math.sin(shot_progress * math.pi))
            
            # Players maintain spacing
            for player in self.players:
                frame_data["players"].append({
                    **player,
                    "x": player["x"] + random.randint(-5, 5),
                    "y": player["y"] + random.randint(-5, 5)
                })
            
            frames.append(frame_data)
        
        return frames
    
    def generate_defensive_stop_play(self):
        """Generate a defensive stop and turnover"""
        frames = []
        
        for frame in range(70):
            frame_data = {
                "players": [],
                "ball": {"x": 400, "y": 200, "with_player": "B1" if frame < 50 else "A3"},
                "play_type": "Defensive Stop",
                "description": f"Defensive pressure and steal (frame {frame}/70)"
            }
            
            # Simulate defensive pressure and steal
            for i, player in enumerate(self.players):
                if player["team"] == "A" and frame >= 50:
                    # Team A gets more aggressive after steal
                    x = player["x"] + random.randint(-20, 20)
                    y = player["y"] + random.randint(-20, 20)
                else:
                    x = player["x"] + random.randint(-10, 10)
                    y = player["y"] + random.randint(-10, 10)
                
                frame_data["players"].append({
                    **player,
                    "x": max(50, min(750, x)),
                    "y": max(50, min(350, y))
                })
            
            frames.append(frame_data)
        
        return frames
    
    def generate_transition_play(self):
        """Generate a transition play"""
        frames = []
        
        for frame in range(90):
            frame_data = {
                "players": [],
                "ball": {"x": 200, "y": 250, "with_player": "A1"},
                "play_type": "Transition",
                "description": f"Transition offense (frame {frame}/90)"
            }
            
            # Transition movement
            progress = frame / 90.0
            
            for player in self.players:
                if player["team"] == "A":
                    # Offense moves up court
                    x = player["x"] + (300 * progress)
                    y = player["y"] + 20 * math.sin(progress * 2 * math.pi)
                else:
                    # Defense retreats
                    x = player["x"] - (200 * progress)
                    y = player["y"] + 15 * math.cos(progress * 2 * math.pi)
                
                frame_data["players"].append({
                    **player,
                    "x": max(50, min(750, x)),
                    "y": max(50, min(350, y))
                })
            
            frames.append(frame_data)
        
        return frames
    
    def update_gnn_analysis(self, frame_data):
        """Update GNN tactical analysis based on current frame"""
        players_data = frame_data["players"]
        ball_data = frame_data["ball"]
        
        # Build interaction graph
        self.interaction_graph.clear()
        
        # Add player nodes
        for player in players_data:
            self.interaction_graph.add_node(
                player["id"],
                team=player["team"],
                position=(player["x"], player["y"]),
                role=player["role"]
            )
        
        # Add edges based on proximity and game context
        for i, p1 in enumerate(players_data):
            for j, p2 in enumerate(players_data[i+1:], i+1):
                distance = math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
                
                # Different edge types based on distance and team
                if distance < 80:
                    if p1["team"] == p2["team"]:
                        edge_type = "teammate_close"
                        weight = 1.0 - (distance / 80)
                    else:
                        edge_type = "opponent_pressure"
                        weight = 1.0 - (distance / 80)
                    
                    self.interaction_graph.add_edge(
                        p1["id"], p2["id"],
                        weight=weight,
                        edge_type=edge_type,
                        distance=distance
                    )
        
        # Update tactical analysis
        team_a_players = [p for p in players_data if p["team"] == "A"]
        team_b_players = [p for p in players_data if p["team"] == "B"]
        
        # Calculate formation compactness
        def calculate_compactness(team_players):
            if len(team_players) < 2:
                return 0
            
            center_x = sum(p["x"] for p in team_players) / len(team_players)
            center_y = sum(p["y"] for p in team_players) / len(team_players)
            
            distances = [
                math.sqrt((p["x"] - center_x)**2 + (p["y"] - center_y)**2)
                for p in team_players
            ]
            return sum(distances) / len(distances)
        
        # Update analysis
        ball_with_player = ball_data.get("with_player", "")
        possession = "Team A" if ball_with_player and ball_with_player.startswith("A") else "Team B" if ball_with_player and ball_with_player.startswith("B") else "Loose Ball"
        
        self.tactical_analysis.update({
            "formation": frame_data.get("play_type", "Unknown"),
            "ball_possession": possession,
            "threat_level": min(1.0, ball_data["x"] / 750),  # Closer to basket = higher threat
            "team_a_compactness": calculate_compactness(team_a_players),
            "team_b_compactness": calculate_compactness(team_b_players),
            "defensive_pressure": len([e for e in self.interaction_graph.edges(data=True) 
                                     if e[2]["edge_type"] == "opponent_pressure"]) / 10.0
        })
    
    def draw_court(self, ax):
        """Draw basketball court"""
        ax.clear()
        ax.set_xlim(0, self.court_width)
        ax.set_ylim(0, self.court_height)
        ax.set_aspect('equal')
        
        # Court outline
        court = patches.Rectangle((25, 25), 750, 350, linewidth=3, 
                                edgecolor='black', facecolor='lightgreen', alpha=0.3)
        ax.add_patch(court)
        
        # Center line
        ax.plot([400, 400], [25, 375], 'k-', linewidth=2)
        
        # Center circle
        center_circle = patches.Circle((400, 200), 50, linewidth=2, 
                                     edgecolor='black', facecolor='none')
        ax.add_patch(center_circle)
        
        # Left basket area
        left_key = patches.Rectangle((25, 150), 100, 100, linewidth=2,
                                   edgecolor='black', facecolor='none')
        ax.add_patch(left_key)
        
        # Right basket area  
        right_key = patches.Rectangle((675, 150), 100, 100, linewidth=2,
                                    edgecolor='black', facecolor='none')
        ax.add_patch(right_key)
        
        # Baskets
        for basket in self.baskets:
            basket_circle = patches.Circle((basket["x"], basket["y"]), 10, 
                                         facecolor='orange', edgecolor='black')
            ax.add_patch(basket_circle)
        
        # Three-point lines (simplified)
        ax.plot([25, 25, 200, 200], [75, 125, 125, 75], 'k-', linewidth=2)
        ax.plot([25, 25, 200, 200], [325, 275, 275, 325], 'k-', linewidth=2)
        ax.plot([775, 775, 600, 600], [75, 125, 125, 75], 'k-', linewidth=2)
        ax.plot([775, 775, 600, 600], [325, 275, 275, 325], 'k-', linewidth=2)
    
    def draw_players_and_ball(self, ax, frame_data):
        """Draw players, ball, and referees"""
        # Draw players
        for player in frame_data["players"]:
            color = 'blue' if player["team"] == "A" else 'red'
            size = 120 if frame_data["ball"].get("with_player") == player["id"] else 80
            
            ax.scatter(player["x"], player["y"], c=color, s=size, 
                      alpha=0.8, edgecolors='black', linewidth=2)
            ax.text(player["x"], player["y"]-20, player["id"], 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw ball
        ball = frame_data["ball"]
        ball_color = 'orange' if ball.get("with_player") else 'darkorange'
        ax.scatter(ball["x"], ball["y"], c=ball_color, s=60, 
                  marker='o', edgecolors='black', linewidth=2)
        
        # Draw referees (simplified - just show 3 referee positions)
        ref_positions = [(200, 100), (400, 350), (600, 100)]
        for i, (x, y) in enumerate(ref_positions):
            ax.scatter(x, y, c='gray', s=40, marker='^', 
                      edgecolors='black', alpha=0.7)
            ax.text(x, y-15, f"R{i+1}", ha='center', va='center', 
                   fontsize=6, color='black')
    
    def draw_gnn_overlay(self, ax, frame_data):
        """Draw GNN analysis overlay"""
        # Update GNN analysis
        self.update_gnn_analysis(frame_data)
        
        # Draw interaction graph edges
        for edge in self.interaction_graph.edges(data=True):
            node1, node2, data = edge
            
            # Get node positions
            pos1 = self.interaction_graph.nodes[node1]["position"]
            pos2 = self.interaction_graph.nodes[node2]["position"]
            
            # Draw edge based on type
            if data["edge_type"] == "teammate_close":
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'g-', alpha=0.6, linewidth=2)
            elif data["edge_type"] == "opponent_pressure":
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'r--', alpha=0.4, linewidth=1)
        
        # Draw GNN analysis text
        analysis_text = f"""GNN Tactical Analysis:
Formation: {self.tactical_analysis['formation']}
Ball Possession: {self.tactical_analysis['ball_possession']}
Threat Level: {self.tactical_analysis['threat_level']:.2f}
Defensive Pressure: {self.tactical_analysis['defensive_pressure']:.2f}
Play: {frame_data.get('description', 'N/A')}"""
        
        ax.text(50, 50, analysis_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
               verticalalignment='bottom')
        
        # Draw possession indicator
        possession_color = 'blue' if self.tactical_analysis['ball_possession'] == 'Team A' else 'red'
        ax.text(650, 50, f"POSSESSION: {self.tactical_analysis['ball_possession']}", 
               fontsize=12, fontweight='bold', color=possession_color,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=possession_color, alpha=0.3))
    
    def animate_frame(self, frame_num):
        """Animation function for each frame"""
        if frame_num >= len(self.play_frames):
            return
        
        frame_data = self.play_frames[frame_num]
        
        # Create subplots
        fig = plt.gcf()
        fig.clear()
        
        # Main court view
        ax1 = fig.add_subplot(111)
        
        # Draw everything
        self.draw_court(ax1)
        self.draw_players_and_ball(ax1, frame_data)
        self.draw_gnn_overlay(ax1, frame_data)
        
        # Set title
        ax1.set_title(f"Basketball GNN Analysis Demo - Frame {frame_num+1}/{len(self.play_frames)}\n"
                     f"Play Type: {frame_data.get('play_type', 'Unknown')}", 
                     fontsize=14, fontweight='bold')
        
        ax1.set_xlabel("Court Width", fontsize=10)
        ax1.set_ylabel("Court Height", fontsize=10)
        
        plt.tight_layout()
    
    def create_demo_video(self, output_path="basketball_gnn_demo.gif"):
        """Create the demo video"""
        print("Creating Basketball GNN Demo Video...")
        
        # Setup figure
        fig = plt.figure(figsize=(16, 10))
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            self.animate_frame,
            frames=len(self.play_frames),
            interval=100,  # 100ms between frames = 10 FPS
            repeat=True,
            blit=False
        )
        
        # Save as GIF
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=self.fps)
        anim.save(output_path, writer=writer)
        
        print(f"Demo video created successfully: {output_path}")
        print(f"Total frames: {len(self.play_frames)}")
        print(f"Duration: {len(self.play_frames) / self.fps:.1f} seconds")
        
        # Also save as MP4 if opencv is available
        try:
            mp4_path = output_path.replace('.gif', '.mp4')
            anim.save(mp4_path, writer='ffmpeg', fps=self.fps)
            print(f"MP4 version saved: {mp4_path}")
        except:
            print("MP4 save failed - ffmpeg not available")
        
        plt.close(fig)
        return output_path

def main():
    """Main function to run the demo"""
    print("Basketball GNN Demo Generator")
    print("="*50)
    
    # Create demo instance
    demo = BasketballGNNDemo()
    
    # Generate the demo video
    output_file = "basketball_gnn_demo.gif"
    demo.create_demo_video(output_file)
    
    # Print summary
    print("\nDemo Summary:")
    print(f"- 10 players (5 per team)")
    print(f"- 3 referees")
    print(f"- 1 ball, 2 baskets") 
    print(f"- 5 different play types")
    print(f"- Real-time GNN tactical analysis")
    print(f"- Interactive graph visualization")
    print(f"- Formation and possession tracking")
    
    return output_file

if __name__ == "__main__":
    main()
