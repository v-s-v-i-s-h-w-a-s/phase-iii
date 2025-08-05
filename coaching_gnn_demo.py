import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec

class CoachingGNNDemo:
    def __init__(self):
        self.court_width = 800
        self.court_height = 400
        self.fps = 1  # Very slow for coaching analysis
        
    def create_guaranteed_score_play(self):
        """Create a play that's guaranteed to score, then show how to stop it"""
        
        # PLAY: Lakers Triangle Offense - Shaq in the post (guaranteed score)
        # SCENARIO: 2001 Lakers vs Spurs - Shaq gets deep position
        
        frames = []
        
        # Frame 1: Setup - Shaq getting position
        frames.append({
            "step": "Initial Setup",
            "description": "Shaq establishes deep post position - This is a guaranteed score",
            "coaching_insight": "PROBLEM: Defense allows Shaq too deep. This is unstoppable.",
            "players": [
                {"id": "Kobe", "team": "LAL", "x": 250, "y": 150, "role": "SG", "has_ball": True},
                {"id": "Fisher", "team": "LAL", "x": 200, "y": 200, "role": "PG", "has_ball": False},
                {"id": "Fox", "team": "LAL", "x": 300, "y": 120, "role": "SF", "has_ball": False},
                {"id": "Horry", "team": "LAL", "x": 450, "y": 180, "role": "PF", "has_ball": False},
                {"id": "Shaq", "team": "LAL", "x": 580, "y": 200, "role": "C", "has_ball": False},  # DEEP POSITION
                {"id": "Parker", "team": "SAS", "x": 220, "y": 210, "role": "PG", "has_ball": False},
                {"id": "Bowen", "team": "SAS", "x": 270, "y": 160, "role": "SG", "has_ball": False},
                {"id": "Duncan", "team": "SAS", "x": 320, "y": 130, "role": "SF", "has_ball": False},
                {"id": "Robinson", "team": "SAS", "x": 470, "y": 190, "role": "PF", "has_ball": False},
                {"id": "Daniels", "team": "SAS", "x": 590, "y": 210, "role": "C", "has_ball": False}  # Behind Shaq
            ],
            "ball": {"x": 250, "y": 150},
            "gnn_analysis": "CRITICAL THREAT: Shaq 3 feet from basket with size advantage",
            "score_probability": 95,
            "defensive_options": "TOO LATE - Position already established"
        })
        
        # Frame 2: Entry pass
        frames.append({
            "step": "Entry Pass",
            "description": "Kobe feeds Shaq in the post - Game over",
            "coaching_insight": "RESULT: Easy 2 points. Defense failed in positioning phase.",
            "players": [
                {"id": "Kobe", "team": "LAL", "x": 250, "y": 150, "role": "SG", "has_ball": False},
                {"id": "Fisher", "team": "LAL", "x": 200, "y": 200, "role": "PG", "has_ball": False},
                {"id": "Fox", "team": "LAL", "x": 300, "y": 120, "role": "SF", "has_ball": False},
                {"id": "Horry", "team": "LAL", "x": 450, "y": 180, "role": "PF", "has_ball": False},
                {"id": "Shaq", "team": "LAL", "x": 580, "y": 200, "role": "C", "has_ball": True},  # RECEIVES BALL
                {"id": "Parker", "team": "SAS", "x": 220, "y": 210, "role": "PG", "has_ball": False},
                {"id": "Bowen", "team": "SAS", "x": 270, "y": 160, "role": "SG", "has_ball": False},
                {"id": "Duncan", "team": "SAS", "x": 320, "y": 130, "role": "SF", "has_ball": False},
                {"id": "Robinson", "team": "SAS", "x": 470, "y": 190, "role": "PF", "has_ball": False},
                {"id": "Daniels", "team": "SAS", "x": 590, "y": 210, "role": "C", "has_ball": False}  # Helpless
            ],
            "ball": {"x": 580, "y": 200},
            "gnn_analysis": "BASKET IMMINENT: No defensive help available, 100% score rate",
            "score_probability": 100,
            "defensive_options": "Send help from Duncan - but creates 4-on-3 elsewhere"
        })
        
        # Frame 3: Dunk
        frames.append({
            "step": "Guaranteed Score",
            "description": "Shaq turns and dunks - Unstoppable from this position",
            "coaching_insight": "SCORED: 2 points Lakers. Defense was doomed from setup.",
            "players": [
                {"id": "Kobe", "team": "LAL", "x": 250, "y": 150, "role": "SG", "has_ball": False},
                {"id": "Fisher", "team": "LAL", "x": 200, "y": 200, "role": "PG", "has_ball": False},
                {"id": "Fox", "team": "LAL", "x": 300, "y": 120, "role": "SF", "has_ball": False},
                {"id": "Horry", "team": "LAL", "x": 450, "y": 180, "role": "PF", "has_ball": False},
                {"id": "Shaq", "team": "LAL", "x": 600, "y": 200, "role": "C", "has_ball": False},  # At rim
                {"id": "Parker", "team": "SAS", "x": 220, "y": 210, "role": "PG", "has_ball": False},
                {"id": "Bowen", "team": "SAS", "x": 270, "y": 160, "role": "SG", "has_ball": False},
                {"id": "Duncan", "team": "SAS", "x": 320, "y": 130, "role": "SF", "has_ball": False},
                {"id": "Robinson", "team": "SAS", "x": 470, "y": 190, "role": "PF", "has_ball": False},
                {"id": "Daniels", "team": "SAS", "x": 590, "y": 210, "role": "C", "has_ball": False}
            ],
            "ball": {"x": 600, "y": 200},
            "gnn_analysis": "BASKET SCORED: Lakers +2, Defense needs immediate adjustment",
            "score_probability": 100,
            "defensive_options": "Reset - prevent this positioning next possession"
        })
        
        # NOW THE COACHING ADJUSTMENTS - Same play but with defensive changes
        
        # Frame 4: COACHING TIMEOUT - Adjustments
        frames.append({
            "step": "TIMEOUT - Defensive Adjustment",
            "description": "Spurs coach makes tactical changes to prevent repeat",
            "coaching_insight": "COACH DECISION: Front Shaq, help from weak side, deny entry pass",
            "players": [
                {"id": "Kobe", "team": "LAL", "x": 250, "y": 150, "role": "SG", "has_ball": True},
                {"id": "Fisher", "team": "LAL", "x": 200, "y": 200, "role": "PG", "has_ball": False},
                {"id": "Fox", "team": "LAL", "x": 300, "y": 120, "role": "SF", "has_ball": False},
                {"id": "Horry", "team": "LAL", "x": 450, "y": 180, "role": "PF", "has_ball": False},
                {"id": "Shaq", "team": "LAL", "x": 550, "y": 200, "role": "C", "has_ball": False},  # SAME ATTEMPT
                {"id": "Parker", "team": "SAS", "x": 220, "y": 210, "role": "PG", "has_ball": False},
                {"id": "Bowen", "team": "SAS", "x": 270, "y": 160, "role": "SG", "has_ball": False},
                {"id": "Duncan", "team": "SAS", "x": 380, "y": 220, "role": "SF", "has_ball": False},  # HELP POSITION
                {"id": "Robinson", "team": "SAS", "x": 500, "y": 190, "role": "PF", "has_ball": False},  # WEAKSIDE HELP
                {"id": "Daniels", "team": "SAS", "x": 570, "y": 190, "role": "C", "has_ball": False}  # FRONTING SHAQ
            ],
            "ball": {"x": 250, "y": 150},
            "gnn_analysis": "DEFENSE ADJUSTED: Fronting position, help defense ready",
            "score_probability": 15,
            "defensive_options": "SUCCESS - Shaq cannot establish deep position"
        })
        
        # Frame 5: Defensive Success
        frames.append({
            "step": "Defense Works",
            "description": "Shaq cannot get deep position, forced to catch further out",
            "coaching_insight": "SUCCESS: Defensive adjustment prevented guaranteed score",
            "players": [
                {"id": "Kobe", "team": "LAL", "x": 250, "y": 150, "role": "SG", "has_ball": False},
                {"id": "Fisher", "team": "LAL", "x": 200, "y": 200, "role": "PG", "has_ball": False},
                {"id": "Fox", "team": "LAL", "x": 300, "y": 120, "role": "SF", "has_ball": False},
                {"id": "Horry", "team": "LAL", "x": 450, "y": 180, "role": "PF", "has_ball": False},
                {"id": "Shaq", "team": "LAL", "x": 480, "y": 200, "role": "C", "has_ball": True},  # FORCED OUT
                {"id": "Parker", "team": "SAS", "x": 220, "y": 210, "role": "PG", "has_ball": False},
                {"id": "Bowen", "team": "SAS", "x": 270, "y": 160, "role": "SG", "has_ball": False},
                {"id": "Duncan", "team": "SAS", "x": 380, "y": 220, "role": "SF", "has_ball": False},
                {"id": "Robinson", "team": "SAS", "x": 500, "y": 170, "role": "PF", "has_ball": False},  # DOUBLE TEAM
                {"id": "Daniels", "team": "SAS", "x": 490, "y": 210, "role": "C", "has_ball": False}  # CONTEST
            ],
            "ball": {"x": 480, "y": 200},
            "gnn_analysis": "TURNOVER LIKELY: Shaq double-teamed, no easy shot available",
            "score_probability": 25,
            "defensive_options": "Force turnover or difficult shot - mission accomplished"
        })
        
        return frames
    
    def create_coaching_demo(self):
        """Create comprehensive coaching demo"""
        
        frames = self.create_guaranteed_score_play()
        
        # Create figure with coaching analysis layout
        fig = plt.figure(figsize=(20, 14))
        
        def animate_coaching_frame(frame_num):
            if frame_num >= len(frames):
                return
                
            frame_data = frames[frame_num]
            fig.clear()
            
            # Create grid layout for coaching view
            gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
            
            # Main court view
            ax1 = fig.add_subplot(gs[0, :2])
            self.draw_coaching_court(ax1, frame_data)
            ax1.set_title(f"COACHING ANALYSIS: {frame_data['step']}", 
                         fontsize=16, fontweight='bold')
            
            # GNN Analysis Panel
            ax2 = fig.add_subplot(gs[0, 2])
            self.draw_gnn_coaching_analysis(ax2, frame_data)
            ax2.set_title("GNN Real-Time Analysis", fontsize=12, fontweight='bold')
            
            # Play Description
            ax3 = fig.add_subplot(gs[1, :])
            self.draw_play_description(ax3, frame_data)
            
            # Coaching Decision Panel
            ax4 = fig.add_subplot(gs[2, :])
            self.draw_coaching_decisions(ax4, frame_data)
            
            plt.tight_layout()
        
        # Setup animation
        anim = FuncAnimation(
            fig, 
            animate_coaching_frame,
            frames=len(frames),
            interval=4000,  # 4 seconds per frame for coaching analysis
            repeat=True,
            blit=False
        )
        
        return anim, frames
    
    def draw_coaching_court(self, ax, frame_data):
        """Draw court with coaching perspective"""
        ax.clear()
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 400)
        ax.set_aspect('equal')
        
        # Court background
        court = patches.Rectangle((25, 25), 750, 350, linewidth=3, 
                                edgecolor='black', facecolor='#F5DEB3', alpha=0.3)
        ax.add_patch(court)
        
        # Paint areas - emphasize scoring zones
        left_paint = patches.Rectangle((25, 150), 120, 100, linewidth=3,
                                     edgecolor='red', facecolor='pink', alpha=0.4)
        right_paint = patches.Rectangle((655, 150), 120, 100, linewidth=3,
                                      edgecolor='red', facecolor='pink', alpha=0.4)
        ax.add_patch(left_paint)
        ax.add_patch(right_paint)
        
        # Baskets
        left_basket = patches.Circle((60, 200), 15, facecolor='orange', edgecolor='red', linewidth=3)
        right_basket = patches.Circle((740, 200), 15, facecolor='orange', edgecolor='red', linewidth=3)
        ax.add_patch(left_basket)
        ax.add_patch(right_basket)
        
        # Three-point lines
        theta = np.linspace(-np.pi/2.5, np.pi/2.5, 50)
        left_3pt_x = 60 + 200 * np.cos(theta)
        left_3pt_y = 200 + 200 * np.sin(theta)
        ax.plot(left_3pt_x, left_3pt_y, 'b-', linewidth=2)
        
        right_3pt_x = 740 - 200 * np.cos(theta)
        right_3pt_y = 200 + 200 * np.sin(theta)
        ax.plot(right_3pt_x, right_3pt_y, 'r-', linewidth=2)
        
        # Draw players with coaching emphasis
        players = frame_data["players"]
        ball = frame_data["ball"]
        
        # Draw players
        for player in players:
            if player["team"] == "LAL":
                color = 'purple'
                marker_size = 150 if player.get("has_ball", False) else 120
            else:
                color = 'black'
                marker_size = 120
            
            # Special emphasis for key players
            if player["id"] == "Shaq":
                marker_size = 200
                edgecolor = 'red'
                linewidth = 4
            elif player["id"] == "Daniels":  # Shaq's defender
                marker_size = 150
                edgecolor = 'blue'
                linewidth = 4
            else:
                edgecolor = 'black'
                linewidth = 2
            
            ax.scatter(player["x"], player["y"], c=color, s=marker_size, 
                      alpha=0.8, edgecolors=edgecolor, linewidth=linewidth, zorder=5)
            
            # Player labels
            ax.text(player["x"], player["y"]-30, f"{player['id']}\n{player['role']}", 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
        
        # Draw ball with emphasis
        ax.scatter(ball["x"], ball["y"], c='orange', s=100, 
                  marker='o', edgecolors='red', linewidth=3, zorder=6)
        
        # Add tactical arrows and annotations
        self.draw_tactical_arrows(ax, frame_data)
        
        # Score probability indicator
        prob = frame_data.get("score_probability", 0)
        color = 'red' if prob > 70 else 'orange' if prob > 30 else 'green'
        ax.text(400, 50, f"SCORE PROBABILITY: {prob}%", 
               ha='center', va='center', fontsize=14, fontweight='bold', color=color,
               bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))
    
    def draw_tactical_arrows(self, ax, frame_data):
        """Draw tactical arrows and movement indicators"""
        step = frame_data["step"]
        
        if "Setup" in step:
            # Show Shaq's deep position threat
            ax.annotate('DANGER ZONE', xy=(580, 200), xytext=(500, 300),
                       arrowprops=dict(arrowstyle='->', lw=3, color='red'),
                       fontsize=12, fontweight='bold', color='red')
            
        elif "Entry Pass" in step:
            # Show pass to Shaq
            ax.annotate('', xy=(580, 200), xytext=(250, 150),
                       arrowprops=dict(arrowstyle='->', lw=4, color='purple'))
            ax.text(400, 100, "ENTRY PASS", ha='center', fontsize=12, 
                   fontweight='bold', color='purple')
            
        elif "Guaranteed Score" in step:
            # Show dunk
            ax.annotate('DUNK!', xy=(600, 200), xytext=(550, 120),
                       arrowprops=dict(arrowstyle='->', lw=4, color='red'),
                       fontsize=14, fontweight='bold', color='red')
            
        elif "Timeout" in step:
            # Show defensive adjustments
            ax.annotate('FRONT HIM', xy=(570, 190), xytext=(600, 120),
                       arrowprops=dict(arrowstyle='->', lw=3, color='blue'),
                       fontsize=11, fontweight='bold', color='blue')
            ax.annotate('HELP DEFENSE', xy=(500, 190), xytext=(450, 120),
                       arrowprops=dict(arrowstyle='->', lw=3, color='blue'),
                       fontsize=11, fontweight='bold', color='blue')
            
        elif "Defense Works" in step:
            # Show successful defense
            ax.annotate('DOUBLE TEAM', xy=(480, 200), xytext=(400, 300),
                       arrowprops=dict(arrowstyle='->', lw=3, color='green'),
                       fontsize=12, fontweight='bold', color='green')
    
    def draw_gnn_coaching_analysis(self, ax, frame_data):
        """Draw GNN analysis from coaching perspective"""
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Create graph analysis
        players = frame_data["players"]
        G = nx.Graph()
        
        # Add nodes
        for player in players:
            G.add_node(player["id"], team=player["team"], 
                      position=(player["x"], player["y"]))
        
        # Add edges based on defensive relationships
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players[i+1:], i+1):
                distance = np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)
                if distance < 100 and p1["team"] != p2["team"]:
                    G.add_edge(p1["id"], p2["id"], weight=1.0 - distance/100)
        
        # GNN Analysis Text
        analysis_text = f"""GNN TACTICAL ANALYSIS:

{frame_data['gnn_analysis']}

THREAT ASSESSMENT:
Score Probability: {frame_data.get('score_probability', 0)}%

GRAPH METRICS:
• Defensive Edges: {len([e for e in G.edges()])}
• Shaq Isolation: {"HIGH" if frame_data.get('score_probability', 0) > 70 else "LOW"}
• Help Defense: {"ACTIVE" if "Works" in frame_data['step'] else "NEEDED"}

COACHING INSIGHT:
{frame_data.get('coaching_insight', 'Analyzing...')}

DEFENSIVE OPTIONS:
{frame_data.get('defensive_options', 'Calculating...')}
"""
        
        ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    def draw_play_description(self, ax, frame_data):
        """Draw detailed play description"""
        ax.clear()
        ax.axis('off')
        
        description_text = f"""
PLAY BREAKDOWN: {frame_data['step']}

WHAT'S HAPPENING: {frame_data['description']}

BASKETBALL CONTEXT: This demonstrates a classic NBA scenario where Shaquille O'Neal gets deep post position.
When Shaq establishes position within 5 feet of the basket, he becomes virtually unstoppable due to his size (7'1", 325 lbs) and strength.

WHY THIS MATTERS: Shows how GNN analysis can help coaches identify problems before they become unstoppable situations.
"""
        
        ax.text(0.05, 0.95, description_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    def draw_coaching_decisions(self, ax, frame_data):
        """Draw coaching decision analysis"""
        ax.clear()
        ax.axis('off')
        
        step = frame_data["step"]
        
        if "Setup" in step or "Entry Pass" in step or "Guaranteed Score" in step:
            coaching_text = f"""
🔴 COACHING FAILURE: {frame_data['coaching_insight']}

THE MISTAKE: Allowing Shaq to establish deep post position (within 5 feet of basket)
CONSEQUENCE: Virtually guaranteed 2 points - Shaq shoots 58% from field, 85% this close
LESSON: Defense must be proactive, not reactive

WHAT SHOULD HAVE HAPPENED:
1. Front Shaq before he gets deep position
2. Help defense ready from weak side
3. Deny the entry pass completely
4. Force Lakers to run a different play

GNN INSIGHT: Once Shaq has deep position, no amount of help defense can stop him.
The key is PREVENTION, not reaction.
"""
        else:
            coaching_text = f"""
🟢 COACHING SUCCESS: {frame_data['coaching_insight']}

THE ADJUSTMENT: Spurs coach recognized the problem and made tactical changes
SOLUTION: Front Shaq, bring help defense, deny deep position
RESULT: Turned 95% score probability into 25% - massive improvement

TACTICAL CHANGES MADE:
1. Daniels fronts Shaq instead of playing behind
2. Duncan provides weak-side help
3. Robinson ready for double team
4. Force Shaq to catch ball further from basket

GNN INSIGHT: Small positional changes create massive tactical advantages.
This is the difference between elite and average coaching.
"""
        
        ax.text(0.05, 0.95, coaching_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen' if "SUCCESS" in coaching_text else 'lightcoral', alpha=0.9))
    
    def create_coaching_video(self, output_path="coaching_gnn_demo.gif"):
        """Create the coaching demo video"""
        print("Creating Coaching GNN Analysis Demo...")
        print("This demo shows a guaranteed scoring play and how coaching can stop it")
        
        anim, frames = self.create_coaching_demo()
        
        # Save as GIF
        print(f"Saving coaching demo to {output_path}...")
        writer = PillowWriter(fps=0.25)  # 4 seconds per frame
        anim.save(output_path, writer=writer)
        
        print(f"Coaching demo created successfully: {output_path}")
        print(f"Total frames: {len(frames)}")
        print(f"Duration: {len(frames) * 4:.1f} seconds")
        
        # Also try MP4
        try:
            mp4_path = output_path.replace('.gif', '.mp4')
            anim.save(mp4_path, writer='ffmpeg', fps=0.25)
            print(f"MP4 version saved: {mp4_path}")
        except Exception as e:
            print(f"MP4 save failed: {e}")
        
        plt.close()
        return output_path

def main():
    """Main function"""
    print("COACHING GNN ANALYSIS DEMO")
    print("="*50)
    print("This demo shows:")
    print("1. A guaranteed scoring play (Shaq in deep post)")
    print("2. Why it's unstoppable once established")
    print("3. How a coach can prevent it with tactical adjustments")
    print("4. Real-time GNN analysis for coaching decisions")
    print("")
    
    demo = CoachingGNNDemo()
    output_file = demo.create_coaching_video()
    
    print("\nWhat you'll see:")
    print("• Frame 1-3: The problem - Shaq gets deep position and scores")
    print("• Frame 4-5: The solution - Coaching adjustments prevent the score")
    print("• Real-time GNN analysis showing probability changes")
    print("• Tactical insights for coaching decisions")
    print("• Before/after comparison of defensive effectiveness")
    
    print(f"\nCoaching demo saved as: {output_file}")
    print("Each frame shows for 4 seconds - time to read all analysis")
    
    return output_file

if __name__ == "__main__":
    main()
