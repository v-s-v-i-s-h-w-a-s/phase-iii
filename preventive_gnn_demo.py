import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import math
import random

class PreventiveGNNDemo:
    def __init__(self):
        # Court dimensions
        self.court_width = 800
        self.court_height = 400
        self.fps = 8
        
        # Demo frames
        self.frames = []
        
        # Create the play scenario: Stephen Curry 3-pointer that goes in, 
        # then show how GNN analysis could have prevented it
        self.create_curry_three_pointer_scenario()
        
    def create_curry_three_pointer_scenario(self):
        """Create Stephen Curry 3-pointer scenario with preventive GNN analysis"""
        
        # PART 1: THE PLAY HAPPENS (Curry scores)
        self.add_title_frame("🏀 LIVE PLAY: Warriors vs Lakers", "Stephen Curry isolation play developing...", 3)
        
        # Setup - Warriors have ball, Curry brings it up
        self.add_play_setup()
        
        # Curry gets the ball and starts his move
        self.add_curry_isolation()
        
        # Curry shoots and scores
        self.add_curry_shot_sequence()
        
        # PART 2: GNN ANALYSIS SHOWS HOW TO PREVENT IT
        self.add_title_frame("🧠 GNN ANALYSIS", "How this play could have been stopped...", 3)
        
        # Show defensive breakdown
        self.add_defensive_analysis()
        
        # Show correct defensive positioning
        self.add_preventive_strategy()
        
        # Show the prevention in action
        self.add_prevention_execution()
        
        print(f"Created demo with {len(self.frames)} frames")
    
    def add_title_frame(self, title, subtitle, duration):
        """Add title frame"""
        for _ in range(duration * self.fps):
            frame = {
                "type": "title",
                "title": title,
                "subtitle": subtitle,
                "players": [],
                "ball": {"x": 400, "y": 200, "with_player": None},
                "analysis": "",
                "arrows": [],
                "threat_level": 0.0
            }
            self.frames.append(frame)
    
    def add_play_setup(self):
        """Add play setup frames"""
        for frame_num in range(24):  # 3 seconds
            
            # Warriors players (Blue team)
            warriors = [
                {"id": "Curry", "x": 200, "y": 200, "team": "Warriors", "role": "PG"},
                {"id": "Klay", "x": 600, "y": 120, "team": "Warriors", "role": "SG"},
                {"id": "Wiggins", "x": 300, "y": 300, "team": "Warriors", "role": "SF"},
                {"id": "Green", "x": 500, "y": 280, "team": "Warriors", "role": "PF"},
                {"id": "Looney", "x": 120, "y": 200, "team": "Warriors", "role": "C"}
            ]
            
            # Lakers players (Red team)
            lakers = [
                {"id": "Russell", "x": 220, "y": 180, "team": "Lakers", "role": "PG"},
                {"id": "Reaves", "x": 580, "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "LeBron", "x": 320, "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 480, "y": 300, "team": "Lakers", "role": "PF"},
                {"id": "Wood", "x": 140, "y": 220, "team": "Lakers", "role": "C"}
            ]
            
            frame = {
                "type": "play",
                "players": warriors + lakers,
                "ball": {"x": 200, "y": 200, "with_player": "Curry"},
                "analysis": f"Warriors bringing ball up court\nCurry has possession\nLakers in standard defense",
                "arrows": [],
                "threat_level": 0.2,
                "description": "Setup: Warriors transition offense"
            }
            
            self.frames.append(frame)
    
    def add_curry_isolation(self):
        """Add Curry isolation sequence"""
        for frame_num in range(32):  # 4 seconds
            progress = frame_num / 32.0
            
            # Curry moves to his spot (right wing, 3-point line)
            curry_x = 200 + (350 * progress)  # Moves to right wing
            curry_y = 200 + (50 * math.sin(progress * math.pi))  # Slight dribble movement
            
            # Other Warriors clear out
            warriors = [
                {"id": "Curry", "x": curry_x, "y": curry_y, "team": "Warriors", "role": "PG"},
                {"id": "Klay", "x": 600 + (50 * progress), "y": 120, "team": "Warriors", "role": "SG"},
                {"id": "Wiggins", "x": 300 - (100 * progress), "y": 300, "team": "Warriors", "role": "SF"},
                {"id": "Green", "x": 500, "y": 280 + (50 * progress), "team": "Warriors", "role": "PF"},
                {"id": "Looney", "x": 120, "y": 200 + (80 * progress), "team": "Warriors", "role": "C"}
            ]
            
            # Lakers defense - Russell follows Curry
            russell_x = 220 + (320 * progress)
            russell_y = 180 + (60 * math.sin(progress * math.pi))
            
            lakers = [
                {"id": "Russell", "x": russell_x, "y": russell_y, "team": "Lakers", "role": "PG"},
                {"id": "Reaves", "x": 580 + (40 * progress), "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "LeBron", "x": 320 - (80 * progress), "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 480, "y": 300 + (40 * progress), "team": "Lakers", "role": "PF"},
                {"id": "Wood", "x": 140, "y": 220 + (60 * progress), "team": "Lakers", "role": "C"}
            ]
            
            # Threat level increases as Curry gets in position
            threat = 0.2 + (0.6 * progress)
            
            analysis_text = f"Curry isolation developing\nSpacing created for 3-point attempt\nDefense: Single coverage on Curry\nTHREAT LEVEL: {threat:.1f}"
            
            if frame_num > 24:
                analysis_text += f"\n⚠️ DANGER: Curry in prime shooting position!"
            
            frame = {
                "type": "play",
                "players": warriors + lakers,
                "ball": {"x": curry_x, "y": curry_y, "with_player": "Curry"},
                "analysis": analysis_text,
                "arrows": [],
                "threat_level": threat,
                "description": f"Isolation: Curry moving to shooting position"
            }
            
            self.frames.append(frame)
    
    def add_curry_shot_sequence(self):
        """Add Curry's shot sequence"""
        for frame_num in range(24):  # 3 seconds
            if frame_num < 8:
                # Shot preparation
                curry_x = 550
                curry_y = 250
                ball_x = curry_x
                ball_y = curry_y
                with_player = "Curry"
                shot_phase = "Setup"
            elif frame_num < 16:
                # Shot release
                curry_x = 550
                curry_y = 250
                # Ball travels toward basket
                shot_progress = (frame_num - 8) / 8.0
                ball_x = 550 + (200 * shot_progress)
                ball_y = 250 - (50 * shot_progress) + (100 * shot_progress * shot_progress)
                with_player = None
                shot_phase = "Release"
            else:
                # Ball goes in
                curry_x = 550
                curry_y = 250
                ball_x = 750  # Basket position
                ball_y = 200  # Basket height
                with_player = None
                shot_phase = "Score!"
            
            # All players watch the shot
            warriors = [
                {"id": "Curry", "x": curry_x, "y": curry_y, "team": "Warriors", "role": "PG"},
                {"id": "Klay", "x": 650, "y": 120, "team": "Warriors", "role": "SG"},
                {"id": "Wiggins", "x": 200, "y": 300, "team": "Warriors", "role": "SF"},
                {"id": "Green", "x": 500, "y": 330, "team": "Warriors", "role": "PF"},
                {"id": "Looney", "x": 120, "y": 280, "team": "Warriors", "role": "C"}
            ]
            
            lakers = [
                {"id": "Russell", "x": 570, "y": 270, "team": "Lakers", "role": "PG"},
                {"id": "Reaves", "x": 620, "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "LeBron", "x": 240, "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 480, "y": 340, "team": "Lakers", "role": "PF"},
                {"id": "Wood", "x": 140, "y": 280, "team": "Lakers", "role": "C"}
            ]
            
            if frame_num >= 16:
                analysis_text = f"🎯 STEPHEN CURRY SCORES!\n3-Point Shot: SUCCESSFUL\nDefense: Failed to contest\nResult: Warriors +3"
                threat = 1.0
            else:
                analysis_text = f"Curry shooting motion\nPhase: {shot_phase}\nContest level: Minimal\nShot quality: Excellent"
                threat = 0.9 + (0.1 * (frame_num / 16))
            
            frame = {
                "type": "play",
                "players": warriors + lakers,
                "ball": {"x": ball_x, "y": ball_y, "with_player": with_player},
                "analysis": analysis_text,
                "arrows": [{"start": (550, 250), "end": (750, 200), "color": "orange", "style": "shot"}] if frame_num >= 8 and frame_num < 16 else [],
                "threat_level": threat,
                "description": f"Shot sequence: {shot_phase}"
            }
            
            self.frames.append(frame)
    
    def add_defensive_analysis(self):
        """Add GNN defensive analysis frames"""
        for frame_num in range(32):  # 4 seconds
            
            # Show the defensive breakdown
            warriors = [
                {"id": "Curry", "x": 550, "y": 250, "team": "Warriors", "role": "PG"},
                {"id": "Klay", "x": 650, "y": 120, "team": "Warriors", "role": "SG"},
                {"id": "Wiggins", "x": 200, "y": 300, "team": "Warriors", "role": "SF"},
                {"id": "Green", "x": 500, "y": 330, "team": "Warriors", "role": "PF"},
                {"id": "Looney", "x": 120, "y": 280, "team": "Warriors", "role": "C"}
            ]
            
            lakers = [
                {"id": "Russell", "x": 570, "y": 270, "team": "Lakers", "role": "PG"},
                {"id": "Reaves", "x": 620, "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "LeBron", "x": 240, "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 480, "y": 340, "team": "Lakers", "role": "PF"},
                {"id": "Wood", "x": 140, "y": 280, "team": "Lakers", "role": "C"}
            ]
            
            # Analysis shows what went wrong
            analysis_phases = [
                "🔍 GNN ANALYSIS: Defensive Breakdown",
                "❌ Russell too far from Curry (5+ feet)",
                "❌ No help defense rotation",
                "❌ Davis stayed in paint instead of stepping up",
                "❌ Curry had 2.3 seconds of clean look",
                "📊 Shot Success Probability: 87%",
                "⚠️ Major defensive errors identified",
                "🧠 GNN Solution: See next sequence..."
            ]
            
            current_analysis = analysis_phases[min(frame_num // 4, len(analysis_phases) - 1)]
            
            # Highlight defensive errors with arrows
            arrows = []
            if frame_num > 8:
                # Show gap between Russell and Curry
                arrows.append({"start": (570, 270), "end": (550, 250), "color": "red", "style": "error"})
            if frame_num > 16:
                # Show where Davis should have been
                arrows.append({"start": (480, 340), "end": (520, 280), "color": "yellow", "style": "should_be"})
            
            frame = {
                "type": "analysis",
                "players": warriors + lakers,
                "ball": {"x": 550, "y": 250, "with_player": None},
                "analysis": current_analysis,
                "arrows": arrows,
                "threat_level": 0.9,
                "description": "GNN Analysis: Defensive breakdown identification"
            }
            
            self.frames.append(frame)
    
    def add_preventive_strategy(self):
        """Add preventive strategy frames"""
        for frame_num in range(32):  # 4 seconds
            
            # Show optimal defensive positioning
            warriors = [
                {"id": "Curry", "x": 550, "y": 250, "team": "Warriors", "role": "PG"},
                {"id": "Klay", "x": 650, "y": 120, "team": "Warriors", "role": "SG"},
                {"id": "Wiggins", "x": 200, "y": 300, "team": "Warriors", "role": "SF"},
                {"id": "Green", "x": 500, "y": 330, "team": "Warriors", "role": "PF"},
                {"id": "Looney", "x": 120, "y": 280, "team": "Warriors", "role": "C"}
            ]
            
            # Corrected Lakers positioning
            progress = min(frame_num / 24.0, 1.0)
            
            # Russell moves closer to Curry
            russell_x = 570 - (15 * progress)  # Closer coverage
            russell_y = 270 - (15 * progress)  # Better angle
            
            # Davis rotates up for help
            davis_x = 480 + (20 * progress)
            davis_y = 340 - (50 * progress)
            
            lakers = [
                {"id": "Russell", "x": russell_x, "y": russell_y, "team": "Lakers", "role": "PG"},
                {"id": "Reaves", "x": 620, "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "LeBron", "x": 240 + (10 * progress), "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": davis_x, "y": davis_y, "team": "Lakers", "role": "PF"},
                {"id": "Wood", "x": 140 + (30 * progress), "y": 280, "team": "Lakers", "role": "C"}
            ]
            
            # Strategy explanation
            strategy_phases = [
                "✅ GNN SOLUTION: Optimal Defense",
                "1️⃣ Russell: Close gap to 2 feet max",
                "2️⃣ Davis: Rotate up for help defense",
                "3️⃣ Wood: Slide over to cover paint",
                "4️⃣ LeBron: Provide weak-side help",
                "📊 New Shot Success Probability: 23%",
                "🎯 Curry forced into difficult shot",
                "✨ GNN prevents easy score!"
            ]
            
            current_strategy = strategy_phases[min(frame_num // 4, len(strategy_phases) - 1)]
            
            # Show movement arrows
            arrows = []
            if frame_num > 8:
                arrows.append({"start": (570, 270), "end": (555, 255), "color": "green", "style": "movement"})
            if frame_num > 16:
                arrows.append({"start": (480, 340), "end": (500, 290), "color": "green", "style": "movement"})
            if frame_num > 24:
                arrows.append({"start": (140, 280), "end": (170, 280), "color": "blue", "style": "support"})
            
            frame = {
                "type": "strategy",
                "players": warriors + lakers,
                "ball": {"x": 550, "y": 250, "with_player": "Curry"},
                "analysis": current_strategy,
                "arrows": arrows,
                "threat_level": 0.8 - (0.6 * progress),  # Threat decreases as defense improves
                "description": "GNN Solution: Preventive defensive strategy"
            }
            
            self.frames.append(frame)
    
    def add_prevention_execution(self):
        """Add prevention execution frames"""
        for frame_num in range(32):  # 4 seconds
            
            # Show the prevention in action
            warriors = [
                {"id": "Curry", "x": 550, "y": 250, "team": "Warriors", "role": "PG"},
                {"id": "Klay", "x": 650, "y": 120, "team": "Warriors", "role": "SG"},
                {"id": "Wiggins", "x": 200, "y": 300, "team": "Warriors", "role": "SF"},
                {"id": "Green", "x": 500, "y": 330, "team": "Warriors", "role": "PF"},
                {"id": "Looney", "x": 120, "y": 280, "team": "Warriors", "role": "C"}
            ]
            
            # Lakers in optimal position
            lakers = [
                {"id": "Russell", "x": 555, "y": 255, "team": "Lakers", "role": "PG"},  # Tight coverage
                {"id": "Reaves", "x": 620, "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "LeBron", "x": 250, "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 500, "y": 290, "team": "Lakers", "role": "PF"},  # Help position
                {"id": "Wood", "x": 170, "y": 280, "team": "Lakers", "role": "C"}
            ]
            
            if frame_num < 16:
                # Curry tries to shoot but faces heavy contest
                ball_pos = {"x": 550, "y": 250, "with_player": "Curry"}
                result_text = "🛡️ DEFENSE EXECUTED\nCurry contested heavily\nShot difficulty: MAXIMUM\nProbable outcome: Miss or pass"
            else:
                # Curry forced to pass or take difficult shot
                ball_pos = {"x": 520, "y": 280, "with_player": None}  # Pass to teammate
                result_text = "✅ SUCCESS!\nCurry forced into pass\nShot prevented\nDefensive win achieved"
            
            arrows = [
                {"start": (555, 255), "end": (550, 250), "color": "red", "style": "pressure"},
                {"start": (500, 290), "end": (525, 265), "color": "red", "style": "help"}
            ]
            
            frame = {
                "type": "prevention",
                "players": warriors + lakers,
                "ball": ball_pos,
                "analysis": result_text,
                "arrows": arrows,
                "threat_level": 0.2,
                "description": "Prevention executed: Curry shot denied"
            }
            
            self.frames.append(frame)
        
        # Add final summary frame
        for _ in range(16):  # 2 seconds
            frame = {
                "type": "summary",
                "players": warriors + lakers,
                "ball": {"x": 400, "y": 200, "with_player": None},
                "analysis": "🏆 GNN COACHING SUCCESS\n\nOriginal play: Curry 3-pointer (87% success)\nGNN Analysis: Identified defensive gaps\nPrevention: Optimal positioning adjustment\nResult: Shot denied (23% success rate)\n\n💡 Proactive coaching beats reactive!",
                "arrows": [],
                "threat_level": 0.0,
                "description": "Summary: GNN preventive coaching demonstrated"
            }
            self.frames.append(frame)
    
    def draw_court(self, ax):
        """Draw basketball court"""
        ax.clear()
        ax.set_xlim(0, self.court_width)
        ax.set_ylim(0, self.court_height)
        ax.set_aspect('equal')
        
        # Court background
        court = patches.Rectangle((25, 25), 750, 350, linewidth=3, 
                                edgecolor='black', facecolor='lightgreen', alpha=0.2)
        ax.add_patch(court)
        
        # Center line and circle
        ax.plot([400, 400], [25, 375], 'k-', linewidth=2)
        center_circle = patches.Circle((400, 200), 50, linewidth=2, 
                                     edgecolor='black', facecolor='none')
        ax.add_patch(center_circle)
        
        # Three-point lines
        # Left side
        three_pt_left = patches.Arc((75, 200), 300, 300, theta1=290, theta2=70, 
                                   linewidth=2, edgecolor='blue')
        ax.add_patch(three_pt_left)
        
        # Right side  
        three_pt_right = patches.Arc((725, 200), 300, 300, theta1=110, theta2=250,
                                    linewidth=2, edgecolor='blue')
        ax.add_patch(three_pt_right)
        
        # Paint areas
        left_paint = patches.Rectangle((25, 150), 100, 100, linewidth=2,
                                     edgecolor='black', facecolor='lightblue', alpha=0.2)
        ax.add_patch(left_paint)
        
        right_paint = patches.Rectangle((675, 150), 100, 100, linewidth=2,
                                      edgecolor='black', facecolor='lightblue', alpha=0.2)
        ax.add_patch(right_paint)
        
        # Baskets
        left_basket = patches.Circle((75, 200), 8, facecolor='orange', edgecolor='black')
        right_basket = patches.Circle((725, 200), 8, facecolor='orange', edgecolor='black')
        ax.add_patch(left_basket)
        ax.add_patch(right_basket)
    
    def draw_frame(self, frame_data, ax):
        """Draw a single frame"""
        self.draw_court(ax)
        
        if frame_data["type"] == "title":
            # Title frame
            ax.text(400, 200, frame_data["title"], 
                   ha='center', va='center', fontsize=24, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=1", facecolor="gold", alpha=0.8))
            ax.text(400, 150, frame_data["subtitle"], 
                   ha='center', va='center', fontsize=16)
            return
        
        # Draw players
        for player in frame_data["players"]:
            color = 'blue' if player["team"] == "Warriors" else 'red'
            size = 150 if frame_data["ball"].get("with_player") == player["id"] else 100
            
            ax.scatter(player["x"], player["y"], c=color, s=size, 
                      alpha=0.8, edgecolors='black', linewidth=2)
            ax.text(player["x"], player["y"]-25, player["id"], 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw ball
        ball = frame_data["ball"]
        if ball.get("with_player"):
            # Ball with player (already shown by larger player circle)
            pass
        else:
            # Free ball
            ax.scatter(ball["x"], ball["y"], c='orange', s=80, 
                      marker='o', edgecolors='black', linewidth=2)
        
        # Draw arrows
        for arrow in frame_data.get("arrows", []):
            start = arrow["start"]
            end = arrow["end"]
            color = arrow["color"]
            style = arrow.get("style", "normal")
            
            if style == "shot":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.8))
            elif style == "error":
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=color, linewidth=3, linestyle='--', alpha=0.7)
            elif style == "movement":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
            elif style == "pressure":
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=color, linewidth=4, alpha=0.8)
            elif style == "help":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle='--'))
        
        # Draw threat level indicator
        threat = frame_data.get("threat_level", 0.0)
        threat_color = 'green' if threat < 0.3 else 'yellow' if threat < 0.7 else 'red'
        threat_rect = patches.Rectangle((650, 350), 100, 20, 
                                      facecolor=threat_color, alpha=0.7, edgecolor='black')
        ax.add_patch(threat_rect)
        ax.text(700, 360, f"THREAT: {threat:.1f}", ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Draw analysis text
        analysis_text = frame_data.get("analysis", "")
        ax.text(50, 350, analysis_text, fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
               verticalalignment='top')
        
        # Set title based on frame type
        if frame_data["type"] == "play":
            title = "🏀 LIVE PLAY: Warriors vs Lakers"
        elif frame_data["type"] == "analysis":
            title = "🧠 GNN ANALYSIS: Defensive Breakdown"
        elif frame_data["type"] == "strategy":
            title = "✅ GNN SOLUTION: Preventive Strategy"
        elif frame_data["type"] == "prevention":
            title = "🛡️ PREVENTION EXECUTED"
        else:
            title = "📊 GNN COACHING DEMO"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    def animate_frame(self, frame_num):
        """Animation function"""
        if frame_num >= len(self.frames):
            return
        
        fig = plt.gcf()
        fig.clear()
        
        ax = fig.add_subplot(111)
        frame_data = self.frames[frame_num]
        
        self.draw_frame(frame_data, ax)
        
        # Add frame counter
        ax.text(750, 50, f"Frame {frame_num+1}/{len(self.frames)}", 
               ha='right', va='bottom', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
    
    def create_demo(self, output_path="preventive_gnn_demo.gif"):
        """Create the demo video"""
        print("Creating Preventive GNN Demo...")
        print(f"Total frames: {len(self.frames)}")
        print(f"Duration: {len(self.frames) / self.fps:.1f} seconds")
        
        # Setup figure
        fig = plt.figure(figsize=(16, 10))
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            self.animate_frame,
            frames=len(self.frames),
            interval=int(1000/self.fps),  # milliseconds between frames
            repeat=True,
            blit=False
        )
        
        # Save animation
        print(f"Saving to {output_path}...")
        writer = PillowWriter(fps=self.fps)
        anim.save(output_path, writer=writer)
        
        print(f"Demo created successfully!")
        
        # Also save as MP4
        try:
            mp4_path = output_path.replace('.gif', '.mp4')
            anim.save(mp4_path, writer='ffmpeg', fps=self.fps)
            print(f"MP4 version saved: {mp4_path}")
        except:
            print("MP4 save skipped - ffmpeg not available")
        
        plt.close(fig)
        return output_path

def main():
    """Main function"""
    print("🏀 Preventive GNN Basketball Demo")
    print("="*50)
    print("Scenario: Stephen Curry 3-pointer")
    print("Shows: Play first, then GNN prevention analysis")
    print("")
    
    # Create and run demo
    demo = PreventiveGNNDemo()
    output_file = demo.create_demo()
    
    print(f"\n📺 Demo Summary:")
    print(f"- Shows Curry 3-pointer that scores")
    print(f"- GNN analysis identifies defensive gaps") 
    print(f"- Demonstrates preventive strategy")
    print(f"- Execution reduces shot success from 87% to 23%")
    print(f"- Total duration: {len(demo.frames) / demo.fps:.1f} seconds")
    print(f"\n🎯 Open {output_file} to see the prevention!")
    
    return output_file

if __name__ == "__main__":
    main()
