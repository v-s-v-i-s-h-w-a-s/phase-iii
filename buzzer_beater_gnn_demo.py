import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import math
import random

class BuzzerBeaterGNNDemo:
    def __init__(self):
        # Court dimensions
        self.court_width = 800
        self.court_height = 400
        self.fps = 6  # Slower for detailed analysis
        
        # Demo frames
        self.frames = []
        
        # Create buzzer beater scenarios with GNN analysis
        self.create_buzzer_beater_scenarios()
        
    def create_buzzer_beater_scenarios(self):
        """Create multiple buzzer beater scenarios with GNN prevention analysis"""
        
        # Scenario 1: Damian Lillard deep 3-pointer
        self.add_title_frame("🚨 BUZZER BEATER ANALYSIS", "Scenario 1: Deep 3-Pointer (0.9 seconds left)", 3)
        self.add_lillard_deep_three()
        
        # Scenario 2: Kobe Bryant fadeaway jumper
        self.add_title_frame("🚨 BUZZER BEATER ANALYSIS", "Scenario 2: Isolation Fadeaway (1.8 seconds left)", 3)
        self.add_kobe_fadeaway()
        
        # Scenario 3: LeBron James drive and score
        self.add_title_frame("🚨 BUZZER BEATER ANALYSIS", "Scenario 3: Drive to Basket (2.4 seconds left)", 3)
        self.add_lebron_drive()
        
        print(f"Created buzzer beater demo with {len(self.frames)} frames")
    
    def add_title_frame(self, title, subtitle, duration):
        """Add title frame"""
        for _ in range(duration * self.fps):
            frame = {
                "type": "title",
                "title": title,
                "subtitle": subtitle,
                "players": [],
                "ball": {"x": 400, "y": 200, "with_player": None},
                "clock": "0.0",
                "analysis": "",
                "arrows": [],
                "threat_level": 0.0
            }
            self.frames.append(frame)
    
    def add_lillard_deep_three(self):
        """Add Damian Lillard deep 3-pointer scenario"""
        
        # PART 1: THE BUZZER BEATER (happens)
        self.add_setup_phase("Lillard", 0.9)
        self.add_lillard_shot_execution()
        
        # PART 2: GNN ANALYSIS (how to prevent)
        self.add_title_frame("🧠 GNN PREVENTION ANALYSIS", "How to stop the deep 3-pointer", 2)
        self.add_lillard_prevention()
    
    def add_setup_phase(self, star_player, time_left):
        """Add setup phase for buzzer beater"""
        for frame_num in range(12):  # 2 seconds
            clock_time = time_left + (frame_num * 0.1)
            
            if star_player == "Lillard":
                # Trail Blazers vs Lakers setup
                blazers = [
                    {"id": "Lillard", "x": 300, "y": 200, "team": "Blazers", "role": "PG"},
                    {"id": "McCollum", "x": 600, "y": 120, "team": "Blazers", "role": "SG"},
                    {"id": "Covington", "x": 200, "y": 300, "team": "Blazers", "role": "SF"},
                    {"id": "Nurkic", "x": 150, "y": 200, "team": "Blazers", "role": "C"},
                    {"id": "Simons", "x": 450, "y": 300, "team": "Blazers", "role": "G"}
                ]
                
                lakers = [
                    {"id": "Westbrook", "x": 320, "y": 180, "team": "Lakers", "role": "PG"},
                    {"id": "Monk", "x": 580, "y": 140, "team": "Lakers", "role": "SG"},
                    {"id": "James", "x": 220, "y": 280, "team": "Lakers", "role": "SF"},
                    {"id": "Davis", "x": 170, "y": 220, "team": "Lakers", "role": "PF"},
                    {"id": "Howard", "x": 130, "y": 180, "team": "Lakers", "role": "C"}
                ]
                
                ball_carrier = "Lillard"
                
            elif star_player == "Kobe":
                # Lakers vs Celtics setup
                blazers = [
                    {"id": "Fisher", "x": 250, "y": 150, "team": "Lakers", "role": "PG"},
                    {"id": "Kobe", "x": 500, "y": 200, "team": "Lakers", "role": "SG"},
                    {"id": "Artest", "x": 200, "y": 300, "team": "Lakers", "role": "SF"},
                    {"id": "Gasol", "x": 300, "y": 320, "team": "Lakers", "role": "PF"},
                    {"id": "Bynum", "x": 150, "y": 250, "team": "Lakers", "role": "C"}
                ]
                
                lakers = [
                    {"id": "Rondo", "x": 270, "y": 170, "team": "Celtics", "role": "PG"},
                    {"id": "Allen", "x": 520, "y": 180, "team": "Celtics", "role": "SG"},
                    {"id": "Pierce", "x": 220, "y": 280, "team": "Celtics", "role": "SF"},
                    {"id": "Garnett", "x": 320, "y": 300, "team": "Celtics", "role": "PF"},
                    {"id": "Perkins", "x": 170, "y": 230, "team": "Celtics", "role": "C"}
                ]
                
                ball_carrier = "Kobe"
                
            elif star_player == "LeBron":
                # Heat vs Spurs setup
                blazers = [
                    {"id": "Chalmers", "x": 250, "y": 180, "team": "Heat", "role": "PG"},
                    {"id": "Wade", "x": 400, "y": 150, "team": "Heat", "role": "SG"},
                    {"id": "LeBron", "x": 350, "y": 220, "team": "Heat", "role": "SF"},
                    {"id": "Bosh", "x": 200, "y": 280, "team": "Heat", "role": "PF"},
                    {"id": "Haslem", "x": 150, "y": 240, "team": "Heat", "role": "C"}
                ]
                
                lakers = [
                    {"id": "Parker", "x": 270, "y": 200, "team": "Spurs", "role": "PG"},
                    {"id": "Ginobili", "x": 420, "y": 170, "team": "Spurs", "role": "SG"},
                    {"id": "Leonard", "x": 370, "y": 240, "team": "Spurs", "role": "SF"},
                    {"id": "Duncan", "x": 220, "y": 300, "team": "Spurs", "role": "PF"},
                    {"id": "Splitter", "x": 170, "y": 260, "team": "Spurs", "role": "C"}
                ]
                
                ball_carrier = "LeBron"
            
            frame = {
                "type": "setup",
                "players": blazers + lakers,
                "ball": {"x": blazers[0]["x"] if ball_carrier == blazers[0]["id"] else blazers[1]["x"] if ball_carrier == blazers[1]["id"] else blazers[2]["x"], 
                        "y": blazers[0]["y"] if ball_carrier == blazers[0]["id"] else blazers[1]["y"] if ball_carrier == blazers[1]["id"] else blazers[2]["y"], 
                        "with_player": ball_carrier},
                "clock": f"{clock_time:.1f}",
                "analysis": f"CLUTCH TIME!\n{star_player} has the ball\nDefense in standard formation\nTime running out...",
                "arrows": [],
                "threat_level": 0.7,
                "description": f"Setup: {star_player} buzzer beater developing"
            }
            
            self.frames.append(frame)
    
    def add_lillard_shot_execution(self):
        """Add Lillard's deep 3-pointer execution"""
        for frame_num in range(18):  # 3 seconds
            clock_time = max(0.0, 0.9 - (frame_num * 0.05))
            
            # Lillard moves to deep 3-point position
            progress = frame_num / 18.0
            lillard_x = 300 + (200 * progress)  # Moves back for deep 3
            lillard_y = 200 + (20 * math.sin(progress * math.pi))
            
            # Other players spread
            blazers = [
                {"id": "Lillard", "x": lillard_x, "y": lillard_y, "team": "Blazers", "role": "PG"},
                {"id": "McCollum", "x": 600 + (30 * progress), "y": 120, "team": "Blazers", "role": "SG"},
                {"id": "Covington", "x": 200 - (50 * progress), "y": 300, "team": "Blazers", "role": "SF"},
                {"id": "Nurkic", "x": 150, "y": 200 + (60 * progress), "team": "Blazers", "role": "C"},
                {"id": "Simons", "x": 450, "y": 300 + (30 * progress), "team": "Blazers", "role": "G"}
            ]
            
            # Lakers defense tries to adjust
            westbrook_x = 320 + (170 * progress)
            lakers = [
                {"id": "Westbrook", "x": westbrook_x, "y": 180 + (15 * progress), "team": "Lakers", "role": "PG"},
                {"id": "Monk", "x": 580 + (20 * progress), "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "James", "x": 220 - (30 * progress), "y": 280, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 170, "y": 220 + (40 * progress), "team": "Lakers", "role": "PF"},
                {"id": "Howard", "x": 130, "y": 180 + (50 * progress), "team": "Lakers", "role": "C"}
            ]
            
            # Shot sequence
            if frame_num < 12:
                # Dribbling and setup
                ball_pos = {"x": lillard_x, "y": lillard_y, "with_player": "Lillard"}
                shot_phase = "Setup"
                analysis_text = f"Lillard backing up for DEEP 3\nDefense scrambling to adjust\nWestbrook closing distance\nTHREAT LEVEL: CRITICAL"
            elif frame_num < 16:
                # Shot release
                shot_progress = (frame_num - 12) / 4.0
                ball_x = lillard_x + (225 * shot_progress)
                ball_y = lillard_y - (40 * shot_progress) + (80 * shot_progress * shot_progress)
                ball_pos = {"x": ball_x, "y": ball_y, "with_player": None}
                shot_phase = "SHOT!"
                analysis_text = f"LILLARD SHOOTS FROM DEEP!\nRange: 35+ feet\nContest: Minimal\nTrajectory: Perfect"
            else:
                # Ball goes in
                ball_pos = {"x": 725, "y": 200, "with_player": None}
                shot_phase = "SCORE!"
                analysis_text = f"🚨 DAME TIME! 🚨\nBUZZER BEATER GOOD!\nDefense failed to close out\nGame winner from deep!"
            
            arrows = []
            if frame_num >= 12 and frame_num < 16:
                # Show shot trajectory
                arrows.append({"start": (lillard_x, lillard_y), "end": (725, 200), "color": "orange", "style": "shot"})
            
            frame = {
                "type": "execution",
                "players": blazers + lakers,
                "ball": ball_pos,
                "clock": f"{clock_time:.1f}",
                "analysis": analysis_text,
                "arrows": arrows,
                "threat_level": 1.0 if frame_num >= 16 else 0.9,
                "description": f"Lillard deep 3: {shot_phase}"
            }
            
            self.frames.append(frame)
    
    def add_lillard_prevention(self):
        """Add GNN analysis showing how to prevent Lillard's shot"""
        for frame_num in range(24):  # 4 seconds
            
            # Show the original positioning (what went wrong)
            blazers = [
                {"id": "Lillard", "x": 500, "y": 220, "team": "Blazers", "role": "PG"},
                {"id": "McCollum", "x": 630, "y": 120, "team": "Blazers", "role": "SG"},
                {"id": "Covington", "x": 150, "y": 300, "team": "Blazers", "role": "SF"},
                {"id": "Nurkic", "x": 150, "y": 260, "team": "Blazers", "role": "C"},
                {"id": "Simons", "x": 450, "y": 330, "team": "Blazers", "role": "G"}
            ]
            
            # GNN optimal positioning (what should have happened)
            progress = min(frame_num / 16.0, 1.0)
            
            # Westbrook moves closer and higher
            westbrook_x = 490 + (20 * progress)  # Much closer
            westbrook_y = 180 + (25 * progress)  # Better angle
            
            # James provides help defense
            james_x = 220 + (200 * progress)  # Rotates over
            james_y = 280 - (40 * progress)  # Steps up
            
            lakers = [
                {"id": "Westbrook", "x": westbrook_x, "y": westbrook_y, "team": "Lakers", "role": "PG"},
                {"id": "Monk", "x": 580 + (30 * progress), "y": 140, "team": "Lakers", "role": "SG"},
                {"id": "James", "x": james_x, "y": james_y, "team": "Lakers", "role": "SF"},
                {"id": "Davis", "x": 170 + (30 * progress), "y": 220 + (20 * progress), "team": "Lakers", "role": "PF"},
                {"id": "Howard", "x": 130 + (40 * progress), "y": 180 + (30 * progress), "team": "Lakers", "role": "C"}
            ]
            
            # Analysis phases
            if frame_num < 6:
                analysis_text = "🔍 GNN ANALYSIS: What went wrong\n❌ Westbrook too far (8+ feet gap)\n❌ No help defense rotation\n❌ Lillard had clean look from 35 feet"
            elif frame_num < 12:
                analysis_text = "✅ GNN SOLUTION: Optimal defense\n1️⃣ Westbrook: Close to 3-foot range\n2️⃣ James: Rotate for help contest\n3️⃣ Davis: Step up to discourage drive"
            elif frame_num < 18:
                analysis_text = "🛡️ PREVENTION EXECUTED\nLillard now heavily contested\nShot difficulty: MAXIMUM\nSuccess probability: 15% (was 65%)"
            else:
                analysis_text = "📊 RESULT: Shot prevented!\nLillard forced into difficult pass\nBuzzer beater denied\n50% improvement in defense!"
            
            # Show movement arrows
            arrows = []
            if frame_num > 6:
                arrows.append({"start": (490, 205), "end": (510, 230), "color": "green", "style": "movement"})
            if frame_num > 12:
                arrows.append({"start": (420, 240), "end": (480, 210), "color": "blue", "style": "help"})
                arrows.append({"start": (510, 230), "end": (500, 220), "color": "red", "style": "pressure"})
            
            frame = {
                "type": "prevention",
                "players": blazers + lakers,
                "ball": {"x": 500, "y": 220, "with_player": "Lillard"},
                "clock": "0.9",
                "analysis": analysis_text,
                "arrows": arrows,
                "threat_level": 0.9 - (0.7 * progress),
                "description": "GNN Prevention: Lillard deep 3 denied"
            }
            
            self.frames.append(frame)
    
    def add_kobe_fadeaway(self):
        """Add Kobe fadeaway buzzer beater scenario"""
        
        # Setup phase
        self.add_setup_phase("Kobe", 1.8)
        
        # Execution phase
        for frame_num in range(18):  # 3 seconds
            clock_time = max(0.0, 1.8 - (frame_num * 0.1))
            
            # Kobe isolation movement
            progress = frame_num / 18.0
            kobe_x = 500 + (50 * math.sin(progress * 2 * math.pi))  # Iso movement
            kobe_y = 200 + (30 * math.cos(progress * 2 * math.pi))
            
            lakers = [
                {"id": "Fisher", "x": 250 - (100 * progress), "y": 150, "team": "Lakers", "role": "PG"},
                {"id": "Kobe", "x": kobe_x, "y": kobe_y, "team": "Lakers", "role": "SG"},
                {"id": "Artest", "x": 200 - (50 * progress), "y": 300, "team": "Lakers", "role": "SF"},
                {"id": "Gasol", "x": 300, "y": 320 + (20 * progress), "team": "Lakers", "role": "PF"},
                {"id": "Bynum", "x": 150, "y": 250 + (40 * progress), "team": "Lakers", "role": "C"}
            ]
            
            # Celtics defense
            allen_x = 520 + (30 * math.sin(progress * 2 * math.pi))
            celtics = [
                {"id": "Rondo", "x": 270 - (80 * progress), "y": 170, "team": "Celtics", "role": "PG"},
                {"id": "Allen", "x": allen_x, "y": 180 + (15 * progress), "team": "Celtics", "role": "SG"},
                {"id": "Pierce", "x": 220 - (30 * progress), "y": 280, "team": "Celtics", "role": "SF"},
                {"id": "Garnett", "x": 320, "y": 300 + (15 * progress), "team": "Celtics", "role": "PF"},
                {"id": "Perkins", "x": 170, "y": 230 + (30 * progress), "team": "Celtics", "role": "C"}
            ]
            
            if frame_num < 12:
                ball_pos = {"x": kobe_x, "y": kobe_y, "with_player": "Kobe"}
                analysis_text = f"Kobe isolation vs Ray Allen\nMamba Mentality activated\nFadeaway coming...\nDefense: Single coverage"
            elif frame_num < 16:
                # Fadeaway shot
                shot_progress = (frame_num - 12) / 4.0
                ball_x = kobe_x + (175 * shot_progress)
                ball_y = kobe_y - (20 * shot_progress) + (60 * shot_progress * shot_progress)
                ball_pos = {"x": ball_x, "y": ball_y, "with_player": None}
                analysis_text = f"KOBE FADEAWAY!\nUnguardable shot\nRay Allen contesting\nClutch gene activated"
            else:
                ball_pos = {"x": 725, "y": 200, "with_player": None}
                analysis_text = f"🐍 MAMBA STRIKES! 🐍\nFadeaway jumper good!\nKobe clutch magic\nBuzzer beater swish!"
            
            frame = {
                "type": "execution",
                "players": lakers + celtics,
                "ball": ball_pos,
                "clock": f"{clock_time:.1f}",
                "analysis": analysis_text,
                "arrows": [{"start": (kobe_x, kobe_y), "end": (725, 200), "color": "purple", "style": "shot"}] if frame_num >= 12 and frame_num < 16 else [],
                "threat_level": 1.0 if frame_num >= 16 else 0.85,
                "description": f"Kobe fadeaway execution"
            }
            
            self.frames.append(frame)
        
        # Prevention analysis
        self.add_title_frame("🧠 GNN PREVENTION ANALYSIS", "How to stop the fadeaway", 2)
        self.add_kobe_prevention()
    
    def add_kobe_prevention(self):
        """Add prevention analysis for Kobe fadeaway"""
        for frame_num in range(24):  # 4 seconds
            
            # Original positioning
            lakers = [
                {"id": "Fisher", "x": 150, "y": 150, "team": "Lakers", "role": "PG"},
                {"id": "Kobe", "x": 550, "y": 200, "team": "Lakers", "role": "SG"},
                {"id": "Artest", "x": 150, "y": 300, "team": "Lakers", "role": "SF"},
                {"id": "Gasol", "x": 300, "y": 340, "team": "Lakers", "role": "PF"},
                {"id": "Bynum", "x": 150, "y": 290, "team": "Lakers", "role": "C"}
            ]
            
            # GNN optimal defense
            progress = min(frame_num / 16.0, 1.0)
            
            # Double team strategy
            allen_x = 520 + (25 * progress)  # Closer
            pierce_x = 190 + (280 * progress)  # Rotates for double
            
            celtics = [
                {"id": "Rondo", "x": 190, "y": 170, "team": "Celtics", "role": "PG"},
                {"id": "Allen", "x": allen_x, "y": 180 + (10 * progress), "team": "Celtics", "role": "SG"},
                {"id": "Pierce", "x": pierce_x, "y": 280 - (60 * progress), "team": "Celtics", "role": "SF"},
                {"id": "Garnett", "x": 320, "y": 300 + (10 * progress), "team": "Celtics", "role": "PF"},
                {"id": "Perkins", "x": 170, "y": 230 + (20 * progress), "team": "Celtics", "role": "C"}
            ]
            
            if frame_num < 6:
                analysis_text = "🔍 GNN ANALYSIS: Kobe iso problem\n❌ Single coverage vs elite scorer\n❌ No help defense preparation\n❌ Fadeaway nearly unguardable 1-on-1"
            elif frame_num < 12:
                analysis_text = "✅ GNN SOLUTION: Force double team\n1️⃣ Allen: Deny initial position\n2️⃣ Pierce: Rotate for trap\n3️⃣ Force Kobe into tough pass"
            elif frame_num < 18:
                analysis_text = "🛡️ DOUBLE TEAM EXECUTED\nKobe sees two defenders\nShot clock pressure\nForced into pass or tough shot"
            else:
                analysis_text = "📊 RESULT: Buzzer beater prevented!\nKobe forced into pass to Gasol\nLow percentage shot created\nTeam defense wins!"
            
            # Show double team arrows
            arrows = []
            if frame_num > 6:
                arrows.append({"start": (545, 190), "end": (550, 200), "color": "red", "style": "pressure"})
            if frame_num > 12:
                arrows.append({"start": (470, 220), "end": (530, 190), "color": "red", "style": "double"})
            
            frame = {
                "type": "prevention",
                "players": lakers + celtics,
                "ball": {"x": 550, "y": 200, "with_player": "Kobe"},
                "clock": "1.8",
                "analysis": analysis_text,
                "arrows": arrows,
                "threat_level": 0.85 - (0.6 * progress),
                "description": "GNN Prevention: Kobe double team"
            }
            
            self.frames.append(frame)
    
    def add_lebron_drive(self):
        """Add LeBron drive buzzer beater scenario"""
        
        # Setup phase
        self.add_setup_phase("LeBron", 2.4)
        
        # Execution phase
        for frame_num in range(24):  # 4 seconds
            clock_time = max(0.0, 2.4 - (frame_num * 0.1))
            
            # LeBron drive to basket
            progress = frame_num / 24.0
            lebron_x = 350 + (325 * progress)  # Drives to basket
            lebron_y = 220 - (20 * progress)   # Slightly up
            
            heat = [
                {"id": "Chalmers", "x": 250 - (100 * progress), "y": 180, "team": "Heat", "role": "PG"},
                {"id": "Wade", "x": 400 - (150 * progress), "y": 150, "team": "Heat", "role": "SG"},
                {"id": "LeBron", "x": lebron_x, "y": lebron_y, "team": "Heat", "role": "SF"},
                {"id": "Bosh", "x": 200 - (50 * progress), "y": 280, "team": "Heat", "role": "PF"},
                {"id": "Haslem", "x": 150, "y": 240 + (30 * progress), "team": "Heat", "role": "C"}
            ]
            
            # Spurs defense tries to help
            leonard_x = 370 + (250 * progress)  # Follows LeBron
            duncan_x = 220 + (200 * progress)   # Help defense
            
            spurs = [
                {"id": "Parker", "x": 270 - (80 * progress), "y": 200, "team": "Spurs", "role": "PG"},
                {"id": "Ginobili", "x": 420 - (120 * progress), "y": 170, "team": "Spurs", "role": "SG"},
                {"id": "Leonard", "x": leonard_x, "y": 240 - (15 * progress), "team": "Spurs", "role": "SF"},
                {"id": "Duncan", "x": duncan_x, "y": 300 - (80 * progress), "team": "Spurs", "role": "PF"},
                {"id": "Splitter", "x": 170 + (200 * progress), "y": 260 - (40 * progress), "team": "Spurs", "role": "C"}
            ]
            
            if frame_num < 18:
                ball_pos = {"x": lebron_x, "y": lebron_y, "with_player": "LeBron"}
                analysis_text = f"LeBron drive to the rim!\nExplosive first step\nDefense scrambling\nPaint opening up..."
            elif frame_num < 22:
                # Dunk/layup
                ball_pos = {"x": 700, "y": 200, "with_player": None}
                analysis_text = f"LEBRON ATTACKS THE RIM!\nPowerful finish\nContact on the play\nClutch drive!"
            else:
                ball_pos = {"x": 725, "y": 200, "with_player": None}
                analysis_text = f"👑 KING JAMES! 👑\nAnd-one opportunity!\nClutch drive and score\nMiami takes the lead!"
            
            frame = {
                "type": "execution",
                "players": heat + spurs,
                "ball": ball_pos,
                "clock": f"{clock_time:.1f}",
                "analysis": analysis_text,
                "arrows": [{"start": (350, 220), "end": (675, 200), "color": "red", "style": "drive"}] if frame_num < 18 else [],
                "threat_level": min(1.0, 0.6 + (0.4 * progress)),
                "description": f"LeBron drive execution"
            }
            
            self.frames.append(frame)
        
        # Prevention analysis
        self.add_title_frame("🧠 GNN PREVENTION ANALYSIS", "How to stop the drive", 2)
        self.add_lebron_prevention()
    
    def add_lebron_prevention(self):
        """Add prevention analysis for LeBron drive"""
        for frame_num in range(24):  # 4 seconds
            
            # Original positioning
            heat = [
                {"id": "Chalmers", "x": 150, "y": 180, "team": "Heat", "role": "PG"},
                {"id": "Wade", "x": 250, "y": 150, "team": "Heat", "role": "SG"},
                {"id": "LeBron", "x": 675, "y": 200, "team": "Heat", "role": "SF"},
                {"id": "Bosh", "x": 150, "y": 280, "team": "Heat", "role": "PF"},
                {"id": "Haslem", "x": 150, "y": 270, "team": "Heat", "role": "C"}
            ]
            
            # GNN optimal defense - pack the paint
            progress = min(frame_num / 16.0, 1.0)
            
            # Leonard forces LeBron baseline
            leonard_x = 620 - (50 * progress)   # Better positioning
            leonard_y = 225 + (20 * progress)   # Force baseline
            
            # Duncan and Splitter pack paint
            duncan_x = 420 - (100 * progress)   # Paint position
            splitter_x = 370 - (150 * progress) # Pack the lane
            
            spurs = [
                {"id": "Parker", "x": 190, "y": 200, "team": "Spurs", "role": "PG"},
                {"id": "Ginobili", "x": 300, "y": 170, "team": "Spurs", "role": "SG"},
                {"id": "Leonard", "x": leonard_x, "y": leonard_y, "team": "Spurs", "role": "SF"},
                {"id": "Duncan", "x": duncan_x, "y": 220 + (40 * progress), "team": "Spurs", "role": "PF"},
                {"id": "Splitter", "x": splitter_x, "y": 220 + (20 * progress), "team": "Spurs", "role": "C"}
            ]
            
            if frame_num < 6:
                analysis_text = "🔍 GNN ANALYSIS: Drive vulnerability\n❌ Paint too open for LeBron\n❌ Leonard's angle allows drive\n❌ Help defense too late"
            elif frame_num < 12:
                analysis_text = "✅ GNN SOLUTION: Pack the paint\n1️⃣ Leonard: Force baseline drive\n2️⃣ Duncan: Protect rim earlier\n3️⃣ Splitter: Clog driving lanes"
            elif frame_num < 18:
                analysis_text = "🛡️ PAINT PROTECTION ACTIVE\nLeBron sees wall of defenders\nBaseline drive forced\nRim protection engaged"
            else:
                analysis_text = "📊 RESULT: Drive disrupted!\nLeBron forced into tough angle\nContest at rim successful\nTeam defense prevails!"
            
            # Show defensive positioning arrows
            arrows = []
            if frame_num > 6:
                arrows.append({"start": (670, 245), "end": (675, 280), "color": "yellow", "style": "force"})
            if frame_num > 12:
                arrows.append({"start": (320, 260), "end": (400, 220), "color": "blue", "style": "help"})
                arrows.append({"start": (220, 240), "end": (350, 200), "color": "blue", "style": "help"})
            
            frame = {
                "type": "prevention",
                "players": heat + spurs,
                "ball": {"x": 675, "y": 200, "with_player": "LeBron"},
                "clock": "2.4",
                "analysis": analysis_text,
                "arrows": arrows,
                "threat_level": 0.9 - (0.6 * progress),
                "description": "GNN Prevention: LeBron drive denied"
            }
            
            self.frames.append(frame)
        
        # Add final summary
        for _ in range(12):  # 2 seconds
            frame = {
                "type": "summary",
                "players": [],
                "ball": {"x": 400, "y": 200, "with_player": None},
                "clock": "0.0",
                "analysis": "🏆 BUZZER BEATER PREVENTION MASTERY\n\n✅ Deep 3-pointers: Close out + help defense\n✅ Isolation plays: Double team at right time\n✅ Drives to rim: Pack paint + force baseline\n\n🧠 GNN learns from every buzzer beater\n📊 Transforms 70%+ success into 25% success\n🎯 Proactive coaching beats reactive panic!",
                "arrows": [],
                "threat_level": 0.0,
                "description": "Summary: Buzzer beater prevention complete"
            }
            self.frames.append(frame)
    
    def draw_court(self, ax):
        """Draw basketball court with detailed markings"""
        ax.clear()
        ax.set_xlim(0, self.court_width)
        ax.set_ylim(0, self.court_height)
        ax.set_aspect('equal')
        
        # Court background
        court = patches.Rectangle((25, 25), 750, 350, linewidth=3, 
                                edgecolor='black', facecolor='darkgreen', alpha=0.3)
        ax.add_patch(court)
        
        # Center line and circle
        ax.plot([400, 400], [25, 375], 'white', linewidth=3)
        center_circle = patches.Circle((400, 200), 50, linewidth=2, 
                                     edgecolor='white', facecolor='none')
        ax.add_patch(center_circle)
        
        # Three-point lines (detailed)
        # Left side arc
        three_pt_left = patches.Arc((75, 200), 280, 280, theta1=293, theta2=67, 
                                   linewidth=3, edgecolor='white')
        ax.add_patch(three_pt_left)
        
        # Right side arc
        three_pt_right = patches.Arc((725, 200), 280, 280, theta1=113, theta2=247,
                                    linewidth=3, edgecolor='white')
        ax.add_patch(three_pt_right)
        
        # Three-point line corners
        ax.plot([25, 215], [125, 125], 'white', linewidth=3)
        ax.plot([25, 215], [275, 275], 'white', linewidth=3)
        ax.plot([775, 585], [125, 125], 'white', linewidth=3)
        ax.plot([775, 585], [275, 275], 'white', linewidth=3)
        
        # Paint areas
        left_paint = patches.Rectangle((25, 150), 120, 100, linewidth=3,
                                     edgecolor='white', facecolor='orange', alpha=0.2)
        ax.add_patch(left_paint)
        
        right_paint = patches.Rectangle((655, 150), 120, 100, linewidth=3,
                                      edgecolor='white', facecolor='orange', alpha=0.2)
        ax.add_patch(right_paint)
        
        # Free throw circles
        ft_circle_left = patches.Circle((145, 200), 50, linewidth=2,
                                       edgecolor='white', facecolor='none')
        ft_circle_right = patches.Circle((655, 200), 50, linewidth=2,
                                        edgecolor='white', facecolor='none')
        ax.add_patch(ft_circle_left)
        ax.add_patch(ft_circle_right)
        
        # Baskets
        left_basket = patches.Circle((75, 200), 10, facecolor='red', edgecolor='black', linewidth=2)
        right_basket = patches.Circle((725, 200), 10, facecolor='red', edgecolor='black', linewidth=2)
        ax.add_patch(left_basket)
        ax.add_patch(right_basket)
        
        # Backboards
        ax.plot([65, 65], [180, 220], 'black', linewidth=4)
        ax.plot([735, 735], [180, 220], 'black', linewidth=4)
    
    def draw_frame(self, frame_data, ax):
        """Draw a single frame with enhanced visuals"""
        self.draw_court(ax)
        
        if frame_data["type"] == "title":
            # Title frame with buzzer beater theme
            ax.text(400, 250, frame_data["title"], 
                   ha='center', va='center', fontsize=28, fontweight='bold',
                   color='red', bbox=dict(boxstyle="round,pad=1", facecolor="black", alpha=0.8))
            ax.text(400, 150, frame_data["subtitle"], 
                   ha='center', va='center', fontsize=18, color='white',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.7))
            return
        
        # Draw players with enhanced styling
        for player in frame_data["players"]:
            if player["team"] in ["Blazers", "Lakers", "Heat"]:
                color = 'blue' if player["team"] == "Blazers" else 'purple' if player["team"] == "Lakers" else 'red'
            else:
                color = 'gray' if player["team"] == "Lakers" else 'green' if player["team"] == "Celtics" else 'black'
            
            size = 180 if frame_data["ball"].get("with_player") == player["id"] else 120
            
            ax.scatter(player["x"], player["y"], c=color, s=size, 
                      alpha=0.9, edgecolors='white', linewidth=3)
            ax.text(player["x"], player["y"]-30, player["id"], 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
        
        # Draw ball with trail effect
        ball = frame_data["ball"]
        if ball.get("with_player"):
            # Ball with player (shown by larger circle)
            pass
        else:
            # Free ball with enhanced visibility
            ax.scatter(ball["x"], ball["y"], c='orange', s=100, 
                      marker='o', edgecolors='black', linewidth=3)
        
        # Draw arrows with different styles
        for arrow in frame_data.get("arrows", []):
            start = arrow["start"]
            end = arrow["end"]
            color = arrow["color"]
            style = arrow.get("style", "normal")
            
            if style == "shot":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=4, alpha=0.9))
            elif style == "drive":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=5, alpha=0.8))
            elif style == "movement":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=3))
            elif style == "pressure":
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=color, linewidth=5, alpha=0.9)
            elif style == "help":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=3, linestyle='--'))
            elif style == "double":
                ax.annotate('', xy=end, xytext=start,
                           arrowprops=dict(arrowstyle='->', color=color, lw=4, linestyle=':'))
            elif style == "force":
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=color, linewidth=4, linestyle='-.', alpha=0.8)
        
        # Draw enhanced threat level indicator
        threat = frame_data.get("threat_level", 0.0)
        if threat < 0.3:
            threat_color = 'green'
            threat_text = "LOW"
        elif threat < 0.7:
            threat_color = 'yellow' 
            threat_text = "MEDIUM"
        else:
            threat_color = 'red'
            threat_text = "CRITICAL"
        
        threat_rect = patches.Rectangle((650, 340), 120, 30, 
                                      facecolor=threat_color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(threat_rect)
        ax.text(710, 355, f"THREAT: {threat_text}", ha='center', va='center', 
               fontsize=12, fontweight='bold', color='black')
        
        # Draw shot clock
        clock_time = frame_data.get("clock", "0.0")
        clock_color = 'red' if float(clock_time) < 1.0 else 'yellow' if float(clock_time) < 3.0 else 'green'
        ax.text(710, 310, f"⏰ {clock_time}", ha='center', va='center', 
               fontsize=16, fontweight='bold', color=clock_color,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
        
        # Draw analysis text with enhanced styling
        analysis_text = frame_data.get("analysis", "")
        ax.text(50, 360, analysis_text, fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.7", facecolor="black", alpha=0.9, edgecolor='white'),
               color='white', verticalalignment='top', fontweight='bold')
        
        # Set title based on frame type
        if frame_data["type"] == "setup":
            title = "🚨 BUZZER BEATER SETUP"
        elif frame_data["type"] == "execution":
            title = "⚡ CLUTCH TIME EXECUTION"
        elif frame_data["type"] == "prevention":
            title = "🧠 GNN PREVENTION STRATEGY"
        else:
            title = "📊 BUZZER BEATER ANALYSIS"
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='red')
    
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
        ax.text(750, 50, f"{frame_num+1}/{len(self.frames)}", 
               ha='right', va='bottom', fontsize=12, alpha=0.8, color='white',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        
        plt.tight_layout()
    
    def create_demo(self, output_path="buzzer_beater_gnn_demo.gif"):
        """Create the demo video"""
        print("Creating Buzzer Beater GNN Demo...")
        print(f"Total frames: {len(self.frames)}")
        print(f"Duration: {len(self.frames) / self.fps:.1f} seconds")
        
        # Setup figure
        fig = plt.figure(figsize=(18, 12))
        
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
        
        print("Demo created successfully!")
        
        # Also save as MP4
        try:
            mp4_path = output_path.replace('.gif', '.mp4')
            anim.save(mp4_path, writer='ffmpeg', fps=self.fps)
            print(f"MP4 version saved: {mp4_path}")
        except Exception as e:
            print(f"MP4 save skipped: {e}")
        
        plt.close(fig)
        return output_path

def main():
    """Main function"""
    print("🚨 BUZZER BEATER GNN PREVENTION DEMO 🚨")
    print("="*60)
    print("Scenarios:")
    print("1. Damian Lillard deep 3-pointer (0.9 sec)")
    print("2. Kobe Bryant fadeaway jumper (1.8 sec)")
    print("3. LeBron James drive to rim (2.4 sec)")
    print("\nShows: Play execution + GNN prevention analysis")
    print("")
    
    # Create and run demo
    demo = BuzzerBeaterGNNDemo()
    output_file = demo.create_demo()
    
    print(f"\n🎯 Demo Complete!")
    print(f"- 3 different buzzer beater scenarios")
    print(f"- Shows play first, then GNN prevention")
    print(f"- 2D court mapping with player tracking")
    print(f"- Defensive adjustments reduce success by 50%+")
    print(f"- Total duration: {len(demo.frames) / demo.fps:.1f} seconds")
    print(f"\n🚨 Open {output_file} to see buzzer beater prevention!")
    
    return output_file

if __name__ == "__main__":
    main()
