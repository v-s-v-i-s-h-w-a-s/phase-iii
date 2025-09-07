"""
Enhanced Basketball Team Classification System - Summary
Shows the improvements and capabilities of the new system
"""

import cv2
import json
from pathlib import Path
import pandas as pd

def display_team_analysis():
    """Display comprehensive analysis of the team classification results"""
    
    print("🏀 ENHANCED BASKETBALL TEAM CLASSIFICATION SYSTEM")
    print("=" * 60)
    print()
    
    # Check for latest analysis files
    analysis_files = list(Path(".").glob("team_classified_analysis_*_team_analysis.json"))
    detection_files = list(Path(".").glob("team_classified_analysis_*_detections.csv"))
    video_files = list(Path(".").glob("team_classified_analysis_*.mp4"))
    
    if not analysis_files:
        print("❌ No team analysis files found. Please run enhanced_team_inference.py first.")
        return
    
    # Load latest analysis
    latest_analysis = sorted(analysis_files)[-1]
    latest_detections = sorted(detection_files)[-1]
    latest_video = sorted(video_files)[-1]
    
    print(f"📊 Analysis Results from: {latest_analysis.name}")
    print(f"🎥 Enhanced Video: {latest_video.name}")
    print(f"📈 Detection Data: {latest_detections.name}")
    print()
    
    # Load and display team analysis
    with open(latest_analysis, 'r') as f:
        team_data = json.load(f)
    
    print("🎯 TEAM CLASSIFICATION RESULTS:")
    print("-" * 40)
    
    total_players = team_data.get('total_players', 0)
    team_counts = team_data.get('team_counts', {})
    team_colors = team_data.get('team_colors', {})
    
    print(f"Total Players Detected: {total_players:,}")
    print()
    
    if team_counts:
        print("Team Distribution:")
        for team, count in team_counts.items():
            if team != 'unknown':
                percentage = (count / total_players) * 100
                print(f"  🔴 {team.upper()}: {count:,} players ({percentage:.1f}%)")
        print()
    
    if team_colors:
        print("Detected Team Colors (RGB):")
        for team, color in team_colors.items():
            # Handle numpy array format
            if isinstance(color, list) and len(color) >= 3:
                rgb = tuple(color[:3][::-1])  # Convert BGR to RGB
                print(f"  🎨 {team.upper()}: {rgb}")
        print()
    
    # Load detection statistics
    if latest_detections.exists():
        df = pd.read_csv(latest_detections)
        
        print("📈 DETECTION STATISTICS:")
        print("-" * 40)
        
        # Overall detection counts
        class_counts = df['class'].value_counts()
        print("Object Detection Summary:")
        for obj_class, count in class_counts.items():
            print(f"  📍 {obj_class.title()}: {count:,} detections")
        print()
        
        # Confidence statistics
        print("Detection Quality:")
        print(f"  📊 Average Confidence: {df['confidence'].mean():.3f}")
        print(f"  📊 Minimum Confidence: {df['confidence'].min():.3f}")
        print(f"  📊 Maximum Confidence: {df['confidence'].max():.3f}")
        print()
        
        # Team-specific statistics for players
        player_df = df[df['class'] == 'player']
        if 'team' in player_df.columns:
            print("Team-Specific Player Statistics:")
            team_conf = player_df.groupby('team')['confidence'].agg(['count', 'mean', 'std']).round(3)
            for team in team_conf.index:
                if team != 'unknown':
                    count = team_conf.loc[team, 'count']
                    avg_conf = team_conf.loc[team, 'mean']
                    print(f"  🎯 {team.upper()}: {count:,} detections, avg confidence: {avg_conf:.3f}")
            print()
    
    # System capabilities
    print("🚀 SYSTEM CAPABILITIES:")
    print("-" * 40)
    print("✅ Automatic team detection based on jersey colors")
    print("✅ Real-time player classification with color-coded bounding boxes")
    print("✅ Handles partial occlusion and varying lighting conditions")
    print("✅ Temporal stability to reduce classification noise")
    print("✅ Comprehensive statistics and analysis")
    print("✅ Visual team legend and detection overlays")
    print("✅ Multi-object detection (players, referees, ball, hoops)")
    print()
    
    print("🎨 VISUAL ENHANCEMENTS:")
    print("-" * 40)
    print("🔴 Team 1: Red bounding boxes")
    print("🔵 Team 2: Blue bounding boxes") 
    print("🟢 Team 3: Green bounding boxes (if detected)")
    print("🟡 Referees: Yellow bounding boxes")
    print("🟠 Ball: Orange bounding boxes")
    print("🟣 Hoops: Purple bounding boxes")
    print("⚫ Unknown: Gray bounding boxes")
    print()
    
    print("💡 KEY IMPROVEMENTS OVER BASIC SYSTEM:")
    print("-" * 40)
    print("1. 🎯 Smart team classification using computer vision")
    print("2. 🎨 Color-based jersey analysis with K-means clustering")
    print("3. 🔄 Temporal stability for consistent team assignments")
    print("4. 📊 Enhanced visualization with team legends")
    print("5. 📈 Detailed analytics and performance metrics")
    print("6. 🛡️ Robust handling of occlusion and lighting changes")
    print("7. ⚡ Real-time processing capabilities")
    print()
    
    file_size = latest_video.stat().st_size / (1024 * 1024) if latest_video.exists() else 0
    print(f"📁 Output Video Size: {file_size:.1f} MB")
    print(f"📂 Generated Files:")
    print(f"   📹 {latest_video.name}")
    print(f"   📊 {latest_detections.name}")
    print(f"   📋 {latest_analysis.name}")
    print()
    
    print("🎯 TO VIEW RESULTS:")
    print("-" * 40)
    print(f"▶️  Play video: {latest_video.name}")
    print("   - Red boxes = Team 1 players")
    print("   - Blue boxes = Team 2 players")
    print("   - Yellow boxes = Referees")
    print("   - Orange boxes = Basketball")
    print("   - Team legend shown in top-left corner")
    print("   - Detection statistics in bottom-left corner")
    print()

if __name__ == "__main__":
    display_team_analysis()
