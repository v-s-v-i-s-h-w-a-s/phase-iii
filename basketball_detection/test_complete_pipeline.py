"""
Test the complete pipeline
"""

import cv2
import numpy as np
from src.improved_team_classifier import ImprovedTeamClassifier
from ultralytics import YOLO

def test_complete_pipeline():
    """Test the complete pipeline from start to finish"""
    print("🔍 Testing Complete Team Classification Pipeline...")
    
    # Initialize classifier
    classifier = ImprovedTeamClassifier()
    model = YOLO('yolo11n.pt')
    
    # Load video
    cap = cv2.VideoCapture('hawks_vs_knicks.mp4')
    
    # Step 1: Collect samples and detect teams
    print("📊 Step 1: Collecting samples and detecting teams...")
    
    frame_count = 0
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players
        results = model(frame, conf=0.5, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id == 0:  # Person class
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        detection = {
                            'bbox': xyxy.tolist(),
                            'confidence': conf,
                            'class': 'player'
                        }
                        detections.append(detection)
        
        # Collect samples
        classifier.collect_color_samples(frame, detections)
        frame_count += 1
    
    # Try team detection
    success = classifier.detect_teams_automatically()
    
    if not success:
        print("❌ Team detection failed")
        cap.release()
        return
    
    print(f"✅ Teams detected successfully!")
    for team_name, profile in classifier.team_profiles.items():
        print(f"   {team_name}: RGB{profile['avg_color']} (threshold: {profile['adaptive_threshold']:.1f})")
    
    # Step 2: Test classification
    print("\n🎯 Step 2: Testing classification...")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, frame = cap.read()
    
    if ret:
        # Detect players
        results = model(frame, conf=0.5, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id == 0:  # Person class
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        
                        detection = {
                            'bbox': xyxy.tolist(),
                            'confidence': conf,
                            'class': 'player'
                        }
                        detections.append(detection)
        
        print(f"   Found {len(detections)} players in test frame")
        
        # Classify each player
        team_counts = {}
        for i, detection in enumerate(detections[:10]):  # Test first 10
            bbox = detection['bbox']
            team = classifier.classify_player_team(frame, bbox)
            team_counts[team] = team_counts.get(team, 0) + 1
            
            if i < 3:  # Show details for first 3
                features = classifier.extract_jersey_features(frame, bbox)
                if features and features['dominant_colors']:
                    color = features['dominant_colors'][0][0]
                    print(f"   Player {i+1}: {team} (jersey: RGB{color})")
        
        print(f"\n📊 Classification Results:")
        for team, count in team_counts.items():
            print(f"   {team}: {count} players")
        
        # Check if we got any team classifications
        non_unknown = sum(count for team, count in team_counts.items() if team != 'unknown')
        total = sum(team_counts.values())
        
        if non_unknown > 0:
            print(f"✅ Success! {non_unknown}/{total} players classified into teams")
        else:
            print(f"❌ All players classified as unknown")
    
    cap.release()

if __name__ == "__main__":
    test_complete_pipeline()
