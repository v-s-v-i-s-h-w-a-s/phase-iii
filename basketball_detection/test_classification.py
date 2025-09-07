"""
Test team classification functionality
"""

import cv2
import numpy as np
from src.improved_team_classifier import ImprovedTeamClassifier
from ultralytics import YOLO

def test_classification():
    """Test actual team classification"""
    print("🔍 Testing Team Classification...")
    
    # Initialize and set up teams (using our detected teams)
    classifier = ImprovedTeamClassifier()
    
    # Manually set up teams for testing (from our previous results)
    classifier.team_profiles = {
        'team_1': {
            'avg_color': np.array([81, 64, 127]),
            'color_std': np.array([13.5, 10.0, 25.3]),
            'avg_hsv': np.array([148.8, 127.0, 136.2]),
            'sample_count': 61,
            'adaptive_threshold': 0.15
        },
        'team_2': {
            'avg_color': np.array([141, 129, 171]),
            'color_std': np.array([23.9, 20.0, 28.6]),
            'avg_hsv': np.array([117.2, 113.5, 164.3]),
            'sample_count': 57,
            'adaptive_threshold': 0.18
        },
        'team_3': {
            'avg_color': np.array([49, 34, 50]),
            'color_std': np.array([15.2, 12.1, 18.4]),
            'avg_hsv': np.array([145.3, 89.2, 78.1]),
            'sample_count': 46,
            'adaptive_threshold': 0.12
        }
    }
    
    # Set up visualization colors
    classifier.team_viz_colors = {
        'team_1': (0, 0, 255),      # Red
        'team_2': (255, 0, 0),      # Blue
        'team_3': (0, 255, 0),      # Green
        'unknown': (128, 128, 128)  # Gray
    }
    
    print("✅ Team profiles loaded manually for testing")
    
    # Load video and test classification
    cap = cv2.VideoCapture('hawks_vs_knicks.mp4')
    model = YOLO('yolo11n.pt')
    
    # Skip to frame 500 for testing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        return
    
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
    
    print(f"🎯 Found {len(detections)} players in frame 500")
    
    # Test classification for each player
    classification_results = {}
    for i, detection in enumerate(detections[:5]):  # Test first 5 players
        bbox = detection['bbox']
        team = classifier.classify_player_team(frame, bbox)
        classification_results[i] = team
        
        print(f"   Player {i+1}: {team}")
        
        # Extract jersey features for debugging
        features = classifier.extract_jersey_features(frame, bbox)
        if features and features['dominant_colors']:
            primary_color = features['dominant_colors'][0][0]
            print(f"      Jersey color: RGB{primary_color}")
    
    cap.release()
    
    # Summary
    teams_found = set(classification_results.values())
    print(f"\n📊 Classification Results:")
    print(f"   - Players classified: {len(classification_results)}")
    print(f"   - Teams identified: {teams_found}")
    
    team_counts = {}
    for team in classification_results.values():
        team_counts[team] = team_counts.get(team, 0) + 1
    
    for team, count in team_counts.items():
        print(f"   - {team}: {count} players")

if __name__ == "__main__":
    test_classification()
