"""
Quick test to diagnose issues with the generalized team classifier
"""

import cv2
import numpy as np
from src.improved_team_classifier import ImprovedTeamClassifier
from ultralytics import YOLO

def test_team_classifier():
    """Test the team classifier on a few frames"""
    print("🔍 Testing Generalized Team Classifier...")
    
    # Initialize components
    classifier = ImprovedTeamClassifier()
    model = YOLO('yolo11n.pt')
    
    # Load video
    cap = cv2.VideoCapture('hawks_vs_knicks.mp4')
    
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    frame_count = 0
    player_detections = []
    
    print("📹 Processing first 100 frames to collect samples...")
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, conf=0.5, verbose=False)
        
        # Extract player detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id == 0:  # Assuming class 0 is person/player
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
        player_detections.extend(detections)
        
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"   Frame {frame_count}: {len(detections)} players, {len(classifier.color_samples)} total samples")
    
    cap.release()
    
    print(f"\n📊 Sample Collection Results:")
    print(f"   - Total frames processed: {frame_count}")
    print(f"   - Total player detections: {len(player_detections)}")
    print(f"   - Color samples collected: {len(classifier.color_samples)}")
    
    # Try team detection
    print(f"\n🎯 Attempting team detection...")
    
    if len(classifier.color_samples) >= classifier.min_samples_for_team_detection:
        success = classifier.detect_teams_automatically()
        
        if success:
            print(f"✅ Team detection successful!")
            print(f"   - Teams detected: {len(classifier.team_profiles)}")
            
            for team_name, profile in classifier.team_profiles.items():
                print(f"   - {team_name}: {profile['avg_color']} ({profile['sample_count']} samples)")
        else:
            print(f"❌ Team detection failed")
            print(f"   - Collected samples: {len(classifier.color_samples)}")
            print(f"   - Required minimum: {classifier.min_samples_for_team_detection}")
            
            # Debug: Check sample quality
            valid_samples = 0
            for sample in classifier.color_samples[:10]:  # Check first 10
                if sample['features']['dominant_colors']:
                    valid_samples += 1
                    colors = sample['features']['dominant_colors']
                    print(f"   Sample: {colors[0][0] if colors else 'No colors'}")
            
            print(f"   - Valid samples (first 10): {valid_samples}/10")
    else:
        print(f"❌ Not enough samples for team detection")
        print(f"   - Collected: {len(classifier.color_samples)}")
        print(f"   - Required: {classifier.min_samples_for_team_detection}")

if __name__ == "__main__":
    test_team_classifier()
