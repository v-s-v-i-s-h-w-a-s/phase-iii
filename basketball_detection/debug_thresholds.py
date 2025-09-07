"""
Debug classification thresholds
"""

import cv2
import numpy as np
from src.improved_team_classifier import ImprovedTeamClassifier
from ultralytics import YOLO

def debug_classification():
    """Debug the classification with detailed logging"""
    print("🔍 Debugging Team Classification Thresholds...")
    
    # Initialize classifier
    classifier = ImprovedTeamClassifier()
    
    # Set up teams (with corrected thresholds)
    classifier.team_profiles = {
        'team_1': {
            'avg_color': np.array([81, 64, 127]),
            'color_std': np.array([13.5, 10.0, 25.3]),
            'avg_hsv': np.array([148.8, 127.0, 136.2]),
            'sample_count': 61,
            'adaptive_threshold': max(50.0, min(150.0, np.mean([13.5, 10.0, 25.3]) * 2.0))
        },
        'team_2': {
            'avg_color': np.array([141, 129, 171]),
            'color_std': np.array([23.9, 20.0, 28.6]),
            'avg_hsv': np.array([117.2, 113.5, 164.3]),
            'sample_count': 57,
            'adaptive_threshold': max(50.0, min(150.0, np.mean([23.9, 20.0, 28.6]) * 2.0))
        },
        'team_3': {
            'avg_color': np.array([49, 34, 50]),
            'color_std': np.array([15.2, 12.1, 18.4]),
            'avg_hsv': np.array([145.3, 89.2, 78.1]),
            'sample_count': 46,
            'adaptive_threshold': max(50.0, min(150.0, np.mean([15.2, 12.1, 18.4]) * 2.0))
        }
    }
    
    # Load video and test one player
    cap = cv2.VideoCapture('hawks_vs_knicks.mp4')
    model = YOLO('yolo11n.pt')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        return
    
    # Get one player detection
    results = model(frame, conf=0.5, verbose=False)
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if cls_id == 0:  # Person class
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    bbox = xyxy.tolist()
                    
                    print(f"🎯 Testing player at bbox: {bbox}")
                    
                    # Extract features
                    features = classifier.extract_jersey_features(frame, bbox)
                    if not features or not features['dominant_colors']:
                        print("❌ No features extracted")
                        continue
                    
                    primary_color = features['dominant_colors'][0][0]
                    hsv_features = features['hsv_features']
                    
                    print(f"   Jersey color: RGB{primary_color}")
                    if hsv_features:
                        print(f"   HSV features: H={hsv_features['hue_mean']:.1f}, S={hsv_features['saturation_mean']:.1f}, V={hsv_features['value_mean']:.1f}")
                    
                    # Test against each team
                    print(f"\n📊 Distance calculations:")
                    
                    best_team = 'unknown'
                    min_distance = float('inf')
                    
                    for team_name, profile in classifier.team_profiles.items():
                        bgr_distance = np.linalg.norm(primary_color - profile['avg_color'])
                        
                        if hsv_features:
                            hsv_distance = np.linalg.norm([
                                hsv_features['hue_mean'] - profile['avg_hsv'][0],
                                hsv_features['saturation_mean'] - profile['avg_hsv'][1],
                                hsv_features['value_mean'] - profile['avg_hsv'][2]
                            ])
                        else:
                            hsv_distance = 0
                        
                        combined_distance = 0.6 * bgr_distance + 0.4 * hsv_distance
                        threshold = profile['adaptive_threshold']  # Use threshold directly
                        
                        print(f"   {team_name}:")
                        print(f"      Team color: RGB{profile['avg_color']}")
                        print(f"      BGR distance: {bgr_distance:.2f}")
                        print(f"      HSV distance: {hsv_distance:.2f}")
                        print(f"      Combined distance: {combined_distance:.2f}")
                        print(f"      Threshold: {threshold:.2f}")
                        print(f"      Match: {'✅' if combined_distance < threshold else '❌'}")
                        
                        if combined_distance < threshold and combined_distance < min_distance:
                            min_distance = combined_distance
                            best_team = team_name
                    
                    print(f"\n🎯 Final classification: {best_team}")
                    
                    # Test with more lenient thresholds
                    print(f"\n🔧 Testing with more lenient thresholds:")
                    for team_name, profile in classifier.team_profiles.items():
                        bgr_distance = np.linalg.norm(primary_color - profile['avg_color'])
                        threshold_lenient = profile['adaptive_threshold'] * 2  # 2x threshold
                        
                        if bgr_distance < threshold_lenient:
                            print(f"   {team_name}: ✅ Match with lenient threshold ({bgr_distance:.2f} < {threshold_lenient:.2f})")
                        else:
                            print(f"   {team_name}: ❌ No match ({bgr_distance:.2f} >= {threshold_lenient:.2f})")
                    
                    cap.release()
                    return
    
    cap.release()
    print("❌ No players found")

if __name__ == "__main__":
    debug_classification()
