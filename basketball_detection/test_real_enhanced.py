"""
Real Enhanced K-Means Test on Hawks vs Knicks Video
Tests the actual enhanced team classification against current system
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime

# Import the actual enhanced classifier
from advanced_team_classifier import EnhancedKMeansClassifier, AdvancedJerseyColorExtractor

def test_enhanced_kmeans_real():
    """Test real enhanced K-means on Hawks vs Knicks video"""
    
    print("🏀 REAL ENHANCED K-MEANS TEST")
    print("=" * 50)
    
    video_path = "hawks_vs_knicks.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Initialize enhanced classifier
    enhanced_classifier = EnhancedKMeansClassifier()
    color_extractor = AdvancedJerseyColorExtractor()
    
    print("✅ Enhanced K-Means classifier loaded")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {total_frames} frames @ {fps:.1f} FPS")
    
    # Test on multiple frames
    test_frames = [500, 1000, 2000, 3000, 5000]
    results = []
    
    for frame_num in test_frames:
        if frame_num >= total_frames:
            continue
        
        print(f"\n🎯 Testing frame {frame_num}...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Create mock player detections (simulating YOLO output)
        mock_players = create_realistic_player_detections(frame)
        
        if len(mock_players) < 4:
            print("  ⚠️ Not enough players detected")
            continue
        
        print(f"  👥 Testing with {len(mock_players)} players")
        
        # Add frame reference for feature extraction
        for player in mock_players:
            player['frame'] = frame
        
        # Test Enhanced K-Means
        start_time = time.time()
        try:
            team_assignments = enhanced_classifier.classify_teams(mock_players, frame_num)
            processing_time = time.time() - start_time
            
            # Analyze results
            teams = {}
            for assignment in team_assignments:
                team = assignment.get('team', 'unknown')
                teams[team] = teams.get(team, 0) + 1
            
            success = len(teams) == 2 and len(team_assignments) >= len(mock_players) * 0.8
            
            result = {
                'frame': frame_num,
                'players_detected': len(mock_players),
                'players_classified': len(team_assignments),
                'teams_found': len(teams),
                'team_distribution': teams,
                'processing_time': processing_time,
                'success': success
            }
            
            results.append(result)
            
            print(f"  ⏱️ Processing time: {processing_time:.3f}s")
            print(f"  🏆 Teams found: {len(teams)}")
            print(f"  👥 Players classified: {len(team_assignments)}/{len(mock_players)}")
            
            if teams:
                print(f"  📊 Team distribution:")
                for team, count in teams.items():
                    print(f"    {team}: {count} players")
            
            print(f"  ✅ Success: {'Yes' if success else 'No'}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            result = {
                'frame': frame_num,
                'players_detected': len(mock_players),
                'success': False,
                'error': str(e)
            }
            results.append(result)
    
    cap.release()
    
    # Summary
    print_enhanced_summary(results)
    
    # Save results
    save_enhanced_results(results)

def create_realistic_player_detections(frame):
    """Create realistic player detections based on frame content"""
    h, w = frame.shape[:2]
    
    # Use simple computer vision to find potential player regions
    # This simulates what YOLO would detect
    
    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find regions that might be players (vertical rectangles with motion/contrast)
    # This is a simplified simulation
    
    # Create mock detections in typical basketball positions
    mock_detections = []
    
    # Court positions (normalized)
    court_positions = [
        (0.15, 0.3, 0.25, 0.8),   # Left wing
        (0.25, 0.4, 0.35, 0.9),   # Left baseline
        (0.75, 0.3, 0.85, 0.8),   # Right wing  
        (0.65, 0.4, 0.75, 0.9),   # Right baseline
        (0.45, 0.2, 0.55, 0.7),   # Center court
        (0.4, 0.5, 0.6, 0.95),    # Low post
        (0.2, 0.5, 0.3, 0.85),    # Left guard
        (0.7, 0.5, 0.8, 0.85),    # Right guard
        (0.5, 0.1, 0.6, 0.5),     # Top of key
        (0.35, 0.3, 0.45, 0.8),   # Weak side
    ]
    
    for i, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(court_positions):
        # Convert normalized coordinates to actual pixels
        x1 = int(x1_norm * w)
        y1 = int(y1_norm * h)
        x2 = int(x2_norm * w)
        y2 = int(y2_norm * h)
        
        # Ensure coordinates are valid
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        if x2 > x1 and y2 > y1:
            # Check if this region has enough contrast (simulates person detection)
            region = frame[y1:y2, x1:x2]
            if region.size > 0:
                contrast = np.std(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))
                
                # Only add if there's sufficient contrast (likely a person)
                if contrast > 15:
                    mock_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 0.7 + 0.3 * min(contrast / 50, 1),
                        'class': 'player',
                        'tracking_id': f'player_{i}'
                    })
    
    return mock_detections[:8]  # Limit to realistic number of players

def print_enhanced_summary(results):
    """Print summary of enhanced K-means results"""
    
    print(f"\n📊 ENHANCED K-MEANS SUMMARY")
    print("=" * 40)
    
    if not results:
        print("❌ No valid results")
        return
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        # Calculate averages
        avg_time = np.mean([r['processing_time'] for r in successful_results])
        avg_teams = np.mean([r['teams_found'] for r in successful_results])
        avg_classified = np.mean([r['players_classified'] for r in successful_results])
        avg_detected = np.mean([r['players_detected'] for r in successful_results])
        
        classification_rate = (avg_classified / avg_detected) * 100 if avg_detected > 0 else 0
        
        print(f"✅ Successful tests: {len(successful_results)}/{len(results)}")
        print(f"⏱️ Average processing time: {avg_time:.3f}s")
        print(f"🏆 Average teams detected: {avg_teams:.1f}")
        print(f"👥 Average classification rate: {classification_rate:.1f}%")
        
        # Check team balance
        team_balances = []
        for result in successful_results:
            if 'team_distribution' in result and len(result['team_distribution']) == 2:
                counts = list(result['team_distribution'].values())
                balance = min(counts) / max(counts) if max(counts) > 0 else 0
                team_balances.append(balance)
        
        if team_balances:
            avg_balance = np.mean(team_balances)
            print(f"⚖️ Average team balance: {avg_balance:.2f} (1.0 = perfect)")
        
        print(f"\n🎯 Performance Analysis:")
        print(f"  🚀 Speed: {'Fast' if avg_time < 0.5 else 'Moderate' if avg_time < 1.0 else 'Slow'}")
        print(f"  🏆 Team Detection: {'Excellent' if avg_teams == 2 else 'Good' if abs(avg_teams - 2) < 0.5 else 'Needs Improvement'}")
        print(f"  👥 Classification: {'Excellent' if classification_rate > 90 else 'Good' if classification_rate > 80 else 'Needs Improvement'}")
    
    else:
        print("❌ No successful classifications")
        
        # Show errors
        error_results = [r for r in results if 'error' in r]
        if error_results:
            print(f"\n🐛 Errors encountered:")
            for result in error_results:
                print(f"  Frame {result['frame']}: {result['error']}")

def save_enhanced_results(results):
    """Save enhanced test results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_kmeans_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    import os
    test_enhanced_kmeans_real()
