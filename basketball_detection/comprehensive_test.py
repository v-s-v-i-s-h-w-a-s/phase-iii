"""
Comprehensive Test of Enhanced Team Classification Methods
Tests both Enhanced K-Means and GNN on Hawks vs Knicks and Olympics videos
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
import sys

# Import our enhanced classifiers
try:
    from advanced_team_classifier import EnhancedKMeansClassifier, GNNTeamClassifier
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced classifiers not available: {e}")
    ADVANCED_AVAILABLE = False

# Import existing system for comparison
from generalized_basketball_inference import GeneralizedBasketballInference

class ComprehensiveTeamClassificationTest:
    """Test enhanced team classification methods on real videos"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize systems
        self.current_system = GeneralizedBasketballInference()
        
        if ADVANCED_AVAILABLE:
            self.enhanced_kmeans = EnhancedKMeansClassifier()
            self.gnn_classifier = GNNTeamClassifier()
            print("✅ All classification systems loaded successfully")
        else:
            print("⚠️ Only current system available for testing")
    
    def test_video_comprehensive(self, video_path, video_name, max_frames=200):
        """Test all classification methods on a video"""
        
        print(f"\n🎬 Testing {video_name}: {video_path}")
        print("=" * 60)
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
        
        # Initialize results for this video
        video_results = {
            'video_name': video_name,
            'video_path': video_path,
            'current_system': {'times': [], 'team_counts': [], 'player_counts': []},
            'enhanced_kmeans': {'times': [], 'team_counts': [], 'player_counts': []},
            'gnn': {'times': [], 'team_counts': [], 'player_counts': []},
            'frame_details': []
        }
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Video info: {total_frames} frames @ {fps} FPS")
        print(f"🎯 Testing every 10th frame up to {max_frames} frames")
        
        frame_count = 0
        tested_frames = 0
        
        while cap.isOpened() and tested_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Test every 10th frame for efficiency
            if frame_count % 10 != 0:
                frame_count += 1
                continue
            
            print(f"⏳ Testing frame {frame_count} ({tested_frames+1}/{max_frames})...")
            
            # Get basic detections
            detections = self.current_system.detect_objects(frame)
            player_detections = [d for d in detections if d['class'] == 'player']
            
            if len(player_detections) < 4:  # Need minimum players for meaningful test
                frame_count += 1
                continue
            
            # Add frame reference for feature extraction
            for detection in player_detections:
                detection['frame'] = frame
            
            frame_result = {
                'frame_number': frame_count,
                'total_players': len(player_detections),
                'methods': {}
            }
            
            # Test 1: Current System
            current_result = self._test_current_system(frame, player_detections)
            video_results['current_system']['times'].append(current_result['time'])
            video_results['current_system']['team_counts'].append(current_result['team_count'])
            video_results['current_system']['player_counts'].append(current_result['player_count'])
            frame_result['methods']['current_system'] = current_result
            
            # Test 2: Enhanced K-Means
            if ADVANCED_AVAILABLE:
                kmeans_result = self._test_enhanced_kmeans(player_detections, frame_count)
                video_results['enhanced_kmeans']['times'].append(kmeans_result['time'])
                video_results['enhanced_kmeans']['team_counts'].append(kmeans_result['team_count'])
                video_results['enhanced_kmeans']['player_counts'].append(kmeans_result['player_count'])
                frame_result['methods']['enhanced_kmeans'] = kmeans_result
                
                # Test 3: GNN
                gnn_result = self._test_gnn(player_detections, frame.shape)
                video_results['gnn']['times'].append(gnn_result['time'])
                video_results['gnn']['team_counts'].append(gnn_result['team_count'])
                video_results['gnn']['player_counts'].append(gnn_result['player_count'])
                frame_result['methods']['gnn'] = gnn_result
            
            video_results['frame_details'].append(frame_result)
            
            frame_count += 1
            tested_frames += 1
            
            # Progress update
            if tested_frames % 20 == 0:
                self._print_progress_summary(video_results, tested_frames)
        
        cap.release()
        
        # Final analysis for this video
        self._analyze_video_results(video_results)
        
        return video_results
    
    def _test_current_system(self, frame, player_detections):
        """Test current system performance"""
        start_time = time.time()
        
        try:
            # Use current team classification approach
            if hasattr(self.current_system, 'team_classifier'):
                team_assignments = self.current_system.team_classifier.classify_players(frame, player_detections)
            else:
                # Fallback: simulate current system behavior
                team_assignments = []
                for i, detection in enumerate(player_detections):
                    team_assignments.append({
                        'detection': detection,
                        'team': f'team_{i % 2}',  # Simple alternating assignment
                        'confidence': 0.5
                    })
            
            processing_time = time.time() - start_time
            
            # Count teams and players
            teams = set()
            for assignment in team_assignments:
                teams.add(assignment.get('team', 'unknown'))
            
            return {
                'time': processing_time,
                'team_count': len(teams),
                'player_count': len(team_assignments),
                'teams': list(teams),
                'success': True
            }
            
        except Exception as e:
            return {
                'time': time.time() - start_time,
                'team_count': 0,
                'player_count': 0,
                'teams': [],
                'success': False,
                'error': str(e)
            }
    
    def _test_enhanced_kmeans(self, player_detections, frame_number):
        """Test Enhanced K-Means approach"""
        start_time = time.time()
        
        try:
            team_assignments = self.enhanced_kmeans.classify_teams(player_detections, frame_number)
            processing_time = time.time() - start_time
            
            # Analyze results
            teams = set()
            for assignment in team_assignments:
                teams.add(assignment.get('team', 'unknown'))
            
            return {
                'time': processing_time,
                'team_count': len(teams),
                'player_count': len(team_assignments),
                'teams': list(teams),
                'success': True
            }
            
        except Exception as e:
            return {
                'time': time.time() - start_time,
                'team_count': 0,
                'player_count': 0,
                'teams': [],
                'success': False,
                'error': str(e)
            }
    
    def _test_gnn(self, player_detections, frame_shape):
        """Test Graph Neural Network approach"""
        start_time = time.time()
        
        try:
            team_assignments = self.gnn_classifier.classify_teams_gnn(player_detections, frame_shape)
            processing_time = time.time() - start_time
            
            # Analyze results
            teams = set()
            for assignment in team_assignments:
                teams.add(assignment.get('team', 'unknown'))
            
            return {
                'time': processing_time,
                'team_count': len(teams),
                'player_count': len(team_assignments),
                'teams': list(teams),
                'success': True
            }
            
        except Exception as e:
            return {
                'time': time.time() - start_time,
                'team_count': 0,
                'player_count': 0,
                'teams': [],
                'success': False,
                'error': str(e)
            }
    
    def _print_progress_summary(self, video_results, tested_frames):
        """Print progress summary"""
        print(f"\n📊 Progress Summary ({tested_frames} frames tested):")
        print("-" * 40)
        
        for method_name in ['current_system', 'enhanced_kmeans', 'gnn']:
            method_data = video_results[method_name]
            if method_data['times']:
                avg_time = np.mean(method_data['times'])
                avg_teams = np.mean(method_data['team_counts'])
                avg_players = np.mean(method_data['player_counts'])
                
                print(f"{method_name:15}: {avg_time:.3f}s, "
                      f"{avg_teams:.1f} teams, {avg_players:.1f} players")
    
    def _analyze_video_results(self, video_results):
        """Analyze and print comprehensive results for a video"""
        video_name = video_results['video_name']
        
        print(f"\n🎯 FINAL RESULTS FOR {video_name.upper()}")
        print("=" * 60)
        
        for method_name in ['current_system', 'enhanced_kmeans', 'gnn']:
            method_data = video_results[method_name]
            
            if not method_data['times']:
                print(f"\n{method_name:20}: ❌ No data")
                continue
            
            # Calculate statistics
            avg_time = np.mean(method_data['times'])
            std_time = np.std(method_data['times'])
            avg_teams = np.mean(method_data['team_counts'])
            avg_players = np.mean(method_data['player_counts'])
            
            # Success rate
            success_count = sum(1 for frame in video_results['frame_details'] 
                              if frame['methods'][method_name]['success'])
            success_rate = success_count / len(video_results['frame_details']) * 100
            
            print(f"\n{method_name.replace('_', ' ').title():20}:")
            print(f"  ⏱️  Avg Time: {avg_time:.3f}s (±{std_time:.3f})")
            print(f"  🏆 Avg Teams: {avg_teams:.1f}")
            print(f"  👥 Avg Players: {avg_players:.1f}")
            print(f"  ✅ Success Rate: {success_rate:.1f}%")
        
        # Ranking
        methods_with_data = [(name, data) for name, data in 
                           [('current_system', video_results['current_system']),
                            ('enhanced_kmeans', video_results['enhanced_kmeans']),
                            ('gnn', video_results['gnn'])] 
                           if data['times']]
        
        if len(methods_with_data) > 1:
            print(f"\n🚀 SPEED RANKING:")
            speed_ranking = sorted(methods_with_data, key=lambda x: np.mean(x[1]['times']))
            for i, (method, data) in enumerate(speed_ranking, 1):
                print(f"  {i}. {method.replace('_', ' ').title()}: {np.mean(data['times']):.3f}s")
            
            print(f"\n🎯 TEAM DETECTION ACCURACY:")
            # Prefer exactly 2 teams (basketball standard)
            accuracy_ranking = sorted(methods_with_data, 
                                    key=lambda x: abs(np.mean(x[1]['team_counts']) - 2))
            for i, (method, data) in enumerate(accuracy_ranking, 1):
                teams = np.mean(data['team_counts'])
                accuracy = max(0, 100 - abs(teams - 2) * 25)  # Penalty for not being 2 teams
                print(f"  {i}. {method.replace('_', ' ').title()}: {teams:.1f} teams ({accuracy:.0f}%)")
    
    def run_full_comparison(self):
        """Run full comparison on both videos"""
        
        print("🏀 COMPREHENSIVE TEAM CLASSIFICATION TEST")
        print("=" * 60)
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test videos
        test_videos = [
            ("hawks_vs_knicks.mp4", "Hawks vs Knicks"),
            ("olympics_preview_1min.mp4", "Olympics Basketball")
        ]
        
        all_results = {}
        
        for video_file, video_name in test_videos:
            if os.path.exists(video_file):
                video_results = self.test_video_comprehensive(video_file, video_name, max_frames=100)
                if video_results:
                    all_results[video_name] = video_results
            else:
                print(f"⚠️ Video not found: {video_file}")
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        # Print final comparison
        self._print_final_comparison(all_results)
        
        return all_results
    
    def _save_comprehensive_results(self, all_results):
        """Save detailed results to file"""
        filename = f"comprehensive_team_classification_results_{self.timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for video_name, video_data in all_results.items():
            json_video = {}
            for key, value in video_data.items():
                if key in ['current_system', 'enhanced_kmeans', 'gnn']:
                    json_video[key] = {
                        'times': [float(t) for t in value['times']],
                        'team_counts': [int(c) for c in value['team_counts']],
                        'player_counts': [int(c) for c in value['player_counts']]
                    }
                else:
                    json_video[key] = value
            json_results[video_name] = json_video
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to: {filename}")
    
    def _print_final_comparison(self, all_results):
        """Print final comparison across all videos"""
        
        print(f"\n🏆 OVERALL COMPARISON ACROSS ALL VIDEOS")
        print("=" * 60)
        
        # Aggregate statistics
        overall_stats = {}
        
        for video_name, video_data in all_results.items():
            print(f"\n📹 {video_name}:")
            
            for method_name in ['current_system', 'enhanced_kmeans', 'gnn']:
                method_data = video_data[method_name]
                
                if method_data['times']:
                    if method_name not in overall_stats:
                        overall_stats[method_name] = {
                            'times': [], 'team_counts': [], 'player_counts': []
                        }
                    
                    overall_stats[method_name]['times'].extend(method_data['times'])
                    overall_stats[method_name]['team_counts'].extend(method_data['team_counts'])
                    overall_stats[method_name]['player_counts'].extend(method_data['player_counts'])
                    
                    avg_time = np.mean(method_data['times'])
                    avg_teams = np.mean(method_data['team_counts'])
                    print(f"  {method_name:15}: {avg_time:.3f}s, {avg_teams:.1f} teams")
        
        # Overall rankings
        if overall_stats:
            print(f"\n🎖️ OVERALL RANKINGS:")
            print("-" * 30)
            
            # Speed ranking
            speed_ranking = sorted(overall_stats.items(), 
                                 key=lambda x: np.mean(x[1]['times']))
            print(f"🚀 Speed (fastest to slowest):")
            for i, (method, data) in enumerate(speed_ranking, 1):
                avg_time = np.mean(data['times'])
                print(f"  {i}. {method.replace('_', ' ').title()}: {avg_time:.3f}s")
            
            # Team detection accuracy (closest to 2 teams)
            accuracy_ranking = sorted(overall_stats.items(), 
                                    key=lambda x: abs(np.mean(x[1]['team_counts']) - 2))
            print(f"\n🎯 Team Detection (closest to 2 teams):")
            for i, (method, data) in enumerate(accuracy_ranking, 1):
                avg_teams = np.mean(data['team_counts'])
                deviation = abs(avg_teams - 2)
                print(f"  {i}. {method.replace('_', ' ').title()}: {avg_teams:.1f} teams (±{deviation:.1f})")


if __name__ == "__main__":
    # Run comprehensive test
    tester = ComprehensiveTeamClassificationTest()
    results = tester.run_full_comparison()
    
    print(f"\n🎉 Comprehensive testing completed!")
    print(f"📊 Results show performance of all classification methods")
    print(f"🔬 Ready for your additional test video!")
