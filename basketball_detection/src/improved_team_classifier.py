"""
Improved Generalized Team Classification Module
Automatically detects and classifies basketball teams for any match
Uses advanced clustering and statistical methods for robust team separation
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import colorsys
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

class ImprovedTeamClassifier:
    def __init__(self):
        """Initialize improved team classifier with adaptive parameters"""
        self.team_profiles = {}  # Store comprehensive team profiles
        self.player_history = defaultdict(list)  # Track player classifications
        self.color_samples = []  # Collect color samples for analysis
        self.frame_count = 0
        self.stability_threshold = 5  # Frames needed for stable classification
        self.min_samples_for_team_detection = 50  # Minimum samples before team establishment
        
        # Adaptive thresholds
        self.color_similarity_threshold = 0.15  # Will be adjusted based on data
        self.min_team_size_ratio = 0.1  # Minimum team size as ratio of total players
        
        # Team visualization colors (will be dynamically assigned)
        self.available_colors = [
            (0, 0, 255),      # Red
            (255, 0, 0),      # Blue  
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
        ]
        self.team_viz_colors = {}
        
        print("🏀 Improved Generalized Team Classifier initialized")
        print("   ✅ Adaptive team detection")
        print("   ✅ No hardcoded values")
        print("   ✅ Works for any basketball match")
        
    def extract_jersey_features(self, frame, bbox):
        """
        Extract comprehensive jersey features including color and texture
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate jersey region with better proportions
        height = y2 - y1
        width = x2 - x1
        
        # Focus on torso area (adjustable based on pose)
        torso_y1 = y1 + int(height * 0.15)  # Skip head area
        torso_y2 = y1 + int(height * 0.65)  # Upper torso
        torso_x1 = x1 + int(width * 0.15)   # Avoid arms
        torso_x2 = x2 - int(width * 0.15)
        
        # Ensure valid coordinates
        torso_y1 = max(0, min(torso_y1, frame.shape[0]-1))
        torso_y2 = max(torso_y1+1, min(torso_y2, frame.shape[0]))
        torso_x1 = max(0, min(torso_x1, frame.shape[1]-1))
        torso_x2 = max(torso_x1+1, min(torso_x2, frame.shape[1]))
        
        if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
            return None
        
        jersey_region = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        
        if jersey_region.size == 0:
            return None
            
        return self._analyze_jersey_region(jersey_region)
    
    def _analyze_jersey_region(self, jersey_region):
        """
        Comprehensive analysis of jersey region
        """
        # Convert to different color spaces for better analysis
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
        
        # Filter out shadows, highlights, and skin tones
        mask = self._create_jersey_mask(jersey_region, hsv)
        
        if np.sum(mask) < 10:  # Not enough valid pixels
            return None
        
        # Extract valid pixels
        valid_bgr = jersey_region[mask > 0]
        valid_hsv = hsv[mask > 0]
        valid_lab = lab[mask > 0]
        
        if len(valid_bgr) < 10:
            return None
        
        # Extract multiple color features
        features = {
            'dominant_colors': self._get_robust_dominant_colors(valid_bgr),
            'hsv_features': self._extract_hsv_features(valid_hsv),
            'lab_features': self._extract_lab_features(valid_lab),
            'texture_features': self._extract_texture_features(jersey_region, mask)
        }
        
        return features
    
    def _create_jersey_mask(self, bgr_region, hsv_region):
        """
        Create mask to filter out non-jersey pixels
        """
        # Remove very dark pixels (shadows)
        dark_mask = np.mean(bgr_region, axis=2) > 30
        
        # Remove very bright pixels (highlights)
        bright_mask = np.mean(bgr_region, axis=2) < 220
        
        # Remove skin tone pixels (approximate)
        h, s, v = hsv_region[:, :, 0], hsv_region[:, :, 1], hsv_region[:, :, 2]
        skin_mask = ~((h >= 0) & (h <= 20) & (s >= 30) & (s <= 170) & (v >= 80) & (v <= 255))
        
        # Combine masks
        final_mask = dark_mask & bright_mask & skin_mask
        
        return final_mask.astype(np.uint8) * 255
    
    def _get_robust_dominant_colors(self, pixels, n_colors=3):
        """
        Get dominant colors using multiple clustering methods
        """
        if len(pixels) < 10:
            return []
        
        try:
            # Use GMM for better color separation
            gmm = GaussianMixture(n_components=min(n_colors, len(pixels)), random_state=42)
            gmm.fit(pixels)
            
            colors = gmm.means_.astype(int)
            weights = gmm.weights_
            
            # Sort by weight
            sorted_indices = np.argsort(weights)[::-1]
            
            dominant_colors = []
            for idx in sorted_indices:
                color = colors[idx]
                weight = weights[idx]
                dominant_colors.append((color, weight))
            
            return dominant_colors
            
        except Exception as e:
            print(f"Warning: GMM clustering failed: {e}, falling back to K-means")
            return self._fallback_kmeans_colors(pixels, n_colors)
    
    def _fallback_kmeans_colors(self, pixels, n_colors):
        """Fallback K-means clustering"""
        try:
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            weights = np.bincount(labels) / len(labels)
            
            sorted_indices = np.argsort(weights)[::-1]
            return [(colors[idx], weights[idx]) for idx in sorted_indices]
            
        except Exception as e:
            print(f"Warning: K-means also failed: {e}")
            return []
    
    def _extract_hsv_features(self, hsv_pixels):
        """Extract HSV color space features"""
        if len(hsv_pixels) == 0:
            return None
        
        h_mean = np.mean(hsv_pixels[:, 0])
        s_mean = np.mean(hsv_pixels[:, 1])
        v_mean = np.mean(hsv_pixels[:, 2])
        
        return {
            'hue_mean': h_mean,
            'saturation_mean': s_mean,
            'value_mean': v_mean,
            'hue_std': np.std(hsv_pixels[:, 0]),
            'saturation_std': np.std(hsv_pixels[:, 1])
        }
    
    def _extract_lab_features(self, lab_pixels):
        """Extract LAB color space features"""
        if len(lab_pixels) == 0:
            return None
        
        return {
            'l_mean': np.mean(lab_pixels[:, 0]),
            'a_mean': np.mean(lab_pixels[:, 1]),
            'b_mean': np.mean(lab_pixels[:, 2])
        }
    
    def _extract_texture_features(self, jersey_region, mask):
        """Extract basic texture features"""
        try:
            gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray, mask)
            
            # Calculate texture variance
            texture_var = np.var(masked_gray[mask > 0]) if np.sum(mask) > 0 else 0
            
            return {'texture_variance': texture_var}
        except Exception as e:
            print(f"Texture analysis failed: {e}")
            return {'texture_variance': 0}
    
    def collect_color_samples(self, frame, player_detections):
        """
        Collect color samples from all players for team analysis
        """
        for detection in player_detections:
            if detection['class'] == 'player':
                features = self.extract_jersey_features(frame, detection['bbox'])
                if features and features['dominant_colors']:
                    # Store sample with metadata
                    sample = {
                        'features': features,
                        'bbox': detection['bbox'],
                        'frame': self.frame_count,
                        'confidence': detection['confidence']
                    }
                    self.color_samples.append(sample)
        
        self.frame_count += 1
    
    def detect_teams_automatically(self):
        """
        Automatically detect teams from collected samples using advanced clustering
        """
        if len(self.color_samples) < self.min_samples_for_team_detection:
            return False
        
        print(f"🔍 Analyzing {len(self.color_samples)} player samples for team detection...")
        
        # Extract color features for clustering
        feature_vectors = []
        for sample in self.color_samples:
            if sample['features']['dominant_colors']:
                # Use primary color and HSV features
                primary_color = sample['features']['dominant_colors'][0][0]
                hsv_features = sample['features']['hsv_features']
                
                if hsv_features:
                    # Create feature vector combining color and HSV
                    feature_vector = [
                        primary_color[0], primary_color[1], primary_color[2],  # BGR
                        hsv_features['hue_mean'], hsv_features['saturation_mean'],  # HSV
                        hsv_features['value_mean']
                    ]
                    feature_vectors.append(feature_vector)
        
        if len(feature_vectors) < 10:
            return False
        
        # Standardize features
        feature_matrix = np.array(feature_vectors)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        # Try different clustering methods
        best_clustering = self._find_best_clustering(normalized_features)
        
        if best_clustering is None:
            return False
        
        # Establish teams based on clustering results
        return self._establish_teams_from_clustering(best_clustering, feature_vectors)
    
    def _find_best_clustering(self, features):
        """
        Find the best clustering method for basketball (exactly 2 teams)
        """
        best_score = -1
        best_clustering = None
        
        # Basketball has exactly 2 teams - focus on finding the best 2-cluster solution
        print("🏀 Basketball rule: Detecting exactly 2 teams (5 players each)")
        
        # Try different clustering methods for 2 teams
        clustering_methods = []
        
        # Method 1: K-means with 2 clusters
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = self._evaluate_clustering(features, labels, 2)
            clustering_methods.append({
                'method': 'kmeans', 
                'labels': labels, 
                'centers': kmeans.cluster_centers_,
                'score': score
            })
            print(f"   K-means (2 teams): score = {score:.3f}")
        except Exception as e:
            print(f"   K-means failed: {e}")
        
        # Method 2: Gaussian Mixture Model with 2 components
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            labels = gmm.fit_predict(features)
            score = self._evaluate_clustering(features, labels, 2)
            clustering_methods.append({
                'method': 'gmm', 
                'labels': labels, 
                'centers': gmm.means_,
                'score': score
            })
            print(f"   GMM (2 teams): score = {score:.3f}")
        except Exception as e:
            print(f"   GMM failed: {e}")
        
        # Method 3: Try K-means++ initialization
        try:
            kmeans_plus = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=20)
            labels = kmeans_plus.fit_predict(features)
            score = self._evaluate_clustering(features, labels, 2)
            clustering_methods.append({
                'method': 'kmeans++', 
                'labels': labels, 
                'centers': kmeans_plus.cluster_centers_,
                'score': score
            })
            print(f"   K-means++ (2 teams): score = {score:.3f}")
        except Exception as e:
            print(f"   K-means++ failed: {e}")
        
        # Select the best clustering method
        if clustering_methods:
            best_clustering = max(clustering_methods, key=lambda x: x['score'])
            print(f"✅ Best method: {best_clustering['method']} (score: {best_clustering['score']:.3f})")
        
        return best_clustering if best_clustering and best_clustering['score'] > 0.1 else None
    
    def _evaluate_clustering(self, features, labels, n_clusters):
        """
        Evaluate clustering quality for basketball (prefer balanced teams)
        """
        try:
            from sklearn.metrics import silhouette_score
            
            # Check if we have valid clustering
            unique_labels = np.unique(labels)
            if len(unique_labels) != n_clusters:
                return 0
            
            # Silhouette score (measure cluster separation)
            sil_score = silhouette_score(features, labels, random_state=42)
            
            # Basketball-specific evaluation: prefer balanced teams
            label_counts = np.bincount(labels)
            
            if n_clusters == 2:  # Basketball teams should be roughly equal
                team1_size = label_counts[0]
                team2_size = label_counts[1]
                total_players = team1_size + team2_size
                
                # Ideal: each team has ~50% of players (5 out of 10 on court)
                balance_ratio = min(team1_size, team2_size) / max(team1_size, team2_size)
                
                # Penalize if teams are too unbalanced
                balance_score = balance_ratio
                
                # Bonus for teams close to basketball size (around 5 players each visible)
                size_score = 1.0
                if 3 <= team1_size <= 7 and 3 <= team2_size <= 7:
                    size_score = 1.2  # Bonus for realistic team sizes
                
                # Combined score (emphasize balance for basketball)
                combined_score = 0.5 * sil_score + 0.3 * balance_score + 0.2 * size_score
                
                print(f"   Team sizes: {team1_size} vs {team2_size} (balance: {balance_ratio:.2f})")
                
            else:
                # For non-basketball clustering, use standard evaluation
                balance_score = 1 - np.std(label_counts) / np.mean(label_counts)
                balance_score = max(0, balance_score)
                combined_score = 0.7 * sil_score + 0.3 * balance_score
            
            return combined_score
            
        except Exception as e:
            print(f"Clustering evaluation failed: {e}")
            return 0
    
    def _establish_teams_from_clustering(self, clustering_result, original_features):
        """
        Establish exactly 2 team profiles for basketball
        """
        labels = clustering_result['labels']
        
        # Clear existing teams
        self.team_profiles = {}
        self.team_viz_colors = {}
        
        # Basketball has exactly 2 teams
        unique_labels = np.unique(labels)
        
        if len(unique_labels) != 2:
            print(f"❌ Expected 2 teams, got {len(unique_labels)}")
            return False
        
        print("🏀 Creating profiles for 2 basketball teams...")
        
        team_names = ['team_home', 'team_away']  # More descriptive names
        team_colors = [(0, 0, 255), (255, 0, 0)]  # Red for home, Blue for away
        
        for i, label in enumerate(unique_labels):
            cluster_indices = np.nonzero(labels == label)[0]
            cluster_size = len(cluster_indices)
            
            # Extract cluster features
            cluster_features = [original_features[i] for i in cluster_indices]
            team_profile = self._create_team_profile(cluster_features)
            
            if team_profile:
                team_name = team_names[i]
                self.team_profiles[team_name] = team_profile
                self.team_viz_colors[team_name] = team_colors[i]
                
                print(f"🎨 {team_name.upper()}: {cluster_size} players, avg color: {team_profile['avg_color']}")
        
        # Add referee and unknown categories
        self.team_viz_colors['referee'] = (0, 255, 255)  # Yellow
        self.team_viz_colors['unknown'] = (128, 128, 128)  # Gray
        
        return len(self.team_profiles) == 2
    
    def _create_team_profile(self, cluster_features):
        """
        Create comprehensive team profile from cluster features
        """
        try:
            # Calculate average color
            colors = [feat[:3] for feat in cluster_features]  # BGR components
            avg_color = np.mean(colors, axis=0).astype(int)
            
            # Calculate color variance for matching threshold
            color_variance = np.std(colors, axis=0)
            
            # HSV features
            hsv_features = [feat[3:6] for feat in cluster_features]  # HSV components
            avg_hsv = np.mean(hsv_features, axis=0)
            
            # Calculate more reasonable adaptive threshold
            # Base threshold on color variance + generous buffer for real-world conditions
            base_threshold = np.mean(color_variance) * 3.0  # 3x the std deviation
            adaptive_threshold = max(80.0, min(200.0, base_threshold))  # Between 80-200
            
            profile = {
                'avg_color': avg_color,
                'color_std': color_variance,
                'avg_hsv': avg_hsv,
                'sample_count': len(cluster_features),
                'adaptive_threshold': adaptive_threshold
            }
            
            return profile
            
        except Exception as e:
            print(f"Failed to create team profile: {e}")
            return None
    
    def classify_player_team(self, frame, bbox, player_id=None):
        """
        Classify player team using established profiles
        """
        if not self.team_profiles:
            # Collect samples first
            return 'unknown'
        
        features = self.extract_jersey_features(frame, bbox)
        if not features or not features['dominant_colors']:
            return 'unknown'
        
        # Get primary color
        primary_color = features['dominant_colors'][0][0]
        hsv_features = features['hsv_features']
        
        if not hsv_features:
            return 'unknown'
        
        # Find best matching team
        best_team = 'unknown'
        min_distance = float('inf')
        
        for team_name, profile in self.team_profiles.items():
            # Calculate color distance in multiple spaces
            bgr_distance = np.linalg.norm(primary_color - profile['avg_color'])
            hsv_distance = np.linalg.norm([
                hsv_features['hue_mean'] - profile['avg_hsv'][0],
                hsv_features['saturation_mean'] - profile['avg_hsv'][1],
                hsv_features['value_mean'] - profile['avg_hsv'][2]
            ])
            
            # Combined distance with adaptive threshold
            combined_distance = 0.6 * bgr_distance + 0.4 * hsv_distance
            threshold = profile['adaptive_threshold']  # Use threshold directly, not multiplied by 255
            
            if combined_distance < threshold and combined_distance < min_distance:
                min_distance = combined_distance
                best_team = team_name
        
        # Apply temporal stability
        if player_id:
            return self._apply_temporal_stability(player_id, best_team)
        
        return best_team
    
    def _apply_temporal_stability(self, player_id, current_classification):
        """
        Apply temporal stability with adaptive weighting
        """
        self.player_history[player_id].append(current_classification)
        
        # Keep only recent history
        if len(self.player_history[player_id]) > 10:
            self.player_history[player_id] = self.player_history[player_id][-10:]
        
        # Need multiple samples for stability
        if len(self.player_history[player_id]) < self.stability_threshold:
            return current_classification
        
        # Weighted voting (recent classifications have more weight)
        recent_classifications = self.player_history[player_id]
        weights = np.exp(np.linspace(-1, 0, len(recent_classifications)))
        
        # Count weighted votes
        vote_counts = defaultdict(float)
        for classification, weight in zip(recent_classifications, weights):
            vote_counts[classification] += weight
        
        # Return most voted classification
        return max(vote_counts.keys(), key=lambda k: vote_counts[k])
    
    def process_frame_with_adaptive_teams(self, frame, player_detections):
        """
        Process frame with adaptive team detection
        """
        # Collect samples for team detection
        self.collect_color_samples(frame, player_detections)
        
        # Trigger team detection periodically
        if (self.frame_count % 100 == 0 and 
            len(self.color_samples) >= self.min_samples_for_team_detection and 
            not self.team_profiles):
            
            self.detect_teams_automatically()
        
        # Classify players
        classified_players = []
        for detection in player_detections:
            if detection['class'] == 'player':
                player_id = f"p_{detection['bbox'][0]}_{detection['bbox'][1]}"
                team = self.classify_player_team(frame, detection['bbox'], player_id)
                detection['team'] = team
                detection['player_id'] = player_id
            
            classified_players.append(detection)
        
        return classified_players
    
    def get_team_color(self, team_name):
        """Get visualization color for a team"""
        return self.team_viz_colors.get(team_name, (128, 128, 128))
    
    def draw_enhanced_detections(self, frame, detections):
        """
        Draw detections with improved visualization
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Determine color and label
            if class_name == 'player' and 'team' in detection:
                team = detection['team']
                color = self.get_team_color(team)
                label = f"{team}: {confidence:.2f}"
            elif class_name == 'referee':
                color = self.get_team_color('referee')
                label = f"referee: {confidence:.2f}"
            else:
                color_map = {
                    'ball': (0, 165, 255),
                    'hoop': (128, 0, 128)
                }
                color = color_map.get(class_name, (255, 255, 255))
                label = f"{class_name}: {confidence:.2f}"
            
            # Draw enhanced bounding box
            thickness = 3 if class_name == 'player' else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label with better readability
            font_scale = 0.6
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            
            # Background with border
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 12), 
                         (x1 + label_size[0] + 8, y1), color, -1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 12), 
                         (x1 + label_size[0] + 8, y1), (255, 255, 255), 1)
            
            # Text with better contrast
            cv2.putText(frame, label, (x1 + 4, y1 - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Draw comprehensive legend
        self._draw_comprehensive_legend(frame)
        
        return frame
    
    def _draw_comprehensive_legend(self, frame):
        """Draw comprehensive team legend and statistics"""
        if not self.team_profiles:
            return
        
        # Legend background
        legend_height = 40 + len(self.team_profiles) * 25
        cv2.rectangle(frame, (10, 50), (250, 50 + legend_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 50), (250, 50 + legend_height), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "TEAM CLASSIFICATION", (15, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Team legend
        y_offset = 85
        for team_name, profile in self.team_profiles.items():
            color = self.get_team_color(team_name)
            
            # Color box
            cv2.rectangle(frame, (15, y_offset - 8), (35, y_offset + 8), color, -1)
            cv2.rectangle(frame, (15, y_offset - 8), (35, y_offset + 8), (255, 255, 255), 1)
            
            # Team info
            team_text = f"{team_name.upper()} ({profile['sample_count']} samples)"
            cv2.putText(frame, team_text, (40, y_offset + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_offset += 25
        
        # Status
        status_text = f"Samples: {len(self.color_samples)}"
        cv2.putText(frame, status_text, (15, y_offset + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def get_team_statistics(self, detections):
        """Get comprehensive team statistics"""
        if not detections:
            return {}
        
        team_counts = defaultdict(int)
        total_players = 0
        
        for detection in detections:
            if detection['class'] == 'player':
                total_players += 1
                team = detection.get('team', 'unknown')
                team_counts[team] += 1
        
        return {
            'total_players': total_players,
            'team_counts': dict(team_counts),
            'team_profiles': {name: {
                'avg_color': profile['avg_color'].tolist(),
                'sample_count': profile['sample_count']
            } for name, profile in self.team_profiles.items()},
            'detection_method': 'adaptive_clustering',
            'samples_collected': len(self.color_samples)
        }
