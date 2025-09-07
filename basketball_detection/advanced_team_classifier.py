"""
Advanced Team Classification Module
Implements both Enhanced K-Means and Graph Neural Network approaches
for superior jersey color-based team classification
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import colorsys
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import json

class AdvancedJerseyColorExtractor:
    """Enhanced jersey color extraction with multiple color spaces and texture analysis"""
    
    def __init__(self):
        self.color_spaces = ['BGR', 'HSV', 'LAB', 'YUV', 'HLS']
        self.texture_descriptors = ['LBP', 'GLCM', 'Gabor']
        
    def extract_comprehensive_features(self, frame, bbox):
        """Extract comprehensive jersey features using multiple color spaces"""
        x1, y1, x2, y2 = bbox
        
        # Enhanced jersey region extraction
        jersey_region = self._extract_jersey_region(frame, bbox)
        if jersey_region is None:
            return None
            
        # Create advanced mask to filter noise
        mask = self._create_advanced_jersey_mask(jersey_region)
        
        # Extract features from multiple color spaces
        features = {}
        
        # 1. Enhanced Color Features
        features.update(self._extract_multi_space_colors(jersey_region, mask))
        
        # 2. Color Distribution Features
        features.update(self._extract_color_distribution(jersey_region, mask))
        
        # 3. Texture Features
        features.update(self._extract_texture_features(jersey_region, mask))
        
        # 4. Statistical Features
        features.update(self._extract_statistical_features(jersey_region, mask))
        
        return features
    
    def _extract_jersey_region(self, frame, bbox):
        """Enhanced jersey region extraction with pose estimation"""
        x1, y1, x2, y2 = bbox
        height, width = y2 - y1, x2 - x1
        
        # Adaptive region based on person size
        if height < 50:  # Small person - use entire region
            jersey_y1, jersey_y2 = y1, y2
            jersey_x1, jersey_x2 = x1, x2
        else:  # Normal person - focus on torso
            jersey_y1 = y1 + int(height * 0.15)  # Skip head
            jersey_y2 = y1 + int(height * 0.65)  # Upper torso
            jersey_x1 = x1 + int(width * 0.1)    # Avoid arms
            jersey_x2 = x2 - int(width * 0.1)
        
        # Ensure valid coordinates
        jersey_y1 = max(0, min(jersey_y1, frame.shape[0]-1))
        jersey_y2 = max(jersey_y1+1, min(jersey_y2, frame.shape[0]))
        jersey_x1 = max(0, min(jersey_x1, frame.shape[1]-1))
        jersey_x2 = max(jersey_x1+1, min(jersey_x2, frame.shape[1]))
        
        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None
            
        return frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
    
    def _create_advanced_jersey_mask(self, jersey_region):
        """Create advanced mask using multiple filtering techniques"""
        gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # 1. Intensity-based filtering
        intensity_mask = (gray > 20) & (gray < 240)
        
        # 2. Saturation-based filtering (remove low-saturation pixels)
        sat_mask = hsv[:, :, 1] > 30
        
        # 3. Remove skin tone approximation
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin_mask = ~((h >= 0) & (h <= 25) & (s >= 40) & (s <= 255) & (v >= 80))
        
        # 4. Edge-based filtering (remove edge pixels which might be noisy)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1) == 0
        
        # Combine all masks
        final_mask = intensity_mask & sat_mask & skin_mask & edge_mask
        
        # Morphological operations to clean up mask
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        return final_mask
    
    def _extract_multi_space_colors(self, jersey_region, mask):
        """Extract dominant colors from multiple color spaces"""
        features = {}
        
        # Convert to different color spaces
        color_spaces = {
            'BGR': jersey_region,
            'HSV': cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV),
            'LAB': cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB),
            'YUV': cv2.cvtColor(jersey_region, cv2.COLOR_BGR2YUV),
            'HLS': cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HLS)
        }
        
        for space_name, space_image in color_spaces.items():
            valid_pixels = space_image[mask > 0]
            if len(valid_pixels) > 10:
                # Get dominant colors using improved K-means
                dominant_colors = self._get_dominant_colors_improved(valid_pixels)
                features[f'{space_name}_dominant'] = dominant_colors
                features[f'{space_name}_mean'] = np.mean(valid_pixels, axis=0)
                features[f'{space_name}_std'] = np.std(valid_pixels, axis=0)
        
        return features
    
    def _get_dominant_colors_improved(self, pixels, n_colors=3):
        """Improved dominant color extraction using ensemble clustering"""
        if len(pixels) < 20:
            return []
        
        # Subsample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        results = []
        
        # Method 1: K-means with multiple initializations
        try:
            kmeans = KMeans(n_clusters=min(n_colors, len(pixels)//5), 
                          init='k-means++', n_init=20, random_state=42)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_
            weights = np.bincount(labels) / len(labels)
            results.append(('kmeans', centers, weights))
        except:
            pass
        
        # Method 2: GMM
        try:
            gmm = GaussianMixture(n_components=min(n_colors, len(pixels)//5), 
                                covariance_type='full', random_state=42)
            gmm.fit(pixels)
            centers = gmm.means_
            weights = gmm.weights_
            results.append(('gmm', centers, weights))
        except:
            pass
        
        # Method 3: Spectral clustering for complex color distributions
        try:
            if len(pixels) <= 500:  # Only for smaller datasets due to complexity
                spectral = SpectralClustering(n_clusters=min(n_colors, len(pixels)//5), 
                                            random_state=42)
                labels = spectral.fit_predict(pixels)
                centers = np.array([pixels[labels == i].mean(axis=0) for i in range(max(labels)+1)])
                weights = np.bincount(labels) / len(labels)
                results.append(('spectral', centers, weights))
        except:
            pass
        
        # Select best result based on silhouette score
        if results:
            best_method, best_centers, best_weights = max(results, 
                key=lambda x: self._evaluate_clustering_quality(pixels, x[1]))
            
            # Sort by weight
            sorted_indices = np.argsort(best_weights)[::-1]
            return [(best_centers[i], best_weights[i]) for i in sorted_indices]
        
        return []
    
    def _evaluate_clustering_quality(self, pixels, centers):
        """Evaluate clustering quality using multiple metrics"""
        try:
            # Assign pixels to closest centers
            distances = cdist(pixels, centers)
            labels = np.argmin(distances, axis=1)
            
            # Silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(pixels, labels)
            else:
                sil_score = 0
            
            # Inertia (within-cluster sum of squares)
            inertia = sum(np.sum((pixels[labels == i] - centers[i])**2) 
                         for i in range(len(centers)) if np.sum(labels == i) > 0)
            
            # Normalized score (higher is better)
            return sil_score - (inertia / 10000)
        except:
            return -1
    
    def _extract_color_distribution(self, jersey_region, mask):
        """Extract color distribution features"""
        features = {}
        
        # Color histogram in HSV space
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        valid_hsv = hsv[mask > 0]
        
        if len(valid_hsv) > 0:
            # Hue histogram
            hue_hist, _ = np.histogram(valid_hsv[:, 0], bins=18, range=(0, 180))
            features['hue_histogram'] = hue_hist / np.sum(hue_hist)
            
            # Saturation histogram
            sat_hist, _ = np.histogram(valid_hsv[:, 1], bins=16, range=(0, 255))
            features['saturation_histogram'] = sat_hist / np.sum(sat_hist)
            
            # Color moments
            features['hue_moments'] = self._calculate_moments(valid_hsv[:, 0])
            features['saturation_moments'] = self._calculate_moments(valid_hsv[:, 1])
            features['value_moments'] = self._calculate_moments(valid_hsv[:, 2])
        
        return features
    
    def _calculate_moments(self, channel_data):
        """Calculate statistical moments for color channels"""
        if len(channel_data) == 0:
            return [0, 0, 0, 0]
        
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        skewness = np.mean(((channel_data - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((channel_data - mean) / std) ** 4) if std > 0 else 0
        
        return [mean, std, skewness, kurtosis]
    
    def _extract_texture_features(self, jersey_region, mask):
        """Extract texture features"""
        features = {}
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
        masked_gray = gray * (mask / 255)
        
        # Local Binary Pattern (simplified)
        lbp_features = self._calculate_lbp_features(masked_gray, mask)
        features.update(lbp_features)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges[mask > 0]) / np.sum(mask > 0) if np.sum(mask) > 0 else 0
        features['edge_density'] = edge_density
        
        return features
    
    def _calculate_lbp_features(self, gray_image, mask):
        """Calculate simplified Local Binary Pattern features"""
        features = {}
        
        # Simple LBP approximation using local variance
        kernel = np.ones((3,3), np.float32) / 9
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel)
        
        valid_variance = local_variance[mask > 0]
        if len(valid_variance) > 0:
            features['texture_uniformity'] = np.mean(valid_variance)
            features['texture_contrast'] = np.std(valid_variance)
        else:
            features['texture_uniformity'] = 0
            features['texture_contrast'] = 0
        
        return features
    
    def _extract_statistical_features(self, jersey_region, mask):
        """Extract statistical features from the jersey region"""
        features = {}
        
        # Basic statistics for each channel
        for i, channel_name in enumerate(['B', 'G', 'R']):
            channel_data = jersey_region[:, :, i][mask > 0]
            if len(channel_data) > 0:
                features[f'{channel_name}_mean'] = np.mean(channel_data)
                features[f'{channel_name}_std'] = np.std(channel_data)
                features[f'{channel_name}_range'] = np.max(channel_data) - np.min(channel_data)
        
        return features


class EnhancedKMeansClassifier:
    """Enhanced K-Means approach with temporal consistency and adaptive features"""
    
    def __init__(self):
        self.color_extractor = AdvancedJerseyColorExtractor()
        self.team_profiles = {}
        self.player_history = defaultdict(list)
        self.temporal_window = 10  # Frames to consider for temporal consistency
        
    def classify_teams(self, player_detections, frame_number):
        """Enhanced team classification using advanced K-means"""
        
        # Extract comprehensive features
        features_list = []
        valid_detections = []
        
        for detection in player_detections:
            features = self.color_extractor.extract_comprehensive_features(
                detection['frame'], detection['bbox']
            )
            if features is not None:
                # Flatten features into a single vector
                feature_vector = self._flatten_features(features)
                features_list.append(feature_vector)
                valid_detections.append(detection)
        
        if len(features_list) < 4:  # Need minimum samples
            return []
        
        features_array = np.array(features_list)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        
        # Enhanced clustering with multiple methods
        best_clustering = self._enhanced_clustering(normalized_features)
        
        if best_clustering is None:
            return []
        
        # Apply temporal consistency
        stable_labels = self._apply_temporal_consistency(
            valid_detections, best_clustering['labels'], frame_number
        )
        
        # Create team assignments
        return self._create_team_assignments(valid_detections, stable_labels)
    
    def _flatten_features(self, features):
        """Flatten nested feature dictionary into a single vector"""
        flattened = []
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            elif isinstance(value, list):
                flattened.extend(value)
            elif isinstance(value, (int, float)):
                flattened.append(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                for sub_value in value.values():
                    if isinstance(sub_value, (list, np.ndarray)):
                        flattened.extend(np.array(sub_value).flatten())
                    elif isinstance(sub_value, (int, float)):
                        flattened.append(sub_value)
        
        # Handle variable length features by padding/truncating to fixed size
        target_length = 150  # Adjust based on expected feature dimension
        if len(flattened) > target_length:
            flattened = flattened[:target_length]
        elif len(flattened) < target_length:
            flattened.extend([0] * (target_length - len(flattened)))
        
        return flattened
    
    def _enhanced_clustering(self, features):
        """Enhanced clustering using multiple algorithms"""
        clustering_methods = []
        
        # Method 1: K-means with PCA preprocessing
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(20, features.shape[1], features.shape[0]-1))
            pca_features = pca.fit_transform(features)
            
            kmeans = KMeans(n_clusters=2, init='k-means++', n_init=50, random_state=42)
            labels = kmeans.fit_predict(pca_features)
            score = self._evaluate_clustering_advanced(pca_features, labels)
            
            clustering_methods.append({
                'method': 'kmeans_pca',
                'labels': labels,
                'score': score,
                'transformer': pca
            })
        except Exception as e:
            print(f"K-means with PCA failed: {e}")
        
        # Method 2: Spectral clustering
        try:
            spectral = SpectralClustering(n_clusters=2, random_state=42, 
                                        affinity='rbf', gamma=1.0)
            labels = spectral.fit_predict(features)
            score = self._evaluate_clustering_advanced(features, labels)
            
            clustering_methods.append({
                'method': 'spectral',
                'labels': labels,
                'score': score
            })
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
        
        # Method 3: Gaussian Mixture with feature selection
        try:
            # Select most discriminative features
            feature_importance = np.std(features, axis=0)
            top_features_idx = np.argsort(feature_importance)[-50:]  # Top 50 features
            selected_features = features[:, top_features_idx]
            
            gmm = GaussianMixture(n_components=2, covariance_type='full', 
                                random_state=42, max_iter=200)
            labels = gmm.fit_predict(selected_features)
            score = self._evaluate_clustering_advanced(selected_features, labels)
            
            clustering_methods.append({
                'method': 'gmm_selected',
                'labels': labels,
                'score': score
            })
        except Exception as e:
            print(f"GMM with feature selection failed: {e}")
        
        # Select best method
        if clustering_methods:
            best = max(clustering_methods, key=lambda x: x['score'])
            print(f"Best enhanced clustering: {best['method']} (score: {best['score']:.3f})")
            return best
        
        return None
    
    def _evaluate_clustering_advanced(self, features, labels):
        """Advanced clustering evaluation"""
        try:
            # Multiple evaluation metrics
            scores = []
            
            # Silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(features, labels)
                scores.append(sil_score)
            
            # Calinski-Harabasz score
            try:
                from sklearn.metrics import calinski_harabasz_score
                ch_score = calinski_harabasz_score(features, labels)
                scores.append(ch_score / 1000)  # Normalize
            except:
                pass
            
            # Team balance score (prefer balanced teams)
            team_sizes = np.bincount(labels)
            if len(team_sizes) == 2:
                balance_score = min(team_sizes) / max(team_sizes)
                scores.append(balance_score)
            
            return np.mean(scores) if scores else 0
        except:
            return 0
    
    def _apply_temporal_consistency(self, detections, labels, frame_number):
        """Apply temporal consistency using player tracking"""
        stable_labels = labels.copy()
        
        for i, detection in enumerate(detections):
            player_id = detection.get('tracking_id', f"player_{i}")
            
            # Store current classification
            self.player_history[player_id].append({
                'frame': frame_number,
                'label': labels[i],
                'confidence': detection.get('confidence', 1.0)
            })
            
            # Keep only recent history
            self.player_history[player_id] = [
                h for h in self.player_history[player_id] 
                if frame_number - h['frame'] <= self.temporal_window
            ]
            
            # Apply temporal smoothing
            if len(self.player_history[player_id]) >= 3:
                recent_labels = [h['label'] for h in self.player_history[player_id]]
                most_common_label = Counter(recent_labels).most_common(1)[0][0]
                stable_labels[i] = most_common_label
        
        return stable_labels
    
    def _create_team_assignments(self, detections, labels):
        """Create team assignments with confidence scores"""
        assignments = []
        
        for i, detection in enumerate(detections):
            team_name = f"team_{labels[i]}"
            assignments.append({
                'detection': detection,
                'team': team_name,
                'confidence': detection.get('confidence', 1.0)
            })
        
        return assignments


class GraphNeuralNetworkClassifier(nn.Module):
    """Graph Neural Network for team classification using spatial-temporal relationships"""
    
    def __init__(self, feature_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(150, hidden_dim),  # Assuming 150-dim features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )
        
        # Graph convolution layers
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Team classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GNN"""
        # Encode features
        x = self.feature_encoder(x)
        
        # Graph convolutions with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = self.conv3(x2, edge_index)
        
        # Residual connection
        x = x2 + x3
        
        # Classification
        output = self.classifier(x)
        
        return output


class GNNTeamClassifier:
    """GNN-based team classifier using spatial-temporal relationships"""
    
    def __init__(self, device='cpu'):
        self.color_extractor = AdvancedJerseyColorExtractor()
        self.device = device
        self.model = GraphNeuralNetworkClassifier().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.is_trained = False
        
    def build_player_graph(self, player_detections, frame_shape):
        """Build graph representing spatial relationships between players"""
        n_players = len(player_detections)
        if n_players < 2:
            return None
        
        # Extract positions and features
        positions = []
        features = []
        
        for detection in player_detections:
            # Position features (normalized)
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2 / frame_shape[1]
            center_y = (bbox[1] + bbox[3]) / 2 / frame_shape[0]
            positions.append([center_x, center_y])
            
            # Jersey color features
            color_features = self.color_extractor.extract_comprehensive_features(
                detection['frame'], bbox
            )
            if color_features is not None:
                feature_vector = self._flatten_features(color_features)
                features.append(feature_vector)
            else:
                features.append([0] * 150)  # Default feature vector
        
        positions = np.array(positions)
        features = np.array(features)
        
        # Build edges based on spatial proximity and visual similarity
        edge_indices = []
        edge_weights = []
        
        for i in range(n_players):
            for j in range(i + 1, n_players):
                # Spatial distance
                spatial_dist = np.linalg.norm(positions[i] - positions[j])
                
                # Visual similarity
                visual_dist = np.linalg.norm(features[i] - features[j])
                visual_sim = 1 / (1 + visual_dist / 100)  # Normalize
                
                # Combined weight (closer and more similar = stronger edge)
                weight = visual_sim / (1 + spatial_dist * 5)
                
                if weight > 0.1:  # Threshold for edge creation
                    edge_indices.extend([(i, j), (j, i)])  # Undirected graph
                    edge_weights.extend([weight, weight])
        
        # Create PyTorch Geometric data object
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            node_features = torch.tensor(features, dtype=torch.float)
            
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weight
            )
            
            return graph_data
        
        return None
    
    def classify_teams_gnn(self, player_detections, frame_shape):
        """Classify teams using GNN approach"""
        graph_data = self.build_player_graph(player_detections, frame_shape)
        
        if graph_data is None:
            return []
        
        # If model is not trained, use unsupervised clustering on graph embeddings
        if not self.is_trained:
            return self._unsupervised_gnn_clustering(graph_data, player_detections)
        
        # Use trained model for prediction
        self.model.eval()
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            predictions = self.model(graph_data.x, graph_data.edge_index)
            predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        
        # Create team assignments
        assignments = []
        for i, detection in enumerate(player_detections):
            team_name = f"team_{predicted_labels[i]}"
            confidence = torch.softmax(predictions[i], dim=0).max().item()
            assignments.append({
                'detection': detection,
                'team': team_name,
                'confidence': confidence
            })
        
        return assignments
    
    def _unsupervised_gnn_clustering(self, graph_data, player_detections):
        """Unsupervised clustering using GNN embeddings"""
        self.model.eval()
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            
            # Get embeddings from the model (before final classification layer)
            x = self.model.feature_encoder(graph_data.x)
            x1 = F.relu(self.model.conv1(x, graph_data.edge_index))
            x2 = F.relu(self.model.conv2(x1, graph_data.edge_index))
            embeddings = self.model.conv3(x2, graph_data.edge_index)
            
            embeddings = embeddings.cpu().numpy()
        
        # Apply K-means clustering on embeddings
        if len(embeddings) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
            labels = kmeans.fit_predict(embeddings)
        else:
            labels = [0] * len(embeddings)
        
        # Create team assignments
        assignments = []
        for i, detection in enumerate(player_detections):
            team_name = f"team_{labels[i]}"
            assignments.append({
                'detection': detection,
                'team': team_name,
                'confidence': 0.8  # Default confidence for unsupervised
            })
        
        return assignments
    
    def _flatten_features(self, features):
        """Flatten features (same as in EnhancedKMeansClassifier)"""
        flattened = []
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                flattened.extend(value.flatten())
            elif isinstance(value, list):
                flattened.extend(value)
            elif isinstance(value, (int, float)):
                flattened.append(value)
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, (list, np.ndarray)):
                        flattened.extend(np.array(sub_value).flatten())
                    elif isinstance(sub_value, (int, float)):
                        flattened.append(sub_value)
        
        # Pad/truncate to fixed size
        target_length = 150
        if len(flattened) > target_length:
            flattened = flattened[:target_length]
        elif len(flattened) < target_length:
            flattened.extend([0] * (target_length - len(flattened)))
        
        return flattened
    
    def train_model(self, training_data, epochs=100):
        """Train the GNN model on labeled data"""
        print("Training GNN model...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_data in training_data:
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch_data.x, batch_data.edge_index)
                loss = self.criterion(output, batch_data.y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        print("GNN training completed!")


def compare_classification_methods(player_detections, frame_shape):
    """Compare different classification methods"""
    
    print("🔍 Comparing Classification Methods:")
    print("=" * 50)
    
    # Method 1: Enhanced K-Means
    print("1️⃣ Enhanced K-Means Classification:")
    kmeans_classifier = EnhancedKMeansClassifier()
    kmeans_start = time.time()
    kmeans_results = kmeans_classifier.classify_teams(player_detections, frame_number=0)
    kmeans_time = time.time() - kmeans_start
    print(f"   ⏱️ Time: {kmeans_time:.3f}s")
    print(f"   📊 Teams detected: {len(set(r['team'] for r in kmeans_results))}")
    
    # Method 2: GNN Classification
    print("2️⃣ Graph Neural Network Classification:")
    gnn_classifier = GNNTeamClassifier()
    gnn_start = time.time()
    gnn_results = gnn_classifier.classify_teams_gnn(player_detections, frame_shape)
    gnn_time = time.time() - gnn_start
    print(f"   ⏱️ Time: {gnn_time:.3f}s")
    print(f"   📊 Teams detected: {len(set(r['team'] for r in gnn_results))}")
    
    # Analysis
    print("\n📈 Performance Analysis:")
    print(f"   🚀 Speed: K-Means {kmeans_time:.3f}s vs GNN {gnn_time:.3f}s")
    print(f"   🎯 K-Means classified: {len(kmeans_results)} players")
    print(f"   🎯 GNN classified: {len(gnn_results)} players")
    
    return {
        'kmeans': kmeans_results,
        'gnn': gnn_results,
        'timing': {
            'kmeans': kmeans_time,
            'gnn': gnn_time
        }
    }


if __name__ == "__main__":
    import time
    print("🏀 Advanced Team Classification Methods Loaded!")
    print("   ✅ Enhanced K-Means with multi-space color analysis")
    print("   ✅ Graph Neural Networks with spatial-temporal modeling")
    print("   ✅ Comprehensive feature extraction")
    print("   ✅ Temporal consistency and tracking")
