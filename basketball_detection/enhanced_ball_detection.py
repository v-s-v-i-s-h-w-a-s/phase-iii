"""
Enhanced Basketball Detection with Improved Ball Detection
Optimized specifically for basketball tracking with better small object detection
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import time
import pandas as pd
from datetime import datetime

class EnhancedBasketballInference:
    def __init__(self, model_path=None):
        """
        Enhanced inference engine with better ball detection
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        if model_path is None:
            model_path = "./models/basketball_yolo11n.pt"
        
        self.model_path = model_path
        
        # Enhanced detection settings for ball detection
        self.conf_threshold = 0.25  # Lower threshold for small objects
        self.iou_threshold = 0.4
        self.ball_conf_threshold = 0.15  # Even lower for ball specifically
        
        # Enhanced colors for better visibility
        self.colors = {
            'player': (255, 100, 100),    # Light blue
            'referee': (100, 255, 100),   # Light green  
            'ball': (0, 255, 255),        # Bright yellow
            'hoop': (255, 0, 255)         # Magenta
        }
        
        # Ball detection enhancements
        self.ball_size_range = (5, 50)  # Expected ball size in pixels
        self.ball_tracking_buffer = []   # Track ball across frames
        
        print("🏀 Enhanced Basketball Inference Engine initialized")
        print(f"   Device: {self.device}")
        print(f"   Enhanced ball detection enabled")
        
    def load_model(self):
        """Load trained model with enhanced settings"""
        if not Path(self.model_path).exists():
            print(f"❌ Model not found: {self.model_path}")
            return False
            
        print("📥 Loading enhanced model...")
        self.model = YOLO(self.model_path)
        
        # Configure model for better small object detection
        self.model.overrides['conf'] = self.conf_threshold
        self.model.overrides['iou'] = self.iou_threshold
        self.model.overrides['imgsz'] = 640  # Higher resolution for small objects
        self.model.overrides['augment'] = True  # Enable augmentation for better detection
        
        print("✅ Enhanced model loaded successfully!")
        return True
    
    def enhance_ball_detection(self, frame, detections):
        """
        Enhanced ball detection using computer vision techniques
        """
        enhanced_detections = detections.copy()
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Basketball color ranges (orange/brown)
        ball_color_ranges = [
            # Orange basketball
            ([5, 100, 100], [25, 255, 255]),
            # Brown basketball
            ([8, 50, 20], [20, 255, 200])
        ]
        
        ball_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Create combined mask for basketball colors
        for (lower, upper) in ball_color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            ball_mask = cv2.bitwise_or(ball_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and circularity
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 2000:  # Size filter
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.3:  # Not circular enough
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (should be close to square)
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
            
            # Check if this detection overlaps with existing ball detections
            overlaps = False
            for detection in enhanced_detections:
                if detection['class'] == 'ball':
                    dx1, dy1, dx2, dy2 = detection['bbox']
                    # Check overlap
                    if not (x + w < dx1 or x > dx2 or y + h < dy1 or y > dy2):
                        overlaps = True
                        break
            
            if not overlaps:
                # Add enhanced ball detection
                confidence = min(0.8, 0.4 + circularity)  # Confidence based on circularity
                enhanced_detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': confidence,
                    'class': 'ball',
                    'class_id': 2,  # Assuming ball is class 2
                    'source': 'enhanced_cv'
                })
        
        return enhanced_detections
    
    def detect_frame_enhanced(self, frame):
        """
        Enhanced detection with better ball detection
        """
        if self.model is None:
            return []
        
        # Standard YOLO detection with multiple scales
        detections = []
        
        # Try multiple image sizes for better small object detection
        sizes = [640, 800, 1024]
        
        for img_size in sizes:
            # Resize frame
            h, w = frame.shape[:2]
            scale = img_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))
            
            # Run inference
            results = self.model.predict(
                resized_frame,
                conf=self.ball_conf_threshold if img_size > 640 else self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=img_size
            )
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Scale back to original size
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                        
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Apply class-specific thresholds
                        min_conf = self.ball_conf_threshold if class_name == 'ball' else self.conf_threshold
                        
                        if confidence >= min_conf:
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence,
                                'class': class_name,
                                'class_id': class_id,
                                'source': f'yolo_{img_size}'
                            })
        
        # Remove duplicates using NMS
        detections = self.non_max_suppression(detections)
        
        # Apply enhanced ball detection
        detections = self.enhance_ball_detection(frame, detections)
        
        return detections
    
    def non_max_suppression(self, detections):
        """Custom NMS to remove duplicate detections"""
        if not detections:
            return detections
        
        # Group by class
        class_groups = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(detection)
        
        final_detections = []
        
        for class_name, class_detections in class_groups.items():
            if len(class_detections) <= 1:
                final_detections.extend(class_detections)
                continue
            
            # Sort by confidence
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS
            keep = []
            for i, detection in enumerate(class_detections):
                keep_detection = True
                x1, y1, x2, y2 = detection['bbox']
                
                for j in range(len(keep)):
                    kept_detection = class_detections[keep[j]]
                    kx1, ky1, kx2, ky2 = kept_detection['bbox']
                    
                    # Calculate IoU
                    intersection_area = max(0, min(x2, kx2) - max(x1, kx1)) * max(0, min(y2, ky2) - max(y1, ky1))
                    box1_area = (x2 - x1) * (y2 - y1)
                    box2_area = (kx2 - kx1) * (ky2 - ky1)
                    union_area = box1_area + box2_area - intersection_area
                    
                    if union_area > 0:
                        iou = intersection_area / union_area
                        if iou > self.iou_threshold:
                            keep_detection = False
                            break
                
                if keep_detection:
                    keep.append(i)
            
            for idx in keep:
                final_detections.append(class_detections[idx])
        
        return final_detections
    
    def draw_enhanced_detections(self, frame, detections):
        """Enhanced visualization with better ball highlighting"""
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            source = detection.get('source', 'yolo')
            
            # Get color for class
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Enhanced visualization for ball
            if class_name == 'ball':
                # Thicker border for ball
                thickness = 4
                # Add circle overlay for better visibility
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(5, min(x2 - x1, y2 - y1) // 2)
                cv2.circle(result_frame, (center_x, center_y), radius, color, 2)
            else:
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label with source info
            label = f"{class_name}: {confidence:.2f}"
            if class_name == 'ball':
                label += f" ({source})"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame

def test_enhanced_detection():
    """Test the enhanced detection system"""
    print("🏀 Testing Enhanced Ball Detection")
    print("=" * 40)
    
    # Test on our sample frames
    inference = EnhancedBasketballInference()
    if not inference.load_model():
        return
    
    # Test on sample frames
    sample_files = list(Path(".").glob("sample_frame_*.jpg"))
    
    for sample_file in sample_files[:3]:  # Test first 3 samples
        print(f"🔍 Testing: {sample_file.name}")
        
        frame = cv2.imread(str(sample_file))
        if frame is None:
            continue
        
        # Run enhanced detection
        detections = inference.detect_frame_enhanced(frame)
        
        # Count by class
        class_counts = {}
        ball_detections = []
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name == 'ball':
                ball_detections.append(detection)
        
        print(f"   Objects detected: {class_counts}")
        if ball_detections:
            for ball in ball_detections:
                print(f"   🏀 Ball found: confidence={ball['confidence']:.3f}, source={ball.get('source', 'unknown')}")
        
        # Save enhanced result
        result_frame = inference.draw_enhanced_detections(frame, detections)
        output_name = f"enhanced_{sample_file.name}"
        cv2.imwrite(output_name, result_frame)
        print(f"   💾 Enhanced result: {output_name}")

if __name__ == "__main__":
    test_enhanced_detection()
