"""
Refined Ball Detection System
Improved basketball detection with better accuracy and reduced false positives
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class RefinedBallDetector:
    def __init__(self, model_path="./models/basketball_yolo11n.pt"):
        """Initialize refined ball detector"""
        self.model_path = model_path
        self.model = None
        
        # Refined thresholds
        self.ball_conf_threshold = 0.2
        self.general_conf_threshold = 0.4
        self.iou_threshold = 0.5
        
        # Ball-specific parameters
        self.min_ball_area = 15  # Minimum ball area in pixels
        self.max_ball_area = 1500  # Maximum ball area in pixels
        self.min_circularity = 0.4  # Minimum circularity for ball candidates
        
        # Color ranges for basketball (HSV)
        self.ball_color_ranges = [
            ([5, 120, 70], [25, 255, 255]),    # Orange basketball
            ([8, 80, 40], [25, 255, 200]),     # Brown basketball
        ]
        
        print("🏀 Refined Ball Detector initialized")
        
    def load_model(self):
        """Load the YOLO model"""
        if not Path(self.model_path).exists():
            print(f"❌ Model not found: {self.model_path}")
            return False
            
        self.model = YOLO(self.model_path)
        print("✅ Model loaded for refined detection")
        return True
    
    def detect_frame(self, frame):
        """Enhanced frame detection with refined ball detection"""
        if self.model is None:
            return []
        
        # Standard YOLO detection
        yolo_detections = self._yolo_detect(frame)
        
        # Enhanced ball detection using computer vision
        cv_ball_detections = self._cv_ball_detect(frame)
        
        # Combine and filter detections
        all_detections = yolo_detections + cv_ball_detections
        
        # Apply non-maximum suppression
        filtered_detections = self._apply_nms(all_detections)
        
        # Post-process ball detections
        final_detections = self._refine_ball_detections(filtered_detections, frame)
        
        return final_detections
    
    def _yolo_detect(self, frame):
        """Standard YOLO detection"""
        results = self.model.predict(
            frame,
            conf=self.ball_conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Apply class-specific thresholds
                    min_conf = self.ball_conf_threshold if class_name == 'ball' else self.general_conf_threshold
                    
                    if confidence >= min_conf:
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class': class_name,
                            'class_id': class_id,
                            'source': 'yolo'
                        })
        
        return detections
    
    def _cv_ball_detect(self, frame):
        """Computer vision based ball detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for basketball colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.ball_color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_ball_area or area > self.max_ball_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (should be roughly square)
            aspect_ratio = w / h
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                continue
            
            # Calculate confidence based on circularity and area
            confidence = min(0.85, 0.3 + (circularity * 0.4) + (min(area, 300) / 300 * 0.15))
            
            ball_candidates.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': confidence,
                'class': 'ball',
                'class_id': 2,  # Assuming ball is class 2
                'source': 'cv',
                'circularity': circularity,
                'area': area
            })
        
        return ball_candidates
    
    def _apply_nms(self, detections):
        """Apply non-maximum suppression"""
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
                        # Use lower IoU threshold for ball detections
                        iou_threshold = 0.3 if class_name == 'ball' else self.iou_threshold
                        if iou > iou_threshold:
                            keep_detection = False
                            break
                
                if keep_detection:
                    keep.append(i)
            
            for idx in keep:
                final_detections.append(class_detections[idx])
        
        return final_detections
    
    def _refine_ball_detections(self, detections, frame):
        """Refine ball detections to reduce false positives"""
        refined = []
        
        for detection in detections:
            if detection['class'] != 'ball':
                refined.append(detection)
                continue
            
            # For ball detections, apply additional validation
            x1, y1, x2, y2 = detection['bbox']
            w, h = x2 - x1, y2 - y1
            
            # Size validation
            if w < 8 or h < 8 or w > 80 or h > 80:
                continue
            
            # Aspect ratio validation
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
            
            # Extract region and validate color
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Check if region contains basketball-like colors
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            color_match = False
            
            for lower, upper in self.ball_color_ranges:
                mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
                color_ratio = np.sum(mask > 0) / mask.size
                if color_ratio > 0.1:  # At least 10% of region should match ball colors
                    color_match = True
                    break
            
            # Keep detection if it passes validation or has high YOLO confidence
            if color_match or (detection['source'] == 'yolo' and detection['confidence'] > 0.6):
                refined.append(detection)
        
        return refined

def test_refined_detector():
    """Test the refined ball detector"""
    print("🏀 Testing Refined Ball Detector")
    print("=" * 40)
    
    detector = RefinedBallDetector()
    if not detector.load_model():
        return
    
    # Test on sample frames
    sample_files = list(Path(".").glob("sample_frame_*.jpg"))
    
    for sample_file in sample_files[:3]:
        print(f"\n🔍 Testing: {sample_file.name}")
        
        frame = cv2.imread(str(sample_file))
        if frame is None:
            continue
        
        # Run refined detection
        detections = detector.detect_frame(frame)
        
        # Count by class
        class_counts = {}
        ball_info = []
        
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if class_name == 'ball':
                ball_info.append({
                    'confidence': detection['confidence'],
                    'source': detection['source'],
                    'bbox': detection['bbox']
                })
        
        print(f"   Total objects: {class_counts}")
        print(f"   Ball detections: {len(ball_info)}")
        
        for i, ball in enumerate(ball_info):
            print(f"     Ball {i+1}: conf={ball['confidence']:.3f}, source={ball['source']}")
        
        # Draw detections
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            source = detection.get('source', 'unknown')
            
            # Color coding
            colors = {
                'player': (255, 100, 100),
                'referee': (100, 255, 100),
                'ball': (0, 255, 255),
                'hoop': (255, 0, 255)
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            thickness = 3 if class_name == 'ball' else 2
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"{class_name}: {confidence:.2f} ({source})"
            cv2.putText(result_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save result
        output_name = f"refined_{sample_file.name}"
        cv2.imwrite(output_name, result_frame)
        print(f"   💾 Refined result: {output_name}")

if __name__ == "__main__":
    test_refined_detector()
