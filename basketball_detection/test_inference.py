"""
Test inference script without video input
Tests the inference engine functionality
"""

from src.inference import BasketballInference
from pathlib import Path
import cv2
import numpy as np

def create_test_image():
    """Create a simple test image for inference"""
    # Create a 640x640 test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(img, (100, 100), (200, 300), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (400, 200), (500, 400), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(img, (320, 320), 30, (0, 165, 255), -1)          # Orange circle
    
    # Add some text
    cv2.putText(img, "Test Basketball Image", (200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_inference():
    """Test the inference system"""
    print("🔍 Testing Basketball Inference System")
    print("=" * 40)
    
    # Check if model exists
    model_path = "./models/basketball_yolo11n.pt"
    if not Path(model_path).exists():
        print("❌ Trained model not found!")
        print("   Please run training first (option 2 in main menu)")
        return
    
    # Initialize inference
    inference = BasketballInference(model_path)
    
    if not inference.load_model():
        print("❌ Failed to load model!")
        return
    
    # Create test image
    test_img = create_test_image()
    
    # Save test image
    cv2.imwrite("test_image.jpg", test_img)
    print("📷 Created test image: test_image.jpg")
    
    # Run detection on test image
    print("🔍 Running detection on test image...")
    detections = inference.detect_frame(test_img)
    
    print(f"✅ Detection complete!")
    print(f"   Found {len(detections)} objects")
    
    if detections:
        for i, detection in enumerate(detections):
            print(f"   Object {i+1}: {detection['class']} (confidence: {detection['confidence']:.3f})")
    
    # Draw detections and save result
    result_img = inference.draw_detections(test_img.copy(), detections)
    cv2.imwrite("test_result.jpg", result_img)
    print("💾 Result saved: test_result.jpg")
    
    return detections

if __name__ == "__main__":
    test_inference()
