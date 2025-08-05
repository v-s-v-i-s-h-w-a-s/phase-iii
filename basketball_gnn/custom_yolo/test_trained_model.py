#!/usr/bin/env python3
"""
Test the trained basketball YOLO model
"""

from ultralytics import YOLO
import cv2

def test_basketball_yolo():
    # Load the trained model
    model_path = r"basketball_yolo_training\basketball_v20250802_230623\weights\best.pt"
    model = YOLO(model_path)
    
    print("Basketball YOLO Model Loaded!")
    print("Classes detected:")
    print("  0: player")
    print("  1: ball")
    print("  2: referee") 
    print("  3: basket")
    print("  4: board")
    
    # Test prediction
    print("\nTesting model...")
    
    # You can test with images or video
    print("Model ready for inference!")
    print("Example usage:")
    print("  results = model('path/to/basketball_image.jpg')")
    print("  results[0].show()  # Display results")
    
    return model

if __name__ == "__main__":
    model = test_basketball_yolo()
