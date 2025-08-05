"""
Advanced Basketball YOLO Training System - Dataset Type 2
========================================================
Training custom YOLO model on the new basketball dataset with:
- 400 training images
- 76 validation images  
- 10 test images
- Classes: ball, basket, player, referee

This will create a basketball-specialized model for accurate tracking
and team differentiation based on jersey colors.
"""

import os
import sys
import time
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json

class BasketballYOLOTrainer:
    """Advanced YOLO training system for basketball detection"""
    
    def __init__(self, dataset_path, output_dir="basketball_type2_training"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_dir = self.output_dir / f"type2_dataset_{self.timestamp}"
        
        # Create output directory
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏀 Basketball YOLO Trainer Initialized")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"💾 Output: {self.training_dir}")
        
    def setup_dataset_config(self):
        """Setup and validate dataset configuration"""
        print("\n📋 Setting up dataset configuration...")
        
        # Read original data.yaml
        data_yaml_path = self.dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
        
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update paths to be absolute
        config['train'] = str(self.dataset_path / "train" / "images")
        config['val'] = str(self.dataset_path / "valid" / "images")
        config['test'] = str(self.dataset_path / "test" / "images")
        
        # Save updated config
        self.config_path = self.training_dir / "data.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Dataset config saved to: {self.config_path}")
        print(f"   Classes: {config['names']}")
        print(f"   Train images: {config['train']}")
        print(f"   Val images: {config['val']}")
        
        return config
    
    def analyze_dataset(self):
        """Analyze dataset composition and statistics"""
        print("\n📊 Analyzing dataset...")
        
        stats = {
            'train': {'images': 0, 'labels': 0, 'objects': 0},
            'valid': {'images': 0, 'labels': 0, 'objects': 0},
            'test': {'images': 0, 'labels': 0, 'objects': 0},
            'class_distribution': {'ball': 0, 'basket': 0, 'player': 0, 'referee': 0}
        }
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.dataset_path / split / "images"
            labels_dir = self.dataset_path / split / "labels"
            
            if images_dir.exists():
                image_files = list(images_dir.glob("*.jpg"))
                stats[split]['images'] = len(image_files)
                
                if labels_dir.exists():
                    label_files = list(labels_dir.glob("*.txt"))
                    stats[split]['labels'] = len(label_files)
                    
                    # Count objects per class
                    for label_file in label_files:
                        try:
                            with open(label_file, 'r') as f:
                                lines = f.readlines()
                                stats[split]['objects'] += len(lines)
                                
                                for line in lines:
                                    class_id = int(line.strip().split()[0])
                                    class_names = ['ball', 'basket', 'player', 'referee']
                                    if 0 <= class_id < len(class_names):
                                        stats['class_distribution'][class_names[class_id]] += 1
                        except:
                            continue
        
        # Print statistics
        print(f"\n📈 Dataset Statistics:")
        for split in ['train', 'valid', 'test']:
            print(f"   {split.capitalize():>5}: {stats[split]['images']:>3} images, "
                  f"{stats[split]['labels']:>3} labels, {stats[split]['objects']:>4} objects")
        
        print(f"\n🎯 Class Distribution:")
        total_objects = sum(stats['class_distribution'].values())
        for class_name, count in stats['class_distribution'].items():
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"   {class_name:>8}: {count:>4} ({percentage:>5.1f}%)")
        
        return stats
    
    def visualize_sample_annotations(self, num_samples=5):
        """Create visualization of sample annotations"""
        print(f"\n🖼️ Creating sample annotation visualizations...")
        
        viz_dir = self.training_dir / "sample_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Colors for each class
        colors = {
            0: (0, 255, 255),    # Yellow for ball
            1: (255, 0, 0),      # Blue for basket
            2: (0, 255, 0),      # Green for player
            3: (0, 0, 255)       # Red for referee
        }
        
        class_names = ['ball', 'basket', 'player', 'referee']
        
        # Get sample images from training set
        train_images = list((self.dataset_path / "train" / "images").glob("*.jpg"))
        samples = train_images[:num_samples] if len(train_images) >= num_samples else train_images
        
        for i, img_path in enumerate(samples):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Load corresponding label
            label_path = self.dataset_path / "train" / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                h, w = img.shape[:2]
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # Draw bounding box
                        color = colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        if 0 <= class_id < len(class_names):
                            label = class_names[class_id]
                            cv2.putText(img, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save visualization
            output_path = viz_dir / f"sample_{i+1}_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), img)
        
        print(f"✅ Sample visualizations saved to: {viz_dir}")
    
    def train_model(self, epochs=100, imgsz=640, batch_size=16):
        """Train the YOLO model with optimized parameters for basketball"""
        print(f"\n🚀 Starting YOLO training...")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        
        # Initialize model
        model = YOLO('yolov8n.pt')  # Start with nano for faster training
        
        # Training parameters optimized for basketball
        training_args = {
            'data': str(self.config_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'name': f'basketball_type2_{self.timestamp}',
            'project': str(self.training_dir),
            'patience': 20,  # Early stopping patience
            'save_period': 10,  # Save checkpoint every 10 epochs
            'workers': 4,
            'device': 'cpu',  # Use CPU for compatibility
            'verbose': True,
            'seed': 42,  # For reproducibility
            
            # Augmentation parameters for basketball
            'hsv_h': 0.015,    # Hue augmentation
            'hsv_s': 0.7,      # Saturation augmentation
            'hsv_v': 0.4,      # Value augmentation
            'degrees': 10.0,   # Rotation degrees
            'translate': 0.1,  # Translation fraction
            'scale': 0.5,      # Scale factor
            'shear': 0.0,      # Shear degrees
            'perspective': 0.0, # Perspective factor
            'flipud': 0.0,     # Vertical flip probability
            'fliplr': 0.5,     # Horizontal flip probability
            'mosaic': 1.0,     # Mosaic augmentation probability
            'mixup': 0.1,      # Mixup augmentation probability
            
            # Optimization parameters
            'lr0': 0.01,       # Initial learning rate
            'lrf': 0.1,        # Final learning rate factor
            'momentum': 0.937, # SGD momentum
            'weight_decay': 0.0005, # Weight decay
            'warmup_epochs': 3.0,   # Warmup epochs
            'warmup_momentum': 0.8, # Warmup momentum
            'warmup_bias_lr': 0.1,  # Warmup bias learning rate
            'box': 7.5,        # Box loss gain
            'cls': 0.5,        # Class loss gain
            'dfl': 1.5,        # DFL loss gain
            
            # Validation parameters
            'val': True,       # Validate during training
            'save_json': True, # Save results in JSON
            'conf': 0.25,      # Confidence threshold for validation
            'iou': 0.7,        # IoU threshold for NMS during validation
            'max_det': 300,    # Maximum detections per image
            'half': False,     # Use half precision
            'dnn': False,      # Use OpenCV DNN backend
            'plots': True,     # Generate training plots
        }
        
        # Start training
        start_time = time.time()
        print(f"⏰ Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            results = model.train(**training_args)
            training_time = time.time() - start_time
            
            print(f"\n✅ Training completed successfully!")
            print(f"⏱️ Total training time: {training_time/60:.1f} minutes")
            
            # Get training results
            self.results = results
            self.model = model
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            raise
    
    def evaluate_model(self):
        """Evaluate the trained model and generate comprehensive metrics"""
        print(f"\n📊 Evaluating trained model...")
        
        # Load the best model
        best_model_path = self.training_dir / f"basketball_type2_{self.timestamp}" / "weights" / "best.pt"
        if not best_model_path.exists():
            print(f"❌ Best model not found at: {best_model_path}")
            return None
        
        model = YOLO(str(best_model_path))
        
        # Run validation
        val_results = model.val(data=str(self.config_path), split='val', save_json=True)
        
        print(f"\n🎯 Validation Results:")
        if hasattr(val_results, 'box'):
            print(f"   mAP50: {val_results.box.map50:.3f}")
            print(f"   mAP50-95: {val_results.box.map:.3f}")
            
            # Per-class metrics
            class_names = ['ball', 'basket', 'player', 'referee']
            if hasattr(val_results.box, 'mp') and len(val_results.box.mp) >= len(class_names):
                print(f"\n📋 Per-Class Precision:")
                for i, class_name in enumerate(class_names):
                    if i < len(val_results.box.mp):
                        print(f"   {class_name:>8}: {val_results.box.mp[i]:.3f}")
            
            if hasattr(val_results.box, 'mr') and len(val_results.box.mr) >= len(class_names):
                print(f"\n📋 Per-Class Recall:")
                for i, class_name in enumerate(class_names):
                    if i < len(val_results.box.mr):
                        print(f"   {class_name:>8}: {val_results.box.mr[i]:.3f}")
        
        # Test on test set if available
        test_images_dir = self.dataset_path / "test" / "images"
        if test_images_dir.exists() and list(test_images_dir.glob("*.jpg")):
            print(f"\n🧪 Testing on test set...")
            test_results = model.val(data=str(self.config_path), split='test')
            
            if hasattr(test_results, 'box'):
                print(f"   Test mAP50: {test_results.box.map50:.3f}")
                print(f"   Test mAP50-95: {test_results.box.map:.3f}")
        
        return val_results
    
    def create_training_summary(self, stats, val_results=None):
        """Create comprehensive training summary"""
        print(f"\n📄 Creating training summary...")
        
        summary = {
            'timestamp': self.timestamp,
            'dataset_path': str(self.dataset_path),
            'training_dir': str(self.training_dir),
            'dataset_stats': stats,
            'model_path': str(self.training_dir / f"basketball_type2_{self.timestamp}" / "weights" / "best.pt"),
            'training_completed': True
        }
        
        if val_results and hasattr(val_results, 'box'):
            summary['validation_results'] = {
                'mAP50': float(val_results.box.map50),
                'mAP50_95': float(val_results.box.map)
            }
        
        # Save summary
        summary_path = self.training_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Training summary saved to: {summary_path}")
        
        # Print final summary
        print(f"\n🏆 TRAINING COMPLETE - SUMMARY")
        print(f"=" * 50)
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"📊 Images: {stats['train']['images']} train, {stats['valid']['images']} val, {stats['test']['images']} test")
        print(f"🎯 Objects: {sum(stats['class_distribution'].values())} total")
        print(f"💾 Model: {summary['model_path']}")
        if val_results and hasattr(val_results, 'box'):
            print(f"📈 Performance: {val_results.box.map50:.1%} mAP50")
        print(f"⏰ Timestamp: {self.timestamp}")
        
        return summary

def main():
    """Main training pipeline for basketball dataset type 2"""
    print("🏀 BASKETBALL YOLO TRAINING - DATASET TYPE 2")
    print("=" * 60)
    print("🎯 Training custom YOLO model for basketball detection")
    print("📊 Classes: ball, basket, player, referee")
    print("🏀 Optimized for team differentiation and tracking")
    print()
    
    # Dataset path
    dataset_path = r"C:\Users\vish\Capstone PROJECT\Phase III\dataset_type2"
    
    try:
        # Initialize trainer
        trainer = BasketballYOLOTrainer(dataset_path)
        
        # Setup dataset
        config = trainer.setup_dataset_config()
        
        # Analyze dataset
        stats = trainer.analyze_dataset()
        
        # Create sample visualizations
        trainer.visualize_sample_annotations()
        
        # Train model
        results = trainer.train_model(epochs=100, imgsz=640, batch_size=16)
        
        # Evaluate model
        val_results = trainer.evaluate_model()
        
        # Create summary
        summary = trainer.create_training_summary(stats, val_results)
        
        print(f"\n🎉 SUCCESS! Basketball YOLO model training completed!")
        print(f"📁 All outputs saved to: {trainer.training_dir}")
        print(f"🎯 Ready for basketball tracking and team differentiation!")
        
        return trainer.training_dir / f"basketball_type2_{trainer.timestamp}" / "weights" / "best.pt"
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
