#!/usr/bin/env python3
"""
Enhanced Basketball YOLO Training with Multiple Datasets
Combines dataset_type2 + ball datasets for superior detection accuracy
"""

import os
import cv2
import numpy as np
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
import json

class EnhancedBasketballTrainer:
    """Enhanced YOLO trainer combining multiple basketball datasets"""
    
    def __init__(self):
        self.project_root = Path(r"C:\Users\vish\Capstone PROJECT\Phase III")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.project_root / f"enhanced_basketball_training/enhanced_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.dataset_type2 = self.project_root / "dataset_type2"
        self.ball_dataset1 = self.project_root / "ball dataset"
        self.ball_dataset2 = self.project_root / "ball datatset 2"
        
        # Combined dataset path
        self.combined_dataset = self.output_dir / "combined_dataset"
        
        # Class mapping for combined dataset
        self.unified_classes = {
            'ball': 0,
            'basket': 1, 
            'player': 2,
            'referee': 3
        }
        
        # Source class mappings
        self.type2_mapping = {
            'ball': 'ball',
            'basket': 'basket', 
            'player': 'player',
            'referee': 'referee'
        }
        
        self.ball_dataset2_mapping = {
            'basketball': 'ball',
            'sports ball': 'ball',
            'rim': 'basket'
        }
        
        print("🏀 Enhanced Basketball YOLO Trainer Initialized")
        print(f"📁 Output: {self.output_dir}")
        print(f"⏰ Timestamp: {self.timestamp}")
        
    def analyze_datasets(self):
        """Analyze all available datasets"""
        print("\n📊 Analyzing Available Datasets...")
        
        datasets_info = {}
        
        # Dataset Type 2
        if self.dataset_type2.exists():
            train_imgs = len(list((self.dataset_type2 / "train/images").glob("*")))
            valid_imgs = len(list((self.dataset_type2 / "valid/images").glob("*")))
            test_imgs = len(list((self.dataset_type2 / "test/images").glob("*")))
            
            datasets_info['dataset_type2'] = {
                'train': train_imgs,
                'valid': valid_imgs,
                'test': test_imgs,
                'total': train_imgs + valid_imgs + test_imgs,
                'classes': ['ball', 'basket', 'player', 'referee']
            }
        
        # Ball Dataset 2
        if self.ball_dataset2.exists():
            train_imgs = len(list((self.ball_dataset2 / "train/images").glob("*")))
            valid_imgs = len(list((self.ball_dataset2 / "valid/images").glob("*")))
            test_imgs = len(list((self.ball_dataset2 / "test/images").glob("*")))
            
            datasets_info['ball_dataset2'] = {
                'train': train_imgs,
                'valid': valid_imgs,
                'test': test_imgs,
                'total': train_imgs + valid_imgs + test_imgs,
                'classes': ['basketball', 'rim', 'sports ball']
            }
        
        # Display analysis
        print("\n📈 Dataset Analysis:")
        for name, info in datasets_info.items():
            print(f"\n   {name.upper()}:")
            print(f"     Train: {info['train']} images")
            print(f"     Valid: {info['valid']} images")
            print(f"     Test: {info['test']} images")
            print(f"     Total: {info['total']} images")
            print(f"     Classes: {info['classes']}")
        
        total_images = sum(info['total'] for info in datasets_info.values())
        print(f"\n🎯 COMBINED TOTAL: {total_images} images")
        
        return datasets_info
    
    def convert_label_format(self, label_path, class_mapping, output_path):
        """Convert label to unified class format"""
        if not label_path.exists():
            return False
            
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    original_class = int(parts[0])
                    
                    # Map to unified class based on source dataset
                    if 'ball' in str(label_path).lower() and 'datatset 2' in str(label_path):
                        # Ball dataset 2 mapping
                        source_classes = ['basketball', 'rim', 'sports ball']
                        if original_class < len(source_classes):
                            source_class_name = source_classes[original_class]
                            if source_class_name in self.ball_dataset2_mapping:
                                unified_class_name = self.ball_dataset2_mapping[source_class_name]
                                unified_class_id = self.unified_classes[unified_class_name]
                                converted_lines.append(f"{unified_class_id} {' '.join(parts[1:])}\n")
                    else:
                        # Dataset type 2 mapping (direct mapping)
                        type2_classes = ['ball', 'basket', 'player', 'referee']
                        if original_class < len(type2_classes):
                            unified_class_id = original_class  # Direct mapping for type2
                            converted_lines.append(f"{unified_class_id} {' '.join(parts[1:])}\n")
            
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)
            return True
            
        except Exception as e:
            print(f"⚠️ Error converting {label_path}: {e}")
            return False
    
    def combine_datasets(self):
        """Combine all datasets into unified format"""
        print("\n🔄 Combining Datasets...")
        
        # Create combined dataset structure
        for split in ['train', 'valid', 'test']:
            (self.combined_dataset / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.combined_dataset / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        image_counter = 0
        dataset_stats = {
            'train': {'images': 0, 'objects': 0},
            'valid': {'images': 0, 'objects': 0},
            'test': {'images': 0, 'objects': 0}
        }
        
        # Process Dataset Type 2
        print("   Processing Dataset Type 2...")
        for split in ['train', 'valid', 'test']:
            source_img_dir = self.dataset_type2 / split / 'images'
            source_lbl_dir = self.dataset_type2 / split / 'labels'
            
            if source_img_dir.exists():
                for img_path in source_img_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Copy image with new name
                        new_name = f"type2_{split}_{image_counter:06d}{img_path.suffix}"
                        dst_img = self.combined_dataset / split / 'images' / new_name
                        shutil.copy2(img_path, dst_img)
                        
                        # Convert and copy label
                        lbl_path = source_lbl_dir / f"{img_path.stem}.txt"
                        dst_lbl = self.combined_dataset / split / 'labels' / f"{new_name.split('.')[0]}.txt"
                        
                        if self.convert_label_format(lbl_path, self.type2_mapping, dst_lbl):
                            # Count objects
                            with open(dst_lbl, 'r') as f:
                                object_count = len(f.readlines())
                            dataset_stats[split]['objects'] += object_count
                        
                        dataset_stats[split]['images'] += 1
                        image_counter += 1
        
        # Process Ball Dataset 2
        print("   Processing Ball Dataset 2...")
        for split in ['train', 'valid', 'test']:
            source_img_dir = self.ball_dataset2 / split / 'images'
            source_lbl_dir = self.ball_dataset2 / split / 'labels'
            
            if source_img_dir.exists():
                for img_path in source_img_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Copy image with new name
                        new_name = f"ball2_{split}_{image_counter:06d}{img_path.suffix}"
                        dst_img = self.combined_dataset / split / 'images' / new_name
                        shutil.copy2(img_path, dst_img)
                        
                        # Convert and copy label
                        lbl_path = source_lbl_dir / f"{img_path.stem}.txt"
                        dst_lbl = self.combined_dataset / split / 'labels' / f"{new_name.split('.')[0]}.txt"
                        
                        if self.convert_label_format(lbl_path, self.ball_dataset2_mapping, dst_lbl):
                            # Count objects
                            with open(dst_lbl, 'r') as f:
                                object_count = len(f.readlines())
                            dataset_stats[split]['objects'] += object_count
                        
                        dataset_stats[split]['images'] += 1
                        image_counter += 1
        
        # Create data.yaml for combined dataset
        data_yaml = {
            'train': str(self.combined_dataset / 'train' / 'images'),
            'val': str(self.combined_dataset / 'valid' / 'images'),
            'test': str(self.combined_dataset / 'test' / 'images'),
            'nc': len(self.unified_classes),
            'names': list(self.unified_classes.keys())
        }
        
        yaml_path = self.combined_dataset / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print("✅ Dataset combination completed!")
        print(f"\n📊 Combined Dataset Statistics:")
        total_images = sum(stats['images'] for stats in dataset_stats.values())
        total_objects = sum(stats['objects'] for stats in dataset_stats.values())
        
        for split, stats in dataset_stats.items():
            print(f"   {split.capitalize()}: {stats['images']} images, {stats['objects']} objects")
        print(f"   Total: {total_images} images, {total_objects} objects")
        
        return yaml_path, dataset_stats
    
    def create_sample_visualizations(self, num_samples=20):
        """Create sample visualizations from combined dataset"""
        print(f"\n🖼️ Creating sample visualizations...")
        
        viz_dir = self.output_dir / "sample_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Yellow, Blue, Green, Red
        class_names = list(self.unified_classes.keys())
        
        for split in ['train', 'valid']:
            img_dir = self.combined_dataset / split / 'images'
            lbl_dir = self.combined_dataset / split / 'labels'
            
            if img_dir.exists():
                images = list(img_dir.glob("*"))
                if images:
                    sample_images = random.sample(images, min(num_samples//2, len(images)))
                    
                    for i, img_path in enumerate(sample_images):
                        # Load image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        # Load labels
                        lbl_path = lbl_dir / f"{img_path.stem}.txt"
                        if lbl_path.exists():
                            with open(lbl_path, 'r') as f:
                                lines = f.readlines()
                            
                            h, w = img.shape[:2]
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cls_id = int(parts[0])
                                    x_center, y_center, width, height = map(float, parts[1:5])
                                    
                                    # Convert to pixel coordinates
                                    x1 = int((x_center - width/2) * w)
                                    y1 = int((y_center - height/2) * h)
                                    x2 = int((x_center + width/2) * w)
                                    y2 = int((y_center + height/2) * h)
                                    
                                    # Draw bounding box
                                    color = colors[cls_id] if cls_id < len(colors) else (255, 255, 255)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Draw label
                                    label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Save visualization
                        output_path = viz_dir / f"{split}_sample_{i:03d}.jpg"
                        cv2.imwrite(str(output_path), img)
        
        print(f"✅ Sample visualizations saved to: {viz_dir}")
    
    def start_training(self, epochs=100, batch_size=16, img_size=640):
        """Start enhanced YOLO training"""
        print(f"\n🚀 Starting Enhanced YOLO Training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {img_size}")
        
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')  # Start with nano for faster training
        
        # Training parameters optimized for basketball detection
        training_args = {
            'data': str(self.combined_dataset / 'data.yaml'),
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': f'enhanced_basketball_{self.timestamp}',
            'project': str(self.output_dir),
            'save': True,
            'save_period': 10,  # Save every 10 epochs
            'patience': 20,     # Early stopping patience
            'workers': 4,
            'device': 'cpu',    # Use CPU for compatibility
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'val': True,
            'plots': True,
            'save_json': True,
            
            # Optimized learning parameters for basketball
            'lr0': 0.01,        # Initial learning rate
            'momentum': 0.937,  # SGD momentum
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights optimized for basketball detection
            'box': 7.5,         # Box loss weight
            'cls': 0.5,         # Classification loss weight
            'dfl': 1.5,         # Distribution focal loss weight
            
            # Data augmentation for basketball scenes
            'hsv_h': 0.015,     # HSV hue augmentation
            'hsv_s': 0.7,       # HSV saturation augmentation
            'hsv_v': 0.4,       # HSV value augmentation
            'degrees': 10.0,    # Rotation degrees
            'translate': 0.1,   # Translation fraction
            'scale': 0.5,       # Scale fraction
            'fliplr': 0.5,      # Horizontal flip probability
            'mosaic': 1.0,      # Mosaic probability
            'mixup': 0.1,       # Mixup probability
        }
        
        print(f"⏰ Training started at: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Start training
            results = model.train(**training_args)
            
            print(f"✅ Training completed successfully!")
            print(f"📁 Results saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None

def main():
    """Main training pipeline"""
    print("🏀 ENHANCED BASKETBALL YOLO TRAINING")
    print("=" * 60)
    print("🎯 Combining multiple datasets for superior detection")
    print("🧠 Enhanced accuracy for ball, player, referee, rim")
    print("🎨 Consistent team color tracking")
    
    # Initialize trainer
    trainer = EnhancedBasketballTrainer()
    
    # Analyze datasets
    datasets_info = trainer.analyze_datasets()
    
    # Combine datasets
    yaml_path, stats = trainer.combine_datasets()
    
    # Create visualizations
    trainer.create_sample_visualizations()
    
    # Start training
    results = trainer.start_training(epochs=150, batch_size=16, img_size=640)
    
    if results:
        print(f"\n🎉 Enhanced training pipeline completed successfully!")
        print(f"📊 Best model will be available at:")
        print(f"   {trainer.output_dir}/enhanced_basketball_{trainer.timestamp}/weights/best.pt")
    else:
        print(f"\n❌ Training pipeline failed!")

if __name__ == "__main__":
    main()
