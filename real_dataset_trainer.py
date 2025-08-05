#!/usr/bin/env python3
"""
Real Dataset YOLO Trainer for Basketball Detection
Training custom YOLO model on user's annotated basketball dataset
"""

import os
import torch
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

class RealBasketballYOLOTrainer:
    def __init__(self, dataset_path):
        """
        Initialize trainer for real annotated basketball dataset
        
        Args:
            dataset_path: Path to dataset folder containing train/, valid/, test/ and data.yaml
        """
        self.dataset_path = Path(dataset_path)
        self.data_yaml_path = self.dataset_path / "data.yaml"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Basketball detection classes
        self.classes = {
            0: 'ball',
            1: 'basket', 
            2: 'player',
            3: 'referee'
        }
        
        print(f"🏀 Real Basketball YOLO Trainer Initialized")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"💻 Device: {self.device}")
        print(f"📋 Classes: {list(self.classes.values())}")
        
        # Verify dataset structure
        self._verify_dataset()
        
    def _verify_dataset(self):
        """Verify dataset structure and files"""
        print("\n🔍 Verifying dataset structure...")
        
        # Check main folders
        required_folders = ['train', 'valid', 'test']
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                print(f"⚠️  Warning: {folder} folder not found")
            else:
                images_path = folder_path / 'images'
                labels_path = folder_path / 'labels'
                
                if images_path.exists() and labels_path.exists():
                    image_count = len(list(images_path.glob('*.jpg')))
                    label_count = len(list(labels_path.glob('*.txt')))
                    print(f"✅ {folder}: {image_count} images, {label_count} labels")
                else:
                    print(f"⚠️  {folder}: Missing images or labels subfolder")
        
        # Check data.yaml
        if self.data_yaml_path.exists():
            print("✅ data.yaml found")
            with open(self.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                print(f"📋 Configured classes: {data_config.get('names', [])}")
        else:
            print("⚠️  data.yaml not found")
            
    def analyze_dataset_stats(self):
        """Analyze dataset statistics"""
        print("\n📊 Analyzing dataset statistics...")
        
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_counts': {name: 0 for name in self.classes.values()},
            'splits': {}
        }
        
        for split in ['train', 'valid', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if images_path.exists() and labels_path.exists():
                image_files = list(images_path.glob('*.jpg'))
                label_files = list(labels_path.glob('*.txt'))
                
                split_annotations = 0
                split_class_counts = {name: 0 for name in self.classes.values()}
                
                # Count annotations per class
                for label_file in label_files:
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            split_annotations += len(lines)
                            
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if class_id in self.classes:
                                        class_name = self.classes[class_id]
                                        split_class_counts[class_name] += 1
                                        stats['class_counts'][class_name] += 1
                    except Exception as e:
                        print(f"⚠️  Error reading {label_file}: {e}")
                
                stats['splits'][split] = {
                    'images': len(image_files),
                    'annotations': split_annotations,
                    'class_counts': split_class_counts
                }
                
                stats['total_images'] += len(image_files)
                stats['total_annotations'] += split_annotations
                
                print(f"📁 {split.upper()}:")
                print(f"   Images: {len(image_files)}")
                print(f"   Annotations: {split_annotations}")
                for class_name, count in split_class_counts.items():
                    print(f"   {class_name}: {count}")
        
        print(f"\n📈 TOTAL DATASET STATS:")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Total Annotations: {stats['total_annotations']}")
        print(f"   Average annotations per image: {stats['total_annotations']/stats['total_images']:.1f}")
        
        print(f"\n🎯 CLASS DISTRIBUTION:")
        for class_name, count in stats['class_counts'].items():
            percentage = (count / stats['total_annotations']) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return stats
    
    def train_model(self, epochs=100, imgsz=640, batch_size=16):
        """
        Train YOLO model on real annotated dataset
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size for training
        """
        print(f"\n🚀 Starting YOLO training on real basketball dataset...")
        print(f"⚙️  Training Parameters:")
        print(f"   Epochs: {epochs}")
        print(f"   Image Size: {imgsz}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Device: {self.device}")
        
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')  # Start with pretrained weights
        
        # Configure CPU-optimized settings if needed
        training_args = {
            'data': str(self.data_yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'project': 'basketball_real_training',
            'name': f'real_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save': True,
            'plots': True,
            'val': True,
            'cache': True,
            'workers': 4 if self.device == 'cpu' else 8,
            'patience': 50,
            'save_period': 10,
            'exist_ok': True
        }
        
        # CPU optimizations
        if self.device == 'cpu':
            training_args.update({
                'workers': 2,
                'batch': min(batch_size, 8),  # Smaller batch for CPU
                'cache': False,  # Disable caching on CPU
                'amp': False,  # Disable mixed precision
            })
            print("🔧 Applied CPU optimizations")
        
        try:
            # Start training
            print(f"\n⏳ Training started at {datetime.now().strftime('%H:%M:%S')}")
            results = model.train(**training_args)
            
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Results saved to: {results.save_dir}")
            
            # Get the best model path
            best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
            last_model_path = Path(results.save_dir) / 'weights' / 'last.pt'
            
            # Validate the best model
            if best_model_path.exists():
                print(f"\n🎯 Validating best model...")
                best_model = YOLO(str(best_model_path))
                
                # Run validation
                val_results = best_model.val(
                    data=str(self.data_yaml_path),
                    device=self.device,
                    plots=True,
                    save_json=True
                )
                
                # Print validation metrics
                self._print_validation_metrics(val_results)
                
                # Test detection on a sample image
                self._test_detection_sample(best_model)
                
                return {
                    'model_path': str(best_model_path),
                    'results_dir': str(results.save_dir),
                    'validation_metrics': val_results,
                    'training_results': results
                }
            else:
                print(f"⚠️  Best model not found at {best_model_path}")
                return None
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def _print_validation_metrics(self, val_results):
        """Print formatted validation metrics"""
        print(f"\n📊 VALIDATION RESULTS:")
        
        if hasattr(val_results, 'box'):
            metrics = val_results.box
            print(f"   mAP50: {metrics.map50:.3f}")
            print(f"   mAP50-95: {metrics.map:.3f}")
            print(f"   Precision: {metrics.mp:.3f}")
            print(f"   Recall: {metrics.mr:.3f}")
            
            # Per-class metrics if available
            if hasattr(metrics, 'ap50') and len(metrics.ap50) > 0:
                print(f"\n🎯 PER-CLASS mAP50:")
                for i, ap in enumerate(metrics.ap50):
                    if i < len(self.classes) and ap > 0:
                        class_name = self.classes[i]
                        print(f"   {class_name}: {ap:.3f}")
    
    def _test_detection_sample(self, model):
        """Test detection on a sample image"""
        print(f"\n🔍 Testing detection on sample image...")
        
        # Find a sample image from validation set
        val_images_path = self.dataset_path / 'valid' / 'images'
        if val_images_path.exists():
            image_files = list(val_images_path.glob('*.jpg'))
            if image_files:
                sample_image = image_files[0]
                print(f"   Sample image: {sample_image.name}")
                
                try:
                    # Run detection
                    results = model(str(sample_image))
                    
                    # Count detections
                    if results and len(results) > 0:
                        detections = results[0]
                        if hasattr(detections, 'boxes') and detections.boxes is not None:
                            detection_count = len(detections.boxes)
                            print(f"   Detections found: {detection_count}")
                            
                            # Count by class
                            if detection_count > 0:
                                class_counts = {}
                                for box in detections.boxes:
                                    class_id = int(box.cls.item())
                                    if class_id in self.classes:
                                        class_name = self.classes[class_id]
                                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                                
                                for class_name, count in class_counts.items():
                                    print(f"   {class_name}: {count}")
                        else:
                            print(f"   No detections found")
                    
                except Exception as e:
                    print(f"   ⚠️  Detection test failed: {e}")
    
    def export_model(self, model_path, export_format='onnx'):
        """
        Export trained model to different formats
        
        Args:
            model_path: Path to trained model
            export_format: Export format (onnx, torchscript, etc.)
        """
        print(f"\n📦 Exporting model to {export_format.upper()}...")
        
        try:
            model = YOLO(model_path)
            export_path = model.export(format=export_format)
            print(f"✅ Model exported to: {export_path}")
            return export_path
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None

def main():
    """Main training function"""
    print("🏀 Real Basketball Dataset YOLO Trainer")
    print("=" * 50)
    
    # Dataset path
    dataset_path = r"C:\Users\vish\Capstone PROJECT\Phase III\dataset"
    
    # Initialize trainer
    trainer = RealBasketballYOLOTrainer(dataset_path)
    
    # Analyze dataset
    stats = trainer.analyze_dataset_stats()
    
    # Train model
    print(f"\n{'='*50}")
    print("🚀 Starting Training Process")
    print(f"{'='*50}")
    
    # Use appropriate settings based on dataset size
    if stats['total_images'] > 1000:
        epochs = 150
        batch_size = 16
    elif stats['total_images'] > 500:
        epochs = 100
        batch_size = 12
    else:
        epochs = 80
        batch_size = 8
    
    print(f"📊 Automatic settings based on dataset size:")
    print(f"   Images: {stats['total_images']} -> Epochs: {epochs}, Batch: {batch_size}")
    
    # Start training
    training_results = trainer.train_model(
        epochs=epochs,
        imgsz=640,
        batch_size=batch_size
    )
    
    if training_results:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Best model: {training_results['model_path']}")
        print(f"📊 Results folder: {training_results['results_dir']}")
        
        # Export model
        exported_path = trainer.export_model(training_results['model_path'])
        if exported_path:
            print(f"📦 Exported model: {exported_path}")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()
 