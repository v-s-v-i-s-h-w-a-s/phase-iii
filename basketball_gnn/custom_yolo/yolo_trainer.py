#!/usr/bin/env python3
"""
Custom YOLO Training for Basketball Objects
Trains a specialized YOLO model for basketball scenes
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import shutil

class BasketballYOLOTrainer:
    """Custom YOLO trainer for basketball object detection."""
    
    def __init__(self, config_path: str = None, auto_generate_dataset: bool = True):
        self.auto_generate_dataset = auto_generate_dataset
        
        # If no config provided and auto-generate is enabled, create dataset first
        if config_path is None and auto_generate_dataset:
            print("🎨 No dataset provided. Generating synthetic basketball dataset...")
            from auto_dataset_generator import AutomatedDatasetGenerator
            
            generator = AutomatedDatasetGenerator()
            config_path = generator.generate_complete_dataset(num_synthetic=1000)
            print(f"✅ Synthetic dataset generated: {config_path}")
        elif config_path is None:
            config_path = "basketball_dataset.yaml"
            
        self.config_path = Path(config_path)
        self.load_config()
        self.model = None
        self.training_results = {}
        
        # Basketball-specific classes
        self.classes = {
            0: "player",
            1: "ball", 
            2: "referee",
            3: "basket",
            4: "board"
        }
        
        # Class-specific colors for visualization
        self.class_colors = {
            0: (255, 0, 0),    # Red for players
            1: (255, 165, 0),  # Orange for ball
            2: (0, 0, 255),    # Blue for referee
            3: (0, 255, 0),    # Green for basket
            4: (128, 0, 128)   # Purple for board
        }
        
    def load_config(self):
        """Load training configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        print(f"📄 Loaded config from {self.config_path}")
        
    def initialize_model(self, model_size: str = "n", pretrained: bool = True):
        """Initialize YOLO model for training."""
        model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        
        print(f"🤖 Initializing YOLOv8{model_size} model...")
        self.model = YOLO(model_name)
        
        if pretrained:
            print("   Using pre-trained weights")
        else:
            print("   Training from scratch")
            
    def train_model(self, epochs: int = None, imgsz: int = None, batch: int = None,
                   patience: int = None, save_period: int = None, 
                   device: str = "auto", workers: int = 8, **kwargs):
        """Train the custom basketball YOLO model."""
        
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # Auto-detect device if not specified
        if device == 'auto':
            import torch
            if torch.cuda.is_available():
                device = '0'  # Use first GPU
            else:
                device = 'cpu'
                print("🖥️  No CUDA GPU detected, using CPU training")
                # Optimize for CPU training
                if batch is None or batch > 8:
                    batch = 4
                    print(f"   Adjusted batch size to {batch} for CPU")
                if workers > 4:
                    workers = 2
                    print(f"   Adjusted workers to {workers} for CPU")
                if imgsz is None or imgsz > 512:
                    imgsz = 416
                    print(f"   Adjusted image size to {imgsz} for CPU")
            
        # Use config values if not provided
        training_params = {
            'data': str(self.config_path),
            'epochs': epochs or self.config.get('train_settings', {}).get('epochs', 100),
            'imgsz': imgsz or self.config.get('train_settings', {}).get('imgsz', 640),
            'batch': batch or self.config.get('train_settings', {}).get('batch', 16),
            'patience': patience or self.config.get('train_settings', {}).get('patience', 50),
            'save_period': save_period or self.config.get('train_settings', {}).get('save_period', 10),
            'device': device,
            'workers': workers,
            'project': 'basketball_yolo_training',
            'name': f'basketball_v{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': device != 'cpu',  # Disable AMP for CPU training
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            **kwargs
        }
        
        print("🚀 Starting basketball YOLO training...")
        print("Training parameters:")
        for key, value in training_params.items():
            print(f"   {key}: {value}")
            
        # Start training
        self.training_results = self.model.train(**training_params)
        
        print("✅ Training completed!")
        return self.training_results
        
    def validate_model(self, data_split: str = "val", save_json: bool = True):
        """Validate the trained model."""
        if self.model is None:
            raise ValueError("No model available for validation")
            
        print(f"🔍 Validating model on {data_split} set...")
        
        validation_results = self.model.val(
            data=str(self.config_path),
            split=data_split,
            save_json=save_json,
            save_hybrid=True,
            conf=0.25,
            iou=0.6,
            max_det=300,
            half=True,
            device="auto",
            dnn=False,
            plots=True,
            verbose=True
        )
        
        print("✅ Validation completed!")
        return validation_results
        
    def export_model(self, format_list: List[str] = None, 
                    export_dir: str = "exported_models"):
        """Export trained model to different formats."""
        if format_list is None:
            format_list = ["onnx", "torchscript", "tflite"]
            
        if self.model is None:
            raise ValueError("No model available for export")
            
        export_dir = Path(export_dir)
        export_dir.mkdir(exist_ok=True)
        
        print(f"📦 Exporting model to formats: {format_list}")
        
        exported_files = {}
        
        for fmt in format_list:
            try:
                print(f"   Exporting to {fmt.upper()}...")
                export_path = self.model.export(
                    format=fmt,
                    imgsz=640,
                    keras=False,
                    optimize=True,
                    half=False,
                    int8=False,
                    dynamic=False,
                    simplify=True,
                    opset=None,
                    workspace=4,
                    nms=True
                )
                
                # Move to export directory
                if export_path:
                    dest_path = export_dir / Path(export_path).name
                    shutil.move(export_path, dest_path)
                    exported_files[fmt] = str(dest_path)
                    print(f"   ✅ {fmt.upper()} exported: {dest_path}")
                    
            except Exception as e:
                print(f"   ❌ Failed to export {fmt.upper()}: {str(e)}")
                
        return exported_files
        
    def test_on_video(self, video_path: str, output_path: str = None,
                     conf_threshold: float = 0.25, save_frames: bool = False):
        """Test the trained model on a basketball video."""
        if self.model is None:
            raise ValueError("No model available for testing")
            
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        if output_path is None:
            output_path = f"basketball_detection_{video_path.stem}.mp4"
            
        print(f"🎥 Testing model on video: {video_path}")
        print(f"   Output will be saved to: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_stats = {class_name: 0 for class_name in self.classes.values()}
        
        print("   Processing frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Annotate frame
            annotated_frame = self.annotate_frame(frame, results[0], detection_stats)
            
            # Write frame
            out.write(annotated_frame)
            
            # Save frame if requested
            if save_frames and frame_count % 30 == 0:  # Save every 30th frame
                frame_filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
                
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"   Processed {frame_count} frames...")
                
        cap.release()
        out.release()
        
        print(f"✅ Video processing complete!")
        print(f"   Total frames processed: {frame_count}")
        print("   Detection statistics:")
        for class_name, count in detection_stats.items():
            print(f"     {class_name}: {count} detections")
            
        return output_path, detection_stats
        
    def annotate_frame(self, frame: np.ndarray, results, detection_stats: Dict):
        """Annotate a frame with detection results."""
        annotated_frame = frame.copy()
        
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                # Get box coordinates and info
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                
                if class_id in self.classes:
                    class_name = self.classes[class_id]
                    color = self.class_colors[class_id]
                    
                    # Update statistics
                    detection_stats[class_name] += 1
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(annotated_frame, 
                                (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), 
                                color, -1)
                    
                    # Text
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                              
        return annotated_frame
        
    def create_training_summary(self, save_path: str = "training_summary.json"):
        """Create a summary of the training process."""
        if not self.training_results:
            print("No training results available")
            return
            
        summary = {
            "model_info": {
                "architecture": "YOLOv8",
                "classes": self.classes,
                "training_date": datetime.now().isoformat(),
                "config_file": str(self.config_path)
            },
            "training_config": self.config.get('train_settings', {}),
            "results": {
                "best_epoch": getattr(self.training_results, 'best_epoch', None),
                "best_fitness": getattr(self.training_results, 'best_fitness', None),
            }
        }
        
        # Add metrics if available
        if hasattr(self.training_results, 'results_dict'):
            summary["metrics"] = self.training_results.results_dict
            
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📊 Training summary saved to: {save_path}")
        
    def load_trained_model(self, model_path: str):
        """Load a previously trained model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"📥 Loading trained model from: {model_path}")
        self.model = YOLO(str(model_path))
        print("✅ Model loaded successfully!")
        
    def compare_with_baseline(self, baseline_model_path: str = "yolov8n.pt",
                            test_video: str = None, conf_threshold: float = 0.25):
        """Compare custom model performance with baseline YOLO."""
        if test_video is None:
            print("No test video provided for comparison")
            return
            
        print("🔄 Comparing custom model with baseline...")
        
        # Test baseline model
        baseline_model = YOLO(baseline_model_path)
        baseline_output, baseline_stats = self.test_model_on_video(
            baseline_model, test_video, "baseline_output.mp4", conf_threshold
        )
        
        # Test custom model
        if self.model is None:
            print("No custom model loaded")
            return
            
        custom_output, custom_stats = self.test_model_on_video(
            self.model, test_video, "custom_output.mp4", conf_threshold
        )
        
        # Create comparison
        comparison = {
            "baseline": {
                "model": baseline_model_path,
                "detections": baseline_stats,
                "output": baseline_output
            },
            "custom": {
                "model": "custom_basketball_model",
                "detections": custom_stats,
                "output": custom_output
            }
        }
        
        print("\n📊 Model Comparison Results:")
        print("=" * 50)
        
        for model_type, results in comparison.items():
            print(f"\n{model_type.upper()} MODEL:")
            total_detections = sum(results["detections"].values())
            print(f"  Total detections: {total_detections}")
            for class_name, count in results["detections"].items():
                print(f"    {class_name}: {count}")
                
        return comparison
        
    def test_model_on_video(self, model, video_path: str, output_path: str, 
                          conf_threshold: float):
        """Helper method to test any model on video."""
        # Similar to test_on_video but for any model
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detection_stats = {class_name: 0 for class_name in self.classes.values()}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Count detections (simplified for comparison)
            if results[0].boxes is not None:
                boxes = results[0].boxes.cpu().numpy()
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in self.classes:
                        class_name = self.classes[class_id]
                        detection_stats[class_name] += 1
                        
            # Simple annotation for comparison
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            frame_count += 1
            
        cap.release()
        out.release()
        
        return output_path, detection_stats


def main():
    """Main function for interactive training."""
    print("🏀 Basketball Custom YOLO Trainer")
    print("=" * 40)
    
    trainer = BasketballYOLOTrainer()
    
    print("\nSelect an action:")
    print("1. Initialize model")
    print("2. Train model")
    print("3. Validate model")
    print("4. Test on video")
    print("5. Export model")
    print("6. Load trained model")
    print("7. Compare with baseline")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == "1":
        model_size = input("Model size (n/s/m/l/x) [n]: ").strip() or "n"
        pretrained = input("Use pretrained weights? (y/n) [y]: ").strip().lower() != "n"
        trainer.initialize_model(model_size, pretrained)
        
    elif choice == "2":
        if trainer.model is None:
            print("Please initialize model first!")
            return
            
        epochs = input("Number of epochs [100]: ").strip()
        epochs = int(epochs) if epochs else 100
        
        trainer.train_model(epochs=epochs)
        trainer.create_training_summary()
        
    elif choice == "3":
        if trainer.model is None:
            model_path = input("Enter model path: ").strip()
            trainer.load_trained_model(model_path)
            
        trainer.validate_model()
        
    elif choice == "4":
        if trainer.model is None:
            model_path = input("Enter model path: ").strip()
            trainer.load_trained_model(model_path)
            
        video_path = input("Enter video path: ").strip()
        conf = input("Confidence threshold [0.25]: ").strip()
        conf = float(conf) if conf else 0.25
        
        trainer.test_on_video(video_path, conf_threshold=conf)
        
    elif choice == "5":
        if trainer.model is None:
            model_path = input("Enter model path: ").strip()
            trainer.load_trained_model(model_path)
            
        formats = input("Export formats (comma-separated) [onnx,torchscript]: ").strip()
        formats = formats.split(",") if formats else ["onnx", "torchscript"]
        
        trainer.export_model(formats)
        
    elif choice == "6":
        model_path = input("Enter model path: ").strip()
        trainer.load_trained_model(model_path)
        
    elif choice == "7":
        if trainer.model is None:
            model_path = input("Enter custom model path: ").strip()
            trainer.load_trained_model(model_path)
            
        video_path = input("Enter test video path: ").strip()
        trainer.compare_with_baseline(test_video=video_path)
        
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
