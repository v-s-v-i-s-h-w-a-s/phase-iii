#!/usr/bin/env python3
"""
Basketball Dataset Preparation and Management
Handles dataset creation, annotation assistance, and validation
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from typing import List, Dict, Tuple, Optional
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

class BasketballDatasetManager:
    """Manages basketball dataset for custom YOLO training."""
    
    def __init__(self, dataset_root: str = "./basketball_dataset"):
        self.dataset_root = Path(dataset_root)
        self.classes = {
            0: "player",
            1: "ball", 
            2: "referee",
            3: "basket",
            4: "board"
        }
        self.class_colors = {
            0: (255, 0, 0),    # Red for players
            1: (255, 165, 0),  # Orange for ball
            2: (0, 0, 255),    # Blue for referee
            3: (0, 255, 0),    # Green for basket
            4: (128, 0, 128)   # Purple for board
        }
        
    def create_dataset_structure(self):
        """Create the required dataset directory structure."""
        print("🏗️ Creating dataset structure...")
        
        # Create main directories
        dirs_to_create = [
            self.dataset_root,
            self.dataset_root / "images" / "train",
            self.dataset_root / "images" / "val", 
            self.dataset_root / "images" / "test",
            self.dataset_root / "labels" / "train",
            self.dataset_root / "labels" / "val",
            self.dataset_root / "labels" / "test",
            self.dataset_root / "annotations",
            self.dataset_root / "raw_videos",
            self.dataset_root / "extracted_frames"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Dataset structure created at: {self.dataset_root}")
        
    def extract_frames_from_video(self, video_path: str, output_dir: str = None, 
                                frame_interval: int = 30, max_frames: int = 1000):
        """Extract frames from basketball video for annotation."""
        if output_dir is None:
            output_dir = self.dataset_root / "extracted_frames"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Extracting frames from {video_name}")
        print(f"   Total frames: {total_frames}, FPS: {fps}")
        print(f"   Extracting every {frame_interval} frames, max {max_frames}")
        
        frame_count = 0
        extracted_count = 0
        
        pbar = tqdm(total=min(total_frames // frame_interval, max_frames))
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
                pbar.update(1)
                
            frame_count += 1
            
        cap.release()
        pbar.close()
        
        print(f"✅ Extracted {extracted_count} frames to {output_dir}")
        return extracted_count
        
    def generate_pseudo_labels(self, image_dir: str, confidence_threshold: float = 0.3):
        """Generate initial pseudo-labels using pre-trained YOLO for players."""
        print("🔍 Generating pseudo-labels with pre-trained YOLO...")
        
        image_dir = Path(image_dir)
        label_dir = image_dir.parent / "labels" / image_dir.name
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Load pre-trained YOLO model
        model = YOLO('yolov8n.pt')
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc="Generating pseudo-labels"):
            # Run detection
            results = model(str(img_path), conf=confidence_threshold, verbose=False)
            
            # Get image dimensions
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Create label file
            label_path = label_dir / f"{img_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # YOLO class 0 is 'person' - we'll map this to 'player'
                            if int(box.cls) == 0:  # Person class
                                # Convert to YOLO format (normalized coordinates)
                                x_center = (box.xywh[0][0] / w).item()
                                y_center = (box.xywh[0][1] / h).item()
                                width = (box.xywh[0][2] / w).item()
                                height = (box.xywh[0][3] / h).item()
                                
                                # Class 0 for player in our custom dataset
                                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                
        print(f"✅ Generated pseudo-labels in {label_dir}")
        
    def visualize_annotations(self, image_path: str, label_path: str = None, 
                            save_path: str = None):
        """Visualize annotations on an image."""
        image_path = Path(image_path)
        
        if label_path is None:
            label_path = image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        else:
            label_path = Path(label_path)
            
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        
        # Read labels if they exist
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        # Convert from normalized to pixel coordinates
                        x_center *= w
                        y_center *= h
                        width *= w
                        height *= h
                        
                        # Calculate top-left corner
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        
                        # Draw bounding box
                        color = np.array(self.class_colors[class_id]) / 255.0
                        rect = patches.Rectangle((x1, y1), width, height, 
                                               linewidth=2, edgecolor=color, 
                                               facecolor='none')
                        ax.add_patch(rect)
                        
                        # Add label
                        class_name = self.classes[class_id]
                        ax.text(x1, y1-5, class_name, color=color, fontsize=10, 
                               weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                                       facecolor='white', alpha=0.7))
        
        ax.set_title(f"Annotations: {image_path.name}")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
        
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                     test_ratio: float = 0.1):
        """Split extracted frames into train/val/test sets."""
        print("📊 Splitting dataset...")
        
        extracted_dir = self.dataset_root / "extracted_frames"
        if not extracted_dir.exists():
            raise ValueError("No extracted frames found. Run extract_frames_from_video first.")
            
        # Get all images
        image_files = list(extracted_dir.glob("*.jpg")) + list(extracted_dir.glob("*.png"))
        
        if len(image_files) == 0:
            raise ValueError("No image files found in extracted_frames directory")
            
        # Shuffle and split
        np.random.shuffle(image_files)
        
        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Move files to appropriate directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split, files in splits.items():
            img_dir = self.dataset_root / "images" / split
            
            print(f"   Moving {len(files)} images to {split} set...")
            for img_file in files:
                dest_path = img_dir / img_file.name
                shutil.copy2(img_file, dest_path)
                
        print(f"✅ Dataset split complete:")
        print(f"   Train: {len(train_files)} images")
        print(f"   Val: {len(val_files)} images") 
        print(f"   Test: {len(test_files)} images")
        
    def create_annotation_template(self, output_file: str = "annotation_instructions.md"):
        """Create annotation instructions and templates."""
        instructions = """# Basketball Dataset Annotation Instructions

## Object Classes

### 1. Player (Class ID: 0)
- **Description**: Any basketball player on the court
- **Include**: All players regardless of team, position, or activity
- **Exclude**: Coaches, bench players clearly off-court
- **Tips**: Include partially visible players, players in motion

### 2. Ball (Class ID: 1)  
- **Description**: The basketball itself
- **Include**: Ball in any state - held, dribbling, in air, bouncing
- **Exclude**: Other balls or objects
- **Tips**: Ball can be small, track carefully even when partially obscured

### 3. Referee (Class ID: 2)
- **Description**: Game officials in referee uniforms
- **Include**: All referees on court
- **Exclude**: Other officials, scorekeepers
- **Tips**: Usually in distinctive striped or colored uniforms

### 4. Basket (Class ID: 3)
- **Description**: Basketball hoop/rim and net
- **Include**: The circular rim, net if visible
- **Exclude**: Just the backboard without rim
- **Tips**: Sometimes only partially visible, include if rim is visible

### 5. Board (Class ID: 4)
- **Description**: The backboard behind the basket
- **Include**: Rectangular backboard surface
- **Exclude**: Support structures, shot clock
- **Tips**: Can be glass or other material, focus on the playing surface

## Annotation Guidelines

### Bounding Box Rules
1. **Tight fitting**: Box should closely fit the object
2. **Complete object**: Include the entire visible object
3. **Overlapping objects**: Create separate boxes for each object
4. **Partial visibility**: Include even if partially occluded

### Quality Standards
- **Consistency**: Maintain consistent box sizes for similar objects
- **Accuracy**: Precise boundary placement
- **Completeness**: Don't skip difficult cases
- **Review**: Double-check annotations before saving

### Common Scenarios
- **Multiple players**: Each player gets separate annotation
- **Player with ball**: Annotate both player and ball separately  
- **Crowd shots**: Only annotate clearly visible court players
- **Multiple camera angles**: Adapt to different perspectives

## File Format
Annotations should be in YOLO format:
```
class_id x_center y_center width height
```
Where coordinates are normalized (0-1) relative to image dimensions.

## Tools Recommended
- LabelImg: https://github.com/tzutalin/labelImg
- Roboflow: https://roboflow.com/
- CVAT: https://github.com/openvinotoolkit/cvat
"""
        
        output_path = self.dataset_root / output_file
        with open(output_path, 'w') as f:
            f.write(instructions)
            
        print(f"📝 Annotation instructions created: {output_path}")
        
    def validate_dataset(self):
        """Validate the dataset structure and annotations."""
        print("🔍 Validating dataset...")
        
        issues = []
        
        # Check directory structure
        required_dirs = [
            "images/train", "images/val", "labels/train", "labels/val"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.dataset_root / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_path}")
                
        # Check image-label pairs
        for split in ['train', 'val']:
            img_dir = self.dataset_root / "images" / split
            label_dir = self.dataset_root / "labels" / split
            
            if img_dir.exists() and label_dir.exists():
                img_files = set(f.stem for f in img_dir.glob("*.jpg"))
                img_files.update(f.stem for f in img_dir.glob("*.png"))
                label_files = set(f.stem for f in label_dir.glob("*.txt"))
                
                missing_labels = img_files - label_files
                missing_images = label_files - img_files
                
                if missing_labels:
                    issues.append(f"{split}: {len(missing_labels)} images without labels")
                if missing_images:
                    issues.append(f"{split}: {len(missing_images)} labels without images")
                    
        # Check annotation format
        for split in ['train', 'val']:
            label_dir = self.dataset_root / "labels" / split
            if label_dir.exists():
                for label_file in label_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    class_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:]]
                                    
                                    if class_id not in self.classes:
                                        issues.append(f"{label_file}:{line_num} - Invalid class ID: {class_id}")
                                    
                                    if not all(0 <= x <= 1 for x in coords):
                                        issues.append(f"{label_file}:{line_num} - Coordinates out of range")
                                        
                                except ValueError:
                                    issues.append(f"{label_file}:{line_num} - Invalid number format")
                            elif line.strip():  # Non-empty line
                                issues.append(f"{label_file}:{line_num} - Invalid format")
                                
        if issues:
            print("❌ Validation issues found:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Dataset validation passed!")
            return True
            
    def get_dataset_stats(self):
        """Get statistics about the dataset."""
        stats = {
            'train': {'images': 0, 'labels': 0, 'objects': {cls: 0 for cls in self.classes.values()}},
            'val': {'images': 0, 'labels': 0, 'objects': {cls: 0 for cls in self.classes.values()}},
            'test': {'images': 0, 'labels': 0, 'objects': {cls: 0 for cls in self.classes.values()}}
        }
        
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_root / "images" / split
            label_dir = self.dataset_root / "labels" / split
            
            if img_dir.exists():
                stats[split]['images'] = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
                
            if label_dir.exists():
                label_files = list(label_dir.glob("*.txt"))
                stats[split]['labels'] = len(label_files)
                
                # Count objects by class
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                if class_id in self.classes:
                                    class_name = self.classes[class_id]
                                    stats[split]['objects'][class_name] += 1
                                    
        return stats
        
    def print_dataset_summary(self):
        """Print a summary of the dataset."""
        stats = self.get_dataset_stats()
        
        print("\n📊 Dataset Summary")
        print("=" * 50)
        
        for split, data in stats.items():
            print(f"\n{split.upper()} SET:")
            print(f"  Images: {data['images']}")
            print(f"  Labels: {data['labels']}")
            print("  Objects:")
            for class_name, count in data['objects'].items():
                print(f"    {class_name}: {count}")
                
        # Total summary
        total_images = sum(data['images'] for data in stats.values())
        total_objects = {}
        for class_name in self.classes.values():
            total_objects[class_name] = sum(data['objects'][class_name] for data in stats.values())
            
        print(f"\nTOTAL:")
        print(f"  Images: {total_images}")
        print("  Objects:")
        for class_name, count in total_objects.items():
            print(f"    {class_name}: {count}")


if __name__ == "__main__":
    # Example usage
    manager = BasketballDatasetManager()
    
    print("🏀 Basketball Dataset Manager")
    print("Choose an option:")
    print("1. Create dataset structure")
    print("2. Extract frames from video")
    print("3. Generate pseudo-labels")
    print("4. Split dataset")
    print("5. Validate dataset")
    print("6. Show dataset summary")
    print("7. Create annotation instructions")
    
    choice = input("Enter choice (1-7): ").strip()
    
    if choice == "1":
        manager.create_dataset_structure()
    elif choice == "2":
        video_path = input("Enter video path: ").strip()
        if os.path.exists(video_path):
            manager.extract_frames_from_video(video_path)
        else:
            print("Video file not found!")
    elif choice == "3":
        manager.generate_pseudo_labels("./basketball_dataset/extracted_frames")
    elif choice == "4":
        manager.split_dataset()
    elif choice == "5":
        manager.validate_dataset()
    elif choice == "6":
        manager.print_dataset_summary()
    elif choice == "7":
        manager.create_annotation_template()
    else:
        print("Invalid choice!")
