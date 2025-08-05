#!/usr/bin/env python3
"""
Automated Basketball Dataset Generator
Automatically creates training data for custom YOLO model
"""

import os
import cv2
import numpy as np
import requests
import zipfile
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple
import tempfile
import shutil
from tqdm import tqdm
import time

class AutomatedDatasetGenerator:
    """Generates basketball training dataset automatically."""
    
    def __init__(self, output_dir: str = "auto_basketball_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Basketball-specific parameters
        self.court_color_ranges = {
            'light_wood': ([15, 50, 100], [25, 255, 255]),
            'dark_wood': ([10, 50, 50], [20, 255, 200]),
            'synthetic': ([80, 30, 30], [120, 255, 255])
        }
        
        self.ball_color_range = ([5, 100, 100], [15, 255, 255])  # Orange basketball
        
        print(f"🏀 Automated Basketball Dataset Generator")
        print(f"📁 Output directory: {self.output_dir}")
        
    def download_sample_videos(self) -> List[str]:
        """Download sample basketball videos for training data extraction."""
        print("📥 Downloading sample basketball videos...")
        
        # Public domain / creative commons basketball videos
        video_urls = [
            "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            # Add more basketball video URLs here
        ]
        
        downloaded_videos = []
        
        for i, url in enumerate(video_urls):
            try:
                video_name = f"basketball_sample_{i+1}.mp4"
                video_path = self.output_dir / video_name
                
                print(f"   Downloading {video_name}...")
                response = requests.get(url, stream=True)
                
                if response.status_code == 200:
                    with open(video_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_videos.append(str(video_path))
                    print(f"   ✅ Downloaded: {video_name}")
                else:
                    print(f"   ❌ Failed to download from {url}")
                    
            except Exception as e:
                print(f"   ❌ Error downloading {url}: {e}")
                
        return downloaded_videos
        
    def generate_synthetic_court_data(self, num_images: int = 500) -> List[str]:
        """Generate synthetic basketball court images with objects."""
        print(f"🎨 Generating {num_images} synthetic basketball images...")
        
        synthetic_dir = self.output_dir / "synthetic_images"
        synthetic_dir.mkdir(exist_ok=True)
        
        generated_images = []
        
        for i in tqdm(range(num_images), desc="Generating synthetic data"):
            # Create court background
            court_img = self._create_synthetic_court()
            
            # Add basketball objects
            court_img, annotations = self._add_basketball_objects(court_img)
            
            # Save image and annotations
            img_name = f"synthetic_court_{i:04d}.jpg"
            img_path = synthetic_dir / img_name
            cv2.imwrite(str(img_path), court_img)
            
            # Save annotations in YOLO format
            label_name = f"synthetic_court_{i:04d}.txt"
            label_path = synthetic_dir / label_name
            self._save_yolo_annotations(annotations, label_path, court_img.shape)
            
            generated_images.append(str(img_path))
            
        print(f"✅ Generated {len(generated_images)} synthetic images")
        return generated_images
        
    def _create_synthetic_court(self) -> np.ndarray:
        """Create a synthetic basketball court image."""
        # Random court dimensions
        width = random.randint(800, 1200)
        height = random.randint(600, 900)
        
        # Random court color
        court_colors = [
            (139, 115, 85),   # Light brown wood
            (101, 67, 33),    # Dark brown wood
            (200, 180, 140),  # Light court
            (160, 140, 100)   # Synthetic court
        ]
        
        court_color = random.choice(court_colors)
        court_img = np.full((height, width, 3), court_color, dtype=np.uint8)
        
        # Add court lines
        line_color = (255, 255, 255)  # White lines
        line_thickness = random.randint(2, 5)
        
        # Center circle
        center_x, center_y = width // 2, height // 2
        cv2.circle(court_img, (center_x, center_y), 
                  random.randint(80, 120), line_color, line_thickness)
        
        # Court boundaries
        cv2.rectangle(court_img, (50, 50), (width-50, height-50), 
                     line_color, line_thickness)
        
        # Three-point lines (simplified)
        arc_radius = random.randint(150, 200)
        cv2.ellipse(court_img, (center_x, height-50), 
                   (arc_radius, arc_radius//2), 0, 0, 180, line_color, line_thickness)
        cv2.ellipse(court_img, (center_x, 50), 
                   (arc_radius, arc_radius//2), 0, 180, 360, line_color, line_thickness)
        
        return court_img
        
    def _add_basketball_objects(self, court_img: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Add basketball objects to court image."""
        annotations = []
        height, width = court_img.shape[:2]
        
        # Add players (2-10 players)
        num_players = random.randint(4, 10)
        for _ in range(num_players):
            player_annotation = self._add_player(court_img)
            if player_annotation:
                annotations.append(player_annotation)
                
        # Add basketball (0-1 balls)
        if random.random() < 0.8:  # 80% chance of ball
            ball_annotation = self._add_basketball(court_img)
            if ball_annotation:
                annotations.append(ball_annotation)
                
        # Add referees (0-2 referees)
        num_refs = random.randint(0, 2)
        for _ in range(num_refs):
            ref_annotation = self._add_referee(court_img)
            if ref_annotation:
                annotations.append(ref_annotation)
                
        # Add baskets (1-2 baskets)
        num_baskets = random.randint(1, 2)
        for _ in range(num_baskets):
            basket_annotation = self._add_basket(court_img)
            if basket_annotation:
                annotations.append(basket_annotation)
                
        return court_img, annotations
        
    def _add_player(self, img: np.ndarray) -> Dict:
        """Add a synthetic player to the image."""
        height, width = img.shape[:2]
        
        # Random player position (avoid edges)
        x = random.randint(100, width - 100)
        y = random.randint(100, height - 100)
        
        # Player size (height varies by perspective)
        player_height = random.randint(80, 150)
        player_width = random.randint(30, 60)
        
        # Player colors (jersey colors)
        jersey_colors = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        jersey_color = random.choice(jersey_colors)
        
        # Draw simplified player (rectangle with head circle)
        # Body
        cv2.rectangle(img, 
                     (x - player_width//2, y - player_height//2),
                     (x + player_width//2, y + player_height//2),
                     jersey_color, -1)
        
        # Head
        head_radius = player_width // 4
        cv2.circle(img, (x, y - player_height//2 - head_radius), 
                  head_radius, (222, 184, 135), -1)  # Skin color
        
        # Create annotation
        annotation = {
            'class_id': 0,  # player
            'x_center': x / width,
            'y_center': y / height,
            'width': player_width / width,
            'height': (player_height + head_radius*2) / height
        }
        
        return annotation
        
    def _add_basketball(self, img: np.ndarray) -> Dict:
        """Add a synthetic basketball to the image."""
        height, width = img.shape[:2]
        
        # Random ball position
        x = random.randint(50, width - 50)
        y = random.randint(50, height - 50)
        
        # Ball size
        ball_radius = random.randint(8, 25)
        
        # Basketball color (orange)
        ball_color = (0, 165, 255)  # BGR format - orange
        
        # Draw basketball
        cv2.circle(img, (x, y), ball_radius, ball_color, -1)
        
        # Add basketball lines
        cv2.line(img, (x - ball_radius, y), (x + ball_radius, y), (0, 0, 0), 2)
        cv2.line(img, (x, y - ball_radius), (x, y + ball_radius), (0, 0, 0), 2)
        
        # Curved lines
        cv2.ellipse(img, (x, y), (ball_radius, ball_radius//2), 
                   45, 0, 180, (0, 0, 0), 2)
        cv2.ellipse(img, (x, y), (ball_radius, ball_radius//2), 
                   -45, 0, 180, (0, 0, 0), 2)
        
        # Create annotation
        annotation = {
            'class_id': 1,  # ball
            'x_center': x / width,
            'y_center': y / height,
            'width': (ball_radius * 2) / width,
            'height': (ball_radius * 2) / height
        }
        
        return annotation
        
    def _add_referee(self, img: np.ndarray) -> Dict:
        """Add a synthetic referee to the image."""
        height, width = img.shape[:2]
        
        # Random referee position (usually on sidelines)
        if random.random() < 0.5:
            x = random.randint(50, 150)  # Left sideline
        else:
            x = random.randint(width-150, width-50)  # Right sideline
        y = random.randint(100, height - 100)
        
        # Referee size
        ref_height = random.randint(70, 120)
        ref_width = random.randint(25, 50)
        
        # Referee colors (striped shirt)
        stripe_color1 = (255, 255, 255)  # White
        stripe_color2 = (0, 0, 0)        # Black
        
        # Draw striped referee
        stripe_width = ref_width // 6
        for i in range(-3, 4):
            color = stripe_color1 if i % 2 == 0 else stripe_color2
            cv2.rectangle(img,
                         (x + i*stripe_width, y - ref_height//2),
                         (x + (i+1)*stripe_width, y + ref_height//2),
                         color, -1)
        
        # Head
        head_radius = ref_width // 4
        cv2.circle(img, (x, y - ref_height//2 - head_radius), 
                  head_radius, (222, 184, 135), -1)
        
        # Create annotation
        annotation = {
            'class_id': 2,  # referee
            'x_center': x / width,
            'y_center': y / height,
            'width': ref_width / width,
            'height': (ref_height + head_radius*2) / height
        }
        
        return annotation
        
    def _add_basket(self, img: np.ndarray) -> Dict:
        """Add a synthetic basketball basket to the image."""
        height, width = img.shape[:2]
        
        # Basket position (usually at ends of court)
        if random.random() < 0.5:
            x = random.randint(50, 150)  # Left side
        else:
            x = random.randint(width-150, width-50)  # Right side
        y = random.randint(100, 300)  # Upper part of court
        
        # Basket dimensions
        rim_width = random.randint(40, 80)
        rim_height = random.randint(8, 15)
        
        # Basket color (orange rim)
        rim_color = (0, 165, 255)  # Orange
        
        # Draw basket rim
        cv2.ellipse(img, (x, y), (rim_width//2, rim_height//2), 
                   0, 0, 360, rim_color, 3)
        
        # Net (simplified)
        net_color = (255, 255, 255)  # White
        for i in range(-2, 3):
            start_x = x + i * (rim_width // 5)
            cv2.line(img, (start_x, y), (start_x, y + 20), net_color, 1)
        
        # Create annotation
        annotation = {
            'class_id': 3,  # basket
            'x_center': x / width,
            'y_center': y / height,
            'width': rim_width / width,
            'height': (rim_height + 20) / height
        }
        
        return annotation
        
    def _save_yolo_annotations(self, annotations: List[Dict], 
                             label_path: Path, img_shape: Tuple):
        """Save annotations in YOLO format."""
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann['class_id']} {ann['x_center']:.6f} "
                       f"{ann['y_center']:.6f} {ann['width']:.6f} "
                       f"{ann['height']:.6f}\n")
                       
    def create_dataset_structure(self):
        """Create proper YOLO dataset structure."""
        print("🏗️ Creating dataset structure...")
        
        # Create directories
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print("✅ Dataset structure created")
        
    def split_synthetic_data(self, synthetic_images: List[str], 
                           train_ratio: float = 0.7, val_ratio: float = 0.2):
        """Split synthetic data into train/val/test sets."""
        print("📊 Splitting synthetic data...")
        
        random.shuffle(synthetic_images)
        
        n_train = int(len(synthetic_images) * train_ratio)
        n_val = int(len(synthetic_images) * val_ratio)
        
        train_images = synthetic_images[:n_train]
        val_images = synthetic_images[n_train:n_train + n_val]
        test_images = synthetic_images[n_train + n_val:]
        
        # Move files to appropriate directories
        for split, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            for img_path in images:
                img_path = Path(img_path)
                label_path = img_path.with_suffix('.txt')
                
                # Move image
                dest_img = self.output_dir / "images" / split / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Move label
                if label_path.exists():
                    dest_label = self.output_dir / "labels" / split / label_path.name
                    shutil.copy2(label_path, dest_label)
                    
        print(f"✅ Data split complete:")
        print(f"   Train: {len(train_images)} images")
        print(f"   Val: {len(val_images)} images")
        print(f"   Test: {len(test_images)} images")
        
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file."""
        yaml_content = f"""# Basketball Custom Dataset Configuration
path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 5

# Class names
names:
  0: player
  1: ball
  2: referee
  3: basket
  4: board

# Training settings
train_settings:
  epochs: 100
  imgsz: 640
  batch: 16
  workers: 8
  patience: 50
"""
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        print(f"📄 Dataset YAML created: {yaml_path}")
        return str(yaml_path)
        
    def generate_complete_dataset(self, num_synthetic: int = 1000) -> str:
        """Generate complete basketball dataset automatically."""
        print("🚀 Generating complete basketball dataset...")
        
        # Create dataset structure
        self.create_dataset_structure()
        
        # Generate synthetic data
        synthetic_images = self.generate_synthetic_court_data(num_synthetic)
        
        # Split data
        self.split_synthetic_data(synthetic_images)
        
        # Create dataset configuration
        yaml_path = self.create_dataset_yaml()
        
        # Clean up temporary synthetic directory
        synthetic_dir = self.output_dir / "synthetic_images"
        if synthetic_dir.exists():
            shutil.rmtree(synthetic_dir)
            
        print("✅ Complete basketball dataset generated!")
        print(f"📁 Dataset location: {self.output_dir}")
        print(f"📄 Configuration file: {yaml_path}")
        
        return yaml_path


def main():
    """Generate basketball dataset automatically."""
    print("🏀 Automated Basketball Dataset Generator")
    print("=" * 40)
    
    generator = AutomatedDatasetGenerator()
    
    num_images = input("Number of synthetic images to generate [1000]: ").strip()
    num_images = int(num_images) if num_images else 1000
    
    print(f"\n🎨 Generating {num_images} synthetic basketball images...")
    print("This will create a complete training dataset automatically.")
    
    dataset_yaml = generator.generate_complete_dataset(num_images)
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"📄 Use this config file for training: {dataset_yaml}")
    print("\n🚀 Ready for training! Run:")
    print("   python yolo_trainer.py")
    print("   Choose option 2: Train model")


if __name__ == "__main__":
    main()
