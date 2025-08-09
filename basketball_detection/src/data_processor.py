"""
Basketball Detection System - Data Processor
Processes and combines basketball datasets from local folders
"""

import os
import yaml
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from tqdm import tqdm
import random

class DataProcessor:
    def __init__(self, dataset_path="../dataset"):
        self.dataset_path = Path(dataset_path)
        self.class_mapping = {
            'basketball': 'ball',
            'sports ball': 'ball', 
            'rim': 'hoop',
            'basket': 'hoop'
        }
        
    def discover_classes(self):
        """Discover classes from all dataset folders"""
        all_classes = set()
        
        # Process folders 1, 2, 3
        for folder in ['1', '2', '3']:
            folder_path = self.dataset_path / folder / "data.yaml"
            if folder_path.exists():
                with open(folder_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        if isinstance(data['names'], list):
                            all_classes.update(data['names'])
                        elif isinstance(data['names'], dict):
                            all_classes.update(data['names'].values())
        
        # Process folder 4 (XML format)
        xml_file = self.dataset_path / "4" / "annotations.xml"
        if xml_file.exists():
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for track in root.findall('.//track'):
                label = track.get('label')
                if label:
                    all_classes.add(label)
        
        # Apply class mapping
        mapped_classes = set()
        for cls in all_classes:
            mapped_cls = self.class_mapping.get(cls, cls)
            mapped_classes.add(mapped_cls)
        
        # Standard basketball classes
        final_classes = ['player', 'referee', 'ball', 'hoop']
        return [cls for cls in final_classes if cls in mapped_classes or cls in all_classes]
    
    def convert_xml_to_yolo(self, xml_file, images_dir, output_dir, class_names):
        """Convert XML annotations to YOLO format"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get original image dimensions
        width = int(root.find('.//original_size/width').text)
        height = int(root.find('.//original_size/height').text)
        
        # Process tracks
        frame_annotations = {}
        for track in root.findall('.//track'):
            label = track.get('label')
            mapped_label = self.class_mapping.get(label, label)
            
            if mapped_label not in class_names:
                continue
                
            class_id = class_names.index(mapped_label)
            
            for box in track.findall('box'):
                frame = int(box.get('frame'))
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                # Convert to YOLO format
                x_center = (xtl + xbr) / 2 / width
                y_center = (ytl + ybr) / 2 / height
                bbox_width = (xbr - xtl) / width
                bbox_height = (ybr - ytl) / height
                
                if frame not in frame_annotations:
                    frame_annotations[frame] = []
                
                frame_annotations[frame].append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # Save annotations and copy images
        saved_files = []
        for frame, annotations in frame_annotations.items():
            # Find image file
            image_file = None
            for ext in ['.PNG', '.jpg', '.jpeg', '.png']:
                potential_file = images_dir / f"frame_{frame:06d}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
            
            if image_file and image_file.exists():
                # Copy image
                new_image_name = f"xml_{frame:06d}.jpg"
                shutil.copy2(image_file, output_dir / "images" / new_image_name)
                
                # Save annotation
                label_file = output_dir / "labels" / f"xml_{frame:06d}.txt"
                with open(label_file, 'w') as f:
                    for annotation in annotations:
                        f.write(annotation + "\n")
                
                saved_files.append(new_image_name)
        
        return saved_files
    
    def convert_yolo_dataset(self, folder_path, output_dir, class_names):
        """Convert existing YOLO dataset with class remapping"""
        # Read original classes
        yaml_file = folder_path / "data.yaml"
        original_classes = []
        
        if yaml_file.exists():
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], list):
                        original_classes = data['names']
                    elif isinstance(data['names'], dict):
                        original_classes = list(data['names'].values())
        
        # Create class mapping
        class_id_mapping = {}
        for orig_id, orig_class in enumerate(original_classes):
            mapped_class = self.class_mapping.get(orig_class, orig_class)
            if mapped_class in class_names:
                class_id_mapping[orig_id] = class_names.index(mapped_class)
        
        saved_files = []
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            images_dir = folder_path / split / "images"
            labels_dir = folder_path / split / "labels"
            
            if not (images_dir.exists() and labels_dir.exists()):
                continue
            
            for image_file in images_dir.glob("*.*"):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label_file = labels_dir / (image_file.stem + '.txt')
                    
                    if label_file.exists():
                        # Read and convert labels
                        converted_labels = []
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    orig_class_id = int(parts[0])
                                    if orig_class_id in class_id_mapping:
                                        new_class_id = class_id_mapping[orig_class_id]
                                        converted_labels.append(f"{new_class_id} {' '.join(parts[1:])}")
                        
                        if converted_labels:
                            # Copy image
                            folder_name = folder_path.name
                            new_image_name = f"{folder_name}_{split}_{image_file.name}"
                            shutil.copy2(image_file, output_dir / "images" / new_image_name)
                            
                            # Save converted labels
                            new_label_file = output_dir / "labels" / f"{folder_name}_{split}_{image_file.stem}.txt"
                            with open(new_label_file, 'w') as f:
                                for label in converted_labels:
                                    f.write(label + "\n")
                            
                            saved_files.append(new_image_name)
        
        return saved_files
    
    def create_dataset(self):
        """Create unified YOLO dataset"""
        print("🏀 Creating Basketball Detection Dataset...")
        
        # Discover classes
        classes = self.discover_classes()
        print(f"📋 Discovered classes: {classes}")
        
        # Create output structure
        output_dir = Path("./data/basketball_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Temporary directory for all data
        temp_dir = Path("./data/temp_all")
        temp_dir.mkdir(parents=True, exist_ok=True)
        (temp_dir / "images").mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        all_files = []
        
        # Process folder 4 (XML)
        xml_file = self.dataset_path / "4" / "annotations.xml"
        images_dir = self.dataset_path / "4" / "images"
        if xml_file.exists() and images_dir.exists():
            print("  Converting XML dataset (folder 4)...")
            files = self.convert_xml_to_yolo(xml_file, images_dir, temp_dir, classes)
            all_files.extend(files)
        
        # Process folders 1, 2, 3 (YOLO format)
        for folder in ['1', '2', '3']:
            folder_path = self.dataset_path / folder
            if folder_path.exists():
                print(f"  Converting YOLO dataset (folder {folder})...")
                files = self.convert_yolo_dataset(folder_path, temp_dir, classes)
                all_files.extend(files)
        
        print(f"📊 Total images processed: {len(all_files)}")
        
        # Split data
        random.shuffle(all_files)
        total = len(all_files)
        train_split = int(0.7 * total)
        val_split = int(0.2 * total)
        
        splits = {
            'train': all_files[:train_split],
            'val': all_files[train_split:train_split + val_split],
            'test': all_files[train_split + val_split:]
        }
        
        # Move files to splits
        for split_name, files in splits.items():
            print(f"  {split_name}: {len(files)} images")
            for file in tqdm(files, desc=f"Moving {split_name}"):
                # Move image
                src_img = temp_dir / "images" / file
                dst_img = output_dir / split_name / "images" / file
                if src_img.exists():
                    shutil.move(str(src_img), str(dst_img))
                
                # Move label
                label_file = file.rsplit('.', 1)[0] + '.txt'
                src_lbl = temp_dir / "labels" / label_file
                dst_lbl = output_dir / split_name / "labels" / label_file
                if src_lbl.exists():
                    shutil.move(str(src_lbl), str(dst_lbl))
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Create dataset.yaml
        dataset_yaml = {
            'train': str(output_dir / 'train'),
            'val': str(output_dir / 'val'),
            'test': str(output_dir / 'test'),
            'nc': len(classes),
            'names': {i: name for i, name in enumerate(classes)}
        }
        
        with open(output_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"✅ Dataset created successfully!")
        print(f"   Path: {output_dir / 'dataset.yaml'}")
        print(f"   Classes: {classes}")
        
        return str(output_dir / 'dataset.yaml'), classes

if __name__ == "__main__":
    processor = DataProcessor()
    dataset_path, classes = processor.create_dataset()
    print(f"Dataset ready: {dataset_path}")
