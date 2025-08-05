#!/usr/bin/env python3
"""
Quick Start: Basketball YOLO Training
Minimal setup for immediate training with sample data
"""

import os
import sys
from pathlib import Path
from dataset_manager import BasketballDatasetManager
from yolo_trainer import BasketballYOLOTrainer

def quick_start_training():
    """Quick start training with minimal setup."""
    print("🚀 Basketball YOLO Quick Start")
    print("=" * 30)
    
    # Step 1: Setup dataset structure
    print("\n1️⃣ Setting up dataset structure...")
    manager = BasketballDatasetManager()
    manager.create_dataset_structure()
    
    # Step 2: Check for video
    video_path = input("Enter path to basketball video (or press Enter to skip): ").strip()
    
    if video_path and Path(video_path).exists():
        print("\n2️⃣ Extracting frames...")
        manager.extract_frames_from_video(video_path, max_frames=200)
        
        print("\n3️⃣ Generating pseudo-labels...")
        manager.generate_pseudo_labels("./basketball_dataset/extracted_frames")
        
        print("\n4️⃣ Splitting dataset...")
        manager.split_dataset()
        
        print("\n5️⃣ Validating dataset...")
        if manager.validate_dataset():
            print("\n6️⃣ Starting training...")
            trainer = BasketballYOLOTrainer()
            trainer.initialize_model("n", pretrained=True)
            trainer.train_model(epochs=50)  # Quick training
            
            print("\n✅ Quick training complete!")
            print("   For better results:")
            print("   1. Manually annotate more images")
            print("   2. Increase epochs to 100+")
            print("   3. Add more diverse data")
        else:
            print("\n❌ Dataset validation failed!")
            print("   Please check the dataset and fix issues")
    else:
        print("\n⚠️ No video provided. Setting up structure only.")
        print("   Manual steps required:")
        print("   1. Add images to basketball_dataset/extracted_frames/")
        print("   2. Annotate images manually")
        print("   3. Run dataset split and validation")
        print("   4. Start training")
    
    # Generate instructions
    manager.create_annotation_template()
    
    print("\n📖 Next steps:")
    print("   - Check TRAINING_WORKFLOW.md for detailed guide")
    print("   - Use annotation_instructions.md for labeling guide")
    print("   - Run python yolo_trainer.py for advanced training options")

if __name__ == "__main__":
    quick_start_training()
