"""
Basketball Detection System
Main entry point for the basketball detection system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_processor import DataProcessor
from src.train_model import BasketballTrainer
from src.inference import BasketballInference

def main():
    """Main function with menu system"""
    print("🏀 Basketball Detection System")
    print("=" * 50)
    print("1. Process Dataset")
    print("2. Train Model")
    print("3. Run Inference")
    print("4. Full Pipeline")
    print("5. Exit")
    print("=" * 50)
    
    while True:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            process_dataset()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_inference()
        elif choice == '4':
            full_pipeline()
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

def process_dataset():
    """Process and create dataset"""
    print("\n📁 Processing Dataset...")
    try:
        processor = DataProcessor()
        dataset_path, classes = processor.create_dataset()
        print(f"✅ Dataset created: {dataset_path}")
        print(f"   Classes: {classes}")
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")

def train_model():
    """Train basketball detection model"""
    print("\n🎯 Training Model...")
    
    # Check if dataset exists
    dataset_path = "./data/basketball_dataset/dataset.yaml"
    if not Path(dataset_path).exists():
        print("❌ Dataset not found! Please process dataset first (option 1).")
        return
    
    try:
        trainer = BasketballTrainer(model_size='n')
        trainer.load_model()
        
        # Training parameters
        epochs = int(input("Enter number of epochs (default 50): ") or "50")
        batch_size = int(input("Enter batch size (default 8): ") or "8")
        
        print(f"Training with {epochs} epochs, batch size {batch_size}")
        
        _, model_path = trainer.train_model(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print(f"✅ Training complete! Model saved: {model_path}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")

def run_inference():
    """Run inference on video"""
    print("\n🔍 Running Inference...")
    
    # Check if model exists
    model_path = "./models/basketball_yolo11n.pt"
    if not Path(model_path).exists():
        print("❌ Trained model not found! Please train model first (option 2).")
        return
    
    try:
        inference = BasketballInference(model_path)
        
        # Find video files
        video_files = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.avi"))
        
        if not video_files:
            print("❌ No video files found! Please add .mp4 or .avi files.")
            return
        
        print("📹 Available videos:")
        for i, video in enumerate(video_files):
            print(f"   {i+1}. {video.name}")
        
        choice = input(f"Select video (1-{len(video_files)}): ").strip()
        try:
            video_idx = int(choice) - 1
            if 0 <= video_idx < len(video_files):
                video_path = str(video_files[video_idx])
                
                results = inference.process_video(video_path)
                
                if results:
                    inference.analyze_detections(results['detections'])
                    print(f"✅ Inference complete! Check outputs/ folder.")
                else:
                    print("❌ Inference failed!")
            else:
                print("❌ Invalid video selection!")
        except ValueError:
            print("❌ Invalid input!")
            
    except Exception as e:
        print(f"❌ Error running inference: {e}")

def full_pipeline():
    """Run complete pipeline"""
    print("\n🚀 Running Full Pipeline...")
    print("This will:")
    print("  1. Process dataset")
    print("  2. Train model") 
    print("  3. Run inference (if video available)")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Pipeline cancelled.")
        return
    
    try:
        # Step 1: Process dataset
        print("\n📁 Step 1: Processing Dataset...")
        processor = DataProcessor()
        dataset_path, classes = processor.create_dataset()
        print(f"✅ Dataset ready: {len(classes)} classes")
        
        # Step 2: Train model
        print("\n🎯 Step 2: Training Model...")
        trainer = BasketballTrainer(model_size='n')
        trainer.load_model()
        
        _, model_path = trainer.train_model(
            dataset_path=dataset_path,
            epochs=30,  # Quick training for demo
            batch_size=8
        )
        print(f"✅ Model trained: {model_path}")
        
        # Step 3: Run inference if video available
        video_files = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.avi"))
        if video_files:
            print("\n🔍 Step 3: Running Inference...")
            inference = BasketballInference(model_path)
            
            # Use first video
            results = inference.process_video(str(video_files[0]))
            if results:
                inference.analyze_detections(results['detections'])
                print("✅ Inference complete!")
        else:
            print("\n⚠️  No video files found for inference step.")
        
        print("\n🎉 Full pipeline complete!")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()
