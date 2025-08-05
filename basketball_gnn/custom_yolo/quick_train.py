#!/usr/bin/env python3
"""
Quick Basketball YOLO Training - No User Input Required
Automatically generates dataset and trains model with sensible defaults
"""

import os
import sys
from pathlib import Path
import time

def quick_train_basketball_yolo():
    """Quick training with no user input required."""
    
    print("🏀 QUICK BASKETBALL YOLO TRAINING")
    print("=" * 40)
    print("Automatically training custom basketball YOLO with:")
    print("• 500 synthetic training images")
    print("• YOLOv8n model (fastest)")
    print("• 50 epochs (quick training)")
    print("• Batch size 8 (memory efficient)")
    print("=" * 40)
    
    try:
        # Import required modules
        from auto_dataset_generator import AutomatedDatasetGenerator
        from yolo_trainer import BasketballYOLOTrainer
        
        start_time = time.time()
        
        # Step 1: Generate Dataset
        print("\n🎨 STEP 1: Generating Synthetic Dataset...")
        dataset_dir = Path("quick_basketball_dataset")
        generator = AutomatedDatasetGenerator(str(dataset_dir))
        dataset_yaml = generator.generate_complete_dataset(num_synthetic=500)
        print(f"✅ Dataset generated: {dataset_yaml}")
        
        # Step 2: Train Model
        print("\n🤖 STEP 2: Training Custom YOLO Model...")
        trainer = BasketballYOLOTrainer(dataset_yaml, auto_generate_dataset=False)
        trainer.initialize_model("n", pretrained=True)
        
        print("🚀 Training started...")
        training_results = trainer.train_model(
            epochs=50,
            batch=8,
            patience=15,
            save_period=10
        )
        
        # Step 3: Find trained model
        print("\n📁 STEP 3: Locating Trained Model...")
        
        # Look for the trained model
        trained_model_path = None
        
        # Check common YOLO training output locations
        potential_paths = [
            Path("basketball_yolo_training"),
            Path("runs/detect"),
            Path("yolo_training_output")
        ]
        
        for base_path in potential_paths:
            if base_path.exists():
                # Look for the most recent training run
                for run_dir in sorted(base_path.glob("basketball_v*"), reverse=True):
                    weights_dir = run_dir / "weights"
                    if weights_dir.exists():
                        best_model = weights_dir / "best.pt"
                        last_model = weights_dir / "last.pt"
                        
                        if best_model.exists():
                            trained_model_path = str(best_model)
                            break
                        elif last_model.exists():
                            trained_model_path = str(last_model)
                            break
                            
                if trained_model_path:
                    break
        
        # Step 4: Summary
        total_time = time.time() - start_time
        
        print("\n🎉 QUICK TRAINING COMPLETE!")
        print("=" * 40)
        print(f"⏱️  Training time: {total_time/60:.1f} minutes")
        print(f"📊 Dataset: 500 synthetic basketball images")
        print(f"🤖 Model: YOLOv8n trained for 50 epochs")
        
        if trained_model_path:
            print(f"🏆 Trained model: {trained_model_path}")
            
            # Create a simple test script
            test_script_content = f'''#!/usr/bin/env python3
"""
Test the trained basketball YOLO model
"""

from ultralytics import YOLO
import cv2

def test_basketball_yolo():
    # Load the trained model
    model = YOLO(r"{trained_model_path}")
    
    print("🏀 Basketball YOLO Model Loaded!")
    print("Classes detected:")
    print("  0: player")
    print("  1: ball")
    print("  2: referee") 
    print("  3: basket")
    print("  4: board")
    
    # Test on webcam or video
    video_path = input("Enter video path (or 0 for webcam): ").strip()
    
    if video_path == "0":
        video_path = 0
        
    # Run inference
    results = model(video_path, show=True, conf=0.3)
    
    print("✅ Testing complete!")

if __name__ == "__main__":
    test_basketball_yolo()
'''
            
            test_script_path = Path("test_trained_model.py")
            with open(test_script_path, 'w') as f:
                f.write(test_script_content)
                
            print(f"🧪 Test script created: {test_script_path}")
            
            # Integration example
            print("\n🔗 GNN INTEGRATION READY!")
            print("Use your trained model with the GNN system:")
            print(f'   python gnn_integration.py')
            print("   Choose option 1 and enter this model path:")
            print(f"   {trained_model_path}")
            
            return {
                'success': True,
                'model_path': trained_model_path,
                'dataset_path': dataset_yaml,
                'training_time': total_time
            }
        else:
            print("⚠️  Could not locate trained model files")
            print("   Training may have failed or files are in unexpected location")
            
            return {
                'success': False,
                'error': 'Model files not found',
                'training_time': total_time
            }
            
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    results = quick_train_basketball_yolo()
    
    if results['success']:
        print("\n🚀 SUCCESS: Your custom basketball YOLO model is ready!")
        print("🏀 It can now detect players, ball, referees, baskets, and backboards!")
    else:
        print(f"\n💡 If training failed, you can still use the existing GNN system")
        print(f"   Run: python ../analyze_video.py your_video.mp4")
