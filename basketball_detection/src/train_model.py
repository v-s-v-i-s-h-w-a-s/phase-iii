"""
Basketball Detection Training Script
Trains YOLOv11 model on basketball detection dataset
"""

import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class BasketballTrainer:
    def __init__(self, model_size='n'):
        """
        Initialize trainer
        Args:
            model_size: YOLOv11 model size ('n', 's', 'm', 'l', 'x')
        """
        self.model_size = model_size
        self.model_name = f'yolo11{model_size}.pt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        
        # Create directories
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"🏀 Basketball Detection Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {self.model_name}")
        
    def load_model(self):
        """Load YOLOv11 model"""
        print(f"📥 Loading {self.model_name}...")
        self.model = YOLO(self.model_name)
        print("✅ Model loaded successfully!")
        
    def train_model(self, dataset_path, epochs=100, imgsz=640, batch_size=16):
        """
        Train the model on basketball dataset
        """
        if not self.model:
            self.load_model()
            
        print(f"🎯 Starting training...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch_size}")
        
        # Create unique run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = "basketball_detection"
        run_name = f"yolo11{self.model_size}_{timestamp}"
        
        # Start training
        results = self.model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=str(self.models_dir / project_name),
            name=run_name,
            device=self.device,
            patience=10,
            save=True,
            plots=True,
            val=True,
            cache=True,
            workers=4,
            verbose=True
        )
        
        # Save best model
        best_model_path = self.models_dir / project_name / run_name / "weights" / "best.pt"
        final_model_path = self.models_dir / f"basketball_yolo11{self.model_size}.pt"
        
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ Best model saved to: {final_model_path}")
        
        return results, str(final_model_path)
    
    def evaluate_model(self, dataset_path):
        """Evaluate trained model"""
        if not self.model:
            print("❌ No model loaded!")
            return
            
        print("📊 Evaluating model...")
        results = self.model.val(data=dataset_path)
        
        print("✅ Evaluation complete!")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        
        return results
    
    def test_inference(self, test_image_path=None):
        """Test model inference"""
        if not self.model:
            print("❌ No model loaded!")
            return
            
        # Use default test image if none provided
        if not test_image_path:
            # Look for a test image in the dataset
            test_dirs = [
                "./data/basketball_dataset/test/images",
                "./data/basketball_dataset/val/images",
                "./data/basketball_dataset/train/images"
            ]
            
            for test_dir in test_dirs:
                test_path = Path(test_dir)
                if test_path.exists():
                    images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
                    if images:
                        test_image_path = str(images[0])
                        break
        
        if not test_image_path:
            print("❌ No test image found!")
            return
            
        print(f"🔍 Testing inference on: {test_image_path}")
        results = self.model.predict(test_image_path, save=True, conf=0.5)
        
        # Print detection results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"   Detected {len(boxes)} objects")
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    print(f"     {class_name}: {confidence:.3f}")
            else:
                print("   No objects detected")
        
        return results

def main():
    """Main training function"""
    print("🏀 Basketball Detection Training System")
    print("=" * 50)
    
    # Initialize trainer
    trainer = BasketballTrainer(model_size='n')  # Start with nano model
    
    # Check for dataset
    dataset_path = "./data/basketball_dataset/dataset.yaml"
    if not Path(dataset_path).exists():
        print("❌ Dataset not found! Please run data_processor.py first.")
        return
    
    # Load dataset info
    with open(dataset_path, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    print(f"📋 Dataset info:")
    print(f"   Classes: {dataset_info['names']}")
    print(f"   Number of classes: {dataset_info['nc']}")
    
    # Load model
    trainer.load_model()
    
    # Train model
    print("\n🎯 Starting training...")
    results, model_path = trainer.train_model(
        dataset_path=dataset_path,
        epochs=50,  # Start with fewer epochs for testing
        imgsz=640,
        batch_size=8   # Smaller batch for GPU memory
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate_model(dataset_path)
    
    # Test inference
    print("\n🔍 Testing inference...")
    trainer.test_inference()
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {model_path}")

if __name__ == "__main__":
    main()
