"""
Quick test script for basketball detection training
"""

from src.train_model import BasketballTrainer
import yaml

# Test with just 5 epochs
trainer = BasketballTrainer(model_size='n')
trainer.load_model()

dataset_path = "./data/basketball_dataset/dataset.yaml"

print("🎯 Quick training test (5 epochs)...")
results, model_path = trainer.train_model(
    dataset_path=dataset_path,
    epochs=5,
    batch_size=4,  # Small batch size
    imgsz=416      # Smaller image size for speed
)

print(f"✅ Test complete! Model saved: {model_path}")
