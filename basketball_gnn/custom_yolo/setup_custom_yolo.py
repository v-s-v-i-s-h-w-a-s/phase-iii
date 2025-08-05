#!/usr/bin/env python3
"""
Custom Basketball YOLO Setup and Training Guide
Complete setup for training custom YOLO models for basketball
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
import json
from datetime import datetime

class BasketballYOLOSetup:
    """Setup manager for custom basketball YOLO training."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.custom_yolo_dir = Path(__file__).parent
        self.dataset_dir = self.custom_yolo_dir / "basketball_dataset"
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            ('ultralytics', 'ultralytics'),
            ('torch', 'torch'), 
            ('torchvision', 'torchvision'),
            ('opencv-python', 'cv2'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('matplotlib', 'matplotlib'),
            ('scikit-learn', 'sklearn'),
            ('tqdm', 'tqdm'),
            ('pillow', 'PIL')
        ]
        
        missing_packages = []
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"   ✅ {package_name}")
            except ImportError:
                print(f"   ❌ {package_name} - Missing")
                missing_packages.append(package_name)
                
        if missing_packages:
            print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
            print("Install with:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
        else:
            print("✅ All dependencies satisfied!")
            return True
            
    def create_training_workflow(self):
        """Create a complete training workflow guide."""
        workflow_file = self.custom_yolo_dir / "TRAINING_WORKFLOW.md"
        
        workflow_content = """# Custom Basketball YOLO Training Workflow

## Overview
This guide walks you through training a custom YOLO model specifically for basketball objects: players, ball, referees, baskets, and backboards.

## Prerequisites
- ✅ Python 3.8+ installed
- ✅ All dependencies installed (run `python setup_custom_yolo.py`)
- ✅ Basketball video footage available
- ✅ Annotation tools ready (LabelImg recommended)

## Step-by-Step Training Process

### Phase 1: Dataset Preparation

#### 1.1 Extract Frames from Videos
```bash
python dataset_manager.py
# Choose option 2: Extract frames from video
# Enter your basketball video path
# This will extract frames at regular intervals
```

#### 1.2 Generate Pseudo-Labels (Optional)
```bash
python dataset_manager.py
# Choose option 3: Generate pseudo-labels
# This creates initial labels using pre-trained YOLO
# Only detects players initially - you'll need to add ball, referees, etc.
```

#### 1.3 Manual Annotation (Required)
**This is the most important step for quality results!**

Recommended tools:
- **LabelImg**: https://github.com/tzutalin/labelImg
- **Roboflow**: https://roboflow.com/ (online)
- **CVAT**: https://github.com/openvinotoolkit/cvat (advanced)

**Annotation Guidelines:**
- **Players (Class 0)**: All basketball players on court
- **Ball (Class 1)**: The basketball in any state
- **Referees (Class 2)**: Game officials in distinctive uniforms  
- **Baskets (Class 3)**: Basketball hoops/rims
- **Boards (Class 4)**: Backboards

**Quality Tips:**
- Aim for 1000+ annotated images minimum
- Include diverse scenarios: different angles, lighting, player positions
- Be consistent with bounding box placement
- Don't skip difficult cases - they improve model robustness

#### 1.4 Split Dataset
```bash
python dataset_manager.py
# Choose option 4: Split dataset
# Automatically splits into train/val/test sets
```

#### 1.5 Validate Dataset
```bash
python dataset_manager.py
# Choose option 5: Validate dataset
# Checks for missing files, format issues
```

### Phase 2: Model Training

#### 2.1 Configure Training Parameters
Edit `basketball_dataset.yaml`:
- Adjust epochs (100-300 recommended)
- Set batch size based on GPU memory
- Configure augmentation settings

#### 2.2 Train the Model
```bash
python yolo_trainer.py
# Choose option 1: Initialize model
# Choose option 2: Train model
```

**Training Tips:**
- Start with YOLOv8n for faster training
- Monitor training plots for overfitting
- Use early stopping if validation loss plateaus
- Save checkpoints regularly

#### 2.3 Validate Performance
```bash
python yolo_trainer.py
# Choose option 3: Validate model
```

#### 2.4 Test on New Videos
```bash
python yolo_trainer.py
# Choose option 4: Test on video
```

### Phase 3: GNN Integration

#### 3.1 Test Custom Model with GNN
```bash
python gnn_integration.py
# Choose option 1: Run complete analysis with custom YOLO
# Enter your trained model path
```

#### 3.2 Compare with Default YOLO
```bash
python gnn_integration.py
# Choose option 3: Compare custom vs default YOLO
```

## Expected Results

### Good Custom YOLO Model Should:
- **Ball Detection**: 70%+ detection rate in basketball videos
- **Player Detection**: 90%+ detection rate with team differentiation
- **Referee Detection**: 80%+ when referees are visible
- **Basket Detection**: 85%+ for visible baskets
- **Low False Positives**: < 5% false detection rate

### Integration Benefits:
- **Better GNN Training**: More accurate object detection improves graph construction
- **Enhanced Team Classification**: Custom model can distinguish referees from players
- **Ball Tracking**: Dedicated ball detection enables possession analysis
- **Court Understanding**: Basket/backboard detection provides spatial context

## Troubleshooting

### Common Issues:

#### Low Detection Performance
- **Solution**: Add more diverse training data
- **Check**: Annotation quality and consistency
- **Try**: Increase training epochs or use larger model (YOLOv8s/m)

#### Model Overfitting
- **Solution**: Add data augmentation
- **Check**: Validation loss curve
- **Try**: Reduce model complexity or add regularization

#### Poor Ball Detection
- **Solution**: Focus on ball annotation quality
- **Check**: Ball visibility in training data
- **Try**: Increase ball detection weight in loss function

#### Memory Issues
- **Solution**: Reduce batch size
- **Check**: GPU memory usage
- **Try**: Use smaller model size or image resolution

### Performance Optimization:

#### For Better Accuracy:
1. Increase dataset size (2000+ images ideal)
2. Add hard negative examples
3. Use test-time augmentation
4. Ensemble multiple models

#### For Faster Training:
1. Use mixed precision training
2. Optimize data loading
3. Use distributed training if multiple GPUs
4. Pre-compute dataset statistics

## Dataset Requirements

### Minimum Dataset Sizes:
- **Players**: 500+ examples across different poses/teams
- **Ball**: 300+ examples in various states (held, flying, bouncing)
- **Referees**: 200+ examples (if referee detection needed)
- **Baskets**: 150+ examples from different angles
- **Boards**: 150+ examples with various backgrounds

### Recommended Diversity:
- **Camera Angles**: Courtside, elevated, broadcast angles
- **Lighting**: Indoor/outdoor, different times of day
- **Teams**: Multiple team colors and uniforms
- **Game Situations**: Defense, offense, transitions, free throws
- **Player Positions**: Guards, forwards, centers in various poses

## Next Steps After Training

1. **Export Model**: Convert to different formats (ONNX, TensorRT)
2. **Optimize for Deployment**: Quantization, pruning
3. **Integration Testing**: Test with full GNN pipeline
4. **Performance Benchmarking**: Compare against baseline
5. **Continuous Improvement**: Collect more data and retrain

## Success Metrics

### Technical Metrics:
- **mAP@0.5**: > 0.7 for all classes
- **Precision**: > 0.8 for players and ball
- **Recall**: > 0.8 for players and ball
- **Inference Speed**: < 50ms per frame on GPU

### Basketball-Specific Metrics:
- **Team Classification Accuracy**: > 85%
- **Ball Possession Detection**: > 80%
- **Basket Recognition**: > 90% when visible
- **False Positive Rate**: < 5%

Remember: Quality annotation is more important than quantity. 500 well-annotated images beat 2000 poorly annotated ones!
"""
        
        with open(workflow_file, 'w', encoding='utf-8') as f:
            f.write(workflow_content)
            
        print(f"📖 Training workflow created: {workflow_file}")
        
    def setup_annotation_tools(self):
        """Provide instructions for setting up annotation tools."""
        print("\n🏷️ Annotation Tools Setup")
        print("=" * 30)
        
        tools_info = {
            "LabelImg": {
                "description": "Simple, desktop annotation tool",
                "install": "pip install labelImg",
                "run": "labelImg",
                "best_for": "Quick annotation, beginners"
            },
            "Roboflow": {
                "description": "Online annotation platform",
                "install": "Create account at roboflow.com",
                "run": "Web browser",
                "best_for": "Team collaboration, advanced features"
            },
            "CVAT": {
                "description": "Advanced annotation platform",
                "install": "Docker: docker run -it --rm -p 8080:8080 openvino/cvat",
                "run": "http://localhost:8080",
                "best_for": "Large datasets, video annotation"
            }
        }
        
        for tool, info in tools_info.items():
            print(f"\n{tool}:")
            print(f"  Description: {info['description']}")
            print(f"  Install: {info['install']}")
            print(f"  Run: {info['run']}")
            print(f"  Best for: {info['best_for']}")
            
    def create_quick_start_script(self):
        """Create a quick start script for immediate training."""
        quick_start_file = self.custom_yolo_dir / "quick_start.py"
        
        quick_start_content = '''#!/usr/bin/env python3
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
    print("\\n1️⃣ Setting up dataset structure...")
    manager = BasketballDatasetManager()
    manager.create_dataset_structure()
    
    # Step 2: Check for video
    video_path = input("Enter path to basketball video (or press Enter to skip): ").strip()
    
    if video_path and Path(video_path).exists():
        print("\\n2️⃣ Extracting frames...")
        manager.extract_frames_from_video(video_path, max_frames=200)
        
        print("\\n3️⃣ Generating pseudo-labels...")
        manager.generate_pseudo_labels("./basketball_dataset/extracted_frames")
        
        print("\\n4️⃣ Splitting dataset...")
        manager.split_dataset()
        
        print("\\n5️⃣ Validating dataset...")
        if manager.validate_dataset():
            print("\\n6️⃣ Starting training...")
            trainer = BasketballYOLOTrainer()
            trainer.initialize_model("n", pretrained=True)
            trainer.train_model(epochs=50)  # Quick training
            
            print("\\n✅ Quick training complete!")
            print("   For better results:")
            print("   1. Manually annotate more images")
            print("   2. Increase epochs to 100+")
            print("   3. Add more diverse data")
        else:
            print("\\n❌ Dataset validation failed!")
            print("   Please check the dataset and fix issues")
    else:
        print("\\n⚠️ No video provided. Setting up structure only.")
        print("   Manual steps required:")
        print("   1. Add images to basketball_dataset/extracted_frames/")
        print("   2. Annotate images manually")
        print("   3. Run dataset split and validation")
        print("   4. Start training")
    
    # Generate instructions
    manager.create_annotation_template()
    
    print("\\n📖 Next steps:")
    print("   - Check TRAINING_WORKFLOW.md for detailed guide")
    print("   - Use annotation_instructions.md for labeling guide")
    print("   - Run python yolo_trainer.py for advanced training options")

if __name__ == "__main__":
    quick_start_training()
'''
        
        with open(quick_start_file, 'w', encoding='utf-8') as f:
            f.write(quick_start_content)
            
        print(f"🚀 Quick start script created: {quick_start_file}")
        
    def create_sample_config(self):
        """Create sample configuration files."""
        print("\n⚙️ Creating sample configurations...")
        
        # Advanced training config
        advanced_config = {
            "model_configurations": {
                "nano": {"size": "n", "speed": "fastest", "accuracy": "lowest", "use_case": "real-time"},
                "small": {"size": "s", "speed": "fast", "accuracy": "good", "use_case": "balanced"},
                "medium": {"size": "m", "speed": "medium", "accuracy": "better", "use_case": "high_accuracy"},
                "large": {"size": "l", "speed": "slow", "accuracy": "high", "use_case": "best_accuracy"},
                "extra_large": {"size": "x", "speed": "slowest", "accuracy": "highest", "use_case": "research"}
            },
            "training_presets": {
                "quick_test": {"epochs": 25, "batch": 16, "patience": 10},
                "standard": {"epochs": 100, "batch": 16, "patience": 20},
                "high_quality": {"epochs": 200, "batch": 8, "patience": 50},
                "production": {"epochs": 300, "batch": 4, "patience": 100}
            },
            "hardware_recommendations": {
                "cpu_only": {"batch_size": 2, "workers": 2, "model_size": "n"},
                "gpu_4gb": {"batch_size": 8, "workers": 4, "model_size": "s"},
                "gpu_8gb": {"batch_size": 16, "workers": 8, "model_size": "m"},
                "gpu_16gb": {"batch_size": 32, "workers": 8, "model_size": "l"}
            }
        }
        
        config_file = self.custom_yolo_dir / "advanced_training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(advanced_config, f, indent=2)
            
        print(f"   ⚙️ Advanced config: {config_file}")
        
    def run_setup(self):
        """Run complete setup process."""
        print("🏀 Basketball Custom YOLO Setup")
        print("=" * 40)
        print(f"Project root: {self.project_root}")
        print(f"Custom YOLO dir: {self.custom_yolo_dir}")
        
        # Check dependencies
        if not self.check_dependencies():
            print("\\n❌ Please install missing dependencies first!")
            return False
            
        # Create workflow guide
        self.create_training_workflow()
        
        # Setup annotation tools info
        self.setup_annotation_tools()
        
        # Create quick start
        self.create_quick_start_script()
        
        # Create sample configs
        self.create_sample_config()
        
        print("\\n✅ Setup complete!")
        print("\\n🎯 Next Steps:")
        print("1. Read TRAINING_WORKFLOW.md for detailed instructions")
        print("2. Run python quick_start.py for immediate training")
        print("3. Or run python dataset_manager.py to start step by step")
        
        return True


def main():
    """Main setup function."""
    setup = BasketballYOLOSetup()
    
    print("Choose setup option:")
    print("1. Complete setup (recommended)")
    print("2. Check dependencies only")
    print("3. Create workflow guide only")
    print("4. Quick start training")
    
    choice = input("\\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        setup.run_setup()
    elif choice == "2":
        setup.check_dependencies()
    elif choice == "3":
        setup.create_training_workflow()
    elif choice == "4":
        # Run quick start
        quick_start_file = setup.custom_yolo_dir / "quick_start.py"
        if quick_start_file.exists():
            exec(open(quick_start_file).read())
        else:
            setup.create_quick_start_script()
            print("Quick start script created. Run it with: python quick_start.py")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
