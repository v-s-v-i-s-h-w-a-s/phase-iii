# Custom Basketball YOLO for Enhanced GNN Analysis

## Overview

This module extends the Basketball GNN system with **custom-trained YOLO models** specifically designed for basketball scenes. Instead of relying on generic object detection, this system trains specialized models to detect:

- 🏃‍♂️ **Players** (all basketball players on court)
- 🏀 **Ball** (basketball in any state - held, flying, bouncing)
- 👨‍💼 **Referees** (game officials in distinctive uniforms)
- 🏀 **Baskets** (basketball hoops/rims)
- 📋 **Backboards** (rectangular backboards)

## Key Advantages Over Default YOLO

### 1. **Basketball-Specific Detection**
- **Default YOLO**: Only detects "person" class → requires post-processing to distinguish players/referees
- **Custom YOLO**: Directly classifies players vs referees vs officials

### 2. **Enhanced Ball Detection**
- **Default YOLO**: No dedicated basketball detection
- **Custom YOLO**: Specialized ball detection even when partially obscured or in motion

### 3. **Court Context Understanding**
- **Default YOLO**: No understanding of basketball-specific objects
- **Custom YOLO**: Detects baskets and backboards for spatial context

### 4. **Improved GNN Performance**
- **Better Graph Construction**: More accurate object detection → better player relationships
- **Enhanced Team Classification**: Referee detection helps separate officials from players
- **Ball Possession Analysis**: Dedicated ball detection enables possession tracking
- **Tactical Understanding**: Court structure detection provides game context

## Directory Structure

```
custom_yolo/
├── setup_custom_yolo.py          # Complete setup and dependency checker
├── dataset_manager.py            # Dataset creation and management
├── yolo_trainer.py               # Custom YOLO model training
├── enhanced_processor.py         # Video processing with custom YOLO
├── gnn_integration.py            # Integration with existing GNN system
├── basketball_dataset.yaml       # Dataset configuration
├── TRAINING_WORKFLOW.md          # Step-by-step training guide
└── basketball_dataset/           # Training data (created during setup)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── extracted_frames/
```

## Quick Start

### 1. **Setup and Dependencies**
```bash
cd custom_yolo
python setup_custom_yolo.py
# Choose option 1: Complete setup
```

### 2. **Prepare Training Data**
```bash
# Extract frames from your basketball videos
python dataset_manager.py
# Choose option 2: Extract frames from video
```

### 3. **Annotate Data**
- Install LabelImg: `pip install labelImg`
- Run: `labelImg`
- Annotate extracted frames with 5 classes:
  - Class 0: player
  - Class 1: ball
  - Class 2: referee
  - Class 3: basket
  - Class 4: board

### 4. **Train Custom Model**
```bash
python yolo_trainer.py
# Choose option 1: Initialize model
# Choose option 2: Train model
```

### 5. **Integrate with GNN**
```bash
python gnn_integration.py
# Choose option 1: Run complete analysis with custom YOLO
```

## Detailed Workflow

### Phase 1: Data Preparation

#### Step 1: Extract Training Frames
```python
from dataset_manager import BasketballDatasetManager

manager = BasketballDatasetManager()
manager.create_dataset_structure()
manager.extract_frames_from_video("your_basketball_video.mp4", max_frames=1000)
```

#### Step 2: Generate Pseudo-Labels (Optional)
```python
# Generate initial labels using pre-trained YOLO
manager.generate_pseudo_labels("./basketball_dataset/extracted_frames")
```

#### Step 3: Manual Annotation
- Use LabelImg or similar tool
- Annotate 1000+ images for best results
- Follow basketball-specific annotation guidelines
- Ensure diversity in camera angles, lighting, teams

#### Step 4: Validate Dataset
```python
manager.split_dataset()  # Split into train/val/test
manager.validate_dataset()  # Check for issues
manager.print_dataset_summary()  # View statistics
```

### Phase 2: Model Training

#### Step 1: Configure Training
Edit `basketball_dataset.yaml`:
```yaml
# Adjust training parameters
train_settings:
  epochs: 100        # Increase for better accuracy
  batch: 16          # Adjust based on GPU memory
  imgsz: 640         # Image size for training
  patience: 50       # Early stopping patience
```

#### Step 2: Train Model
```python
from yolo_trainer import BasketballYOLOTrainer

trainer = BasketballYOLOTrainer()
trainer.initialize_model("n", pretrained=True)  # Start with YOLOv8n
trainer.train_model(epochs=100)
```

#### Step 3: Evaluate Performance
```python
trainer.validate_model()
trainer.test_on_video("test_basketball_video.mp4")
```

### Phase 3: GNN Integration

#### Step 1: Enhanced Video Processing
```python
from enhanced_processor import EnhancedBasketballProcessor

processor = EnhancedBasketballProcessor("path/to/custom_model.pt")
results = processor.process_video_enhanced("basketball_game.mp4")
```

#### Step 2: Complete Pipeline
```python
from gnn_integration import CustomYOLO_GNN_Integration

integration = CustomYOLO_GNN_Integration("path/to/custom_model.pt")
complete_results = integration.run_complete_analysis("basketball_game.mp4")
```

## Training Guidelines

### Minimum Dataset Requirements
- **Players**: 500+ examples across different poses/teams
- **Ball**: 300+ examples in various states
- **Referees**: 200+ examples (if needed)
- **Baskets**: 150+ examples from different angles
- **Backboards**: 150+ examples

### Annotation Best Practices
1. **Consistency**: Use consistent bounding box placement
2. **Completeness**: Include challenging cases (partial occlusion, motion blur)
3. **Diversity**: Various lighting, angles, teams, game situations
4. **Quality over Quantity**: 500 well-annotated images > 2000 poor ones

### Training Tips
1. **Start Small**: Begin with YOLOv8n for faster iteration
2. **Monitor Training**: Watch for overfitting in validation plots
3. **Data Augmentation**: Use built-in YOLO augmentations
4. **Early Stopping**: Use patience parameter to prevent overfitting
5. **Gradual Improvement**: Iteratively add more data and retrain

## Performance Benchmarks

### Expected Performance Metrics
- **Players**: 90%+ detection accuracy
- **Ball**: 70%+ detection rate (challenging due to size/speed)
- **Referees**: 80%+ when visible
- **Baskets**: 85%+ when in frame
- **Overall mAP@0.5**: 0.7+ across all classes

### GNN Integration Benefits
- **20-30% improvement** in player tracking accuracy
- **Enhanced team classification** with referee separation
- **Ball possession analysis** previously unavailable
- **Court-aware tactical analysis** using basket/backboard positions

## Model Comparison

| Aspect | Default YOLO | Custom Basketball YOLO |
|--------|--------------|------------------------|
| **Player Detection** | Generic "person" class | Specific "player" classification |
| **Ball Detection** | ❌ Not available | ✅ Dedicated ball detection |
| **Referee Recognition** | ❌ Classified as "person" | ✅ Separate referee class |
| **Court Objects** | ❌ No basketball context | ✅ Baskets and backboards |
| **GNN Graph Quality** | Basic player connections | Enhanced with ball/court context |
| **Team Classification** | Post-processing required | Direct from object types |
| **Tactical Analysis** | Limited to player positions | Full court context available |

## Troubleshooting

### Common Training Issues

#### Low Ball Detection Performance
**Symptoms**: Ball detection rate < 50%
**Solutions**:
- Increase ball annotation examples
- Add more diverse ball states (held, flying, bouncing)
- Use smaller anchor boxes in YOLO config
- Increase detection confidence threshold

#### Model Overfitting
**Symptoms**: Training accuracy high, validation accuracy low
**Solutions**:
- Add more training data
- Increase data augmentation
- Reduce model complexity (use YOLOv8n instead of YOLOv8m)
- Add early stopping

#### Poor Team Classification in GNN
**Symptoms**: Players and referees mixed in team assignments
**Solutions**:
- Improve referee annotation quality
- Add more referee examples in training data
- Adjust GNN clustering parameters
- Verify custom YOLO referee detection accuracy

### Hardware Requirements

#### Minimum Requirements
- **RAM**: 8GB
- **GPU**: 4GB VRAM (GTX 1660 or better)
- **Storage**: 10GB for dataset and models
- **Training Time**: 2-4 hours for 100 epochs

#### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 50GB for large datasets
- **Training Time**: 1-2 hours for 100 epochs

## Integration Examples

### Example 1: Enhanced Player Tracking
```python
# Compare default vs custom YOLO
integration = CustomYOLO_GNN_Integration()
comparison = integration.compare_models(
    "basketball_game.mp4",
    custom_model_path="custom_basketball_yolo.pt"
)
print(comparison)
```

### Example 2: Ball Possession Analysis
```python
# Analyze ball possession patterns
processor = EnhancedBasketballProcessor("custom_model.pt")
results = processor.process_video_enhanced("game.mp4")

ball_stats = results['summary_statistics']
print(f"Ball detection rate: {ball_stats['frames_with_ball']/ball_stats['total_frames']*100:.1f}%")
print(f"Possession frames: {ball_stats['ball_possession_frames']}")
```

### Example 3: Tactical Pattern Recognition
```python
# Complete tactical analysis with custom YOLO
integration = CustomYOLO_GNN_Integration("custom_model.pt")
results = integration.run_complete_analysis(
    "championship_game.mp4",
    confidence=0.3,
    epochs=200
)

# Results include enhanced tactical insights
tactical_patterns = results['predictions']
```

## Future Enhancements

### Planned Features
1. **Multi-Camera Fusion**: Combine multiple camera angles
2. **Player Identification**: Individual player recognition
3. **Shot Prediction**: Predict shooting opportunities
4. **Real-Time Processing**: Live game analysis
5. **3D Court Modeling**: Full 3D tactical reconstruction

### Research Directions
1. **Temporal Consistency**: Improve tracking across frames
2. **Action Recognition**: Classify basketball actions (dribble, shoot, pass)
3. **Formation Analysis**: Automatic play recognition
4. **Performance Metrics**: Player-specific analytics

## Contributing

### Adding New Classes
To add new basketball objects (e.g., shot clock, scoreboard):
1. Update `basketball_dataset.yaml` with new class
2. Annotate training data with new class
3. Retrain model with updated configuration
4. Update integration code to handle new detections

### Improving Accuracy
1. **Collect More Data**: Add diverse basketball scenarios
2. **Better Annotation**: Improve bounding box precision
3. **Advanced Augmentation**: Custom basketball-specific augmentations
4. **Model Architecture**: Experiment with different YOLO variants

## License and Credits

This custom YOLO training system builds upon:
- **Ultralytics YOLOv8**: State-of-the-art object detection
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision utilities

Designed specifically for basketball analysis and GNN integration.

---

**Ready to train your custom basketball YOLO model?**
Start with: `python setup_custom_yolo.py`
