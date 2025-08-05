# Custom Basketball YOLO Training Workflow

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
