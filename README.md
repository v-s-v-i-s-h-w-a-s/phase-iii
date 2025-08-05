# Basketball Analysis System - Phase III

A comprehensive basketball video analysis system using custom-trained YOLO models, Graph Neural Networks (GNN), and advanced computer vision techniques for real-time game analysis and tactical visualization.

## 🏀 Project Overview

This project provides an end-to-end basketball analysis system that can:
- Detect and track players, ball, referees, and basketball rims using custom YOLO models
- Perform 5v5 team assignment with jersey color analysis
- Generate 2D tactical court visualizations with accurate player positioning
- Create side-by-side analysis videos with original footage and tactical overlays
- Simulate basketball plays using Graph Neural Networks
- Provide comprehensive game statistics and insights

## 🎯 Key Features

### 1. Enhanced Basketball Detection
- **Custom YOLO Training**: Enhanced basketball-specific object detection with 775% improvement over base models
- **Multi-class Detection**: Detects players, basketball, referees, and rims with high accuracy (82.5% mAP50)
- **Team Assignment**: Intelligent jersey color analysis for automatic team separation

### 2. Advanced Tracking Systems
- **5v5 Player Tracking**: Maintains exactly 5 players per team with consistent player IDs
- **Ball Trajectory**: Real-time ball tracking with trajectory visualization
- **Referee Tracking**: Automatic referee detection and positioning

### 3. 2D Court Visualization
- **Realistic Court Design**: Wooden basketball court with authentic proportions and markings
- **Accurate Positioning**: Perspective-corrected player mapping to 2D court coordinates
- **Transparent Overlays**: Semi-transparent tactical overlays on original footage

### 4. Graph Neural Network Analysis
- **Play Pattern Recognition**: GNN-based basketball play analysis and prediction
- **Tactical Insights**: Advanced game situation analysis and coaching recommendations
- **Real-time Decision Making**: Live analysis of game situations and optimal plays

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for YOLO training)
- FFmpeg for video processing

### Required Dependencies
```bash
pip install ultralytics opencv-python numpy scikit-learn torch torchvision torchaudio
pip install matplotlib seaborn pandas pathlib datetime
```

### Model Setup
The system uses custom-trained YOLO models. Ensure you have the trained model weights:
- `enhanced_basketball_training/enhanced_20250803_174000/enhanced_basketball_20250803_174000/weights/best.pt`

## 🚀 Usage

### Quick Analysis
```python
# Run analysis on any basketball video
python analyze_any_basketball_video.py
```

### Custom Video Analysis
```python
from side_by_side_basketball_analyzer import create_side_by_side_analysis

# Analyze custom video
output_video, stats = create_side_by_side_analysis(
    video_path="your_video.mp4",
    max_frames=500
)
```

### Enhanced Tracking (5v5 Fixed)
```python
# Run fixed 5v5 player tracking
python fixed_basketball_team_tracker.py
```

### Realistic Court Overlay
```python
# Create analysis with realistic wooden court overlay
python realistic_court_basketball_tracker.py
```

### GNN Basketball Simulation
```python
# Run Graph Neural Network basketball analysis
python gnn_basketball_simulation.py
```
- `basketball_demo_jersey.py` - Jersey detection demo

## Quick Test Results
```
✅ Enhanced basketball intelligence module imported successfully
✅ Test video found: enhanced_basketball_test_20250803_175335.mp4
✅ Enhanced model found and loaded successfully
📹 Video Properties: 1280x720, 17487 frames, 29.00 FPS, 603.00 seconds
✅ Detection working: Successfully detecting and tracking basketball objects
```

## Usage
1. **Quick test original**: `python quick_test.py`
2. **Quick test improved**: `python quick_improved_test.py` 🆕
3. **Run full improved detection**: `python run_improved_test.py` 🆕
4. **Run enhanced detection**: `python enhanced_basketball_intelligence.py`
5. **Test the model**: `python test_real_model.py`
6. **Train enhanced model**: `python enhanced_basketball_training.py`

## Current Focus
✅ **LABELING ACCURACY IMPROVED!** - The new improved detection system shows much better results
- Better class mapping (ball, rim, player, referee)
- Automatic team color detection and assignment
- Enhanced tracking with ball trail visualization  
- Optimized confidence thresholds for each object type
- Processing 300 frames: 1795 detections (247 balls, 1378 players, 170 referees)
