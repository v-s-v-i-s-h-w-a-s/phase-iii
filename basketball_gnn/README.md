# 🏀 Basketball GNN - Tactical Analysis with Graph Neural Networks

## 📋 Table of Contents
- [Overview](#overview)
- [What Makes This Special](#what-makes-this-special)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage Guide](#detailed-usage-guide)
- [Understanding the Results](#understanding-the-results)
- [Advanced Configuration](#advanced-configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## 📖 Overview

The **Basketball GNN (Graph Neural Network) System** is a revolutionary AI-powered tool that analyzes basketball gameplay using cutting-edge machine learning techniques. Unlike traditional methods that rely on jersey colors or manual annotations, this system uses **Graph Neural Networks** to understand player relationships, team formations, and tactical patterns purely from movement data.

### � What It Does

1. **🔍 Automatic Player Detection**: Uses YOLOv8 to detect players in basketball videos
2. **👥 Intelligent Team Classification**: Groups players into teams without using jersey colors
3. **📊 Tactical Analysis**: Analyzes formations, movement patterns, and strategic plays
4. **🧠 Machine Learning**: Trains GNN models to understand basketball dynamics
5. **📈 Visualization**: Creates comprehensive charts, graphs, and annotated videos

### 🏆 Key Achievements

- **Zero Manual Annotation**: No need to manually label players or teams
- **Color-Independent**: Works regardless of jersey colors or lighting conditions
- **Real-time Processing**: Can analyze live or recorded basketball footage
- **Professional Quality**: Produces publication-ready analysis and visualizations
- **Extensible**: Can be adapted for other team sports

## 🌟 What Makes This Special

### Traditional Approaches vs. Our GNN Solution

| Traditional Method | Basketball GNN |
|---|---|
| ❌ Relies on jersey colors | ✅ Uses player interaction patterns |
| ❌ Fails with similar colors | ✅ Works with any color combination |
| ❌ Manual team labeling | ✅ Automatic team discovery |
| ❌ Basic position tracking | ✅ Advanced tactical analysis |
| ❌ Static analysis | ✅ Dynamic relationship modeling |

### 🧬 The Science Behind It

**Graph Neural Networks (GNNs)** represent players as nodes and their interactions as edges in a dynamic graph. The AI learns:

- **Spatial Relationships**: How players position relative to each other
- **Temporal Patterns**: How formations evolve over time
- **Team Dynamics**: Which players work together vs. against each other
- **Tactical Intelligence**: Strategic patterns and play recognition

## �️ System Architecture

```
📁 basketball_gnn/
├── 🎥 video_processor.py          # Video → Player Tracking Data
├── 📊 main.py                     # Main Pipeline Orchestrator
├── 🔧 analyze_video.py            # Simple Video Analysis Script
├── 📋 requirements.txt            # Dependencies
├── ⚙️ config.json                 # Configuration Settings
├── 
├── 🧠 gnn_model/                  # Neural Network Components
│   ├── model.py                   # GCN & GraphSAGE Architectures
│   ├── train.py                   # Training Pipeline
│   └── predict.py                 # Inference & Analysis
│
├── 📈 graph_builder/              # Graph Construction
│   └── build_graph.py            # Convert Tracking → Graphs
│
├── 🎨 visualization/              # Plotting & Animation
│   └── visualize.py              # Charts, Courts, Animations
│
├── 🛠️ utils/                      # Helper Functions
│   ├── yolo_tracking_parser.py   # Data Processing
│   └── pose_loader.py            # Pose Data Integration
│
├── 📊 results/                    # Analysis Outputs
├── 🎬 video_analysis_*/           # Video Processing Results
└── 🤖 models/                     # Trained AI Models
```

## 🚀 Installation

### Prerequisites

- **Python 3.9+** (Tested on 3.13)
- **Windows/Linux/macOS**
- **4GB+ RAM** recommended
- **GPU optional** (CUDA supported)

### Step 1: Clone or Download

```bash
# If using git
git clone <repository-url>
cd basketball_gnn

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test the system
python test_setup.py
```

Expected output:
```
✅ All imports successful
✅ Quick demo working
✅ System ready for analysis
```

## ⚡ Quick Start

### 🎯 Analyze Your First Video

The easiest way to get started:

```bash
# Basic analysis (recommended for first try)
python analyze_video.py your_basketball_video.mp4

# With custom settings
python analyze_video.py your_video.mp4 --max_frames 500 --confidence 0.7 --epochs 30
```

### 📁 What You'll Get

After running, check these folders:
- **`video_analysis_[name]/`** - Tracking data and annotated video
- **`results/`** - Tactical analysis charts and animations
- **`models/`** - Trained AI model for future use

## 📚 Detailed Usage Guide

### 🎬 Method 1: Simple Video Analysis (Recommended)

**Best for**: First-time users, quick analysis

```bash
# Basic command
python analyze_video.py path/to/your/video.mp4

# Common options
python analyze_video.py video.mp4 --max_frames 300    # Limit processing time
python analyze_video.py video.mp4 --confidence 0.8    # Higher accuracy
python analyze_video.py video.mp4 --epochs 40         # Better AI training
```

**Parameters Explained:**
- `--max_frames`: Number of frames to process (300 = ~10 seconds at 30fps)
- `--confidence`: Player detection threshold (0.5-0.9, higher = fewer false positives)
- `--epochs`: AI training iterations (20-50, more = better learning)

### 🔧 Method 2: Advanced Pipeline

**Best for**: Researchers, custom workflows

```bash
# Step 1: Extract player tracking from video
python video_processor.py your_video.mp4 --output_dir tracking_output

# Step 2: Run GNN analysis on tracking data
python main.py --tracking tracking_output/tracking_data.csv --train --epochs 50

# Step 3: Analyze with existing model
python main.py --tracking tracking_output/tracking_data.csv
```

### 🧪 Method 3: Demo Mode

**Best for**: Testing, learning the system

```bash
# Run with dummy data to see how it works
python main.py --demo --train --epochs 20
```

## 📊 Understanding the Results

### 🎥 Video Outputs

**`annotated_video.mp4`**
- Your original video with AI-detected player bounding boxes
- Different colors represent different detected players
- Confidence scores shown above each detection

### 📈 Analysis Charts

**`formation_analysis.png`**
- **Formation Stability**: How consistent team formations are
- **Transition Patterns**: How formations change over time
- **Team Cohesion**: Measures of team coordination

**`sequence_visualization.png`**
- **Player Trajectories**: Movement paths on basketball court
- **Interaction Networks**: Connections between players
- **Temporal Progression**: How relationships evolve

**`tactical_animation.gif`**
- **Animated Analysis**: Moving visualization of player interactions
- **Formation Evolution**: How teams adapt their positions
- **Key Moments**: Highlighted tactical transitions

### 📊 Data Files

**`tracking_data.csv`** - Raw player data:
```csv
frame,player_id,x,y,vx,vy,team,confidence
0,0,640,360,2.1,-1.5,0,0.85
0,1,200,400,0.5,3.2,1,0.78
...
```

**`metadata.json`** - Processing information:
```json
{
  "fps": 30,
  "resolution": [1280, 720],
  "total_detections": 1523,
  "confidence_threshold": 0.6
}
```

### 🧠 AI Model Metrics

The system reports several key metrics:

**Training Metrics:**
- **Contrastive Loss**: How well the AI distinguishes between teams (lower = better)
- **Formation Coherence**: How well formations are maintained (higher = better)
- **Silhouette Score**: Quality of team clustering (0.5+ is good, 0.7+ is excellent)

**Analysis Results:**
- **Formation Stability**: Average consistency of team formations (-1 to 1, higher = more stable)
- **Transition Rate**: How often formations change (0 to 1, higher = more dynamic)
- **Player Interaction Strength**: Connectivity between teammates vs opponents

## ⚙️ Advanced Configuration

### 🔧 Custom Configuration File

Create `config.json` for advanced settings:

```json
{
  "graph_builder": {
    "proximity_threshold": 150.0,    // Distance for player connections (pixels)
    "min_players": 3,                // Minimum players to build graph
    "max_players": 12               // Maximum players to track
  },
  "model_type": "gcn",              // "gcn" or "graphsage"
  "hidden_channels": 64,            // Neural network size
  "out_channels": 32,              // Output embedding size
  "learning_rate": 0.01,           // Training speed
  "batch_size": 16,                // Training batch size
  "plot_training": true,           // Show training plots
  "create_animation": true         // Generate GIF animations
}
```

Then run with:
```bash
python main.py --video your_video.mp4 --config config.json
```

### 🎯 Optimization for Different Videos

**High-Quality Analysis** (slow but accurate):
```bash
python analyze_video.py video.mp4 --max_frames 1000 --confidence 0.8 --epochs 50
```

**Fast Preview** (quick but basic):
```bash
python analyze_video.py video.mp4 --max_frames 100 --confidence 0.5 --epochs 10
```

**Balanced Approach** (recommended):
```bash
python analyze_video.py video.mp4 --max_frames 300 --confidence 0.6 --epochs 25
```

## 🔌 API Reference

### Core Classes

#### `BasketballVideoProcessor`
```python
from video_processor import BasketballVideoProcessor

processor = BasketballVideoProcessor()
tracking_csv = processor.process_video(
    video_path="game.mp4",
    output_dir="output",
    max_frames=300,
    confidence_threshold=0.6
)
```

#### `BasketballGNNPipeline`
```python
from main import BasketballGNNPipeline

config = {...}  # Your configuration
pipeline = BasketballGNNPipeline(config)

# Run complete analysis
results = pipeline.run_complete_pipeline(
    tracking_path="tracking_data.csv",
    train_new_model=True
)
```

#### `BasketballGNNPredictor`
```python
from gnn_model.predict import BasketballGNNPredictor

predictor = BasketballGNNPredictor("models/trained_model.pth")
embeddings = predictor.predict_embeddings(graphs)
clusters = predictor.cluster_players(graphs, n_clusters=2)
```

### Key Functions

**Data Processing:**
```python
# Load and process tracking data
from utils.yolo_tracking_parser import YOLOTrackingParser

parser = YOLOTrackingParser()
df = parser.parse_csv("tracking.csv")
filtered_df = parser.filter_tracking_data(df, min_track_length=10)
```

**Graph Building:**
```python
# Convert tracking data to graphs
from graph_builder.build_graph import BasketballGraphBuilder

builder = BasketballGraphBuilder(proximity_threshold=150.0)
graphs = builder.build_sequence_graphs(tracking_data)
```

**Visualization:**
```python
# Create analysis visualizations
from visualization.visualize import BasketballVisualizer

viz = BasketballVisualizer()
viz.visualize_sequence(graphs, team_labels)
viz.plot_formation_analysis(formation_data)
viz.create_animation(graphs, team_labels)
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### ❌ "No detections found in video"
**Solution:**
```bash
# Lower confidence threshold
python analyze_video.py video.mp4 --confidence 0.4

# Check video quality - ensure players are clearly visible
# Try different lighting conditions or camera angles
```

#### ❌ "Import errors" or "Module not found"
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Should be 3.9+

# Verify installation
python test_setup.py
```

#### ❌ "Processing too slow"
**Solution:**
```bash
# Reduce frame count
python analyze_video.py video.mp4 --max_frames 200

# Use smaller video resolution
# Close other applications
# Consider using GPU if available
```

#### ❌ "Poor team classification"
**Solution:**
```bash
# Increase training epochs
python analyze_video.py video.mp4 --epochs 40

# Adjust proximity threshold in config.json
# Ensure video has clear team separation
# Try different confidence thresholds
```

#### ❌ "Memory errors"
**Solution:**
```bash
# Reduce batch size in config.json: "batch_size": 8
# Process fewer frames: --max_frames 200
# Close other applications
# Use lower resolution video
```

### Performance Optimization

**For Faster Processing:**
- Use lower resolution videos (720p instead of 1080p)
- Reduce `max_frames` parameter
- Set `batch_size` to 8 or lower
- Disable animation creation: `"create_animation": false`

**For Better Accuracy:**
- Increase `epochs` to 40-50
- Use higher `confidence` threshold (0.7-0.8)
- Increase `proximity_threshold` for larger courts
- Process more frames for longer analysis

## 🔬 Technical Details

### Graph Neural Network Architecture

**Node Features (per player):**
- Position coordinates (x, y)
- Velocity components (vx, vy)
- Team assignment (learned)

**Edge Construction:**
- Proximity-based connections (distance < threshold)
- Temporal connections (same player across frames)
- Dynamic graph topology

**Learning Objective:**
- Contrastive learning for team separation
- Formation coherence regularization
- Unsupervised representation learning

### Model Types

**GCN (Graph Convolutional Network):**
- Standard graph convolution layers
- Good for stable, structured data
- Faster training and inference

**GraphSAGE:**
- Sampling and aggregating approach
- Better for large, dynamic graphs
- More robust to varying graph sizes

### Training Process

1. **Graph Construction**: Convert tracking data to temporal graphs
2. **Contrastive Learning**: Learn to separate teams without labels
3. **Formation Analysis**: Measure spatial coherence within teams
4. **Clustering**: Group players based on learned representations
5. **Validation**: Evaluate using silhouette score and formation metrics

## 🎯 Use Cases

### 🏀 Basketball Analysis
- **Coaching**: Analyze team formations and player movements
- **Scouting**: Evaluate player performance and team dynamics
- **Broadcasting**: Generate insights for commentary and analysis
- **Research**: Study basketball strategy and game evolution

### 🔬 Sports Science
- **Performance Analysis**: Measure player efficiency and coordination
- **Injury Prevention**: Identify risky movement patterns
- **Training Optimization**: Design drills based on game analysis
- **Strategy Development**: Discover new tactical approaches

### 🤖 AI Research
- **Graph Neural Networks**: Benchmark for temporal graph learning
- **Unsupervised Learning**: Team discovery without labels
- **Computer Vision**: Multi-object tracking and classification
- **Sports Analytics**: Domain-specific AI applications

## 📖 Example Workflows

### Workflow 1: Quick Game Analysis
```bash
# 1. Analyze a game highlight
python analyze_video.py game_highlight.mp4 --max_frames 300

# 2. Review results in results/ folder
# 3. Watch annotated_video.mp4 to verify detections
# 4. Analyze formation_analysis.png for tactical insights
```

### Workflow 2: Player Performance Study
```bash
# 1. Process multiple game segments
python analyze_video.py player_segment1.mp4 --epochs 30
python analyze_video.py player_segment2.mp4 --epochs 30

# 2. Compare tracking_data.csv files
# 3. Analyze player movement patterns
# 4. Generate performance reports
```

### Workflow 3: Team Strategy Analysis
```bash
# 1. Analyze full game quarters
python analyze_video.py quarter1.mp4 --max_frames 500 --epochs 40
python analyze_video.py quarter2.mp4 --max_frames 500 --epochs 40

# 2. Compare formation stability across quarters
# 3. Identify tactical adaptations
# 4. Generate strategy recommendations
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repo-url>
cd basketball_gnn

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black .
```

### Adding New Features

1. **New Sports**: Adapt graph builder for other team sports
2. **Advanced Metrics**: Add new tactical analysis functions
3. **Visualization**: Create new chart types and animations
4. **Models**: Implement new GNN architectures
5. **Data Sources**: Support additional tracking data formats

### Code Structure Guidelines

- **Modular Design**: Keep components separate and reusable
- **Documentation**: Document all functions and classes
- **Testing**: Write tests for new functionality
- **Performance**: Optimize for large video processing
- **Configuration**: Make new features configurable

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Geometric**: For excellent GNN implementation
- **Ultralytics**: For YOLOv8 object detection
- **Basketball Community**: For inspiring this tactical analysis tool
- **AI Research Community**: For advancing graph neural networks

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check this README for comprehensive guidance
- **Community**: Join discussions about basketball analytics and AI

---

**🏀 Ready to revolutionize basketball analysis? Start with your first video today!**

```bash
python analyze_video.py your_basketball_video.mp4
```

**Made with ❤️ by the Basketball GNN Team**
- ✅ Predict player movements and tactical shifts
- ✅ Analyze team coordination and formation stability

## 🛠️ Tech Stack

| Component | Tools | Purpose |
|-----------|-------|---------|
| **Core ML** | PyTorch, PyTorch Geometric | GNN implementation |
| **Vision** | OpenCV, YOLOv8, DeepSORT | Player detection & tracking |
| **Pose** | MediaPipe, OpenPose | Movement analysis |
| **Data** | Pandas, NumPy | Data processing |
| **Viz** | Matplotlib, NetworkX, Seaborn | Visualization |

## 📁 Project Structure

```
basketball_gnn/
├── data/
│   ├── raw_videos/           # Input video files
│   ├── extracted_frames/     # Frame extraction
│   ├── frame_graphs/         # Generated graphs (.pt files)
│   ├── player_tracks.csv     # Tracking data
│   └── pose_keypoints.csv    # Pose estimation data
├── gnn_model/
│   ├── model.py             # GNN architectures (GCN, GraphSAGE)
│   ├── train.py             # Training with contrastive loss
│   └── predict.py           # Inference and clustering
├── graph_builder/
│   ├── build_graph.py       # Convert frame data → graphs
│   └── features.py          # Node feature extraction
├── vis/
│   └── visualize_graph.py   # Graph visualization & animation
├── utils/
│   ├── yolo_tracking_parser.py  # Parse YOLO/DeepSORT output
│   └── pose_loader.py       # Load MediaPipe/OpenPose data
├── main.py                  # End-to-end pipeline
└── requirements.txt         # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create the project directory
cd basketball_gnn

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Demo with Dummy Data

```bash
# Test the complete pipeline with synthetic data
python main.py --demo --train --epochs 50
```

This will:
- Generate dummy player tracking data
- Build graphs from the data
- Train a GCN model
- Perform tactical analysis
- Create visualizations

### 3. Use Your Own Data

#### Option A: From CSV Tracking Data
```bash
python main.py --tracking data/player_tracks.csv --train --epochs 100
```

#### Option B: From YOLO Detection Folder
```bash
python main.py --tracking path/to/yolo_detections/ --pose path/to/pose_data/ --train
```

### 4. Expected Data Formats

#### Tracking Data (CSV):
```csv
frame_id,player_id,x,y,confidence
1,1,312,209,0.95
1,2,118,320,0.87
2,1,315,205,0.93
...
```

#### Pose Data (CSV - optional):
```csv
frame_id,player_id,nose_x,nose_y,left_shoulder_x,left_shoulder_y,...
1,1,320,200,310,220,...
```

## 🧠 Model Architecture

### GNN Models Available:
- **GCN (Graph Convolutional Network)**: Basic message passing
- **GraphSAGE**: Scalable for larger graphs
- **Formation Classifier**: Graph-level tactical pattern recognition

### Training Objectives:
1. **Contrastive Loss**: Connected players should have similar embeddings
2. **Formation Coherence**: Spatial proximity correlates with embedding similarity
3. **Unsupervised Learning**: No manual team labels required

## 📊 Features Extracted

### Node Features (per player):
- **Position**: Normalized (x, y) coordinates
- **Velocity**: Movement vector from previous frame
- **Court Zone**: Relative position (left/center/right court)
- **Distance**: To court center, baskets, teammates
- **Pose** (optional): Body orientation, arm positions, stance

### Graph Construction:
- **Nodes**: Players in each frame
- **Edges**: Based on proximity threshold or k-nearest neighbors
- **Temporal**: Sequence of graphs over time

## 🎯 Analysis Outputs

### 1. Player Clustering
- Team assignment based on learned embeddings
- Dynamic role detection (guard, forward, center)

### 2. Formation Analysis
- Formation stability over time
- Cluster compactness metrics
- Team centroid tracking

### 3. Tactical Patterns
- Formation transition detection
- Player role changes
- Team coordination metrics

### 4. Visualizations
- Interactive graph plots
- Player movement animations
- Formation analysis charts
- Embedding space visualization

## 📈 Performance Metrics

- **Silhouette Score**: Clustering quality
- **Formation Stability**: Consistency over time
- **Spatial Agreement**: Correlation with actual team positions
- **Contrastive Loss**: Model training progress

## 🔧 Configuration

Create a `config.json` file:

```json
{
  "graph_builder": {
    "proximity_threshold": 150.0,
    "min_players": 3,
    "max_players": 12
  },
  "model_type": "gcn",
  "hidden_channels": 64,
  "out_channels": 32,
  "num_epochs": 100,
  "learning_rate": 0.01,
  "batch_size": 16,
  "create_animation": true
}
```

Run with custom config:
```bash
python main.py --config config.json --tracking your_data.csv
```

## 🧪 Testing Individual Components

### Test Graph Building:
```bash
cd graph_builder
python build_graph.py
```

### Test Model Training:
```bash
cd gnn_model
python train.py
```

### Test Visualization:
```bash
cd vis
python visualize_graph.py
```

## 📋 Extending the System

### Add New GNN Architecture:
1. Implement in `gnn_model/model.py`
2. Update `create_model()` factory function
3. Test with `python train.py`

### Add New Features:
1. Extend `graph_builder/features.py`
2. Update feature extraction in `build_graph.py`
3. Adjust model input dimensions

### Custom Loss Functions:
1. Add to `gnn_model/train.py`
2. Combine with existing objectives

## 🎬 Use Cases

### Basketball Analytics:
- **Team Performance**: Measure formation effectiveness
- **Player Analysis**: Role consistency, positioning
- **Opponent Scouting**: Tactical pattern recognition
- **Training**: Formation drill effectiveness

### Broader Sports:
- Soccer/Football tactical analysis
- Hockey line formations
- Volleyball rotation patterns

## 🔍 Troubleshooting

### Common Issues:

1. **No graphs generated**: Check tracking data format and minimum player requirements
2. **Training fails**: Ensure sufficient data and adjust batch size
3. **Poor clustering**: Try different proximity thresholds or model parameters
4. **Memory issues**: Reduce batch size or use CPU training

### Debug Mode:
```bash
python main.py --demo --train --epochs 10  # Quick test
```

## 📚 Research References

- **Graph Neural Networks**: [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- **Sports Analytics**: Graph-based team formation analysis
- **Contrastive Learning**: Self-supervised representation learning
- **Basketball Analytics**: Player tracking and tactical analysis

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-analysis`
3. Commit changes: `git commit -am 'Add new tactical pattern detection'`
4. Push to branch: `git push origin feature/new-analysis`
5. Submit pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🚀 Future Enhancements

- **Temporal GNNs**: Model time dependencies directly
- **Attention Mechanisms**: Focus on key player interactions
- **Real-time Analysis**: Live game tactical insights
- **Multi-sport Support**: Extend to other team sports
- **3D Court Analysis**: Incorporate player height and jumping

---

**Ready to revolutionize basketball analytics with GNNs? Start with the demo and explore the tactical insights! 🏀⚡**
