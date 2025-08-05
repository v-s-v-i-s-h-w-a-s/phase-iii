# 🏀 Basketball GNN Video Analysis Guide

## Quick Start: Analyze Your Basketball Video

### Option 1: Simple Video Analysis (Recommended)
```bash
# Basic analysis with default settings
python analyze_video.py your_basketball_video.mp4

# With custom settings
python analyze_video.py your_video.mp4 --max_frames 500 --confidence 0.7 --epochs 30
```

### Option 2: Step-by-step Analysis
```bash
# Step 1: Extract tracking data from video
python video_processor.py your_video.mp4 --output_dir video_output

# Step 2: Run GNN analysis on extracted data
python main.py --tracking video_output/tracking_data.csv --train --epochs 50
```

## What You'll Get

### Generated Files:
- **Annotated Video**: Shows detected players with bounding boxes
- **Tracking Data**: CSV file with player positions and movements
- **Tactical Visualizations**: Formation analysis plots and graphs
- **Trained GNN Model**: For future analysis of similar videos

### Analysis Results:
- **Player Clustering**: Teams identified without jersey colors
- **Formation Analysis**: Team formations and their stability
- **Tactical Patterns**: Movement patterns and transitions
- **Interactive Graphs**: Player relationship networks

## Parameters You Can Adjust

### `--max_frames` (default: 300)
- Controls how many frames to process
- More frames = better analysis but longer processing time
- Start with 200-500 frames for testing

### `--confidence` (default: 0.6)
- Player detection confidence threshold (0.0-1.0)
- Higher = fewer false detections but might miss some players
- Lower = more detections but more noise
- Try 0.5-0.8 depending on video quality

### `--epochs` (default: 20)
- Number of training epochs for the GNN
- More epochs = better learning but longer training
- 20-50 is usually sufficient for good results

## Video Requirements

### Best Results With:
- Clear basketball court view
- Multiple players visible
- Good lighting and resolution
- Stable camera angle
- Duration: 30 seconds to 5 minutes

### Supported Formats:
- MP4, AVI, MOV, MKV
- Resolution: 720p or higher recommended
- Frame rate: 24-60 FPS

## Example Commands

```bash
# Quick test with short clip
python analyze_video.py game_clip.mp4 --max_frames 100

# High-quality analysis
python analyze_video.py full_game.mp4 --max_frames 1000 --confidence 0.8 --epochs 50

# Fast processing for preview
python analyze_video.py test_video.mp4 --max_frames 50 --epochs 10
```

## Output Explanation

### Tracking Data Columns:
- `frame`: Frame number in video
- `player_id`: Unique player identifier
- `x`, `y`: Player position coordinates
- `vx`, `vy`: Player velocity (calculated)
- `team`: Predicted team assignment (0 or 1)
- `confidence`: Detection confidence score

### Visualization Files:
- `formation_analysis.png`: Team formation patterns over time
- `sequence_visualization.png`: Player positions and movements
- `annotated_video.mp4`: Original video with player detections

## Troubleshooting

### Common Issues:

**"No detections found"**
- Lower confidence threshold: `--confidence 0.4`
- Check video quality and lighting
- Ensure players are clearly visible

**"Processing too slow"**
- Reduce max_frames: `--max_frames 200`
- Use smaller video resolution
- Close other applications

**"Poor team classification"**
- Increase training epochs: `--epochs 40`
- Try different confidence threshold
- Ensure video has clear team separation

### Performance Tips:
- Start with small frame counts for testing
- Use GPU if available (automatic detection)
- Ensure sufficient disk space (~500MB per analysis)
- Close unnecessary applications during processing

## Advanced Usage

### Custom Configuration:
Create a `config.json` file to customize analysis parameters:

```json
{
  "graph_builder": {
    "proximity_threshold": 200.0,
    "min_players": 3,
    "max_players": 12
  },
  "model_type": "gcn",
  "hidden_channels": 64,
  "learning_rate": 0.01
}
```

Then run: `python main.py --video your_video.mp4 --config config.json`

### Using Pre-trained Models:
If you have a trained model, skip training:
```bash
python main.py --tracking your_data.csv  # Uses existing model
```

### Batch Processing:
Process multiple videos with a script:
```bash
for video in *.mp4; do
    python analyze_video.py "$video" --max_frames 300
done
```

## Next Steps

1. **Start Simple**: Try the basic command first
2. **Experiment**: Adjust parameters based on your video
3. **Compare Results**: Try different settings to see what works best
4. **Scale Up**: Once satisfied, process longer clips or full games

Happy analyzing! 🚀
