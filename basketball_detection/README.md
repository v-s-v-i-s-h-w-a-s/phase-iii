# Basketball Detection System

A comprehensive YOLOv11-based basketball detection and tracking system for detecting players, referees, basketball, and hoop in video footage.

## Features

- **Multi-Object Detection**: Detects players, referees, basketball, and hoop
- **Real-time Processing**: Optimized for real-time video analysis
- **GPU Acceleration**: CUDA support for faster training and inference
- **Dataset Processing**: Converts multiple dataset formats to unified YOLO format
- **Video Analytics**: Generates detection statistics and tracking data

## System Requirements

- NVIDIA GPU with CUDA 12.6+ support
- Python 3.11+
- 8GB+ GPU memory (RTX 4060 or better)
- 16GB+ system RAM

## Installation

1. **Create GPU Environment**:
```bash
conda create -n basketball-gpu python=3.11
conda activate basketball-gpu
```

2. **Install PyTorch with CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. **Install Dependencies**:
```bash
pip install ultralytics opencv-python pandas matplotlib seaborn pyyaml tqdm
```

4. **Verify GPU Setup**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## Project Structure

```
basketball_detection/
├── main.py                 # Main entry point
├── src/
│   ├── data_processor.py   # Dataset processing
│   ├── train_model.py      # Model training
│   └── inference.py        # Video inference
├── models/                 # Trained models
├── data/                   # Processed datasets
└── outputs/                # Detection results
```

## Usage

### Quick Start

Run the main script for interactive menu:

```bash
python main.py
```

### Dataset Processing

The system processes datasets from the `../dataset/` folder:
- Folders 1, 2, 3: YOLO format datasets
- Folder 4: XML annotation format

```python
from src.data_processor import DataProcessor

processor = DataProcessor()
dataset_path, classes = processor.create_dataset()
```

### Model Training

Train YOLOv11 on basketball data:

```python
from src.train_model import BasketballTrainer

trainer = BasketballTrainer(model_size='n')  # n, s, m, l, x
trainer.load_model()
results, model_path = trainer.train_model(
    dataset_path="./data/basketball_dataset/dataset.yaml",
    epochs=50,
    batch_size=8
)
```

### Video Inference

Process videos with trained model:

```python
from src.inference import BasketballInference

inference = BasketballInference("./models/basketball_yolo11n.pt")
results = inference.process_video("video.mp4")
```

## Detection Classes

The system detects 4 main classes:
- **Player**: Basketball players
- **Referee**: Game referees  
- **Ball**: Basketball
- **Hoop**: Basketball hoop/rim

## Performance

- **Training Speed**: ~10-15 FPS on RTX 4060
- **Inference Speed**: ~30-45 FPS on RTX 4060
- **Model Accuracy**: 85%+ mAP@0.5 on test set

## Output

### Video Output
- Annotated video with bounding boxes
- Real-time detection confidence scores
- Frame-by-frame object tracking

### Analytics Output
- CSV file with all detections
- Detection statistics by class
- Confidence score analysis
- Temporal tracking data

## Training Parameters

Recommended settings for RTX 4060:

```python
# Nano model (fastest)
model_size = 'n'
batch_size = 8
epochs = 50
imgsz = 640

# Small model (balanced)
model_size = 's'
batch_size = 6
epochs = 75
imgsz = 640

# Medium model (best accuracy)
model_size = 'm'
batch_size = 4
epochs = 100
imgsz = 640
```

## Troubleshooting

### GPU Memory Issues
- Reduce batch_size to 4 or 2
- Use 'n' model size instead of 's' or 'm'
- Reduce image size to 416

### Dataset Issues
- Ensure dataset folders exist in `../dataset/`
- Check YAML files have correct format
- Verify image and label file pairs match

### Model Loading Issues
- Train model first using option 2 in main menu
- Check model path exists: `./models/basketball_yolo11n.pt`
- Verify CUDA setup if using GPU

## File Formats

### Supported Video Formats
- MP4, AVI, MOV, MKV
- 720p, 1080p, 4K resolution
- 24-60 FPS

### Dataset Formats
- YOLO: .txt annotation files
- XML: CVAT annotation format
- Images: JPG, PNG, BMP

## Results Analysis

The system generates comprehensive analytics:

```python
# Detection summary
Total detections: 1,234
Class distribution:
  player: 856 (69.4%)
  ball: 234 (19.0%)
  referee: 89 (7.2%)
  hoop: 55 (4.4%)

Average confidence: 0.847
Processing speed: 34.2 FPS
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Test with basketball videos
4. Submit pull request

## License

MIT License - See LICENSE file for details.

## Contact

For questions or support, please create an issue in the repository.
