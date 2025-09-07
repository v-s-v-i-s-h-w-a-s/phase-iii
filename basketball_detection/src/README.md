# src/ - Core Modules

This folder contains the core logic for detection, classification, and analytics.

- `team_classifier.py`: Basic color-based team classification
- `improved_team_classifier.py`: Advanced team classification (adaptive thresholds, multi-space, basketball rules)
- `inference.py`: Main inference logic, including 3D-2D mapping
- `data_processor.py`: Data loading, preprocessing, and 3D mapping utilities
- `train_model.py`: Model training scripts

**3D-2D Mapping:**
- See `inference.py` and `data_processor.py` for code that projects 2D detections onto a 3D basketball court model using homography/camera calibration.
