# Basketball Play Creator - Setup Guide

## Project Overview

This is an AI-powered basketball play creator that converts natural language descriptions into interactive 2D court animations using Graph Neural Networks (GNNs) and Natural Language Processing (NLP).

## Architecture

```
basketball_play_creator/
├── backend/          # Flask API server (Python)
├── frontend/         # React web interface  
├── gnn_engine/       # PyTorch Geometric GNN simulation
├── nlp_parser/       # spaCy-based NLP processing
├── data/            # Training data and play database
└── models/          # Trained ML models
```

## Installation & Setup

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- Git

### Backend Setup

1. **Navigate to backend directory:**
```bash
cd basketball_play_creator/backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

5. **Set environment variables:**
```bash
# Windows
set FLASK_APP=app.py
set FLASK_ENV=development

# macOS/Linux
export FLASK_APP=app.py  
export FLASK_ENV=development
```

6. **Start Flask server:**
```bash
flask run
```

The backend API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd basketball_play_creator/frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Start React development server:**
```bash  
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

### Basic Play Creation

1. **Open the web interface** at `http://localhost:3000`

2. **Enter a play description** in natural language:
   - "PG passes to SG, center sets screen, cut to basket"
   - "Point guard dribbles left, power forward sets pick, shooting guard cuts to corner"
   - "Run a pick and roll with center, then pass to wing for three-pointer"

3. **Select formation** (5-out, 4-out-1-in, etc.)

4. **Click "Parse Play"** to convert text to structured actions

5. **Click "Simulate"** to run GNN-based simulation

6. **View results** including success probability, tactical analysis, and optimization suggestions

### Advanced Features

#### Natural Language Processing
- Supports basketball terminology and abbreviations
- Handles position names (PG, SG, SF, PF, C, 1-5)
- Recognizes actions (pass, screen, cut, dribble, shoot)
- Identifies locations (paint, corner, wing, top of key)

#### GNN Simulation
- Models player interactions using graph neural networks
- Predicts player movements and role changes
- Calculates score probabilities based on court position
- Provides tactical analysis (spacing, ball movement, tempo)

#### Interactive Court
- Drag and drop player positioning
- Real-time simulation visualization
- Formation templates
- Player role indicators

## API Endpoints

### Parse Play
```
POST /api/parse_play
{
  "description": "PG passes to SG, center sets screen",
  "formation": "5-out"
}
```

### Simulate Play  
```
POST /api/simulate_play
{
  "players": [...],
  "actions": [...],
  "duration": 8.0
}
```

### Optimize Play
```
POST /api/optimize_play
{
  "simulation_result": {...},
  "original_description": "..."
}
```

### Get Plays Database
```
GET /api/plays
```

### Analyze Formation
```
POST /api/analyze_formation
{
  "formation": "5-out",
  "players": [...]
}
```

## Development

### Project Structure

**Backend (Python/Flask)**
- `app.py` - Main Flask application with API endpoints
- `requirements.txt` - Python dependencies

**GNN Engine (PyTorch)**
- `gnn_engine/simulator.py` - Main GNN simulation engine
- `gnn_engine/__init__.py` - Module exports

**NLP Parser (spaCy)**
- `nlp_parser/play_parser.py` - Natural language processing
- `nlp_parser/__init__.py` - Module exports  

**Frontend (React)**
- `frontend/src/App.js` - Main React component
- `frontend/src/index.js` - React application entry
- `frontend/package.json` - Node.js dependencies

### Key Technologies

**Backend:**
- Flask - Web framework
- PyTorch Geometric - Graph neural networks
- spaCy - Natural language processing  
- NumPy/SciPy - Scientific computing
- Socket.IO - Real-time communication

**Frontend:**
- React - UI framework
- Material-UI - Component library
- Konva.js/React-Konva - 2D canvas graphics
- Axios - HTTP client
- Socket.IO-Client - Real-time updates

### Data Flow

1. **Input:** User enters natural language play description
2. **NLP:** spaCy parser converts text to structured actions
3. **GNN:** PyTorch Geometric simulates player interactions  
4. **Visualization:** React/Konva renders 2D court animation
5. **Analysis:** Backend provides tactical insights and optimization

## Training

### GNN Model Training
The GNN model can be trained on basketball play data:

```python
from gnn_engine import PlaySimulator, BasketballGNN
import torch

# Initialize model and simulator
model = BasketballGNN()
simulator = PlaySimulator()

# Train on play data
# (Training data and code would be added here)

# Save trained model
simulator.save_model('models/basketball_gnn.pth')
```

### NLP Model Enhancement
The NLP parser can be enhanced with basketball-specific training:

```python
from nlp_parser import PlayParser
import spacy

# Load parser
parser = PlayParser()

# Add custom basketball patterns
# (Custom training code would be added here)
```

## Troubleshooting

### Common Issues

**Backend not starting:**
- Check Python version (3.8+)
- Verify virtual environment is activated
- Install missing dependencies: `pip install -r requirements.txt`

**spaCy model missing:**
- Download model: `python -m spacy download en_core_web_sm`

**Frontend not starting:**
- Check Node.js version (16+)  
- Install dependencies: `npm install`
- Clear cache: `npm cache clean --force`

**CORS errors:**
- Backend and frontend must run on different ports
- Flask CORS is configured for `localhost:3000`

**Socket.IO connection issues:**
- Ensure both backend and frontend are running
- Check firewall settings
- Verify WebSocket support

### Performance Optimization

**Backend:**
- Use GPU acceleration for PyTorch if available
- Cache parsed plays to avoid re-processing
- Implement request rate limiting

**Frontend:**  
- Optimize Konva.js rendering for large numbers of objects
- Implement lazy loading for play database
- Use React.memo for expensive components

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature-name`  
6. Submit pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check troubleshooting section above
- Search existing GitHub issues
- Create new issue with detailed description and steps to reproduce
