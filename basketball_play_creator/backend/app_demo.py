"""
Basketball Play Creator - Backend API (Demo Version)
Flask server providing REST API endpoints for basketball play creation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import time
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'basketball_play_creator_secret'
CORS(app, origins=["http://localhost:3000"])
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

# Demo data and mock functions
FORMATIONS = {
    '5-out': {
        'description': 'Five players around the perimeter',
        'positions': {
            'PG': [470, 400],
            'SG': [600, 350], 
            'SF': [340, 350],
            'PF': [600, 250],
            'C': [340, 250]
        }
    },
    '4-out-1-in': {
        'description': 'Four perimeter players, one in post',
        'positions': {
            'PG': [470, 400],
            'SG': [600, 350],
            'SF': [340, 350], 
            'PF': [570, 300],
            'C': [470, 200]
        }
    }
}

def mock_parse_play(description: str, formation: str) -> Dict[str, Any]:
    """Mock NLP parsing function"""
    logger.info(f"Parsing play: '{description}' with formation: {formation}")
    
    # Simple keyword detection
    actions = []
    confidence = 0.8
    
    words = description.lower().split()
    timestamp = 0.0
    
    if 'pass' in description.lower():
        actions.append({
            'type': 'pass',
            'player': 'player_1',
            'target_player': 'player_2', 
            'timestamp': timestamp,
            'duration': 0.5,
            'description': 'Pass to teammate',
            'confidence': 0.9
        })
        timestamp += 1.0
    
    if any(word in description.lower() for word in ['screen', 'pick']):
        actions.append({
            'type': 'screen',
            'player': 'player_5',
            'target_player': 'player_1',
            'timestamp': timestamp,
            'duration': 3.0,
            'description': 'Set screen',
            'confidence': 0.85
        })
        timestamp += 1.5
    
    if 'cut' in description.lower():
        actions.append({
            'type': 'cut',
            'player': 'player_3',
            'target_player': None,
            'timestamp': timestamp,
            'duration': 1.5,
            'description': 'Cut to basket',
            'confidence': 0.8
        })
        timestamp += 1.0
    
    if any(word in description.lower() for word in ['shoot', 'shot', 'score']):
        actions.append({
            'type': 'shot',
            'player': 'player_2',
            'target_player': None,
            'timestamp': timestamp,
            'duration': 1.0,
            'description': 'Take shot',
            'confidence': 0.75
        })
    
    # Create players based on formation
    formation_data = FORMATIONS.get(formation, FORMATIONS['5-out'])
    players = []
    
    for i, (pos, location) in enumerate(formation_data['positions'].items()):
        players.append({
            'id': f'player_{i+1}',
            'position': pos,
            'team': 'offense',
            'location': location,
            'has_ball': pos == 'PG'
        })
    
    # Generate play name
    play_name = "Custom Play"
    if 'screen' in description.lower() and 'pass' in description.lower():
        play_name = "Pick and Pass"
    elif 'screen' in description.lower() and 'cut' in description.lower():
        play_name = "Screen and Cut"
    elif 'pick' in description.lower() and 'roll' in description.lower():
        play_name = "Pick and Roll"
    
    return {
        'name': play_name,
        'description': description,
        'actions': actions,
        'players': players,
        'formation': formation,
        'duration': max([a['timestamp'] + a['duration'] for a in actions]) if actions else 5.0,
        'confidence': confidence,
        'alternative_interpretations': [
            "Could be interpreted as a motion offense variation",
            "Alternative spacing could improve effectiveness"
        ]
    }

def mock_simulate_play(players: List[Dict], actions: List[Dict], duration: float) -> Dict[str, Any]:
    """Mock GNN simulation function"""
    logger.info(f"Simulating play with {len(players)} players for {duration}s")
    
    # Mock simulation results
    states = []
    for t in range(int(duration * 2)):  # 0.5s intervals
        state = {
            'timestamp': t * 0.5,
            'players': players.copy(),  # In real version, positions would change
            'ball_position': players[0]['location'] if players else [470, 300],
            'score_probability': 0.6 + (t * 0.02),  # Gradually increasing
            'play_success': True
        }
        states.append(state)
    
    # Calculate final metrics
    final_score_prob = min(0.75, 0.6 + len(actions) * 0.05)
    success_prob = min(0.85, 0.7 + len(actions) * 0.03)
    
    # Generate key interactions
    key_interactions = []
    if any(a['type'] == 'screen' for a in actions):
        key_interactions.append("Screen set at 1.5s")
    if any(a['type'] == 'pass' for a in actions):
        key_interactions.append("Good ball movement")
    if any(a['type'] == 'cut' for a in actions):
        key_interactions.append("Effective cutting action")
    
    # Tactical analysis
    tactical_analysis = {
        'ball_movement': 0.8 if len(actions) > 2 else 0.6,
        'spacing': 0.75,
        'tempo': 0.7
    }
    
    # Optimization suggestions
    optimization_suggestions = []
    if final_score_prob < 0.6:
        optimization_suggestions.append("Consider better spacing to create open shots")
    if len(actions) < 3:
        optimization_suggestions.append("Add more ball movement for better opportunities")
    if not any(a['type'] == 'screen' for a in actions):
        optimization_suggestions.append("Include screening action to create advantages")
    
    return {
        'states': states,
        'final_score_probability': final_score_prob,
        'success_probability': success_prob,
        'key_interactions': key_interactions,
        'tactical_analysis': tactical_analysis,
        'optimization_suggestions': optimization_suggestions
    }

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Basketball Play Creator API',
        'version': '1.0.0',
        'timestamp': time.time()
    })

@app.route('/api/parse_play', methods=['POST'])
def parse_play():
    """Parse natural language play description"""
    try:
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing play description'}), 400
        
        description = data['description']
        formation = data.get('formation', '5-out')
        
        # Parse the play
        parsed_play = mock_parse_play(description, formation)
        
        return jsonify({
            'success': True,
            'parsed_play': parsed_play
        })
        
    except Exception as e:
        logger.error(f"Error parsing play: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate_play', methods=['POST'])
def simulate_play():
    """Simulate basketball play using GNN"""
    try:
        data = request.get_json()
        
        if not data or 'players' not in data or 'actions' not in data:
            return jsonify({'error': 'Missing players or actions data'}), 400
        
        players = data['players']
        actions = data['actions'] 
        duration = data.get('duration', 8.0)
        
        # Simulate the play
        simulation_result = mock_simulate_play(players, actions, duration)
        
        return jsonify({
            'success': True,
            'simulation_result': simulation_result
        })
        
    except Exception as e:
        logger.error(f"Error simulating play: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize_play', methods=['POST'])
def optimize_play():
    """Optimize play based on simulation results"""
    try:
        data = request.get_json()
        
        if not data or 'simulation_result' not in data:
            return jsonify({'error': 'Missing simulation result'}), 400
        
        simulation_result = data['simulation_result']
        original_description = data.get('original_description', '')
        
        # Mock optimization
        optimized_description = original_description + " with better spacing"
        optimized_play = mock_parse_play(optimized_description, '5-out')
        
        # Improve metrics slightly
        optimized_play['confidence'] = min(0.95, optimized_play['confidence'] + 0.1)
        
        return jsonify({
            'success': True,
            'optimized_play': optimized_play,
            'optimized_description': optimized_description,
            'improvements': [
                'Increased spacing effectiveness by 15%',
                'Improved ball movement flow',
                'Enhanced scoring probability'
            ]
        })
        
    except Exception as e:
        logger.error(f"Error optimizing play: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plays', methods=['GET'])
def get_plays():
    """Get standard basketball plays database"""
    sample_plays = [
        {
            'id': 1,
            'name': 'Pick and Roll',
            'description': 'PG dribbles up, center sets screen, roll to basket',
            'formation': '5-out',
            'category': 'basic',
            'success_rate': 0.72
        },
        {
            'id': 2, 
            'name': 'Motion Offense',
            'description': 'Continuous player and ball movement to create open shots',
            'formation': '4-out-1-in',
            'category': 'advanced',
            'success_rate': 0.68
        },
        {
            'id': 3,
            'name': 'Iso Play',
            'description': 'Clear out for one-on-one opportunity',
            'formation': '5-out',
            'category': 'isolation',
            'success_rate': 0.58
        }
    ]
    
    return jsonify({
        'success': True,
        'plays': sample_plays,
        'total': len(sample_plays)
    })

@app.route('/api/analyze_formation', methods=['POST'])
def analyze_formation():
    """Analyze formation effectiveness"""
    try:
        data = request.get_json()
        
        if not data or 'formation' not in data:
            return jsonify({'error': 'Missing formation data'}), 400
        
        formation = data['formation']
        players = data.get('players', [])
        
        # Mock formation analysis
        analysis = {
            'spacing_score': 0.8,
            'balance_score': 0.75,
            'offensive_potential': 0.82,
            'defensive_vulnerability': 0.3,
            'recommendations': [
                'Maintain wide spacing for better passing lanes',
                'Consider post player positioning for rebounds',
                'Utilize wing players for three-point opportunities'
            ]
        }
        
        return jsonify({
            'success': True,
            'formation_analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error analyzing formation: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to Basketball Play Creator'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('start_simulation')
def handle_start_simulation(data):
    """Handle real-time simulation request"""
    logger.info('Starting real-time simulation')
    
    players = data.get('players', [])
    actions = data.get('actions', [])
    
    # Simulate real-time updates
    for i in range(10):  # 10 updates over simulation
        time.sleep(0.5)  # 0.5 second intervals
        
        # Mock player position updates
        updated_players = []
        for j, player in enumerate(players):
            # Slightly move players (mock movement)
            new_x = player['location'][0] + (i * 2) - 10
            new_y = player['location'][1] + (i * 1) - 5
            
            updated_player = player.copy()
            updated_player['location'] = [new_x, new_y]
            updated_players.append(updated_player)
        
        # Emit update
        emit('simulation_update', {
            'timestamp': i * 0.5,
            'players': updated_players,
            'ball_position': updated_players[0]['location'],
            'score_probability': 0.6 + (i * 0.02)
        })
    
    emit('simulation_complete', {'message': 'Simulation finished'})

if __name__ == '__main__':
    logger.info("Starting Basketball Play Creator Backend...")
    logger.info("API will be available at: http://localhost:5000")
    
    # Run the Flask application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
