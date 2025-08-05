"""
Basketball Play Creator - Backend API Server
Flask application providing REST endpoints for play creation and simulation
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_parser.play_parser import PlayParser
from gnn_engine.simulator import PlaySimulator
from data.play_database import PlayDatabase

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'basketball_play_creator_2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
play_parser = PlayParser()
play_simulator = PlaySimulator()
play_database = PlayDatabase()

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'active',
        'service': 'Basketball Play Creator API',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/parse_play', methods=['POST'])
def parse_play():
    """
    Parse natural language play description into structured format
    
    Request body:
    {
        "description": "PG passes to SG, center sets screen, cut to basket",
        "formation": "5-out" (optional),
        "context": {"quarter": 4, "score_diff": 2} (optional)
    }
    """
    try:
        data = request.get_json()
        description = data.get('description', '')
        formation = data.get('formation', 'default')
        context = data.get('context', {})
        
        if not description:
            return jsonify({'error': 'Play description is required'}), 400
        
        logger.info(f"Parsing play: {description}")
        
        # Parse the natural language description
        parsed_play = play_parser.parse(description, formation, context)
        
        return jsonify({
            'success': True,
            'parsed_play': parsed_play,
            'confidence': parsed_play.get('confidence', 0.0),
            'suggestions': parsed_play.get('alternative_interpretations', [])
        })
        
    except Exception as e:
        logger.error(f"Error parsing play: {str(e)}")
        return jsonify({'error': f'Failed to parse play: {str(e)}'}), 500

@app.route('/api/simulate_play', methods=['POST'])
def simulate_play():
    """
    Simulate play using GNN engine
    
    Request body:
    {
        "parsed_play": {...},
        "initial_positions": [...],
        "simulation_params": {...}
    }
    """
    try:
        data = request.get_json()
        parsed_play = data.get('parsed_play', {})
        initial_positions = data.get('initial_positions', [])
        params = data.get('simulation_params', {})
        
        if not parsed_play:
            return jsonify({'error': 'Parsed play data is required'}), 400
        
        logger.info(f"Simulating play: {parsed_play.get('play_name', 'Unknown')}")
        
        # Run GNN simulation
        simulation_result = play_simulator.simulate(
            parsed_play, 
            initial_positions, 
            params
        )
        
        return jsonify({
            'success': True,
            'simulation': simulation_result,
            'metadata': {
                'duration': simulation_result.get('duration', 0),
                'success_probability': simulation_result.get('success_probability', 0.5),
                'key_events': simulation_result.get('events', [])
            }
        })
        
    except Exception as e:
        logger.error(f"Error simulating play: {str(e)}")
        return jsonify({'error': f'Failed to simulate play: {str(e)}'}), 500

@app.route('/api/optimize_play', methods=['POST'])
def optimize_play():
    """
    Optimize play using ML recommendations
    
    Request body:
    {
        "parsed_play": {...},
        "constraints": {...},
        "objectives": ["score_probability", "time_efficiency"]
    }
    """
    try:
        data = request.get_json()
        parsed_play = data.get('parsed_play', {})
        constraints = data.get('constraints', {})
        objectives = data.get('objectives', ['score_probability'])
        
        logger.info(f"Optimizing play with objectives: {objectives}")
        
        # Run optimization
        optimization_result = play_simulator.optimize_play(
            parsed_play,
            constraints,
            objectives
        )
        
        return jsonify({
            'success': True,
            'optimization': optimization_result,
            'recommendations': optimization_result.get('recommendations', []),
            'improvements': optimization_result.get('improvements', {})
        })
        
    except Exception as e:
        logger.error(f"Error optimizing play: {str(e)}")
        return jsonify({'error': f'Failed to optimize play: {str(e)}'}), 500

@app.route('/api/plays', methods=['GET'])
def get_plays():
    """Get play library"""
    try:
        category = request.args.get('category', 'all')
        search = request.args.get('search', '')
        limit = int(request.args.get('limit', 50))
        
        plays = play_database.get_plays(category, search, limit)
        
        return jsonify({
            'success': True,
            'plays': plays,
            'count': len(plays)
        })
        
    except Exception as e:
        logger.error(f"Error fetching plays: {str(e)}")
        return jsonify({'error': f'Failed to fetch plays: {str(e)}'}), 500

@app.route('/api/plays', methods=['POST'])
def save_play():
    """Save a new play to the database"""
    try:
        data = request.get_json()
        play_data = data.get('play', {})
        
        if not play_data.get('name'):
            return jsonify({'error': 'Play name is required'}), 400
        
        saved_play = play_database.save_play(play_data)
        
        return jsonify({
            'success': True,
            'play': saved_play,
            'id': saved_play.get('id')
        })
        
    except Exception as e:
        logger.error(f"Error saving play: {str(e)}")
        return jsonify({'error': f'Failed to save play: {str(e)}'}), 500

@app.route('/api/plays/<play_id>', methods=['GET'])
def get_play(play_id):
    """Get specific play by ID"""
    try:
        play = play_database.get_play_by_id(play_id)
        
        if not play:
            return jsonify({'error': 'Play not found'}), 404
        
        return jsonify({
            'success': True,
            'play': play
        })
        
    except Exception as e:
        logger.error(f"Error fetching play {play_id}: {str(e)}")
        return jsonify({'error': f'Failed to fetch play: {str(e)}'}), 500

@app.route('/api/analyze_formation', methods=['POST'])
def analyze_formation():
    """Analyze team formation and provide insights"""
    try:
        data = request.get_json()
        positions = data.get('positions', [])
        team_data = data.get('team_data', {})
        
        if not positions:
            return jsonify({'error': 'Player positions are required'}), 400
        
        analysis = play_simulator.analyze_formation(positions, team_data)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'insights': analysis.get('insights', []),
            'recommendations': analysis.get('recommendations', [])
        })
        
    except Exception as e:
        logger.error(f"Error analyzing formation: {str(e)}")
        return jsonify({'error': f'Failed to analyze formation: {str(e)}'}), 500

@app.route('/api/predict_outcome', methods=['POST'])
def predict_outcome():
    """Predict play outcome probability"""
    try:
        data = request.get_json()
        play_state = data.get('play_state', {})
        context = data.get('context', {})
        
        prediction = play_simulator.predict_outcome(play_state, context)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': prediction.get('confidence', 0.0),
            'factors': prediction.get('key_factors', [])
        })
        
    except Exception as e:
        logger.error(f"Error predicting outcome: {str(e)}")
        return jsonify({'error': f'Failed to predict outcome: {str(e)}'}), 500

# WebSocket events for real-time features
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to Basketball Play Creator'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('live_simulation')
def handle_live_simulation(data):
    """Handle real-time play simulation"""
    try:
        simulation_id = data.get('simulation_id')
        play_data = data.get('play_data')
        
        # Run real-time simulation
        result = play_simulator.simulate_realtime(play_data)
        
        emit('simulation_update', {
            'simulation_id': simulation_id,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in live simulation: {str(e)}")
        emit('error', {'message': f'Simulation error: {str(e)}'})

@socketio.on('request_suggestions')
def handle_suggestion_request(data):
    """Handle real-time play suggestions"""
    try:
        context = data.get('context', {})
        current_state = data.get('current_state', {})
        
        suggestions = play_simulator.get_suggestions(current_state, context)
        
        emit('suggestions', {
            'suggestions': suggestions,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        emit('error', {'message': f'Suggestion error: {str(e)}'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Basketball Play Creator API Server...")
    
    # Initialize database
    play_database.initialize()
    
    # Start server
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
