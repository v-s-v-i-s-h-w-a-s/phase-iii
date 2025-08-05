"""
Basketball Play Creator - Simple Demo Server
"""

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

import json
import time

if not HAS_FLASK:
    print("Flask not installed. Please install Flask:")
    print("pip install Flask Flask-CORS")
    exit(1)

app = Flask(__name__)
CORS(app)

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Basketball Play Creator Demo',
        'timestamp': time.time()
    })

@app.route('/api/parse_play', methods=['POST'])
def parse_play():
    data = request.get_json()
    description = data.get('description', '')
    
    return jsonify({
        'success': True,
        'parsed_play': {
            'name': 'Demo Play',
            'description': description,
            'confidence': 0.85,
            'actions': [
                {'type': 'pass', 'player': 'player_1', 'timestamp': 0.0},
                {'type': 'screen', 'player': 'player_5', 'timestamp': 1.0},
                {'type': 'cut', 'player': 'player_3', 'timestamp': 2.0}
            ],
            'players': [
                {'id': 'player_1', 'position': 'PG', 'location': [470, 400], 'has_ball': True},
                {'id': 'player_2', 'position': 'SG', 'location': [600, 350], 'has_ball': False},
                {'id': 'player_3', 'position': 'SF', 'location': [340, 350], 'has_ball': False},
                {'id': 'player_4', 'position': 'PF', 'location': [570, 300], 'has_ball': False},
                {'id': 'player_5', 'position': 'C', 'location': [470, 200], 'has_ball': False}
            ],
            'formation': '5-out',
            'duration': 5.0
        }
    })

@app.route('/api/simulate_play', methods=['POST'])
def simulate_play():
    return jsonify({
        'success': True,
        'simulation_result': {
            'final_score_probability': 0.75,
            'success_probability': 0.82,
            'key_interactions': ['Screen set at 1.0s', 'Good ball movement'],
            'tactical_analysis': {
                'ball_movement': 0.8,
                'spacing': 0.75,
                'tempo': 0.7
            },
            'optimization_suggestions': [
                'Consider better spacing',
                'Add more movement'
            ]
        }
    })

if __name__ == '__main__':
    print("🏀 Starting Basketball Play Creator Demo Server...")
    print("🌐 Server will run at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
