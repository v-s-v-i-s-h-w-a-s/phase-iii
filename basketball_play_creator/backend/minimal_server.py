from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Basketball Play Creator API is running!',
        'status': 'success',
        'endpoints': [
            '/api/health',
            '/api/parse_play',
            '/api/simulate_play'
        ]
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'Basketball Play Creator'})

@app.route('/api/parse_play', methods=['POST'])
def parse_play():
    return jsonify({
        'success': True,
        'parsed_play': {
            'name': 'Pick and Pass',
            'confidence': 0.85,
            'actions': [
                {'type': 'pass', 'player': 'PG', 'target': 'SG', 'timestamp': 0.0},
                {'type': 'screen', 'player': 'C', 'timestamp': 1.0},
                {'type': 'cut', 'player': 'SF', 'timestamp': 2.0},
                {'type': 'shot', 'player': 'SG', 'timestamp': 3.5}
            ]
        }
    })

if __name__ == '__main__':
    print("🏀 Starting Basketball Play Creator Backend...")
    print("🚀 Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
