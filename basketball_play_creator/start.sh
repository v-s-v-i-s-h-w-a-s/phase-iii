#!/bin/bash

# Basketball Play Creator - Start Script
echo "🏀 Starting Basketball Play Creator..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python; then
    echo "❌ Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js not found. Please install Node.js 16+ and try again."
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm not found. Please install npm and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start backend
echo "🚀 Starting backend..."
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip install -r requirements.txt

# Download spaCy model if not present
echo "🔧 Checking spaCy model..."
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
    echo "📦 Downloading spaCy model..."
    python -m spacy download en_core_web_sm
}

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Start Flask in background
echo "🚀 Starting Flask server..."
flask run --host=0.0.0.0 --port=5000 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:5000/api/health >/dev/null 2>&1; then
    echo "✅ Backend started successfully"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo "🚀 Starting frontend..."
cd ../frontend

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
npm install

# Start React in background  
echo "🚀 Starting React server..."
npm start &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 10

# Check if frontend is running
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "✅ Frontend started successfully"
else
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Basketball Play Creator is now running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Wait for user to stop
wait
