@echo off
echo 🏀 Starting Basketball Play Creator...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm not found. Please install npm and try again.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Get script directory and navigate to it
cd /d "%~dp0"

REM Start backend
echo 🚀 Starting backend...
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install backend dependencies
echo 📦 Installing backend dependencies...
pip install -r requirements.txt

REM Download spaCy model if not present
echo 🔧 Checking spaCy model...
python -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
if errorlevel 1 (
    echo 📦 Downloading spaCy model...
    python -m spacy download en_core_web_sm
)

REM Set environment variables
set FLASK_APP=app.py
set FLASK_ENV=development

REM Start Flask in background
echo 🚀 Starting Flask server...
start /B flask run --host=0.0.0.0 --port=5000

REM Wait for backend to start
echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak >nul

REM Check if backend is running
curl -s http://localhost:5000/api/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Backend failed to start
    pause
    exit /b 1
) else (
    echo ✅ Backend started successfully
)

REM Start frontend
echo 🚀 Starting frontend...
cd ..\frontend

REM Install frontend dependencies
echo 📦 Installing frontend dependencies...
npm install

REM Start React
echo 🚀 Starting React server...
start /B npm start

REM Wait for frontend to start
echo ⏳ Waiting for frontend to start...
timeout /t 10 /nobreak >nul

echo.
echo 🎉 Basketball Play Creator is now running!
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:5000
echo.
echo Press any key to open the application in your browser...
pause >nul

REM Open browser
start http://localhost:3000

echo.
echo Press any key to stop all servers and exit...
pause >nul

REM Kill background processes (simplified - may need manual cleanup)
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1

echo ✅ Application stopped
