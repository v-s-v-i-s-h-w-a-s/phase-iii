"""
Basketball Detection Web UI
Flask-based web interface for basketball video analysis
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import cv2
import tempfile
import threading
import time
from pathlib import Path
import yt_dlp
from werkzeug.utils import secure_filename
import json
from datetime import datetime

# Import our detection system
from src.inference import BasketballInference

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = 'basketball_detection_2025'

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)
Path('static/results').mkdir(parents=True, exist_ok=True)

# Global variables for progress tracking
processing_status = {}

class VideoProcessor:
    def __init__(self):
        self.model_path = "./models/basketball_yolo11n.pt"
        
    def download_youtube_video(self, url, output_path):
        """Download video from YouTube"""
        try:
            ydl_opts = {
                'format': 'best[height<=720]',  # Max 720p for processing speed
                'outtmpl': str(output_path / '%(title)s.%(ext)s'),
                'max_filesize': 500 * 1024 * 1024,  # 500MB limit
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
            return filename, info.get('title', 'Downloaded Video')
            
        except Exception as e:
            raise Exception(f"Failed to download video: {str(e)}")
    
    def process_video(self, video_path, task_id):
        """Process video and update progress"""
        try:
            processing_status[task_id] = {
                'status': 'initializing',
                'progress': 0,
                'message': 'Loading model...'
            }
            
            # Initialize inference
            inference = BasketballInference(self.model_path)
            if not inference.load_model():
                processing_status[task_id]['status'] = 'error'
                processing_status[task_id]['message'] = 'Failed to load model'
                return
            
            processing_status[task_id]['message'] = 'Analyzing video...'
            processing_status[task_id]['progress'] = 10
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            processing_status[task_id]['total_frames'] = total_frames
            processing_status[task_id]['fps'] = fps
            
            # Process video with progress tracking
            output_path = Path(app.config['OUTPUT_FOLDER']) / f"result_{task_id}.mp4"
            
            # Custom progress callback
            def progress_callback(frame_num, total_frames):
                progress = int((frame_num / total_frames) * 80) + 10  # 10-90%
                processing_status[task_id]['progress'] = progress
                processing_status[task_id]['message'] = f'Processing frame {frame_num}/{total_frames}'
            
            # Process video
            results = self.process_video_with_callback(
                video_path, str(output_path), inference, progress_callback
            )
            
            if results:
                processing_status[task_id]['status'] = 'completed'
                processing_status[task_id]['progress'] = 100
                processing_status[task_id]['message'] = 'Analysis complete!'
                processing_status[task_id]['output_video'] = str(output_path)
                processing_status[task_id]['results'] = results
                
                # Generate summary
                summary = self.generate_summary(results['detections'])
                processing_status[task_id]['summary'] = summary
                
            else:
                processing_status[task_id]['status'] = 'error'
                processing_status[task_id]['message'] = 'Video processing failed'
                
        except Exception as e:
            processing_status[task_id]['status'] = 'error'
            processing_status[task_id]['message'] = f'Error: {str(e)}'
    
    def process_video_with_callback(self, video_path, output_path, inference, progress_callback):
        """Process video with progress updates"""
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = inference.detect_frame(frame)
            
            # Draw detections
            result_frame = inference.draw_detections(frame.copy(), detections)
            
            # Add frame info
            info_text = f"Frame: {frame_count+1}/{total_frames} | Objects: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(result_frame)
            
            # Store detections
            for detection in detections:
                detection['frame'] = frame_count
                detection['timestamp'] = frame_count / fps
                all_detections.append(detection.copy())
            
            frame_count += 1
            
            # Update progress every 30 frames
            if frame_count % 30 == 0:
                progress_callback(frame_count, total_frames)
        
        cap.release()
        out.release()
        
        return {
            'detections': all_detections,
            'stats': {
                'total_frames': frame_count,
                'total_detections': len(all_detections)
            }
        }
    
    def generate_summary(self, detections):
        """Generate analysis summary"""
        if not detections:
            return {'total': 0}
        
        import pandas as pd
        df = pd.DataFrame(detections)
        
        summary = {
            'total_detections': len(df),
            'class_distribution': df['class'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'detection_timeline': []
        }
        
        # Timeline data (every 30 seconds)
        if 'timestamp' in df.columns:
            df['time_bucket'] = (df['timestamp'] // 30) * 30
            timeline = df.groupby(['time_bucket', 'class']).size().unstack(fill_value=0)
            
            for time_bucket in timeline.index:
                timeline_entry = {'time': int(time_bucket)}
                for class_name in timeline.columns:
                    timeline_entry[class_name] = int(timeline.loc[time_bucket, class_name])
                summary['detection_timeline'].append(timeline_entry)
        
        return summary

# Initialize processor
processor = VideoProcessor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        video_path = Path(app.config['UPLOAD_FOLDER']) / safe_filename
        
        file.save(video_path)
        
        # Start processing
        task_id = f"upload_{timestamp}"
        processing_status[task_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Video uploaded, starting analysis...'
        }
        
        # Start processing in background
        thread = threading.Thread(
            target=processor.process_video,
            args=(str(video_path), task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/youtube', methods=['POST'])
def process_youtube():
    """Handle YouTube URL processing"""
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            return jsonify({'error': 'No YouTube URL provided'}), 400
        
        task_id = f"youtube_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        processing_status[task_id] = {
            'status': 'downloading',
            'progress': 0,
            'message': 'Downloading video from YouTube...'
        }
        
        def download_and_process():
            try:
                # Download video
                upload_path = Path(app.config['UPLOAD_FOLDER'])
                video_file, video_title = processor.download_youtube_video(youtube_url, upload_path)
                
                processing_status[task_id]['video_title'] = video_title
                processing_status[task_id]['progress'] = 5
                
                # Start processing
                processor.process_video(video_file, task_id)
                
            except Exception as e:
                processing_status[task_id]['status'] = 'error'
                processing_status[task_id]['message'] = str(e)
        
        # Start in background
        thread = threading.Thread(target=download_and_process)
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status"""
    status = processing_status.get(task_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/download/<task_id>')
def download_result(task_id):
    """Download processed video"""
    status = processing_status.get(task_id)
    if not status or status['status'] != 'completed':
        return jsonify({'error': 'Video not ready'}), 404
    
    output_path = status.get('output_video')
    if output_path and Path(output_path).exists():
        return send_file(output_path, as_attachment=True, 
                        download_name=f'basketball_analysis_{task_id}.mp4')
    
    return jsonify({'error': 'Output file not found'}), 404

@app.route('/results/<task_id>')
def view_results(task_id):
    """View analysis results"""
    status = processing_status.get(task_id)
    if not status:
        return jsonify({'error': 'Task not found'}), 404
    
    return render_template('results.html', task_id=task_id, status=status)

if __name__ == '__main__':
    print("🏀 Basketball Detection Web UI")
    print("Starting server at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
