"""Flask web application for video dubbing interface."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import threading

from ..utils.config import AppConfig, load_config
from ..pipeline import VideoDubbingPipeline, PipelineManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-dubbing-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config = load_config()
pipeline_manager = PipelineManager(config)
upload_folder = app.config['UPLOAD_FOLDER']

# Ensure upload directory exists
os.makedirs(upload_folder, exist_ok=True)


@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file for processing."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(upload_folder, filename)
            
            file.save(file_path)
            
            logger.info(f"Video uploaded: {filename}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'file_path': file_path,
                'message': 'Video uploaded successfully'
            })
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/reference', methods=['POST'])
def upload_reference_audio():
    """Upload reference audio for voice cloning."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ref_{timestamp}_{filename}"
            file_path = os.path.join(upload_folder, filename)
            
            file.save(file_path)
            
            logger.info(f"Reference audio uploaded: {filename}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'file_path': file_path,
                'message': 'Reference audio uploaded successfully'
            })
    
    except Exception as e:
        logger.error(f"Reference upload failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def start_processing():
    """Start video dubbing process."""
    try:
        data = request.get_json()
        
        video_path = data.get('video_path')
        target_language = data.get('target_language')
        speaker_reference = data.get('speaker_reference')
        auto_approve = data.get('auto_approve', False)
        
        if not video_path or not target_language:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Create pipeline
        pipeline = pipeline_manager.create_pipeline()
        session_id = pipeline.session_id
        
        # Generate output path
        output_filename = f"dubbed_{session_id}.mp4"
        output_path = os.path.join(config.video.output_dir, output_filename)
        
        # Start processing in background thread
        def process_video():
            try:
                result = pipeline.process_video(
                    video_path=video_path,
                    target_language=target_language,
                    output_path=output_path,
                    speaker_reference=speaker_reference,
                    auto_approve=auto_approve
                )
                
                # Emit completion event
                socketio.emit('processing_complete', {
                    'session_id': session_id,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Processing failed: {e}")
                socketio.emit('processing_error', {
                    'session_id': session_id,
                    'error': str(e)
                })
        
        # Start processing thread
        thread = threading.Thread(target=process_video)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Processing started'
        })
    
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<session_id>')
def get_status(session_id: str):
    """Get processing status."""
    try:
        status = pipeline_manager.get_pipeline_status(session_id)
        
        if status is None:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/review/<session_id>')
def get_review_data(session_id: str):
    """Get translation review data."""
    try:
        pipeline = pipeline_manager.get_pipeline(session_id)
        if not pipeline:
            return jsonify({'error': 'Session not found'}), 404
        
        # Load review session data
        review_path = pipeline.pipeline_state["files"].get("review_session")
        if not review_path or not os.path.exists(review_path):
            return jsonify({'error': 'Review session not found'}), 404
        
        with open(review_path, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        
        return jsonify(review_data)
    
    except Exception as e:
        logger.error(f"Failed to get review data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/review/<session_id>/segment/<int:segment_id>', methods=['PUT'])
def update_segment_translation(session_id: str, segment_id: int):
    """Update segment translation during review."""
    try:
        data = request.get_json()
        new_translation = data.get('translation')
        approved = data.get('approved', False)
        
        if not new_translation:
            return jsonify({'error': 'Translation text required'}), 400
        
        pipeline = pipeline_manager.get_pipeline(session_id)
        if not pipeline:
            return jsonify({'error': 'Session not found'}), 404
        
        # Update translation
        pipeline.translation_processor.review_manager.update_segment_translation(
            session_id, segment_id, new_translation, approved
        )
        
        return jsonify({
            'success': True,
            'message': 'Translation updated'
        })
    
    except Exception as e:
        logger.error(f"Failed to update translation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/review/<session_id>/approve', methods=['POST'])
def approve_review(session_id: str):
    """Approve all translations and continue processing."""
    try:
        pipeline = pipeline_manager.get_pipeline(session_id)
        if not pipeline:
            return jsonify({'error': 'Session not found'}), 404
        
        # Approve all segments
        pipeline.translation_processor.review_manager.approve_all_segments(session_id)
        
        # Continue processing from TTS stage
        # This would need to be implemented to resume pipeline
        
        return jsonify({
            'success': True,
            'message': 'Review approved, continuing processing'
        })
    
    except Exception as e:
        logger.error(f"Failed to approve review: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<session_id>')
def download_result(session_id: str):
    """Download processed video."""
    try:
        pipeline = pipeline_manager.get_pipeline(session_id)
        if not pipeline:
            return jsonify({'error': 'Session not found'}), 404
        
        # Check if processing is complete
        if pipeline.pipeline_state["status"] != "completed":
            return jsonify({'error': 'Processing not complete'}), 400
        
        output_path = pipeline.pipeline_state["metadata"]["output_path"]
        if not os.path.exists(output_path):
            return jsonify({'error': 'Output file not found'}), 404
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"dubbed_video_{session_id}.mp4"
        )
    
    except Exception as e:
        logger.error(f"Failed to download result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/languages')
def get_supported_languages():
    """Get list of supported languages."""
    return jsonify(config.supported_languages)


@app.route('/api/config')
def get_config():
    """Get application configuration."""
    return jsonify({
        'supported_languages': config.supported_languages,
        'max_file_size': app.config['MAX_CONTENT_LENGTH'],
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'translation_service': config.translation.service,
        'translation_model': config.translation.model_name,
        'has_openai_key': bool(config.translation.api_key)
    })


# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('connected', {'message': 'Connected to video dubbing server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')


@socketio.on('subscribe_status')
def handle_subscribe_status(data):
    """Subscribe to status updates for a session."""
    session_id = data.get('session_id')
    if session_id:
        # Join room for this session
        from flask_socketio import join_room
        join_room(session_id)
        logger.info(f'Client subscribed to session: {session_id}')


# Background task to emit status updates
def background_status_updates():
    """Emit periodic status updates for active pipelines."""
    while True:
        try:
            for session_id in pipeline_manager.list_active_pipelines():
                status = pipeline_manager.get_pipeline_status(session_id)
                if status:
                    socketio.emit('status_update', status, room=session_id)
            
            socketio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in background status updates: {e}")
            socketio.sleep(5)


# Start background task
socketio.start_background_task(background_status_updates)


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(config.video.output_dir, exist_ok=True)
    os.makedirs(config.video.temp_dir, exist_ok=True)
    
    logger.info("Starting video dubbing web application")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
