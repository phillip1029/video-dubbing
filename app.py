#!/usr/bin/env python3
"""
Video Dubbing Application - Web Interface Entry Point

This script starts the Flask web application for the video dubbing interface.
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web import app, socketio


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('temp', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    # RunPod environment detection
    is_runpod = os.getenv('RUNPOD_POD_ID') is not None
    if is_runpod:
        print(f"🚀 Starting Video Dubbing Application on RunPod")
        print(f"Pod ID: {os.getenv('RUNPOD_POD_ID')}")
        print(f"Public IP: {os.getenv('RUNPOD_PUBLIC_IP', 'Not available')}")
        print(f"Access URL: http://{os.getenv('RUNPOD_PUBLIC_IP', 'localhost')}:{port}")
    else:
        print("Starting Video Dubbing Web Application...")
        print(f"Access the application at: http://localhost:{port}")
    
    # Start the web application
    socketio.run(
        app, 
        debug=debug, 
        host=host, 
        port=port,
        allow_unsafe_werkzeug=True
    )
