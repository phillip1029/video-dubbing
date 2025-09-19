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
    
    print("Starting Video Dubbing Web Application...")
    print("Access the application at: http://localhost:5000")
    
    # Start the web application
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
