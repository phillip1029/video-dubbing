#!/usr/bin/env python3
"""
RunPod Web Interface for Video Dubbing Application

This script configures the web application for RunPod deployment with
proper networking and GPU utilization.
"""

import os
import sys
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.web import app, socketio
from src.utils.config import load_config


def setup_runpod_environment():
    """Setup environment for RunPod deployment."""
    
    # Get RunPod environment variables
    pod_id = os.getenv('RUNPOD_POD_ID', 'local')
    public_ip = os.getenv('RUNPOD_PUBLIC_IP', 'localhost')
    
    # Configure Flask for RunPod
    app.config.update({
        'SECRET_KEY': os.getenv('SECRET_KEY', f'video-dubbing-{pod_id}'),
        'MAX_CONTENT_LENGTH': 1024 * 1024 * 1024,  # 1GB for large videos
        'UPLOAD_FOLDER': '/workspace/uploads',
    })
    
    # Ensure directories exist
    os.makedirs('/workspace/uploads', exist_ok=True)
    os.makedirs('/workspace/output', exist_ok=True)
    os.makedirs('/workspace/temp', exist_ok=True)
    os.makedirs('/workspace/models', exist_ok=True)
    os.makedirs('/workspace/logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/workspace/logs/app.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Video Dubbing Application on RunPod")
    logger.info(f"Pod ID: {pod_id}")
    logger.info(f"Public IP: {public_ip}")
    
    # Log GPU information
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("No GPU available")
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
    
    return logger


@app.route('/health')
def health_check():
    """Health check endpoint for RunPod."""
    import torch
    
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "pod_id": os.getenv('RUNPOD_POD_ID', 'local')
    }


@app.route('/api/system/info')
def system_info():
    """Get system information."""
    import torch
    import psutil
    
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,
                "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,
            })
    
    return {
        "pod_id": os.getenv('RUNPOD_POD_ID', 'local'),
        "public_ip": os.getenv('RUNPOD_PUBLIC_IP', 'localhost'),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total / 1024**3,
        "memory_available": psutil.virtual_memory().available / 1024**3,
        "gpu_info": gpu_info
    }


if __name__ == '__main__':
    # Setup RunPod environment
    logger = setup_runpod_environment()
    
    # Load configuration
    config = load_config('/workspace/config.yaml')
    
    # Get port from environment
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting web server on {host}:{port}")
    logger.info(f"Access URL: http://{os.getenv('RUNPOD_PUBLIC_IP', 'localhost')}:{port}")
    
    # Start the web application
    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True
    )
