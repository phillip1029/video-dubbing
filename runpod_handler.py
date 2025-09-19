#!/usr/bin/env python3
"""
RunPod Serverless Handler for Video Dubbing Application

This handler provides a serverless API interface for RunPod deployments.
"""

import os
import sys
import json
import tempfile
import logging
from typing import Dict, Any
import base64

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import runpod
from src.utils.config import AppConfig, load_config
from src.pipeline import VideoDubbingPipeline


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file_from_url(url: str, local_path: str):
    """Download file from URL to local path."""
    import requests
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        f.write(response.content)


def upload_file_to_url(file_path: str, upload_url: str = None):
    """Upload file to cloud storage and return URL."""
    # This is a placeholder - implement based on your preferred storage
    # Options: AWS S3, Google Cloud Storage, Azure Blob, etc.
    
    if upload_url:
        # If upload URL provided, use it
        import requests
        with open(file_path, 'rb') as f:
            response = requests.put(upload_url, data=f)
            response.raise_for_status()
        return upload_url
    else:
        # Return local path as fallback
        return file_path


def process_video_dubbing(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process video dubbing job.
    
    Expected job input:
    {
        "video_url": "https://example.com/video.mp4",
        "target_language": "es",
        "speaker_reference_url": "https://example.com/voice.wav" (optional),
        "auto_approve": true,
        "openai_api_key": "sk-..." (optional),
        "config_overrides": {} (optional)
    }
    """
    try:
        logger.info("Starting video dubbing job")
        logger.info(f"Job input: {json.dumps(job, indent=2)}")
        
        # Validate required inputs
        if "video_url" not in job or "target_language" not in job:
            return {
                "error": "Missing required parameters: video_url and target_language"
            }
        
        video_url = job["video_url"]
        target_language = job["target_language"]
        speaker_reference_url = job.get("speaker_reference_url")
        auto_approve = job.get("auto_approve", True)
        openai_api_key = job.get("openai_api_key")
        config_overrides = job.get("config_overrides", {})
        
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Load configuration
        config = load_config("config.yaml")
        
        # Apply config overrides
        if config_overrides:
            logger.info(f"Applying config overrides: {config_overrides}")
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "input_video.mp4")
        output_path = os.path.join(temp_dir, "dubbed_video.mp4")
        speaker_reference_path = None
        
        # Download input video
        logger.info(f"Downloading video from: {video_url}")
        download_file_from_url(video_url, video_path)
        
        # Download speaker reference if provided
        if speaker_reference_url:
            logger.info(f"Downloading speaker reference from: {speaker_reference_url}")
            speaker_reference_path = os.path.join(temp_dir, "speaker_reference.wav")
            download_file_from_url(speaker_reference_url, speaker_reference_path)
        
        # Create pipeline
        pipeline = VideoDubbingPipeline(config)
        
        # Process video
        logger.info("Starting video dubbing pipeline")
        result = pipeline.process_video(
            video_path=video_path,
            target_language=target_language,
            output_path=output_path,
            speaker_reference=speaker_reference_path,
            auto_approve=auto_approve
        )
        
        if result["success"]:
            # Upload result video
            upload_url = job.get("output_upload_url")
            output_url = upload_file_to_url(output_path, upload_url)
            
            logger.info("Video dubbing completed successfully")
            
            return {
                "success": True,
                "output_url": output_url,
                "session_id": result["session_id"],
                "processing_time": result["pipeline_state"]["metadata"].get("duration", 0),
                "pipeline_state": result["pipeline_state"]
            }
        else:
            logger.error(f"Video dubbing failed: {result['error']}")
            return {
                "success": False,
                "error": result["error"],
                "session_id": result.get("session_id")
            }
    
    except Exception as e:
        logger.error(f"Job processing failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.
    """
    job_input = job.get("input", {})
    
    # Log GPU information
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("No GPU available")
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
    
    # Process the job
    return process_video_dubbing(job_input)


if __name__ == "__main__":
    # For local testing
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test job
        test_job = {
            "input": {
                "video_url": "https://example.com/test_video.mp4",
                "target_language": "es",
                "auto_approve": True
            }
        }
        result = handler(test_job)
        print(json.dumps(result, indent=2))
    else:
        # Start RunPod serverless
        logger.info("Starting RunPod serverless handler")
        runpod.serverless.start({"handler": handler})
