#!/usr/bin/env python3
"""
Video Dubbing Application - Command Line Interface

This script provides a command-line interface for the video dubbing application.
It can process videos in batch mode or start the web interface.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import AppConfig, load_config
from src.pipeline import VideoDubbingPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_dubbing.log')
        ]
    )


def process_video_cli(args):
    """Process video using command line interface."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input video file not found: {args.input}")
        return 1
    
    if args.target_language not in config.supported_languages:
        logger.error(f"Unsupported language: {args.target_language}")
        logger.info(f"Supported languages: {list(config.supported_languages.keys())}")
        return 1
    
    # Setup output path
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_dubbed_{args.target_language}.mp4")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        logger.info("Starting video dubbing pipeline")
        logger.info(f"Input: {args.input}")
        logger.info(f"Target language: {args.target_language}")
        logger.info(f"Output: {args.output}")
        
        # Create pipeline
        pipeline = VideoDubbingPipeline(config)
        
        # Process video
        result = pipeline.process_video(
            video_path=args.input,
            target_language=args.target_language,
            output_path=args.output,
            speaker_reference=args.speaker_reference,
            auto_approve=True  # CLI mode auto-approves translations
        )
        
        if result["success"]:
            logger.info(f"Video dubbing completed successfully!")
            logger.info(f"Output saved to: {result['output_path']}")
            
            # Print pipeline statistics
            metadata = result["pipeline_state"]["metadata"]
            duration = metadata.get("duration", 0)
            logger.info(f"Processing time: {duration:.2f} seconds")
            
            return 0
        else:
            logger.error(f"Video dubbing failed: {result['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


def start_web_interface(args):
    """Start the web interface."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.web import app, socketio
        
        logger.info("Starting video dubbing web interface")
        logger.info(f"Server will be available at: http://localhost:{args.port}")
        
        # Create necessary directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('temp', exist_ok=True)
        
        # Start web server
        socketio.run(
            app, 
            debug=args.debug, 
            host=args.host, 
            port=args.port
        )
        
    except ImportError as e:
        logger.error(f"Web interface dependencies not available: {e}")
        logger.error("Please install web dependencies: pip install flask flask-socketio")
        return 1
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video Dubbing Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video via command line
  python main.py process -i video.mp4 -l es -o dubbed_video.mp4
  
  # Process with voice cloning
  python main.py process -i video.mp4 -l es -s reference_voice.wav
  
  # Start web interface
  python main.py web
  
  # Start web interface on custom port
  python main.py web --port 8080
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('-c', '--config', type=str,
                       help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process video via CLI')
    process_parser.add_argument('-i', '--input', required=True,
                               help='Input video file path')
    process_parser.add_argument('-l', '--target-language', required=True,
                               help='Target language code (e.g., es, fr, de)')
    process_parser.add_argument('-o', '--output',
                               help='Output video file path (auto-generated if not specified)')
    process_parser.add_argument('-s', '--speaker-reference',
                               help='Reference audio file for voice cloning')
    process_parser.add_argument('--resume', type=str,
                               help='Session ID to resume a previous run')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='0.0.0.0',
                           help='Host to bind to (default: 0.0.0.0)')
    web_parser.add_argument('--port', type=int, default=5000,
                           help='Port to bind to (default: 5000)')
    web_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'process':
        return process_video_cli(args)
    elif args.command == 'web':
        return start_web_interface(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
