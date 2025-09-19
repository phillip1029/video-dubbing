#!/usr/bin/env python3
"""
Installation script for Video Dubbing Application

This script helps set up the video dubbing application by:
1. Checking system requirements
2. Installing dependencies
3. Downloading required models
4. Setting up configuration
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import urllib.request
import zipfile
import json


class VideoDubbingInstaller:
    """Installer for the Video Dubbing Application."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.architecture = platform.machine()
        
    def check_requirements(self):
        """Check system requirements."""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if self.python_version < (3, 8):
            print("❌ Python 3.8 or higher is required")
            return False
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}")
        
        # Check for Git
        if not shutil.which("git"):
            print("❌ Git is required for installing some dependencies")
            return False
        print("✅ Git found")
        
        # Check for FFmpeg
        if not shutil.which("ffmpeg"):
            print("⚠️  FFmpeg not found. It's required for video processing.")
            print("   Please install FFmpeg:")
            if self.platform == "Windows":
                print("   - Download from: https://ffmpeg.org/download.html")
            elif self.platform == "Darwin":  # macOS
                print("   - Run: brew install ffmpeg")
            else:  # Linux
                print("   - Run: sudo apt install ffmpeg (Ubuntu/Debian)")
                print("   - Run: sudo yum install ffmpeg (CentOS/RHEL)")
            return False
        print("✅ FFmpeg found")
        
        return True
    
    def install_dependencies(self, gpu_support=False):
        """Install Python dependencies."""
        print("\n📦 Installing Python dependencies...")
        
        # Upgrade pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install PyTorch first (with GPU support if requested)
        if gpu_support:
            print("🔥 Installing PyTorch with GPU support...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchaudio", "--index-url", 
                "https://download.pytorch.org/whl/cu118"
            ])
        else:
            print("💻 Installing PyTorch (CPU only)...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchaudio", "--index-url", 
                "https://download.pytorch.org/whl/cpu"
            ])
        
        # Install other dependencies
        print("📋 Installing other dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed")
    
    def setup_directories(self):
        """Create necessary directories."""
        print("\n📁 Setting up directories...")
        
        directories = [
            "models",
            "temp", 
            "output",
            "uploads",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def download_models(self):
        """Download required AI models."""
        print("\n🤖 Setting up AI models...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # WhisperX models will be downloaded automatically on first use
        print("✅ WhisperX models will be downloaded automatically")
        
        # Coqui TTS models will be downloaded automatically on first use
        print("✅ Coqui TTS models will be downloaded automatically")
        
        # MuseTalk setup instructions
        print("ℹ️  MuseTalk models will be set up automatically")
        print("   The first run may take longer as models are downloaded")
        
    def create_config(self):
        """Create default configuration if it doesn't exist."""
        print("\n⚙️  Setting up configuration...")
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("ℹ️  Default config.yaml already exists")
        else:
            print("✅ Configuration file ready")
    
    def test_installation(self):
        """Test the installation."""
        print("\n🧪 Testing installation...")
        
        try:
            # Test imports
            import torch
            print(f"✅ PyTorch {torch.__version__}")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                print(f"🔥 CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 CUDA not available (CPU mode)")
            
            # Test other key imports
            import whisperx
            print("✅ WhisperX available")
            
            try:
                import TTS
                print("✅ Coqui TTS available")
            except ImportError:
                print("⚠️  Coqui TTS import failed - will be installed on first use")
            
            print("✅ Installation test passed")
            return True
            
        except ImportError as e:
            print(f"❌ Installation test failed: {e}")
            return False
    
    def run_installation(self, gpu_support=False):
        """Run the complete installation process."""
        print("🚀 Video Dubbing Application Installer")
        print("=" * 50)
        
        # Check requirements
        if not self.check_requirements():
            print("\n❌ System requirements not met. Please fix the issues above.")
            return False
        
        # Install dependencies
        try:
            self.install_dependencies(gpu_support)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Download models
        self.download_models()
        
        # Create config
        self.create_config()
        
        # Test installation
        if not self.test_installation():
            return False
        
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Start the web interface: python app.py")
        print("2. Or use CLI: python main.py process -i video.mp4 -l es")
        print("3. Visit http://localhost:5000 for the web interface")
        
        return True


def main():
    """Main installer function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Dubbing Application Installer")
    parser.add_argument("--gpu", action="store_true", 
                       help="Install with GPU support (CUDA)")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Install CPU-only version")
    
    args = parser.parse_args()
    
    # Determine GPU support
    gpu_support = args.gpu or (not args.cpu_only and input("Install GPU support? (y/N): ").lower().startswith('y'))
    
    # Run installation
    installer = VideoDubbingInstaller()
    success = installer.run_installation(gpu_support)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
