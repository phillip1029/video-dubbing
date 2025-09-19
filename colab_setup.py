#!/usr/bin/env python3
"""
Quick setup script for running Video Dubbing on Google Colab
"""

import os
import subprocess
import sys

def setup_colab_environment():
    """Setup the video dubbing environment on Google Colab."""
    
    print("🚀 Setting up Video Dubbing on Google Colab")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running on Google Colab")
    except ImportError:
        print("⚠️ Not running on Google Colab")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Available: {gpu_name}")
        else:
            print("❌ No GPU detected! Enable GPU runtime:")
            print("   Runtime → Change runtime type → T4 GPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    os.system("apt-get update -qq")
    os.system("apt-get install -y -qq ffmpeg git-lfs")
    
    # Install Python dependencies
    print("\n🐍 Installing Python dependencies...")
    
    # PyTorch with CUDA
    os.system("pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Core packages
    packages = [
        "openai>=1.12.0",
        "whisperx", 
        "TTS>=0.22.0",
        "flask flask-socketio",
        "ffmpeg-python opencv-python librosa soundfile pydub moviepy",
        "transformers>=4.30.0", 
        "face-alignment scipy scikit-image pillow imageio",
        "pyyaml python-dotenv click pandas numpy tqdm requests"
    ]
    
    for package in packages:
        print(f"Installing {package.split()[0]}...")
        os.system(f"pip install -q {package}")
    
    print("✅ Installation completed!")
    return True

def quick_test():
    """Quick test of the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        import torch
        import whisperx
        import TTS
        print("✅ Core packages imported successfully")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    success = setup_colab_environment()
    
    if success:
        test_success = quick_test()
        
        if test_success:
            print("\n🎉 Setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Upload your video file")
            print("2. Set your OpenAI API key (optional)")
            print("3. Run the dubbing pipeline")
            print("\n💡 Use the Jupyter cells above or run:")
            print("   from colab_setup import quick_start")
            print("   quick_start()")
        else:
            print("\n❌ Setup completed but tests failed")
    else:
        print("\n❌ Setup failed")

def quick_start():
    """Quick start function for immediate use."""
    print("🎬 Video Dubbing Quick Start")
    print("=" * 30)
    
    # Import required modules
    sys.path.append('/content/video-dubbing/src')
    
    from google.colab import files
    import os
    
    # Upload video
    print("📤 Upload your video file:")
    uploaded = files.upload()
    
    if uploaded:
        video_filename = list(uploaded.keys())[0]
        video_path = f"/content/{video_filename}"
        
        # Select language
        languages = {
            "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
            "pt": "Portuguese", "ru": "Russian", "ja": "Japanese"
        }
        
        print("\n🌍 Available languages:")
        for code, name in languages.items():
            print(f"  {code}: {name}")
        
        target_lang = input("Enter language code: ").strip()
        
        if target_lang in languages:
            print(f"\n🚀 Processing {video_filename} → {languages[target_lang]}")
            
            # Set up API key
            from getpass import getpass
            api_key = getpass("OpenAI API key (optional): ")
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
            
            # Run processing
            from src.utils.config import AppConfig
            from src.pipeline import VideoDubbingPipeline
            
            config = AppConfig()
            config.translation.service = "openai" if api_key else "google"
            config.whisperx.model_size = "medium"  # Colab-optimized
            
            pipeline = VideoDubbingPipeline(config)
            
            output_path = f"/content/dubbed_{os.path.splitext(video_filename)[0]}.mp4"
            
            result = pipeline.process_video(
                video_path=video_path,
                target_language=target_lang,
                output_path=output_path,
                auto_approve=True
            )
            
            if result["success"]:
                print("✅ Success! Downloading result...")
                files.download(output_path)
            else:
                print(f"❌ Failed: {result['error']}")
        else:
            print("❌ Invalid language code")
    else:
        print("❌ No video uploaded")

if __name__ == "__main__":
    main()
