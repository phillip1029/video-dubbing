# 🎬 Video Dubbing on Google Colab - Complete Guide

Run the AI-powered video dubbing application on Google Colab with free GPU access!

## 🚀 Quick Start Options

### Option 1: Use the Jupyter Notebook (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/video-dubbing/blob/main/Video_Dubbing_Colab.ipynb)

1. **Click the badge above** to open the notebook in Colab
2. **Enable GPU**: Runtime → Change runtime type → T4 GPU → Save
3. **Run all cells** step by step
4. **Upload your video** and select target language
5. **Download the result**!

### Option 2: Quick Setup Script
```python
# In a new Colab notebook, run:
!git clone https://github.com/yourusername/video-dubbing.git
%cd video-dubbing
!python colab_setup.py

# Then run quick start:
from colab_setup import quick_start
quick_start()
```

### Option 3: Manual Setup
```python
# 1. Install dependencies
!apt-get update -qq && apt-get install -y -qq ffmpeg
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install whisperx TTS openai flask

# 2. Clone repository
!git clone https://github.com/yourusername/video-dubbing.git
%cd video-dubbing

# 3. Upload video and process
from google.colab import files
uploaded = files.upload()  # Upload your video
# ... (see notebook for full code)
```

## 💡 Why Google Colab?

### ✅ **Advantages:**
- **🆓 Free GPU Access**: T4 GPU (16GB VRAM) for free
- **⚡ Fast Setup**: No local installation required
- **🌐 Web-based**: Works from any browser
- **📱 Mobile Friendly**: Can run on tablets/phones
- **🔄 Easy Sharing**: Share notebooks with others
- **💾 Cloud Storage**: Integrated with Google Drive

### ⚠️ **Limitations:**
- **⏰ Session Limits**: 12 hours max (free tier)
- **💾 Disk Space**: ~78GB available
- **📤 File Size**: Large videos may hit limits
- **🔄 Temporary**: Files deleted after session
- **🐌 CPU Fallback**: Slower without GPU

## 🛠️ Setup Requirements

### **System Requirements:**
- **GPU**: Enable T4 GPU runtime (free) or A100/V100 (Pro)
- **RAM**: ~12GB available in Colab
- **Storage**: ~20GB for models and temp files
- **Internet**: Stable connection for model downloads

### **Input Specifications:**
- **Video Formats**: MP4, AVI, MOV, MKV
- **File Size**: < 100MB (free), < 500MB (Pro)
- **Duration**: < 30 minutes recommended
- **Audio**: Clear speech, minimal background noise

## ⚙️ Configuration for Colab

### **Optimized Settings:**
```python
# Colab-optimized configuration
config = AppConfig()

# WhisperX settings
config.whisperx.model_size = "medium"  # Balance speed/quality
config.whisperx.batch_size = 8         # Reduced for memory
config.whisperx.device = "cuda"

# Translation settings  
config.translation.service = "openai"  # Best quality
config.translation.model_name = "gpt-4o"

# TTS settings
config.tts.device = "cuda"
config.tts.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
```

### **Memory Management:**
```python
# For memory-constrained processing
config.whisperx.model_size = "small"   # Use smaller model
config.whisperx.batch_size = 4         # Reduce batch size

# Clear GPU memory between stages
import torch
torch.cuda.empty_cache()
```

## 🎯 Performance Expectations

### **Processing Times (T4 GPU):**
| Video Length | WhisperX | Translation | TTS | Total |
|--------------|----------|-------------|-----|-------|
| 1 minute | 30 seconds | 10 seconds | 1 minute | ~2 minutes |
| 5 minutes | 2 minutes | 30 seconds | 4 minutes | ~7 minutes |
| 15 minutes | 5 minutes | 1 minute | 10 minutes | ~16 minutes |
| 30 minutes | 10 minutes | 2 minutes | 20 minutes | ~32 minutes |

### **Quality vs Speed:**
| Model Size | Quality | Speed | Memory |
|------------|---------|-------|--------|
| tiny | ⭐⭐ | ⭐⭐⭐⭐⭐ | 0.5GB |
| small | ⭐⭐⭐ | ⭐⭐⭐⭐ | 1GB |
| medium | ⭐⭐⭐⭐ | ⭐⭐⭐ | 2GB |
| large-v3 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 4GB |

## 📁 File Management

### **Input Files:**
```python
# Upload from local computer
from google.colab import files
uploaded = files.upload()

# Or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
video_path = '/content/drive/MyDrive/video.mp4'
```

### **Output Files:**
```python
# Download directly
files.download('/content/dubbed_video.mp4')

# Or save to Google Drive
import shutil
shutil.copy('/content/dubbed_video.mp4', '/content/drive/MyDrive/')
```

### **Temporary Storage:**
- **Models**: Downloaded automatically (~5GB total)
- **Temp files**: Audio extracts, segments (~2x video size)
- **Output**: Final dubbed video (~same as input size)

## 🌐 Web Interface on Colab

### **Launch Web Interface:**
```python
# Install ngrok for public access
!pip install pyngrok

# Start web app with ngrok tunnel
import threading
from pyngrok import ngrok

def start_app():
    os.system('python app.py')

# Start in background
thread = threading.Thread(target=start_app, daemon=True)
thread.start()

# Create public URL
public_url = ngrok.connect(5000)
print(f"🌐 Access at: {public_url}")
```

### **Web Interface Features:**
- **📤 Drag & drop upload**
- **📊 Real-time progress**
- **✏️ Translation editing**
- **🎵 Voice cloning**
- **📥 Direct download**

## 🔧 Troubleshooting

### **Common Issues:**

1. **GPU Not Available**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU → Save
   ```

2. **Out of Memory**
   ```python
   # Use smaller models
   config.whisperx.model_size = "small"
   config.whisperx.batch_size = 4
   
   # Clear memory
   torch.cuda.empty_cache()
   ```

3. **Session Timeout**
   ```python
   # Keep session alive (run in cell)
   import time
   while True:
       print("Keeping session alive...")
       time.sleep(300)  # 5 minutes
   ```

4. **Large File Upload**
   ```python
   # Use Google Drive for large files
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Model Download Fails**
   ```python
   # Retry with better connection
   !pip install --upgrade whisperx TTS
   
   # Or download manually
   !wget https://model-url -O /content/model.bin
   ```

### **Performance Optimization:**

1. **Faster Processing**
   ```python
   # Use smaller, faster models
   config.whisperx.model_size = "medium"  # Instead of large
   config.whisperx.batch_size = 16        # Increase if memory allows
   ```

2. **Better Quality**
   ```python
   # Use larger models (if memory allows)
   config.whisperx.model_size = "large-v3"
   config.translation.service = "openai"
   config.translation.model_name = "gpt-4o"
   ```

3. **Memory Management**
   ```python
   # Process in smaller chunks
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

## 💰 Colab Plans Comparison

### **Free Tier:**
- **GPU**: T4 (16GB VRAM)
- **Session**: 12 hours max
- **Features**: All core functionality
- **Best for**: Testing, small videos (< 15 min)

### **Colab Pro ($9.99/month):**
- **GPU**: A100, V100 (40GB+ VRAM)
- **Session**: 24 hours, priority access
- **Features**: Background execution
- **Best for**: Production use, long videos

### **Colab Pro+ ($49.99/month):**
- **GPU**: Premium access, longer sessions
- **Features**: Even more compute time
- **Best for**: Heavy production workloads

## 🚀 Advanced Usage

### **Batch Processing:**
```python
# Process multiple videos
video_files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
target_langs = ['es', 'fr', 'de']

for video in video_files:
    for lang in target_langs:
        result = pipeline.process_video(
            video_path=video,
            target_language=lang,
            output_path=f"dubbed_{video}_{lang}.mp4"
        )
```

### **Custom Voice Cloning:**
```python
# Upload reference voice sample
voice_uploaded = files.upload()
voice_path = list(voice_uploaded.keys())[0]

# Use in dubbing
result = pipeline.process_video(
    video_path=video_path,
    target_language="es",
    output_path=output_path,
    speaker_reference=voice_path
)
```

### **API Integration:**
```python
# Set up for API use
os.environ['OPENAI_API_KEY'] = 'your-key'

# Process via API
import requests
result = requests.post('http://localhost:5000/api/process', 
                      json={
                          'video_path': video_path,
                          'target_language': 'es'
                      })
```

## 📖 Additional Resources

- **📓 Colab Notebook**: [Video_Dubbing_Colab.ipynb](Video_Dubbing_Colab.ipynb)
- **🐙 GitHub Repo**: [github.com/yourusername/video-dubbing](https://github.com/yourusername/video-dubbing)
- **📚 Documentation**: [README.md](README.md)
- **🎥 Demo Videos**: [examples/](examples/)

## 🤝 Getting Help

1. **Check Issues**: Common problems and solutions
2. **Discussions**: Community Q&A
3. **Discord**: Real-time help
4. **Documentation**: Comprehensive guides

---

**Happy Dubbing on Colab!** 🎬✨

*Ready to start? Click the Colab badge at the top!*
