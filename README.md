# Video Dubbing Application

A comprehensive AI-powered video dubbing application that translates videos into different languages using state-of-the-art AI models for speech recognition, translation, text-to-speech synthesis, and lip synchronization.

## 🎯 Features

- **🎤 High-Quality Speech Recognition**: WhisperX for accurate transcription with word-level alignment
- **🌐 AI-Powered Translation**: OpenAI GPT-4/GPT-5 for context-aware, high-quality translations with user review interface
- **🗣️ Natural Voice Synthesis**: Coqui XTTS for realistic text-to-speech with voice cloning
- **💋 Advanced Lip Sync**: MuseTalk for high-fidelity lip synchronization
- **⚡ Real-time Processing**: Live progress tracking and status updates
- **🖥️ User-Friendly Interface**: Both web interface and command-line options
- **🔄 Translation Review**: Manual review and editing of translations before synthesis

## 🏗️ Architecture

- **ASR + Alignment**: WhisperX for high-quality speech recognition and word-level alignment
- **Translation**: OpenAI GPT-4/GPT-5 with context-aware translation, plus Google Translate and HuggingFace models
- **TTS**: Coqui XTTS v2 for natural-sounding multilingual speech synthesis
- **Lip Sync**: MuseTalk for real-time, high-fidelity lip synchronization
- **Pipeline**: Orchestrated processing with progress tracking and error handling

## 🚀 Quick Start

### Automated Installation
```bash
# Run the installer (recommended)
python install.py

# Or with GPU support
python install.py --gpu
```

### OpenAI API Setup
For best translation quality, set your OpenAI API key:
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key"

# Option 2: Copy and edit environment file
cp env.example .env
# Edit .env file with your API key

# Option 3: Set in config.yaml
translation:
  service: "openai"
  api_key: "your-openai-api-key"
```

### Manual Installation
1. **System Requirements**:
   - Python 3.8+
   - FFmpeg
   - Git
   - CUDA (optional, for GPU acceleration)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Directories**:
   ```bash
   mkdir models temp output uploads
   ```

### Usage Options

#### 🖥️ Web Interface (Recommended)
```bash
python app.py
```
Then visit `http://localhost:5000`

#### 💻 Command Line Interface
```bash
# Basic usage
python main.py process -i video.mp4 -l es -o dubbed_video.mp4

# With voice cloning
python main.py process -i video.mp4 -l es -s reference_voice.wav

# See all options
python main.py --help
```

## 🌍 Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| en   | English  | es   | Spanish  | fr   | French   |
| de   | German   | it   | Italian  | pt   | Portuguese |
| ru   | Russian  | ja   | Japanese | ko   | Korean   |
| zh   | Chinese  | ar   | Arabic   | hi   | Hindi    |
| nl   | Dutch    | pl   | Polish   | tr   | Turkish  |
| cs   | Czech    | hu   | Hungarian |      |          |

## 📋 Processing Pipeline

1. **📹 Video Analysis**: Extract audio and analyze video properties
2. **🎤 Speech Recognition**: Transcribe audio using WhisperX with word-level timing
3. **🌐 Translation**: Translate text with optional human review
4. **✅ Review & Approval**: Edit and approve translations via web interface
5. **🗣️ Speech Synthesis**: Generate natural speech using Coqui XTTS
6. **📏 Audio Alignment**: Adjust timing to match original video duration
7. **💋 Lip Synchronization**: Sync lip movements using MuseTalk
8. **🎬 Final Output**: Combine everything into the final dubbed video

## 🛠️ Configuration

Edit `config.yaml` to customize:
```yaml
# WhisperX settings
whisperx:
  model_size: "large-v3"
  device: "auto"

# TTS settings  
tts:
  model_name: "tts_models/multilingual/multi-dataset/xtts_v2"
  device: "auto"

# Translation settings
translation:
  service: "openai"          # openai (recommended), google, huggingface
  model_name: "gpt-4o"       # gpt-4o, gpt-4, gpt-3.5-turbo
  api_key: "your-api-key"    # or set OPENAI_API_KEY env var
  max_chunk_size: 1000
```

## 📁 Project Structure

```
video-dubbing/
├── src/                    # Main application code
│   ├── asr/               # WhisperX speech recognition
│   ├── translation/       # Translation services
│   ├── tts/              # Coqui TTS integration
│   ├── lip_sync/         # MuseTalk lip synchronization
│   ├── pipeline/         # Main processing pipeline
│   ├── web/              # Flask web interface
│   └── utils/            # Utilities and configuration
├── models/               # Downloaded AI models
├── temp/                 # Temporary processing files
├── output/               # Generated dubbed videos
├── uploads/              # Uploaded source videos
├── main.py              # CLI entry point
├── app.py               # Web interface entry point
├── install.py           # Installation script
├── config.yaml          # Configuration file
└── requirements.txt     # Python dependencies
```

## 🔧 Advanced Usage

### Voice Cloning
Provide a reference audio sample (10-30 seconds) to clone a specific voice:
```bash
python main.py process -i video.mp4 -l es -s target_voice_sample.wav
```

### Batch Processing
Process multiple videos programmatically:
```python
from src.pipeline import VideoDubbingPipeline
from src.utils.config import load_config

config = load_config()
pipeline = VideoDubbingPipeline(config)

result = pipeline.process_video(
    video_path="input.mp4",
    target_language="es", 
    output_path="output.mp4",
    speaker_reference="voice.wav"
)
```

### Translation Quality Options

**🥇 OpenAI GPT (Recommended)**
- **Models**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Quality**: Highest quality, context-aware translations
- **Features**: Maintains tone, cultural context, natural speech patterns
- **Cost**: Pay-per-use API (requires OpenAI API key)

**🥈 Google Translate**
- **Quality**: Good general translations
- **Features**: Fast, supports many languages
- **Cost**: Free (with rate limits)

**🥉 HuggingFace Models**
- **Models**: mBART, T5, custom models
- **Quality**: Good for supported language pairs
- **Features**: Offline capability, no API costs
- **Requirements**: Higher GPU memory usage

### Custom Translation Review
The web interface allows manual review and editing of translations:
1. Upload video and select target language
2. Wait for transcription and AI translation
3. Review and edit translations in the web interface
4. Approve and continue processing

## 🎛️ Web Interface Features

- **📤 Drag & Drop Upload**: Easy video file uploading
- **📊 Real-time Progress**: Live processing status and progress bars
- **✏️ Translation Editor**: Review and edit translations before synthesis
- **🎵 Voice Cloning**: Upload reference audio for voice cloning
- **📱 Responsive Design**: Works on desktop and mobile devices
- **🔄 WebSocket Updates**: Real-time status updates via WebSocket

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```yaml
   # In config.yaml, reduce batch sizes:
   whisperx:
     batch_size: 8
   ```

2. **FFmpeg Not Found**:
   - Windows: Download from https://ffmpeg.org/
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

3. **Model Download Fails**:
   - Check internet connection
   - Models are downloaded automatically on first use
   - Large models may take time to download

### Performance Tips

- **GPU Acceleration**: Use CUDA-enabled PyTorch for faster processing
- **Model Size**: Use smaller WhisperX models for faster processing
- **Batch Size**: Adjust batch sizes based on available memory
- **CPU Cores**: Set appropriate worker counts for your system

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- **WhisperX**: For excellent speech recognition and alignment
- **Coqui TTS**: For high-quality multilingual text-to-speech
- **MuseTalk**: For advanced lip synchronization technology
- **OpenAI Whisper**: For the foundation of speech recognition
- **HuggingFace**: For translation models and infrastructure

## 📞 Support

- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Join GitHub Discussions for questions
- 📧 **Contact**: Email for commercial support

---

**Note**: This application requires significant computational resources. GPU acceleration is highly recommended for optimal performance.
