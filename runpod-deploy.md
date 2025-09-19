# RunPod Deployment Guide - Video Dubbing Application

This guide will help you deploy the Video Dubbing Application to [RunPod](https://console.runpod.io/) for high-performance GPU processing.

## 🚀 Quick Deployment Options

### Option 1: RunPod Template Deployment (Recommended)

1. **Visit RunPod Console**: Go to [https://console.runpod.io/](https://console.runpod.io/)

2. **Create New Pod**:
   - Click "Deploy" → "New Pod"
   - Choose GPU type (RTX 4090, A100, etc.)
   - Select "Custom" template

3. **Configure Container**:
   ```
   Container Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   Container Disk: 50GB minimum (for models and temp files)
   Volume Path: /workspace
   Expose HTTP Ports: 5000, 8080
   ```

4. **Environment Variables**:
   ```
   OPENAI_API_KEY=your-openai-api-key
   RUNPOD_POD_ID=${RUNPOD_POD_ID}
   RUNPOD_PUBLIC_IP=${RUNPOD_PUBLIC_IP}
   ```

5. **Startup Commands**:
   ```bash
   cd /workspace && \
   git clone https://github.com/yourusername/video-dubbing.git . && \
   pip install -r requirements.txt && \
   python runpod_web.py
   ```

### Option 2: Docker Image Deployment

1. **Build and Push Docker Image**:
   ```bash
   # Build the image
   docker build -t your-username/video-dubbing:latest .
   
   # Push to registry
   docker push your-username/video-dubbing:latest
   ```

2. **Deploy on RunPod**:
   - Use your custom image: `your-username/video-dubbing:latest`
   - Expose ports: 5000, 8080
   - Set environment variables as above

### Option 3: Serverless Deployment

1. **Deploy as Serverless Function**:
   ```bash
   # Use runpod_handler.py for serverless deployment
   runpod deploy --name video-dubbing-serverless
   ```

## 📋 Recommended RunPod Configurations

### For Development/Testing
- **GPU**: RTX 3080/4080 (10-16GB VRAM)
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **Cost**: ~$0.50-1.00/hour

### For Production
- **GPU**: RTX 4090/A100 (24GB+ VRAM)
- **CPU**: 16+ cores  
- **RAM**: 64GB+
- **Storage**: 100GB+ SSD
- **Cost**: ~$1.50-3.00/hour

### For Batch Processing
- **GPU**: Multiple A100s (40-80GB VRAM)
- **CPU**: 32+ cores
- **RAM**: 128GB+
- **Storage**: 200GB+ SSD
- **Cost**: ~$5.00-10.00/hour

## 🔧 Configuration Files

### Dockerfile
- Optimized for RunPod's PyTorch base image
- Includes all dependencies and GPU drivers
- Configures proper CUDA environment

### runpod_handler.py
- Serverless API interface
- Handles video download/upload
- GPU-optimized processing

### runpod_web.py
- Web interface configured for RunPod
- Health checks and system monitoring
- Proper networking setup

## 🌐 Access Your Application

Once deployed, access your application at:
```
http://[your-pod-ip]:5000
```

### Available Endpoints:
- `/` - Main web interface
- `/health` - Health check
- `/api/config` - Configuration info
- `/api/system/info` - System information

## 🛠️ Post-Deployment Setup

### 1. Set OpenAI API Key
```bash
# Via environment variable (recommended)
export OPENAI_API_KEY="your-api-key"

# Or update config.yaml
nano /workspace/config.yaml
```

### 2. Configure Storage
```bash
# Create directories
mkdir -p /workspace/{models,temp,output,uploads,logs}

# Set permissions
chmod 777 /workspace/{temp,output,uploads,logs}
```

### 3. Download Models (Optional)
Models are downloaded automatically on first use, but you can pre-download:
```bash
# Pre-download WhisperX models
python -c "import whisperx; whisperx.load_model('large-v3')"

# Pre-download Coqui TTS models
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

## 🔍 Monitoring and Debugging

### Check Logs
```bash
# Application logs
tail -f /workspace/logs/app.log

# Container logs
docker logs video-dubbing-runpod
```

### GPU Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi
```

### System Information
Visit: `http://[your-pod-ip]:5000/api/system/info`

## 📊 Performance Optimization

### GPU Memory Management
```python
# In config.yaml
whisperx:
  batch_size: 8  # Reduce if OOM
  compute_type: "float16"  # Use for memory efficiency

tts:
  device: "cuda"  # Ensure GPU usage
```

### Concurrent Processing
```python
# Enable concurrent processing
max_workers = 2  # Adjust based on GPU memory
```

## 🔒 Security Considerations

### Environment Variables
```bash
# Keep API keys secure
export OPENAI_API_KEY="sk-..."

# Use RunPod secrets for production
# https://docs.runpod.io/pods/configuration/environment-variables
```

### Network Security
- RunPod provides secure networking
- Use HTTPS in production
- Implement rate limiting for public APIs

## 💰 Cost Optimization

### Auto-Scaling
- Use RunPod's auto-scaling features
- Scale down during low usage
- Consider spot instances for batch processing

### Storage Management
```bash
# Clean up temp files regularly
find /workspace/temp -type f -mtime +1 -delete

# Monitor disk usage
df -h /workspace
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Out of Memory**:
   ```yaml
   # Reduce batch sizes in config.yaml
   whisperx:
     batch_size: 4
   ```

2. **Model Download Fails**:
   ```bash
   # Check internet connectivity
   curl -I https://huggingface.co
   
   # Manual download
   python install.py
   ```

3. **Port Access Issues**:
   ```bash
   # Ensure ports are exposed in RunPod
   # Check firewall settings
   ```

### Getting Help
- RunPod Documentation: [https://docs.runpod.io/](https://docs.runpod.io/)
- RunPod Discord: [https://discord.gg/runpod](https://discord.gg/runpod)
- GitHub Issues: Report application-specific issues

## 🎯 Next Steps

1. **Scale Up**: Add more GPUs for faster processing
2. **Automate**: Set up CI/CD for automatic deployments
3. **Monitor**: Implement comprehensive logging and monitoring
4. **Optimize**: Fine-tune for your specific use cases

---

**Ready to deploy?** Visit [RunPod Console](https://console.runpod.io/) and get started! 🚀
