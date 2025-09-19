# Video Dubbing Application - RunPod Deployment
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for RunPod
RUN pip install --no-cache-dir \
    runpod \
    python-dotenv \
    gunicorn \
    eventlet

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models temp output uploads logs

# Set environment variables
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
ENV CUDA_VISIBLE_DEVICES="0"

# Expose ports
EXPOSE 5000 8080

# Create startup script
RUN echo '#!/bin/bash\n\
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}\n\
export RUNPOD_POD_ID=${RUNPOD_POD_ID:-local}\n\
export RUNPOD_PUBLIC_IP=${RUNPOD_PUBLIC_IP:-localhost}\n\
\n\
# Start the web application\n\
echo "Starting Video Dubbing Application on RunPod..."\n\
echo "Pod ID: $RUNPOD_POD_ID"\n\
echo "Public IP: $RUNPOD_PUBLIC_IP"\n\
echo "Access URL: http://$RUNPOD_PUBLIC_IP:5000"\n\
\n\
# Run the application\n\
if [ "$1" = "web" ]; then\n\
    python app.py\n\
elif [ "$1" = "api" ]; then\n\
    python runpod_handler.py\n\
else\n\
    python app.py\n\
fi' > /workspace/start.sh

RUN chmod +x /workspace/start.sh

# Default command
CMD ["/workspace/start.sh", "web"]
