"""Configuration management for video dubbing application."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml


@dataclass
class WhisperXConfig:
    """Configuration for WhisperX ASR."""
    model_size: str = "large-v3"
    device: str = "auto"  # auto, cuda, cpu
    compute_type: str = "float16"  # float16, int8, float32
    batch_size: int = 16
    chunk_size: int = 30
    return_char_alignments: bool = True


@dataclass
class TranslationConfig:
    """Configuration for translation service."""
    service: str = "google"  # google, openai, huggingface
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    max_chunk_size: int = 1000


@dataclass
class TTSConfig:
    """Configuration for Coqui TTS."""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "auto"
    speaker_wav: Optional[str] = None
    language: str = "en"
    speed: float = 1.0


@dataclass
class MuseTalkConfig:
    """Configuration for MuseTalk lip sync."""
    model_path: str = "models/musetalk"
    device: str = "auto"
    fps: int = 25
    resolution: tuple = (512, 512)


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    temp_dir: str = "temp"
    output_dir: str = "output"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"


@dataclass
class AppConfig:
    """Main application configuration."""
    whisperx: WhisperXConfig = WhisperXConfig()
    translation: TranslationConfig = TranslationConfig()
    tts: TTSConfig = TTSConfig()
    musetalk: MuseTalkConfig = MuseTalkConfig()
    video: VideoConfig = VideoConfig()
    
    supported_languages: Dict[str, str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = {
                "en": "English",
                "es": "Spanish", 
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
                "ar": "Arabic",
                "hi": "Hindi"
            }


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from file or use defaults."""
    config = AppConfig()
    
    # Load from YAML file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Update config with YAML values
        if 'translation' in config_dict:
            trans_config = config_dict['translation']
            config.translation.service = trans_config.get('service', config.translation.service)
            config.translation.model_name = trans_config.get('model_name', config.translation.model_name)
            config.translation.api_key = trans_config.get('api_key', config.translation.api_key)
            config.translation.max_chunk_size = trans_config.get('max_chunk_size', config.translation.max_chunk_size)
        
        if 'whisperx' in config_dict:
            whisper_config = config_dict['whisperx']
            config.whisperx.model_size = whisper_config.get('model_size', config.whisperx.model_size)
            config.whisperx.device = whisper_config.get('device', config.whisperx.device)
            config.whisperx.compute_type = whisper_config.get('compute_type', config.whisperx.compute_type)
            config.whisperx.batch_size = whisper_config.get('batch_size', config.whisperx.batch_size)
        
        if 'tts' in config_dict:
            tts_config = config_dict['tts']
            config.tts.model_name = tts_config.get('model_name', config.tts.model_name)
            config.tts.device = tts_config.get('device', config.tts.device)
            config.tts.language = tts_config.get('language', config.tts.language)
            config.tts.speed = tts_config.get('speed', config.tts.speed)
    
    # Override with environment variables if available
    if os.getenv('OPENAI_API_KEY'):
        config.translation.api_key = os.getenv('OPENAI_API_KEY')
    
    return config


def save_config(config: AppConfig, config_path: str):
    """Save configuration to file."""
    # Convert to dict for YAML serialization
    config_dict = {
        "whisperx": {
            "model_size": config.whisperx.model_size,
            "device": config.whisperx.device,
            "compute_type": config.whisperx.compute_type,
            "batch_size": config.whisperx.batch_size,
        },
        "translation": {
            "service": config.translation.service,
            "model_name": config.translation.model_name,
        },
        "tts": {
            "model_name": config.tts.model_name,
            "device": config.tts.device,
            "language": config.tts.language,
        },
        "musetalk": {
            "model_path": config.musetalk.model_path,
            "device": config.musetalk.device,
            "fps": config.musetalk.fps,
        }
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
