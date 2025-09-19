"""Coqui XTTS text-to-speech processor."""

import os
import torch
import logging
import tempfile
from typing import Dict, List, Optional, Union
import numpy as np
import soundfile as sf
from pathlib import Path

try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except ImportError:
    TTS = None
    Xtts = None
    XttsConfig = None

from ..utils.config import TTSConfig


class CoquiTTSProcessor:
    """Coqui XTTS processor for text-to-speech generation."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.device = self._get_device()
        
        self.tts = None
        self.model = None
        self.speaker_embedding = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Supported languages for XTTS v2
        self.supported_languages = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "pl": "Polish",
            "tr": "Turkish",
            "ru": "Russian",
            "nl": "Dutch",
            "cs": "Czech",
            "ar": "Arabic",
            "zh": "Chinese",
            "ja": "Japanese",
            "hu": "Hungarian",
            "ko": "Korean"
        }
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    def load_model(self):
        """Load Coqui TTS model."""
        if TTS is None:
            raise ImportError("TTS package is required. Install with: pip install TTS")
        
        try:
            self.logger.info(f"Loading Coqui TTS model: {self.config.model_name}")
            
            # Initialize TTS
            self.tts = TTS(
                model_name=self.config.model_name,
                progress_bar=True,
                gpu=self.device == "cuda"
            )
            
            self.logger.info("Coqui TTS model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Coqui TTS model: {e}")
            raise
    
    def prepare_speaker_embedding(self, speaker_wav_path: str):
        """Prepare speaker embedding from reference audio."""
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker reference file not found: {speaker_wav_path}")
        
        try:
            self.logger.info(f"Preparing speaker embedding from: {speaker_wav_path}")
            
            # The speaker embedding will be computed automatically by XTTS
            # when we provide the speaker_wav parameter
            self.config.speaker_wav = speaker_wav_path
            
            self.logger.info("Speaker embedding prepared")
            
        except Exception as e:
            self.logger.error(f"Failed to prepare speaker embedding: {e}")
            raise
    
    def synthesize_text(self, text: str, output_path: str, 
                       language: Optional[str] = None,
                       speaker_wav: Optional[str] = None) -> str:
        """Synthesize speech from text."""
        if self.tts is None:
            self.load_model()
        
        # Use provided parameters or fall back to config
        lang = language or self.config.language
        speaker_ref = speaker_wav or self.config.speaker_wav
        
        if lang not in self.supported_languages:
            raise ValueError(f"Language '{lang}' not supported. Supported: {list(self.supported_languages.keys())}")
        
        try:
            self.logger.info(f"Synthesizing text in {lang}: {text[:50]}...")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Synthesize using XTTS
            if speaker_ref and os.path.exists(speaker_ref):
                # Clone voice from reference
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_ref,
                    language=lang,
                    split_sentences=True
                )
            else:
                # Use default speaker
                self.logger.warning("No speaker reference provided, using default speaker")
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=lang,
                    split_sentences=True
                )
            
            self.logger.info(f"Speech synthesized successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def synthesize_segments(self, segments: List[Dict], output_dir: str,
                           language: str, speaker_wav: Optional[str] = None) -> List[Dict]:
        """Synthesize speech for multiple text segments."""
        self.logger.info(f"Synthesizing {len(segments)} segments")
        
        os.makedirs(output_dir, exist_ok=True)
        synthesized_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            try:
                # Generate output filename
                segment_id = segment.get("id", i)
                output_filename = f"segment_{segment_id:04d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # Synthesize this segment
                audio_path = self.synthesize_text(
                    text=text,
                    output_path=output_path,
                    language=language,
                    speaker_wav=speaker_wav
                )
                
                # Create segment with audio info
                synthesized_segment = segment.copy()
                synthesized_segment["audio_path"] = audio_path
                synthesized_segment["synthesized"] = True
                
                # Get audio duration
                try:
                    import librosa
                    audio_duration = librosa.get_duration(filename=audio_path)
                    synthesized_segment["audio_duration"] = audio_duration
                except ImportError:
                    self.logger.warning("librosa not available for duration calculation")
                
                synthesized_segments.append(synthesized_segment)
                
            except Exception as e:
                self.logger.error(f"Failed to synthesize segment {i}: {e}")
                # Keep segment without audio
                failed_segment = segment.copy()
                failed_segment["synthesized"] = False
                failed_segment["error"] = str(e)
                synthesized_segments.append(failed_segment)
        
        self.logger.info(f"Synthesis completed for {len(synthesized_segments)} segments")
        return synthesized_segments
    
    def concatenate_audio_segments(self, segments: List[Dict], 
                                  output_path: str, 
                                  silence_duration: float = 0.1) -> str:
        """Concatenate synthesized audio segments with timing."""
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            raise ImportError("librosa and soundfile are required for audio concatenation")
        
        self.logger.info("Concatenating audio segments")
        
        # Collect all audio data
        audio_data = []
        sample_rate = None
        
        for segment in segments:
            if not segment.get("synthesized", False):
                continue
            
            audio_path = segment.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                continue
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if needed
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            audio_data.append(audio)
            
            # Add silence between segments
            if len(audio_data) > 1:
                silence_samples = int(silence_duration * sample_rate)
                silence = np.zeros(silence_samples)
                audio_data.append(silence)
        
        if not audio_data:
            raise ValueError("No audio data to concatenate")
        
        # Concatenate all audio
        final_audio = np.concatenate(audio_data)
        
        # Save concatenated audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, final_audio, sample_rate)
        
        self.logger.info(f"Audio concatenated successfully: {output_path}")
        return output_path
    
    def adjust_speech_rate(self, audio_path: str, target_duration: float,
                          output_path: str) -> str:
        """Adjust speech rate to match target duration."""
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            raise ImportError("librosa and soundfile are required for speech rate adjustment")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        current_duration = len(audio) / sr
        
        # Calculate rate adjustment
        rate = current_duration / target_duration
        
        if abs(rate - 1.0) > 0.05:  # Only adjust if difference is significant
            self.logger.info(f"Adjusting speech rate by factor {rate:.2f}")
            
            # Time-stretch audio
            audio_adjusted = librosa.effects.time_stretch(audio, rate=rate)
            
            # Save adjusted audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio_adjusted, sr)
            
            self.logger.info(f"Speech rate adjusted: {output_path}")
            return output_path
        else:
            # No significant adjustment needed
            if audio_path != output_path:
                import shutil
                shutil.copy2(audio_path, output_path)
            return output_path
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def cleanup(self):
        """Clean up loaded models to free memory."""
        self.tts = None
        self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("TTS models cleaned up")


class TTSBatchProcessor:
    """Batch processor for handling multiple TTS jobs."""
    
    def __init__(self, config: TTSConfig, max_workers: int = 2):
        self.config = config
        self.max_workers = max_workers
        self.processor = CoquiTTSProcessor(config)
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, text_segments: List[str], output_dir: str,
                     language: str, speaker_wav: Optional[str] = None) -> List[str]:
        """Process multiple text segments in batch."""
        # For now, process sequentially to avoid memory issues
        # In the future, this could be parallelized with careful memory management
        
        audio_paths = []
        
        for i, text in enumerate(text_segments):
            output_path = os.path.join(output_dir, f"batch_{i:04d}.wav")
            
            try:
                audio_path = self.processor.synthesize_text(
                    text=text,
                    output_path=output_path,
                    language=language,
                    speaker_wav=speaker_wav
                )
                audio_paths.append(audio_path)
            except Exception as e:
                self.logger.error(f"Failed to process batch item {i}: {e}")
                audio_paths.append(None)
        
        return audio_paths
