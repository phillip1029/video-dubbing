"""WhisperX ASR and alignment processor."""

import os
import torch
import whisperx
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..utils.config import WhisperXConfig


class WhisperXProcessor:
    """WhisperX processor for ASR and word-level alignment."""
    
    def __init__(self, config: WhisperXConfig):
        self.config = config
        self.device = self._get_device()
        self.compute_type = self._get_compute_type()
        
        self.model = None
        self.align_model = None
        self.metadata = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    def _get_compute_type(self) -> str:
        """Determine the best compute type for the device."""
        if self.device == "cpu":
            return "int8"
        return self.config.compute_type
    
    def load_model(self):
        """Load WhisperX model and alignment model."""
        try:
            self.logger.info(f"Loading WhisperX model: {self.config.model_size}")
            self.model = whisperx.load_model(
                self.config.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.logger.info("WhisperX model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load WhisperX model: {e}")
            raise
    
    def load_align_model(self, language_code: str):
        """Load alignment model for specific language."""
        try:
            self.logger.info(f"Loading alignment model for language: {language_code}")
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            self.logger.info("Alignment model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load alignment model: {e}")
            # Some languages might not have alignment models
            self.align_model = None
            self.metadata = None
    
    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio file using WhisperX."""
        if self.model is None:
            self.load_model()
        
        try:
            self.logger.info(f"Transcribing audio: {audio_path}")
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                chunk_size=self.config.chunk_size
            )
            
            self.logger.info("Transcription completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def align_transcript(self, result: Dict, audio_path: str, 
                        language_code: str) -> Dict:
        """Perform word-level alignment on transcript."""
        if self.align_model is None:
            self.load_align_model(language_code)
        
        if self.align_model is None:
            self.logger.warning(f"No alignment model available for {language_code}")
            return result
        
        try:
            self.logger.info("Performing word-level alignment")
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Perform alignment
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=self.config.return_char_alignments
            )
            
            self.logger.info("Alignment completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Alignment failed: {e}")
            return result  # Return original result if alignment fails
    
    def process_audio(self, audio_path: str, language_code: str = "auto") -> Dict:
        """Complete ASR pipeline: transcribe and align."""
        self.logger.info(f"Starting ASR processing for: {audio_path}")
        
        # Transcribe
        result = self.transcribe(audio_path)
        
        # Detect language if auto
        if language_code == "auto":
            language_code = result.get("language", "en")
            self.logger.info(f"Detected language: {language_code}")
        
        # Align
        aligned_result = self.align_transcript(result, audio_path, language_code)
        
        # Add metadata
        final_result = {
            "language": language_code,
            "segments": aligned_result.get("segments", result.get("segments", [])),
            "word_segments": aligned_result.get("word_segments", [])
        }
        
        self.logger.info("ASR processing completed")
        return final_result
    
    def extract_text_segments(self, result: Dict) -> List[Dict]:
        """Extract text segments with timing information."""
        segments = []
        
        for segment in result.get("segments", []):
            segment_data = {
                "id": segment.get("id", 0),
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
                "words": []
            }
            
            # Add word-level timing if available
            for word in segment.get("words", []):
                word_data = {
                    "word": word.get("word", ""),
                    "start": word.get("start", 0.0),
                    "end": word.get("end", 0.0),
                    "score": word.get("score", 0.0)
                }
                segment_data["words"].append(word_data)
            
            segments.append(segment_data)
        
        return segments
    
    def save_transcript(self, result: Dict, output_path: str):
        """Save transcript to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        transcript_data = {
            "language": result.get("language", "unknown"),
            "segments": self.extract_text_segments(result)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Transcript saved to: {output_path}")
    
    def load_transcript(self, transcript_path: str) -> Dict:
        """Load transcript from JSON file."""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_full_text(self, result: Dict) -> str:
        """Extract full text from transcript result."""
        segments = result.get("segments", [])
        return " ".join([segment.get("text", "").strip() for segment in segments])
    
    def cleanup(self):
        """Clean up loaded models to free memory."""
        self.model = None
        self.align_model = None
        self.metadata = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Models cleaned up")
