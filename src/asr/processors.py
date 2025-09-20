import os
import torch
import whisperx
import logging
from typing import Dict, List, Optional
import json

from ..utils.config import ASRConfig
from .base import BaseASRProcessor

class WhisperXProcessor(BaseASRProcessor):
    """WhisperX processor for ASR and word-level alignment."""
    
    def __init__(self, config: ASRConfig):
        super().__init__()
        self.config = config
        self.device = self._get_device()
        self.compute_type = self._get_compute_type()
        
        self.model = None
        self.align_model = None
        self.metadata = None
    
    def _get_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    def _get_compute_type(self) -> str:
        if self.device == "cpu":
            return "int8"
        return self.config.compute_type
    
    def load_model(self):
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
        try:
            self.logger.info(f"Loading alignment model for language: {language_code}")
            self.align_model, self.metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            self.logger.info("Alignment model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load alignment model: {e}")
            self.align_model = None
            self.metadata = None
    
    def transcribe(self, audio_path: str) -> Dict:
        if self.model is None:
            self.load_model()
        
        try:
            self.logger.info(f"Transcribing audio: {audio_path}")
            audio = whisperx.load_audio(audio_path)
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
    
    def align_transcript(self, result: Dict, audio_path: str, language_code: str) -> Dict:
        current_lang = self.metadata.get("language") if self.metadata else None
        if self.align_model is None or (current_lang != language_code):
            self.load_align_model(language_code)
        
        if self.align_model is None:
            self.logger.warning(f"No alignment model for {language_code}. Skipping alignment.")
            return result
        
        try:
            self.logger.info("Performing word-level alignment")
            audio = whisperx.load_audio(audio_path)
            aligned_result = whisperx.align(
                result["segments"],
                self.align_model,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=self.config.return_char_alignments
            )
            self.logger.info("Alignment completed")
            return aligned_result
            
        except Exception as e:
            self.logger.error(f"Alignment failed: {e}")
            return result
    
    def process_audio(self, audio_path: str, language_code: str = "auto") -> Dict:
        self.logger.info(f"Starting WhisperX ASR processing for: {audio_path}")
        
        result = self.transcribe(audio_path)
        
        detected_language = result.get("language", "en")
        if language_code == "auto":
            language_code = detected_language
            self.logger.info(f"Detected language: {language_code}")
        
        aligned_result = self.align_transcript(result, audio_path, language_code)
        
        # Ensure 'segments' key exists even if alignment fails
        final_segments = aligned_result.get("segments", result.get("segments", []))
        
        return {
            "language": language_code,
            "text": self.get_full_text(aligned_result),
            "segments": final_segments,
            "word_segments": aligned_result.get("word_segments", [])
        }

    def cleanup(self):
        self.model = None
        self.align_model = None
        self.metadata = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("WhisperX models cleaned up")

try:
    import openai
except ImportError:
    openai = None

class OpenAIWhisperAPIProcessor(BaseASRProcessor):
    """ASR processor using OpenAI's Whisper API."""

    def __init__(self, config: ASRConfig):
        super().__init__()
        self.config = config
        if openai is None:
            raise ImportError("OpenAI package required. 'pip install openai'")
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required for Whisper API.")
        
        self.client = openai.OpenAI(api_key=self.config.api_key)

    def process_audio(self, audio_path: str, language_code: str = "auto") -> Dict:
        self.logger.info(f"Transcribing with OpenAI Whisper API: {audio_path}")
        try:
            with open(audio_path, "rb") as audio_file:
                transcript_obj = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                    language=language_code if language_code != "auto" else None,
                )
            
            transcript = transcript_obj.to_dict()
            self.logger.info("Whisper API transcription successful.")
            return self._format_result(transcript)

        except Exception as e:
            self.logger.error(f"Whisper API transcription failed: {e}")
            raise

    def _format_result(self, result: Dict) -> Dict:
        formatted_segments = []
        for segment in result.get("segments", []):
            formatted_segment = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "words": []
            }
            formatted_segments.append(formatted_segment)

        word_idx = 0
        for segment in formatted_segments:
            while word_idx < len(result.get("words", [])):
                word = result.get("words", [])[word_idx]
                if word["start"] >= segment["start"] and word["end"] <= segment["end"]:
                    segment["words"].append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "score": -1.0
                    })
                    word_idx += 1
                else:
                    break
        
        return {
            "language": result.get("language", "en"),
            "text": result.get("text", ""),
            "segments": formatted_segments,
            "word_segments": result.get("words", [])
        }
