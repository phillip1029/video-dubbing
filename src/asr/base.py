from abc import ABC, abstractmethod
from typing import Dict, List
import json
import os
import logging

class BaseASRProcessor(ABC):
    """Base class for ASR processors."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process_audio(self, audio_path: str, language_code: str = "auto") -> Dict:
        """Transcribe audio and return segments with timestamps."""
        pass

    def get_full_text(self, result: Dict) -> str:
        """Extract full text from a transcript result."""
        if "text" in result:
            return result["text"]
        segments = result.get("segments", [])
        return " ".join([segment.get("text", "").strip() for segment in segments])

    def extract_text_segments(self, result: Dict) -> List[Dict]:
        """Extract text segments with timing and word-level details."""
        segments = []
        
        for i, segment_data in enumerate(result.get("segments", [])):
            segment = {
                "id": segment_data.get("id", i),
                "start": segment_data.get("start", 0.0),
                "end": segment_data.get("end", 0.0),
                "text": segment_data.get("text", "").strip(),
                "words": []
            }
            
            for word_data in segment_data.get("words", []):
                word = {
                    "word": word_data.get("word", ""),
                    "start": word_data.get("start", 0.0),
                    "end": word_data.get("end", 0.0),
                    "score": word_data.get("score", 0.0)
                }
                segment["words"].append(word)
            
            segments.append(segment)
        
        return segments

    def save_transcript(self, result: Dict, output_path: str):
        """Save transcript to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        transcript_data = {
            "language": result.get("language", "unknown"),
            "text": self.get_full_text(result),
            "segments": self.extract_text_segments(result)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Transcript saved to: {output_path}")

    def cleanup(self):
        """Clean up resources."""
        pass
