"""Translation services for video dubbing application."""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import json

# Translation services
try:
    from googletrans import Translator as GoogleTranslator
except ImportError:
    GoogleTranslator = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    pipeline = None

try:
    import openai
except ImportError:
    openai = None

from ..utils.config import TranslationConfig


class BaseTranslator(ABC):
    """Base class for translation services."""
    
    @abstractmethod
    def translate(self, text: str, target_language: str, 
                 source_language: str = "auto") -> str:
        """Translate text to target language."""
        pass
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        pass


class GoogleTranslationService(BaseTranslator):
    """Google Translate service implementation."""
    
    def __init__(self):
        if GoogleTranslator is None:
            raise ImportError("googletrans package is required for Google translation")
        
        self.translator = GoogleTranslator()
        self.logger = logging.getLogger(__name__)
    
    def translate(self, text: str, target_language: str, 
                 source_language: str = "auto") -> str:
        """Translate text using Google Translate."""
        try:
            result = self.translator.translate(
                text, 
                dest=target_language, 
                src=source_language
            )
            return result.text
        except Exception as e:
            self.logger.error(f"Google translation failed: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Detect language using Google Translate."""
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown"


class OpenAITranslationService(BaseTranslator):
    """OpenAI GPT-based translation service."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if openai is None:
            raise ImportError("openai package is required for OpenAI translation")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Language name mapping for better prompts
        self.language_names = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese (Simplified)",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "cs": "Czech",
            "hu": "Hungarian"
        }
    
    def translate(self, text: str, target_language: str, 
                 source_language: str = "auto") -> str:
        """Translate text using OpenAI GPT models."""
        try:
            target_lang_name = self.language_names.get(target_language, target_language)
            source_lang_name = self.language_names.get(source_language, "the source language")
            
            # Create context-aware prompt for video dubbing
            if source_language == "auto":
                prompt = f"""Translate the following text to {target_lang_name}.

This is from a video transcript, so:
- Maintain natural speaking tone and rhythm
- Keep cultural context appropriate for the target language
- Preserve emotional tone and intent
- Use conversational language suitable for dubbing
- If there are names or proper nouns, keep them unless there's a standard translation

Text to translate:
"{text}"

Translation:"""
            else:
                prompt = f"""Translate the following {source_lang_name} text to {target_lang_name}.

This is from a video transcript, so:
- Maintain natural speaking tone and rhythm
- Keep cultural context appropriate for the target language
- Preserve emotional tone and intent
- Use conversational language suitable for dubbing
- If there are names or proper nouns, keep them unless there's a standard translation

Text to translate:
"{text}"

Translation:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional translator specializing in video dubbing and localization. You provide high-quality, natural translations that maintain the original tone and are suitable for voice acting."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=2000
            )
            
            translation = response.choices[0].message.content.strip()
            
            # Remove any quotation marks that might be added
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]
            
            return translation
            
        except Exception as e:
            self.logger.error(f"OpenAI translation failed: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Detect language using OpenAI GPT."""
        try:
            prompt = f"""Detect the language of the following text and respond with only the ISO 639-1 language code (e.g., 'en', 'es', 'fr'):

Text: "{text}"

Language code:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a language detection expert. Respond only with the ISO 639-1 language code."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            detected_lang = response.choices[0].message.content.strip().lower()
            
            # Validate the response is a valid language code
            if detected_lang in self.language_names:
                return detected_lang
            else:
                self.logger.warning(f"Invalid language code detected: {detected_lang}")
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown"
    
    def translate_with_context(self, segments: List[Dict], target_language: str,
                              source_language: str = "auto") -> List[str]:
        """Translate multiple segments with context awareness."""
        try:
            target_lang_name = self.language_names.get(target_language, target_language)
            
            # Prepare context for better translation
            full_context = " ".join([seg.get("text", "") for seg in segments[:5]])  # Use first 5 segments for context
            
            translations = []
            
            for i, segment in enumerate(segments):
                text = segment.get("text", "").strip()
                if not text:
                    translations.append("")
                    continue
                
                # Include previous segment for context if available
                context_prompt = ""
                if i > 0 and translations:
                    prev_original = segments[i-1].get("text", "")
                    prev_translation = translations[-1]
                    context_prompt = f"""

Previous segment for context:
Original: "{prev_original}"
Translation: "{prev_translation}"
"""
                
                prompt = f"""Translate the following text to {target_lang_name}.

This is segment {i+1} from a video transcript. {context_prompt}

Current segment to translate:
"{text}"

Guidelines:
- Maintain natural speaking tone suitable for dubbing
- Keep consistent with previous translations
- Preserve emotional tone and intent
- Use conversational language
- Keep names and proper nouns unless there's a standard translation

Translation:"""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional video dubbing translator. Provide natural, contextually aware translations suitable for voice acting."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                translation = response.choices[0].message.content.strip()
                
                # Clean up the translation
                if translation.startswith('"') and translation.endswith('"'):
                    translation = translation[1:-1]
                
                translations.append(translation)
            
            return translations
            
        except Exception as e:
            self.logger.error(f"Context-aware translation failed: {e}")
            raise


class HuggingFaceTranslationService(BaseTranslator):
    """HuggingFace transformer models for translation."""
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        if pipeline is None:
            raise ImportError("transformers package is required for HuggingFace translation")
        
        self.model_name = model_name
        self.translator = None
        self.logger = logging.getLogger(__name__)
        
        # Language mapping for mBART
        self.lang_mapping = {
            "en": "en_XX",
            "es": "es_XX", 
            "fr": "fr_XX",
            "de": "de_DE",
            "it": "it_IT",
            "pt": "pt_XX",
            "ru": "ru_RU",
            "ja": "ja_XX",
            "ko": "ko_KR",
            "zh": "zh_CN",
            "ar": "ar_AR",
            "hi": "hi_IN"
        }
    
    def _load_model(self):
        """Load the translation model."""
        if self.translator is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1
                
            self.logger.info(f"Loading HuggingFace model: {self.model_name}")
            self.translator = pipeline(
                "translation",
                model=self.model_name,
                device=device
            )
    
    def translate(self, text: str, target_language: str, 
                 source_language: str = "auto") -> str:
        """Translate text using HuggingFace model."""
        self._load_model()
        
        try:
            # Map language codes for mBART
            src_lang = self.lang_mapping.get(source_language, source_language)
            tgt_lang = self.lang_mapping.get(target_language, target_language)
            
            result = self.translator(
                text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=512
            )
            
            return result[0]['translation_text']
        except Exception as e:
            self.logger.error(f"HuggingFace translation failed: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Language detection not implemented for HuggingFace."""
        return "unknown"


class TranslationReviewManager:
    """Manages translation review and approval process."""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_review_session(self, segments: List[Dict], 
                             session_id: str) -> str:
        """Create a review session with original and translated segments."""
        review_data = {
            "session_id": session_id,
            "created_at": time.time(),
            "status": "pending_review",
            "segments": []
        }
        
        for i, segment in enumerate(segments):
            review_segment = {
                "id": i,
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "original_text": segment.get("text", ""),
                "translated_text": segment.get("translated_text", ""),
                "approved": False,
                "edited": False,
                "comments": ""
            }
            review_data["segments"].append(review_segment)
        
        # Save review session
        review_path = os.path.join(self.temp_dir, f"review_{session_id}.json")
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Review session created: {review_path}")
        return review_path
    
    def load_review_session(self, session_id: str) -> Dict:
        """Load a review session."""
        review_path = os.path.join(self.temp_dir, f"review_{session_id}.json")
        with open(review_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_review_session(self, review_data: Dict):
        """Save review session changes."""
        session_id = review_data["session_id"]
        review_path = os.path.join(self.temp_dir, f"review_{session_id}.json")
        
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)
    
    def update_segment_translation(self, session_id: str, segment_id: int,
                                  new_translation: str, approved: bool = False):
        """Update a segment's translation."""
        review_data = self.load_review_session(session_id)
        
        if 0 <= segment_id < len(review_data["segments"]):
            segment = review_data["segments"][segment_id]
            segment["translated_text"] = new_translation
            segment["approved"] = approved
            segment["edited"] = True
            
            self.save_review_session(review_data)
            self.logger.info(f"Updated segment {segment_id} in session {session_id}")
    
    def approve_segment(self, session_id: str, segment_id: int):
        """Approve a segment translation."""
        review_data = self.load_review_session(session_id)
        
        if 0 <= segment_id < len(review_data["segments"]):
            review_data["segments"][segment_id]["approved"] = True
            self.save_review_session(review_data)
    
    def approve_all_segments(self, session_id: str):
        """Approve all segments in a session."""
        review_data = self.load_review_session(session_id)
        
        for segment in review_data["segments"]:
            segment["approved"] = True
        
        review_data["status"] = "approved"
        self.save_review_session(review_data)
        self.logger.info(f"All segments approved for session {session_id}")
    
    def get_approved_segments(self, session_id: str) -> List[Dict]:
        """Get all approved segments."""
        review_data = self.load_review_session(session_id)
        
        approved_segments = []
        for segment in review_data["segments"]:
            if segment["approved"]:
                approved_segments.append({
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["translated_text"],
                    "original_text": segment["original_text"]
                })
        
        return approved_segments


class TranslationProcessor:
    """Main translation processor with review capabilities."""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.translator = self._create_translator()
        self.review_manager = TranslationReviewManager()
        self.logger = logging.getLogger(__name__)
    
    def _create_translator(self) -> BaseTranslator:
        """Create translator based on configuration."""
        if self.config.service == "openai":
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required for OpenAI translation service")
            model_name = self.config.model_name or "gpt-4o"
            return OpenAITranslationService(self.config.api_key, model_name)
        elif self.config.service == "google":
            return GoogleTranslationService()
        elif self.config.service == "huggingface":
            model_name = self.config.model_name or "facebook/mbart-large-50-many-to-many-mmt"
            return HuggingFaceTranslationService(model_name)
        else:
            raise ValueError(f"Unsupported translation service: {self.config.service}")
    
    def translate_segments(self, segments: List[Dict], target_language: str,
                          source_language: str = "auto") -> List[Dict]:
        """Translate transcript segments."""
        self.logger.info(f"Translating {len(segments)} segments to {target_language}")
        
        # Use context-aware translation for OpenAI
        if isinstance(self.translator, OpenAITranslationService):
            return self._translate_segments_with_context(segments, target_language, source_language)
        
        # Regular translation for other services
        translated_segments = []
        
        for segment in segments:
            original_text = segment.get("text", "").strip()
            if not original_text:
                continue
            
            try:
                # Split long text into chunks if necessary
                if len(original_text) > self.config.max_chunk_size:
                    chunks = self._split_text(original_text)
                    translated_chunks = []
                    
                    for chunk in chunks:
                        translated_chunk = self.translator.translate(
                            chunk, target_language, source_language
                        )
                        translated_chunks.append(translated_chunk)
                    
                    translated_text = " ".join(translated_chunks)
                else:
                    translated_text = self.translator.translate(
                        original_text, target_language, source_language
                    )
                
                translated_segment = segment.copy()
                translated_segment["translated_text"] = translated_text
                translated_segments.append(translated_segment)
                
            except Exception as e:
                self.logger.error(f"Failed to translate segment: {e}")
                # Keep original text if translation fails
                translated_segment = segment.copy()
                translated_segment["translated_text"] = original_text
                translated_segments.append(translated_segment)
        
        self.logger.info("Translation completed")
        return translated_segments
    
    def _translate_segments_with_context(self, segments: List[Dict], target_language: str,
                                        source_language: str = "auto") -> List[Dict]:
        """Translate segments using OpenAI's context-aware translation."""
        try:
            self.logger.info("Using context-aware OpenAI translation")
            
            # Extract text segments for context-aware translation
            text_segments = [segment.get("text", "").strip() for segment in segments if segment.get("text", "").strip()]
            
            if not text_segments:
                return segments
            
            # Get context-aware translations
            translations = self.translator.translate_with_context(
                segments, target_language, source_language
            )
            
            # Combine translations with original segments
            translated_segments = []
            translation_idx = 0
            
            for segment in segments:
                original_text = segment.get("text", "").strip()
                if not original_text:
                    translated_segments.append(segment)
                    continue
                
                translated_segment = segment.copy()
                if translation_idx < len(translations):
                    translated_segment["translated_text"] = translations[translation_idx]
                    translation_idx += 1
                else:
                    translated_segment["translated_text"] = original_text
                
                translated_segments.append(translated_segment)
            
            self.logger.info("Context-aware translation completed")
            return translated_segments
            
        except Exception as e:
            self.logger.error(f"Context-aware translation failed, falling back to regular translation: {e}")
            # Fallback to regular translation
            return self.translate_segments(segments, target_language, source_language)
    
    def _split_text(self, text: str) -> List[str]:
        """Split long text into smaller chunks."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < self.config.max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_translation_review(self, segments: List[Dict], 
                                 target_language: str,
                                 session_id: str) -> str:
        """Create translation and review session."""
        # Translate segments
        translated_segments = self.translate_segments(segments, target_language)
        
        # Create review session
        review_path = self.review_manager.create_review_session(
            translated_segments, session_id
        )
        
        return review_path
