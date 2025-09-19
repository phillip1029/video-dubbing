"""ASR module for video dubbing application."""

from .base import BaseASRProcessor
from .processors import WhisperXProcessor, OpenAIWhisperAPIProcessor

__all__ = ["BaseASRProcessor", "WhisperXProcessor", "OpenAIWhisperAPIProcessor"]
