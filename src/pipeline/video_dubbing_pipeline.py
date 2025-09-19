"""Main video dubbing pipeline orchestrating all components."""

import os
import time
import uuid
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..utils.config import AppConfig
from ..utils.video_utils import VideoProcessor
from ..asr import WhisperXProcessor, OpenAIWhisperAPIProcessor
from ..translation import TranslationProcessor, TranslationReviewManager
from ..tts import CoquiTTSProcessor
from ..lip_sync import MuseTalkProcessor


class VideoDubbingPipeline:
    """Complete video dubbing pipeline."""
    
    def __init__(self, config: AppConfig, session_id: Optional[str] = None, resume: bool = False):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.resume = resume
        
        self.session_dir = Path(self.config.video.sessions_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.video_processor = VideoProcessor(str(self.session_dir / "temp"))
        
        if config.asr.service == 'whisperx':
            self.asr_processor = WhisperXProcessor(config.asr)
        elif config.asr.service == 'openai_api':
            self.asr_processor = OpenAIWhisperAPIProcessor(config.asr)
        else:
            raise ValueError(f"Unsupported ASR service: {config.asr.service}")

        self.translation_processor = TranslationProcessor(config.translation)
        self.tts_processor = CoquiTTSProcessor(config.tts)
        self.lipsync_processor = MuseTalkProcessor(config.musetalk)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self.pipeline_state = self._load_or_initialize_state()
    
    def _load_or_initialize_state(self) -> Dict:
        """Load state from file if resuming, otherwise initialize a new state."""
        state_file = self.session_dir / "pipeline_state.json"
        if self.resume and state_file.exists():
            self.logger.info(f"Resuming pipeline from session: {self.session_id}")
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        self.logger.info(f"Initializing new pipeline session: {self.session_id}")
        return {
            "session_id": self.session_id,
            "status": "initialized",
            "current_stage": None,
            "progress": 0.0,
            "stages": {
                "asr": {"status": "pending", "progress": 0.0},
                "translation": {"status": "pending", "progress": 0.0},
                "review": {"status": "pending", "progress": 0.0},
                "tts": {"status": "pending", "progress": 0.0},
                "lipsync": {"status": "pending", "progress": 0.0},
                "finalize": {"status": "pending", "progress": 0.0}
            },
            "files": {},
            "metadata": {}
        }

    def _save_state(self):
        """Save the current pipeline state to a file."""
        state_file = self.session_dir / "pipeline_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_state, f, indent=2, ensure_ascii=False)

    def process_video(self, video_path: str, target_language: str,
                     output_path: str, speaker_reference: Optional[str] = None,
                     auto_approve: bool = False) -> Dict:
        """Process complete video dubbing pipeline."""
        try:
            self.logger.info(f"Starting video dubbing pipeline for session: {self.session_id}")
            self.logger.info(f"Video: {video_path}, Language: {target_language}")

            if not self.resume:
                self.pipeline_state["metadata"] = {
                    "input_video": video_path,
                    "target_language": target_language,
                    "output_path": output_path,
                    "speaker_reference": speaker_reference,
                    "start_time": time.time()
                }

            self.pipeline_state["status"] = "running"
            
            # Stage 1: Extract audio and get video info
            if self.pipeline_state["stages"]["asr"]["status"] != "completed":
                self._update_stage("asr", "running", 0.1)
                audio_path, video_info = self._extract_audio_and_info(video_path)
            else:
                self.logger.info("Skipping audio extraction (already complete).")
                audio_path = self.pipeline_state["files"]["original_audio"]
                video_info = self.pipeline_state["files"]["video_info"]

            # Stage 2: Speech recognition and alignment
            if self.pipeline_state["stages"]["asr"]["status"] != "completed":
                self._update_stage("asr", "running", 0.3)
                transcript_result = self._perform_asr(audio_path)
                self._update_stage("asr", "completed", 1.0)
            else:
                self.logger.info("Skipping ASR (already complete).")
                with open(self.pipeline_state["files"]["transcript"], 'r', encoding='utf-8') as f:
                    transcript_result = json.load(f)
            
            # Stage 3: Translation
            if self.pipeline_state["stages"]["translation"]["status"] != "completed":
                self._update_stage("translation", "running", 0.1)
                translated_segments = self._perform_translation(
                    transcript_result, target_language
                )
                self._update_stage("translation", "completed", 1.0)
            else:
                self.logger.info("Skipping translation (already complete).")
                with open(self.pipeline_state["files"]["translation"], 'r', encoding='utf-8') as f:
                    translated_segments = json.load(f)

            # Stage 4: Review (skip if auto_approve)
            if self.pipeline_state["stages"]["review"]["status"] != "completed":
                if auto_approve:
                    approved_segments = self._auto_approve_review(translated_segments)
                    self._update_stage("review", "completed", 1.0)
                else:
                    self._update_stage("review", "running", 0.5)
                    # For now, auto-approve all segments in non-interactive resume
                    # In a real implementation, this would wait for user review
                    approved_segments = self._auto_approve_review(translated_segments)
            else:
                self.logger.info("Skipping review (already complete).")
                with open(self.pipeline_state["files"]["approved_translation"], 'r', encoding='utf-8') as f:
                    approved_segments = json.load(f)

            # Stage 5: Text-to-speech generation
            if self.pipeline_state["stages"]["tts"]["status"] != "completed":
                self._update_stage("tts", "running", 0.1)
                tts_result = self._perform_tts(
                    approved_segments, target_language, speaker_reference
                )

                # Stage 6: Audio alignment and adjustment
                self._update_stage("tts", "running", 0.7)
                aligned_audio = self._align_audio(
                    tts_result, video_info["duration"]
                )
                self._update_stage("tts", "completed", 1.0)
            else:
                self.logger.info("Skipping TTS and alignment (already complete).")
                aligned_audio = self.pipeline_state["files"]["aligned_audio"]

            # Stage 7: Lip synchronization
            if self.pipeline_state["stages"]["lipsync"]["status"] != "completed":
                self._update_stage("lipsync", "running", 0.1)
                synced_video = self._perform_lipsync(
                    video_path, aligned_audio
                )
                self._update_stage("lipsync", "completed", 1.0)
            else:
                self.logger.info("Skipping lip-sync (already complete).")
                synced_video = self.pipeline_state["files"]["lipsynced_video"]
            
            # Stage 8: Finalization
            self._update_stage("finalize", "running", 0.1)
            final_result = self._finalize_output(synced_video, output_path)
            self._update_stage("finalize", "completed", 1.0)
            
            # Complete pipeline
            self.pipeline_state["status"] = "completed"
            self.pipeline_state["progress"] = 1.0
            self.pipeline_state["metadata"]["end_time"] = time.time()
            self.pipeline_state["metadata"]["duration"] = (
                self.pipeline_state["metadata"].get("end_time", time.time()) - 
                self.pipeline_state["metadata"]["start_time"]
            )
            self._save_state()
            
            self.logger.info("Video dubbing pipeline completed successfully")
            return {
                "success": True,
                "output_path": final_result,
                "session_id": self.session_id,
                "pipeline_state": self.pipeline_state
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.pipeline_state["status"] = "failed"
            self.pipeline_state["error"] = str(e)
            
            return {
                "success": False,
                "error": str(e),
                "session_id": self.session_id,
                "pipeline_state": self.pipeline_state
            }
    
    def _update_stage(self, stage_name: str, status: str, progress: float):
        """Update pipeline stage status and progress."""
        self.pipeline_state["stages"][stage_name]["status"] = status
        self.pipeline_state["stages"][stage_name]["progress"] = progress
        self.pipeline_state["current_stage"] = stage_name
        
        # Calculate overall progress
        stage_weights = {
            "asr": 0.2,
            "translation": 0.15,
            "review": 0.05,
            "tts": 0.3,
            "lipsync": 0.25,
            "finalize": 0.05
        }
        
        total_progress = 0.0
        for stage, weight in stage_weights.items():
            stage_progress = self.pipeline_state["stages"][stage]["progress"]
            total_progress += stage_progress * weight
        
        self.pipeline_state["progress"] = min(total_progress, 1.0)
        
        self.logger.info(f"Stage {stage_name}: {status} ({progress:.1%})")
        self._save_state()
    
    def _extract_audio_and_info(self, video_path: str) -> Tuple[str, Dict]:
        """Extract audio and get video information."""
        self.logger.info("Extracting audio and video information")
        
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)
        
        # Extract audio
        audio_path = self.session_dir / "original_audio.wav"
        self.video_processor.extract_audio(video_path, str(audio_path))
        
        self.pipeline_state["files"]["original_audio"] = str(audio_path)
        self.pipeline_state["files"]["video_info"] = video_info
        
        self._update_stage("asr", "running", 0.5)
        return audio_path, video_info
    
    def _perform_asr(self, audio_path: str) -> Dict:
        """Perform automatic speech recognition."""
        self.logger.info("Performing speech recognition and alignment")

        duration_sec = self.video_processor.get_audio_duration(audio_path)
        
        final_result = {}

        if self.config.asr.split_long_audio and duration_sec > self.config.asr.split_threshold_min * 60:
            self.logger.info(f"Audio is long ({duration_sec/60:.1f} min), splitting into {self.config.asr.split_chunk_duration_min}-min chunks.")
            
            chunk_output_dir = Path(self.config.video.temp_dir) / f"chunks_{self.session_id}"
            audio_chunks = self.video_processor.split_audio(
                audio_path, 
                chunk_duration_min=self.config.asr.split_chunk_duration_min, 
                output_dir=str(chunk_output_dir)
            )
            
            all_segments = []
            full_text = []
            detected_language = "auto"
            
            time_offset = 0.0

            for i, chunk_path in enumerate(audio_chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}: {chunk_path}")
                result = self.asr_processor.process_audio(chunk_path, language_code=detected_language)

                if i == 0 and detected_language == "auto":
                    detected_language = result.get("language", "en")
                    self.logger.info(f"Detected language from first chunk: {detected_language}")

                for segment in result.get("segments", []):
                    segment["start"] += time_offset
                    segment["end"] += time_offset
                    for word in segment.get("words", []):
                        if "start" in word and word["start"] is not None:
                            word["start"] += time_offset
                        if "end" in word and word["end"] is not None:
                            word["end"] += time_offset
                    all_segments.append(segment)
                
                full_text.append(self.asr_processor.get_full_text(result))
                
                chunk_duration = self.video_processor.get_audio_duration(chunk_path)
                time_offset += chunk_duration
            
            final_result = {
                "language": detected_language,
                "text": " ".join(full_text),
                "segments": all_segments
            }

        else:
            final_result = self.asr_processor.process_audio(audio_path)

        # Save transcript
        transcript_path = self.session_dir / "transcript.json"
        self.asr_processor.save_transcript(final_result, str(transcript_path))
        
        self.pipeline_state["files"]["transcript"] = str(transcript_path)
        self.pipeline_state["metadata"]["source_language"] = final_result.get("language", "unknown")
        
        return final_result
    
    def _perform_translation(self, transcript_result: Dict, 
                           target_language: str) -> List[Dict]:
        """Perform translation of transcript segments."""
        self.logger.info(f"Translating to {target_language}")
        
        segments = self.asr_processor.extract_text_segments(transcript_result)
        translated_segments = self.translation_processor.translate_segments(
            segments, target_language
        )
        
        # Save translated segments
        translation_path = self.session_dir / "translation.json"
        
        with open(translation_path, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)
        
        self.pipeline_state["files"]["translation"] = str(translation_path)
        
        self._update_stage("translation", "completed", 1.0)
        return translated_segments
    
    def _create_review_session(self, translated_segments: List[Dict]) -> str:
        """Create translation review session."""
        self.logger.info("Creating translation review session")
        
        review_path = self.session_dir / f"review_session.json"
        
        self.translation_processor.review_manager.temp_dir = str(self.session_dir)
        self.translation_processor.review_manager.create_review_session(
            translated_segments, self.session_id
        )
        
        self.pipeline_state["files"]["review_session"] = str(review_path)
        return str(review_path)
    
    def _auto_approve_review(self, translated_segments: List[Dict]) -> List[Dict]:
        """Auto-approve all translations for automated processing."""
        self.logger.info("Auto-approving translations")

        approved_path = self.session_dir / "approved_translation.json"
        
        with open(approved_path, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)
        
        self.pipeline_state["files"]["approved_translation"] = str(approved_path)

        self._update_stage("review", "completed", 1.0)
        return translated_segments
    
    def _perform_tts(self, segments: List[Dict], target_language: str,
                    speaker_reference: Optional[str] = None) -> Dict:
        """Perform text-to-speech synthesis."""
        self.logger.info("Generating speech synthesis")
        
        # Prepare TTS output directory
        tts_output_dir = self.session_dir / "tts_segments"
        tts_output_dir.mkdir(exist_ok=True)
        
        # Synthesize segments
        synthesized_segments = self.tts_processor.synthesize_segments(
            segments, str(tts_output_dir), target_language, speaker_reference
        )
        
        self.pipeline_state["files"]["tts_segments"] = [
            {**s, 'audio_path': str(Path(s['audio_path']))} if 'audio_path' in s else s
            for s in synthesized_segments
        ]
        self.pipeline_state["files"]["tts_output_dir"] = str(tts_output_dir)
        
        self._update_stage("tts", "running", 0.9)
        return {
            "segments": synthesized_segments,
            "output_dir": str(tts_output_dir)
        }
    
    def _align_audio(self, tts_result: Dict, target_duration: float) -> str:
        """Align and adjust audio to match video duration."""
        self.logger.info("Aligning audio with video duration")
        
        # Concatenate audio segments
        audio_output_path = self.session_dir / "concatenated_audio.wav"
        
        self.tts_processor.concatenate_audio_segments(
            tts_result["segments"], str(audio_output_path)
        )
        
        # Adjust speed if necessary
        aligned_audio_path = self.session_dir / "aligned_audio.wav"
        
        self.video_processor.adjust_audio_speed(
            str(audio_output_path), target_duration, str(aligned_audio_path)
        )
        
        self.pipeline_state["files"]["aligned_audio"] = str(aligned_audio_path)
        
        self._update_stage("tts", "completed", 1.0)
        return str(aligned_audio_path)
    
    def _perform_lipsync(self, video_path: str, audio_path: str) -> str:
        """Perform lip synchronization."""
        self.logger.info("Performing lip synchronization")
        
        # Create temporary output for lip-sync
        temp_output = self.session_dir / "lipsynced.mp4"
        
        synced_video = self.lipsync_processor.synthesize_lip_sync(
            video_path, audio_path, str(temp_output)
        )
        
        self.pipeline_state["files"]["lipsynced_video"] = str(synced_video)
        
        self._update_stage("lipsync", "completed", 1.0)
        return str(synced_video)
    
    def _finalize_output(self, synced_video: str, final_output_path: str) -> str:
        """Finalize output video."""
        self.logger.info("Finalizing output")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        
        # Copy to final location
        import shutil
        shutil.copy2(synced_video, final_output_path)
        
        # Clean up temporary files if configured
        if hasattr(self.config, 'cleanup_temp_files') and self.config.cleanup_temp_files:
            self._cleanup_temp_files()
        
        self._update_stage("finalize", "completed", 1.0)
        return final_output_path
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        self.logger.info("Cleaning up temporary files")
        
        try:
            # Clean up video processor temp files
            self.video_processor.cleanup_temp_files()
            
            # Clean up ASR processor
            self.asr_processor.cleanup()
            
            # Clean up TTS processor
            self.tts_processor.cleanup()
            
            # Clean up lip-sync processor
            self.lipsync_processor.cleanup()
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        return self.pipeline_state.copy()
    
    def save_pipeline_state(self, file_path: Optional[str] = None):
        """Save pipeline state to file."""
        if file_path is None:
            file_path = self.session_dir / "pipeline_state.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_state, f, indent=2, ensure_ascii=False)
    
    def load_pipeline_state(self, file_path: Optional[str] = None):
        """Load pipeline state from file."""
        if file_path is None:
            file_path = self.session_dir / "pipeline_state.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            self.pipeline_state = json.load(f)
            self.session_id = self.pipeline_state["session_id"]


class PipelineManager:
    """Manages multiple pipeline instances."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.active_pipelines: Dict[str, VideoDubbingPipeline] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_or_create_pipeline(self, session_id: Optional[str] = None, resume: bool = False) -> VideoDubbingPipeline:
        """Get an existing pipeline or create a new one."""
        if session_id and session_id in self.active_pipelines:
            return self.active_pipelines[session_id]

        if session_id and not resume:
             self.logger.warning(f"Session ID {session_id} provided but resume is false. Starting a new session.")
        
        pipeline = VideoDubbingPipeline(self.config, session_id=session_id, resume=resume)
        self.active_pipelines[pipeline.session_id] = pipeline
        
        self.logger.info(f"Loaded pipeline for session: {pipeline.session_id}")
        return pipeline

    def get_pipeline(self, session_id: str) -> Optional[VideoDubbingPipeline]:
        """Get pipeline by session ID."""
        return self.active_pipelines.get(session_id)
    
    def get_pipeline_status(self, session_id: str) -> Optional[Dict]:
        """Get pipeline status by session ID."""
        pipeline = self.get_pipeline(session_id)
        if pipeline:
            return pipeline.get_pipeline_status()
        return None
    
    def cleanup_pipeline(self, session_id: str):
        """Clean up and remove pipeline."""
        if session_id in self.active_pipelines:
            pipeline = self.active_pipelines[session_id]
            pipeline._cleanup_temp_files()
            del self.active_pipelines[session_id]
            
            self.logger.info(f"Cleaned up pipeline: {session_id}")
    
    def list_active_pipelines(self) -> List[str]:
        """List all active pipeline session IDs."""
        return list(self.active_pipelines.keys())
