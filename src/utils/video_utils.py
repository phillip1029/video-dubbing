"""Video processing utilities."""

import os
import subprocess
from typing import Tuple, Optional, List, Dict
import ffmpeg
import cv2
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import logging
import math


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def extract_audio(self, video_path: str, audio_path: Optional[str] = None, acodec: str = 'pcm_s16le', ar: str = '16000', **kwargs) -> str:
        """Extract audio from video file with specified codec."""
        if audio_path is None:
            ext = "wav" if acodec == 'pcm_s16le' else "m4a"
            video_name = Path(video_path).stem
            audio_path = str(self.temp_dir / f"{video_name}_audio.{ext}")
        
        try:
            output_params = {
                'acodec': acodec,
                'ac': 1,
                'ar': ar,
                **kwargs
            }
            stream = ffmpeg.input(video_path).audio
            stream = ffmpeg.output(stream, audio_path, **output_params)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            
            self.logger.info(f"Extracted audio to {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            self.logger.error(f"Failed to extract audio: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information using ffprobe."""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'video': {},
                'audio': {}
            }
            
            if video_stream:
                info['video'] = {
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate']),
                    'codec': video_stream['codec_name']
                }
            
            if audio_stream:
                info['audio'] = {
                    'sample_rate': int(audio_stream['sample_rate']),
                    'channels': int(audio_stream['channels']),
                    'codec': audio_stream['codec_name'],
                    'duration': float(audio_stream.get('duration', probe['format']['duration']))
                }
            
            return info
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to get video info: {e}")
    
    def split_video_audio(self, video_path: str) -> Tuple[str, str]:
        """Split video into separate video and audio files."""
        video_name = Path(video_path).stem
        video_only_path = str(self.temp_dir / f"{video_name}_video_only.mp4")
        audio_path = str(self.temp_dir / f"{video_name}_audio.m4a")
        
        # Extract video without audio
        (
            ffmpeg
            .input(video_path)
            .output(video_only_path, vcodec='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Extract audio
        audio_path = self.extract_audio(video_path, audio_path)
        
        return video_only_path, audio_path

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            self.logger.warning(f"Pydub failed to get duration for {audio_path}: {e}, trying ffprobe.")
            try:
                probe = ffmpeg.probe(audio_path)
                return float(probe['format']['duration'])
            except ffmpeg.Error as e:
                self.logger.error(f"Failed to get audio duration with ffprobe: {e}")
                return 0.0

    def split_audio(self, audio_path: str, target_chunk_size_mb: Optional[float] = None, chunk_duration_min: int = 15, output_dir: Optional[str] = None) -> List[str]:
        """Split audio into chunks based on size or duration."""
        output_dir_path = Path(output_dir) if output_dir else self.temp_dir
        output_dir_path.mkdir(exist_ok=True)
        self.logger.info(f"Splitting audio file: {audio_path}")

        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        input_filename = os.path.splitext(os.path.basename(audio_path))[0]
        input_format = os.path.splitext(audio_path)[1].lstrip('.')

        # If target_chunk_size_mb is provided, calculate chunk_duration_ms based on bitrate
        if target_chunk_size_mb is not None:
            bitrate_kbps = self.get_audio_bitrate(audio_path) / 1000
            if bitrate_kbps > 0:
                target_chunk_size_kb = target_chunk_size_mb * 1024
                # Duration (s) = Size (kb) / Bitrate (kbps)
                chunk_duration_s = target_chunk_size_kb / bitrate_kbps
                chunk_duration_ms = int(chunk_duration_s * 1000)
                self.logger.info(f"Calculated chunk duration for {target_chunk_size_mb}MB target: {chunk_duration_s:.1f}s")
            else:
                self.logger.warning("Could not determine bitrate. Falling back to duration-based split.")
                chunk_duration_ms = chunk_duration_min * 60 * 1000
        else:
             chunk_duration_ms = chunk_duration_min * 60 * 1000

        chunk_paths = []
        for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms)):
            end_ms = start_ms + chunk_duration_ms
            chunk = audio[start_ms:end_ms]
            
            chunk_filename = f"{input_filename}_chunk_{i:03d}.{input_format}"
            chunk_path = os.path.join(output_dir_path, chunk_filename)
            
            self.logger.info(f"Exporting chunk {i}: {chunk_path}")
            
            # Explicitly handle m4a format with the correct codec
            if input_format == 'm4a':
                chunk.export(chunk_path, format="ipod", codec="aac")
            else:
                chunk.export(chunk_path, format=input_format)

            chunk_paths.append(chunk_path)
            
        return chunk_paths

    def get_audio_bitrate(self, audio_path: str) -> int:
        """Get audio bitrate in bits per second."""
        try:
            self.logger.info(f"Probing for audio bitrate: {audio_path}")
            probe = ffmpeg.probe(audio_path)
            # Find the audio stream and get the bit_rate
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            if audio_stream and 'bit_rate' in audio_stream:
                bitrate = int(audio_stream['bit_rate'])
                self.logger.info(f"Detected bitrate: {bitrate / 1000:.0f} kbps")
                return bitrate
            self.logger.warning("Bitrate not found in audio stream.")
            return 0
        except ffmpeg.Error as e:
            self.logger.error(f"Failed to probe audio bitrate: {e.stderr}")
            return 0
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during bitrate probing: {e}")
            return 0

    def get_audio_info(self, audio_path: str) -> Dict:
        """
        Get audio information using ffprobe.
        """
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next((stream for stream in probe['streams'] 
                               if stream['codec_type'] == 'audio'), None)
            
            info = {
                'sample_rate': int(audio_stream['sample_rate']),
                'channels': int(audio_stream['channels']),
                'codec': audio_stream['codec_name'],
                'duration': float(audio_stream.get('duration', probe['format']['duration']))
            }
            
            return info
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to get audio info: {e}")
    
    def merge_video_audio(self, video_path: str, audio_path: str, 
                         output_path: str, video_codec: str = "libx264",
                         audio_codec: str = "aac") -> str:
        """Merge video and audio files."""
        try:
            (
                ffmpeg
                .output(
                    ffmpeg.input(video_path),
                    ffmpeg.input(audio_path),
                    output_path,
                    vcodec=video_codec,
                    acodec=audio_codec,
                    strict='experimental'
                )
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to merge video and audio: {e}")
    
    def adjust_audio_speed(self, audio_path: str, target_duration: float, 
                          output_path: Optional[str] = None) -> str:
        """Adjust audio speed to match target duration."""
        if output_path is None:
            audio_name = Path(audio_path).stem
            output_path = str(self.temp_dir / f"{audio_name}_adjusted.wav")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        current_duration = len(audio) / sr
        
        # Calculate speed adjustment factor
        speed_factor = current_duration / target_duration
        
        # Adjust speed using librosa
        if speed_factor != 1.0:
            audio_adjusted = librosa.effects.time_stretch(audio, rate=speed_factor)
        else:
            audio_adjusted = audio
        
        # Save adjusted audio
        sf.write(output_path, audio_adjusted, sr)
        return output_path
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None,
                      fps: Optional[float] = None) -> str:
        """Extract frames from video."""
        if output_dir is None:
            video_name = Path(video_path).stem
            output_dir = str(self.temp_dir / f"{video_name}_frames")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        input_stream = ffmpeg.input(video_path)
        if fps:
            input_stream = ffmpeg.filter(input_stream, 'fps', fps=fps)
        
        (
            input_stream
            .output(f"{output_dir}/frame_%06d.png")
            .overwrite_output()
            .run(quiet=True)
        )
        
        return output_dir
    
    def frames_to_video(self, frames_dir: str, output_path: str, 
                       fps: float = 25, video_codec: str = "libx264") -> str:
        """Convert frame sequence back to video."""
        try:
            (
                ffmpeg
                .input(f"{frames_dir}/frame_%06d.png", framerate=fps)
                .output(output_path, vcodec=video_codec, pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to create video from frames: {e}")
    
    def cleanup_temp_files(self, patterns: list = None):
        """Clean up temporary files."""
        if patterns is None:
            patterns = ["*.wav", "*.mp4", "*.png", "frame_*"]
        
        for pattern in patterns:
            for file_path in self.temp_dir.glob(pattern):
                try:
                    file_path.unlink()
                except OSError:
                    pass  # File might be in use


def get_available_devices():
    """Get available processing devices."""
    devices = {"cpu": True}
    
    try:
        import torch
        if torch.cuda.is_available():
            devices["cuda"] = True
            devices["gpu_count"] = torch.cuda.device_count()
        else:
            devices["cuda"] = False
    except ImportError:
        devices["cuda"] = False
    
    return devices
