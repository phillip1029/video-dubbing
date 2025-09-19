"""Video processing utilities."""

import os
import subprocess
from typing import Tuple, Optional
import ffmpeg
import cv2
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def extract_audio(self, video_path: str, audio_path: Optional[str] = None) -> str:
        """Extract audio from video file."""
        if audio_path is None:
            video_name = Path(video_path).stem
            audio_path = str(self.temp_dir / f"{video_name}_audio.wav")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to extract audio: {e}")
    
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
                    'codec': audio_stream['codec_name']
                }
            
            return info
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to get video info: {e}")
    
    def split_video_audio(self, video_path: str) -> Tuple[str, str]:
        """Split video into separate video and audio files."""
        video_name = Path(video_path).stem
        video_only_path = str(self.temp_dir / f"{video_name}_video_only.mp4")
        audio_path = str(self.temp_dir / f"{video_name}_audio.wav")
        
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
