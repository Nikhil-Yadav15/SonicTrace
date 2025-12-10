# utils/ffmpeg_handler.py

"""
FFmpeg handler for audio format conversion
Supports custom FFmpeg path on Windows
"""

import os
import subprocess
from pathlib import Path
import platform
import json
from config import Config

class FFmpegHandler:
    """Handle FFmpeg operations with custom path support"""
    
    def __init__(self, ffmpeg_path=None):
        """
        Initialize FFmpeg handler
        
        Args:
            ffmpeg_path: Custom path to ffmpeg executable
                        If None, tries Config.FFMPEG_PATH then auto-detect
        """
        self.ffmpeg_path = self._find_ffmpeg(ffmpeg_path)
        self.ffprobe_path = self._find_ffprobe()
        
        if self.ffmpeg_path:
            version = self.get_version()
            print(f"✅ FFmpeg found: {self.ffmpeg_path}")
            if version:
                print(f"   Version: {version}")
        else:
            print("⚠️ FFmpeg not found - limited audio format support")
    
    def _find_ffmpeg(self, custom_path=None):
        """Find FFmpeg executable"""
        
        # Priority 1: Custom path provided
        if custom_path:
            if self._is_valid_ffmpeg(custom_path):
                return str(Path(custom_path).resolve())
        
        # Priority 2: Config path
        if Config.FFMPEG_PATH:
            if self._is_valid_ffmpeg(Config.FFMPEG_PATH):
                return str(Path(Config.FFMPEG_PATH).resolve())
        
        # Priority 3: Common Windows locations
        if platform.system() == "Windows":
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                Path.home() / "ffmpeg" / "bin" / "ffmpeg.exe",
                Path(__file__).parent.parent / "bin" / "ffmpeg.exe",
            ]
            
            for path in common_paths:
                if self._is_valid_ffmpeg(path):
                    return str(Path(path).resolve())
        
        # Priority 4: System PATH
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=3
            )
            if result.returncode == 0:
                return 'ffmpeg'
        except:
            pass
        
        return None
    
    def _find_ffprobe(self):
        """Find ffprobe executable (same directory as ffmpeg)"""
        if not self.ffmpeg_path or self.ffmpeg_path == 'ffmpeg':
            try:
                result = subprocess.run(
                    ['ffprobe', '-version'],
                    capture_output=True,
                    timeout=3
                )
                if result.returncode == 0:
                    return 'ffprobe'
            except:
                pass
            return None
        
        ffmpeg_path = Path(self.ffmpeg_path)
        
        if platform.system() == "Windows":
            probe_path = ffmpeg_path.parent / 'ffprobe.exe'
        else:
            probe_path = ffmpeg_path.parent / 'ffprobe'
        
        return str(probe_path) if probe_path.exists() else None
    
    def _is_valid_ffmpeg(self, path):
        """Check if path is valid FFmpeg executable"""
        try:
            path = Path(path)
            if not path.exists():
                return False
            
            result = subprocess.run(
                [str(path), '-version'],
                capture_output=True,
                timeout=3
            )
            return result.returncode == 0
        except:
            return False
    
    def is_available(self):
        """Check if FFmpeg is available"""
        return self.ffmpeg_path is not None
    
    def get_version(self):
        """Get FFmpeg version string"""
        if not self.ffmpeg_path:
            return None
        
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                # Extract version from first line
                first_line = result.stdout.split('\n')[0]
                return first_line.split('version')[1].split()[0] if 'version' in first_line else first_line
        except:
            pass
        
        return None
    
    def convert_to_wav(self, input_path, output_path=None, sample_rate=None, mono=True):
        """
        Convert audio to WAV format
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output WAV (auto-generated if None)
            sample_rate: Target sample rate (default: Config.SAMPLE_RATE)
            mono: Convert to mono if True
        
        Returns:
            Path to output WAV file
        """
        if not self.is_available():
            raise RuntimeError("FFmpeg not available")
        
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        else:
            output_path = Path(output_path)
        
        if sample_rate is None:
            sample_rate = Config.SAMPLE_RATE
        
        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            '-i', str(input_path),
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1' if mono else '2',  # Channels
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            
            if not output_path.exists():
                raise RuntimeError("Output file was not created")
            
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg conversion timed out (>5 minutes)")
    
    def get_audio_info(self, audio_path):
        """
        Get detailed audio file information
        
        Returns:
            dict with keys: duration, sample_rate, channels, codec, bitrate
        """
        if not self.ffprobe_path:
            return None
        
        cmd = [
            self.ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            # Extract audio stream info
            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                return None
            
            format_info = data.get('format', {})
            
            return {
                'duration': float(format_info.get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'format': format_info.get('format_name', 'unknown'),
                'size_mb': float(format_info.get('size', 0)) / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"Warning: Could not get audio info: {e}")
            return None
    
    def extract_segment(self, input_path, output_path, start_time, end_time):
        """
        Extract audio segment
        
        Args:
            input_path: Source audio file
            output_path: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Path to extracted segment
        """
        if not self.is_available():
            raise RuntimeError("FFmpeg not available")
        
        duration = end_time - start_time
        
        cmd = [
            self.ffmpeg_path,
            '-i', str(input_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Copy codec (fast)
            '-y',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Segment extraction failed")
            
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Segment extraction timed out")


# Global instance
ffmpeg = FFmpegHandler()
