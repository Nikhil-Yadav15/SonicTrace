# utils/audio_loader.py

"""
Multi-method audio loader with automatic fallback
Tries: torchaudio → soundfile → librosa → FFmpeg conversion
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from pathlib import Path
from config import Config
from utils.ffmpeg_handler import ffmpeg
import tempfile

class AudioLoader:
    """Load audio with multiple fallback methods"""
    
    def __init__(self):
        self.target_sr = Config.SAMPLE_RATE
        self.ffmpeg = ffmpeg
        
    def load(self, audio_path, sr=None, mono=True):
        """
        Load audio file with automatic method selection
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (default: Config.SAMPLE_RATE)
            mono: Convert to mono if True
        
        Returns:
            tuple: (waveform: np.ndarray, sample_rate: int)
        """
        audio_path = Path(audio_path)
        target_sr = sr or self.target_sr
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate format
        if not self._is_supported_format(audio_path):
            raise ValueError(
                f"Unsupported format: {audio_path.suffix}\n"
                f"Supported: {', '.join(Config.SUPPORTED_FORMATS)}"
            )
        
        # Try different loading methods
        methods = [
            ('torchaudio', self._load_torchaudio),
            ('soundfile', self._load_soundfile),
            ('librosa', self._load_librosa),
        ]
        
        # Add FFmpeg if available
        if self.ffmpeg.is_available():
            methods.append(('FFmpeg', self._load_ffmpeg))
        
        last_error = None
        
        for method_name, method_func in methods:
            try:
                waveform, sample_rate = method_func(audio_path)
                
                # Convert to mono if needed
                if mono and len(waveform.shape) > 1:
                    waveform = waveform.mean(axis=0)
                elif mono and waveform.ndim == 1:
                    pass  # Already mono
                
                # Resample if needed
                if sample_rate != target_sr:
                    waveform = librosa.resample(
                        waveform,
                        orig_sr=sample_rate,
                        target_sr=target_sr
                    )
                    sample_rate = target_sr
                
                # Normalize to [-1, 1]
                if waveform.max() > 1.0 or waveform.min() < -1.0:
                    waveform = waveform / np.max(np.abs(waveform))
                
                print(f"✅ Loaded audio using {method_name}")
                return waveform, sample_rate
                
            except Exception as e:
                last_error = e
                continue
        
        # All methods failed
        error_msg = (
            f"Could not load audio file: {audio_path}\n"
            f"Tried: {', '.join([m[0] for m in methods])}\n"
            f"Last error: {last_error}"
        )
        
        if not self.ffmpeg.is_available():
            error_msg += "\n\n💡 Install FFmpeg for better format support"
        
        raise RuntimeError(error_msg)
    
    def _is_supported_format(self, path):
        """Check if file format is supported"""
        ext = path.suffix.lower().lstrip('.')
        return ext in Config.SUPPORTED_FORMATS
    
    def _load_torchaudio(self, path):
        """Load using torchaudio"""
        waveform, sr = torchaudio.load(str(path))
        waveform = waveform.numpy()
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0)
        else:
            waveform = waveform[0]
        
        return waveform, sr
    
    def _load_soundfile(self, path):
        """Load using soundfile"""
        waveform, sr = sf.read(str(path), dtype='float32')
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        
        return waveform, sr
    
    def _load_librosa(self, path):
        """Load using librosa"""
        waveform, sr = librosa.load(
            str(path),
            sr=None,  # Preserve original sample rate
            mono=False
        )
        
        # librosa returns mono by default, but let's be safe
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=0)
        
        return waveform, sr
    
    def _load_ffmpeg(self, path):
        """Load using FFmpeg conversion"""
        # Convert to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Convert to WAV
            self.ffmpeg.convert_to_wav(path, tmp_path, sample_rate=self.target_sr)
            
            # Load the WAV file
            waveform, sr = sf.read(tmp_path, dtype='float32')
            
            # Convert to mono
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            
            return waveform, sr
            
        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except:
                pass
    
    def get_duration(self, audio_path):
        """
        Get audio duration in seconds
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            float: Duration in seconds
        """
        # Try FFprobe first (fastest)
        if self.ffmpeg.ffprobe_path:
            info = self.ffmpeg.get_audio_info(audio_path)
            if info:
                return info['duration']
        
        # Fallback: Load and calculate
        try:
            waveform, sr = self.load(audio_path)
            return len(waveform) / sr
        except:
            return 0.0
    
    def validate_audio_file(self, audio_path, max_size_mb=None):
        """
        Validate audio file
        
        Args:
            audio_path: Path to audio file
            max_size_mb: Maximum file size in MB
        
        Returns:
            dict: {'valid': bool, 'error': str or None, 'info': dict}
        """
        audio_path = Path(audio_path)
        
        # Check existence
        if not audio_path.exists():
            return {
                'valid': False,
                'error': 'File not found',
                'info': None
            }
        
        # Check format
        if not self._is_supported_format(audio_path):
            return {
                'valid': False,
                'error': f'Unsupported format: {audio_path.suffix}',
                'info': None
            }
        
        # Check size
        max_size_mb = max_size_mb or Config.MAX_FILE_SIZE_MB
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return {
                'valid': False,
                'error': f'File too large: {size_mb:.1f} MB (max: {max_size_mb} MB)',
                'info': {'size_mb': size_mb}
            }
        
        # Try to get audio info
        info = self.ffmpeg.get_audio_info(audio_path) if self.ffmpeg.is_available() else {}
        
        if not info:
            # Try loading a small portion
            try:
                waveform, sr = self.load(audio_path)
                info = {
                    'duration': len(waveform) / sr,
                    'sample_rate': sr,
                    'size_mb': size_mb
                }
            except Exception as e:
                return {
                    'valid': False,
                    'error': f'Cannot load audio: {str(e)[:100]}',
                    'info': None
                }
        
        return {
            'valid': True,
            'error': None,
            'info': info
        }


# Global instance
audio_loader = AudioLoader()
