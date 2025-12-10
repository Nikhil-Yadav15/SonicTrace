# utils/audio_preprocessor.py

"""
Audio preprocessing utilities
Normalization, segmentation, feature extraction
"""

import numpy as np
import librosa
from typing import Tuple, List

class AudioPreprocessor:
    """Audio preprocessing and feature extraction"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def normalize_audio(self, waveform, method='peak'):
        """
        Normalize audio waveform
        
        Args:
            waveform: Audio waveform
            method: 'peak' or 'rms'
        
        Returns:
            Normalized waveform
        """
        if method == 'peak':
            # Peak normalization to [-1, 1]
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                return waveform / max_val
            return waveform
        
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(waveform ** 2))
            if rms > 0:
                target_rms = 0.1
                return waveform * (target_rms / rms)
            return waveform
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def remove_silence(self, waveform, top_db=20, frame_length=2048, hop_length=512):
        """
        Remove silence from audio
        
        Args:
            waveform: Audio waveform
            top_db: Threshold in decibels
        
        Returns:
            Trimmed waveform
        """
        trimmed, _ = librosa.effects.trim(
            waveform,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed
    
    def extract_segments(self, waveform, segment_timestamps, sample_rate=None):
        """
        Extract audio segments from timestamps
        
        Args:
            waveform: Full audio waveform
            segment_timestamps: List of (start, end) tuples in seconds
            sample_rate: Sample rate (uses self.sample_rate if None)
        
        Returns:
            List of audio segments
        """
        sr = sample_rate or self.sample_rate
        segments = []
        
        for start, end in segment_timestamps:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(waveform), end_sample)
            
            if end_sample > start_sample:
                segment = waveform[start_sample:end_sample]
                segments.append(segment)
        
        return segments
    
    def pad_or_trim(self, waveform, target_length):
        """
        Pad or trim waveform to target length
        
        Args:
            waveform: Audio waveform
            target_length: Target length in samples
        
        Returns:
            Padded or trimmed waveform
        """
        current_length = len(waveform)
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            return np.pad(waveform, (0, padding), mode='constant')
        elif current_length > target_length:
            # Trim
            return waveform[:target_length]
        else:
            return waveform
    
    def apply_preemphasis(self, waveform, coef=0.97):
        """
        Apply pre-emphasis filter
        
        Args:
            waveform: Audio waveform
            coef: Pre-emphasis coefficient
        
        Returns:
            Filtered waveform
        """
        return np.append(waveform[0], waveform[1:] - coef * waveform[:-1])
    
    def calculate_energy(self, waveform, frame_length=2048, hop_length=512):
        """
        Calculate energy contour
        
        Returns:
            Energy values over time
        """
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return rms[0]
    
    def detect_voice_activity_simple(self, waveform, energy_threshold=0.02):
        """
        Simple voice activity detection based on energy
        
        Args:
            waveform: Audio waveform
            energy_threshold: Minimum energy threshold
        
        Returns:
            Binary mask (1 = voice, 0 = silence)
        """
        energy = self.calculate_energy(waveform)
        return (energy > energy_threshold).astype(int)
    
    def resample(self, waveform, orig_sr, target_sr):
        """
        Resample audio
        
        Args:
            waveform: Audio waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate
        
        Returns:
            Resampled waveform
        """
        if orig_sr == target_sr:
            return waveform
        
        return librosa.resample(
            waveform,
            orig_sr=orig_sr,
            target_sr=target_sr
        )


# Global instance
audio_preprocessor = AudioPreprocessor()
