# core/overlap_detector.py

"""
Overlap Detection - Detect when multiple speakers talk simultaneously
Uses spectral analysis and multi-pitch detection
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple
from scipy import signal as scipy_signal
from config import Config

class OverlapDetector:
    """Detect overlapping speech segments"""
    
    def __init__(self):
        """Initialize overlap detector"""
        self.sample_rate = Config.SAMPLE_RATE
        self.pitch_threshold = Config.OVERLAP_PITCH_THRESHOLD
        self.energy_threshold = Config.OVERLAP_ENERGY_THRESHOLD
        
        print("✅ Overlap Detector initialized")
    
    def detect_overlap(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = None,
        method: str = 'auto'
    ) -> Tuple[bool, float, Dict]:
        """
        Detect if audio segment contains overlapping speakers
        
        Args:
            audio_segment: Audio waveform segment
            sample_rate: Sample rate (uses config if None)
            method: Detection method ('spectral', 'pitch', 'energy', 'auto')
        
        Returns:
            (is_overlap: bool, confidence: float, metadata: dict)
        """
        sr = sample_rate or self.sample_rate
        
        # Skip very short segments
        if len(audio_segment) < sr * 0.3:  # Less than 300ms
            return False, 0.0, {'reason': 'segment_too_short'}
        
        # Skip silence
        if np.max(np.abs(audio_segment)) < 0.01:
            return False, 0.0, {'reason': 'silence'}
        
        # Choose detection method
        if method == 'auto':
            # Use combined approach
            return self._detect_overlap_combined(audio_segment, sr)
        elif method == 'spectral':
            return self._detect_overlap_spectral(audio_segment, sr)
        elif method == 'pitch':
            return self._detect_overlap_pitch(audio_segment, sr)
        elif method == 'energy':
            return self._detect_overlap_energy(audio_segment, sr)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_overlap_combined(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[bool, float, Dict]:
        """
        Combined overlap detection using multiple methods
        
        Returns:
            (is_overlap, confidence, metadata)
        """
        metadata = {}
        
        # Method 1: Pitch-based detection
        pitch_overlap, pitch_conf, pitch_meta = self._detect_overlap_pitch(audio, sr)
        metadata['pitch'] = pitch_meta
        
        # Method 2: Spectral analysis
        spectral_overlap, spectral_conf, spectral_meta = self._detect_overlap_spectral(audio, sr)
        metadata['spectral'] = spectral_meta
        
        # Method 3: Energy distribution
        energy_overlap, energy_conf, energy_meta = self._detect_overlap_energy(audio, sr)
        metadata['energy'] = energy_meta
        
        # Combine decisions
        votes = sum([pitch_overlap, spectral_overlap, energy_overlap])
        
        # Need at least 2 out of 3 methods to agree
        is_overlap = votes >= 2
        
        # Combined confidence (weighted average)
        confidence = (pitch_conf * 0.4 + spectral_conf * 0.4 + energy_conf * 0.2)
        
        metadata['votes'] = votes
        metadata['method'] = 'combined'
        
        return is_overlap, confidence, metadata
    
    def _detect_overlap_pitch(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[bool, float, Dict]:
        """
        Detect overlap using multi-pitch detection
        
        Multiple simultaneous pitches indicate multiple speakers
        """
        try:
            # Extract pitches using librosa's piptrack
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=sr,
                fmin=50,   # Minimum fundamental frequency (low male voice)
                fmax=400,  # Maximum fundamental frequency (high female voice)
                threshold=0.1
            )
            
            # Count simultaneous pitches at each time frame
            max_simultaneous = 0
            total_frames = pitches.shape[1]
            multi_pitch_frames = 0
            
            for time_idx in range(total_frames):
                # Count active pitches in this frame
                active_pitches = np.sum(magnitudes[:, time_idx] > 0.1)
                max_simultaneous = max(max_simultaneous, active_pitches)
                
                if active_pitches >= 2:
                    multi_pitch_frames += 1
            
            # Calculate percentage of frames with multiple pitches
            multi_pitch_ratio = multi_pitch_frames / max(total_frames, 1)
            
            # Decision: overlap if multiple pitches detected frequently
            is_overlap = max_simultaneous >= self.pitch_threshold
            
            # Confidence based on consistency
            confidence = min(0.9, 0.5 + multi_pitch_ratio * 0.4)
            
            metadata = {
                'max_simultaneous_pitches': int(max_simultaneous),
                'multi_pitch_ratio': float(multi_pitch_ratio),
                'threshold': self.pitch_threshold
            }
            
            return is_overlap, confidence, metadata
            
        except Exception as e:
            # Fallback on error
            return False, 0.0, {'error': str(e)}
    
    def _detect_overlap_spectral(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[bool, float, Dict]:
        """
        Detect overlap using spectral analysis
        
        Overlapping speech has wider spectral bandwidth and
        more complex spectral patterns
        """
        try:
            # Compute spectral features
            
            # 1. Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            mean_bandwidth = np.mean(bandwidth)
            
            # 2. Spectral contrast (measures peaks vs valleys)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            mean_contrast = np.mean(contrast)
            
            # 3. Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            std_centroid = np.std(centroid)
            
            # 4. Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            mean_rolloff = np.mean(rolloff)
            
            # Decision logic:
            # Overlap typically has:
            # - Wider bandwidth (>3500 Hz) ← INCREASED from 2000
            # - Higher spectral variation
            # - More complex spectral structure
            
            is_overlap = False
            confidence = 0.5
            
            # More conservative thresholds
            if mean_bandwidth > 3500:  # ← CHANGED from 2000
                is_overlap = True
                confidence += 0.2
            
            if std_centroid > 500:  # ← CHANGED from 300
                is_overlap = True
                confidence += 0.15
            
            if mean_contrast > 30:  # ← CHANGED from 25
                confidence += 0.1
            
            confidence = min(0.9, confidence)
            
            metadata = {
                'bandwidth': float(mean_bandwidth),
                'spectral_contrast': float(mean_contrast),
                'centroid_std': float(std_centroid),
                'rolloff': float(mean_rolloff)
            }
            
            return is_overlap, confidence, metadata
            
        except Exception as e:
            return False, 0.0, {'error': str(e)}
    
    def _detect_overlap_energy(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[bool, float, Dict]:
        """
        Detect overlap using energy distribution
        
        Overlapping speech has more distributed energy across
        frequency bands
        """
        try:
            # Compute STFT
            D = np.abs(librosa.stft(audio))
            
            # Split into frequency bands
            n_bins = D.shape[0]
            low_band = D[:n_bins//3, :]      # Low frequencies
            mid_band = D[n_bins//3:2*n_bins//3, :]  # Mid frequencies
            high_band = D[2*n_bins//3:, :]   # High frequencies
            
            # Calculate energy in each band
            low_energy = np.mean(low_band ** 2)
            mid_energy = np.mean(mid_band ** 2)
            high_energy = np.mean(high_band ** 2)
            
            total_energy = low_energy + mid_energy + high_energy
            
            # Energy distribution (entropy-like measure)
            if total_energy > 0:
                low_ratio = low_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
                
                # Calculate energy entropy
                # More distributed energy suggests overlap
                ratios = [low_ratio, mid_ratio, high_ratio]
                ratios = [r for r in ratios if r > 0]
                energy_entropy = -sum(r * np.log(r) for r in ratios)
            else:
                energy_entropy = 0
            
            # Overall energy (RMS)
            rms = np.sqrt(np.mean(audio ** 2))
            
            # Decision: overlap if high energy + distributed spectrum
            is_overlap = (rms > self.energy_threshold and energy_entropy > 0.8)
            
            confidence = min(0.85, 0.5 + energy_entropy * 0.3)
            
            metadata = {
                'rms_energy': float(rms),
                'energy_entropy': float(energy_entropy),
                'low_energy_ratio': float(low_ratio) if total_energy > 0 else 0,
                'mid_energy_ratio': float(mid_ratio) if total_energy > 0 else 0,
                'high_energy_ratio': float(high_ratio) if total_energy > 0 else 0
            }
            
            return is_overlap, confidence, metadata
            
        except Exception as e:
            return False, 0.0, {'error': str(e)}
    
    def classify_segments(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        sample_rate: int = None,
        show_progress: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify segments as single-speaker or overlap
        
        Args:
            audio: Full audio waveform
            segments: List of speech segments from VAD
            sample_rate: Sample rate
            show_progress: Show progress messages
        
        Returns:
            (single_segments, overlap_segments)
        """
        sr = sample_rate or self.sample_rate
        
        if show_progress:
            print("\n🔀 Detecting overlapping speakers...")
        
        single_segments = []
        overlap_segments = []
        
        for idx, segment in enumerate(segments):
            # Extract audio for this segment
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            audio_segment = audio[start_sample:end_sample]
            
            # Detect overlap
            is_overlap, confidence, metadata = self.detect_overlap(
                audio_segment,
                sr,
                method='auto'
            )
            
            # Create segment copy with overlap info
            seg = segment.copy()
            seg['overlap_detected'] = is_overlap
            seg['overlap_confidence'] = confidence
            seg['overlap_metadata'] = metadata
            
            # Classify
            if is_overlap:
                overlap_segments.append(seg)
            else:
                single_segments.append(seg)
        
        if show_progress:
            print(f"   Single-speaker: {len(single_segments)} segments")
            print(f"   Overlap: {len(overlap_segments)} segments")
            
            if overlap_segments:
                total_overlap_duration = sum(s['duration'] for s in overlap_segments)
                total_duration = sum(s['duration'] for s in segments)
                overlap_percentage = (total_overlap_duration / total_duration) * 100
                print(f"   Overlap ratio: {overlap_percentage:.1f}%")
        
        return single_segments, overlap_segments
    
    def get_overlap_statistics(
        self,
        overlap_segments: List[Dict]
    ) -> Dict:
        """
        Calculate statistics about overlap segments
        
        Returns:
            Dictionary with overlap statistics
        """
        if not overlap_segments:
            return {
                'total_overlaps': 0,
                'total_overlap_duration': 0.0,
                'avg_overlap_duration': 0.0,
                'avg_confidence': 0.0
            }
        
        durations = [s['duration'] for s in overlap_segments]
        confidences = [s.get('overlap_confidence', 0) for s in overlap_segments]
        
        return {
            'total_overlaps': len(overlap_segments),
            'total_overlap_duration': sum(durations),
            'avg_overlap_duration': np.mean(durations),
            'min_overlap_duration': min(durations),
            'max_overlap_duration': max(durations),
            'avg_confidence': np.mean(confidences)
        }


# Global instance
overlap_detector = OverlapDetector()
