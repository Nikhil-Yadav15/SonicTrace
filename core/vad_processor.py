# core/vad_processor.py

"""
Voice Activity Detection using Silero VAD
Detects speech segments in audio
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import time
from config import Config

class VADProcessor:
    """Voice Activity Detection processor using Silero VAD"""
    
    def __init__(self):
        """Initialize Silero VAD model"""
        self.model = None
        self.utils = None
        self.sample_rate = 16000  # Silero VAD expects 16kHz
        self.device = Config.get_device()
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model"""
        print("Loading Silero VAD model...")
        start_time = time.time()
        
        try:
            # Load from torch hub
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Move to device
            model.to(self.device)
            model.eval()
            
            self.model = model
            self.utils = utils
            
            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            
            elapsed = time.time() - start_time
            print(f"✅ Silero VAD loaded in {elapsed:.2f}s on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD: {e}")
    
    def detect_speech(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = None,
        min_silence_duration_ms: int = None,
        speech_pad_ms: int = 30,
        return_seconds: bool = True
    ) -> List[Dict]:
        """
        Detect speech segments in audio
        
        Args:
            waveform: Audio waveform (numpy array)
            sample_rate: Sample rate of audio
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration between segments
            speech_pad_ms: Padding around speech segments
            return_seconds: Return timestamps in seconds (vs samples)
        
        Returns:
            List of speech segments with metadata
        """
        
        # Use config defaults if not specified
        if min_speech_duration_ms is None:
            min_speech_duration_ms = Config.VAD_MIN_SPEECH_DURATION_MS
        if min_silence_duration_ms is None:
            min_silence_duration_ms = Config.VAD_MIN_SILENCE_DURATION_MS
        
        # Ensure correct sample rate
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Silero VAD requires {self.sample_rate}Hz audio. "
                f"Got {sample_rate}Hz. Please resample first."
            )
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(waveform).float()
        
        # Get speech timestamps
        try:
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=return_seconds
            )
        except Exception as e:
            raise RuntimeError(f"VAD detection failed: {e}")
        
        # Convert to our format with metadata
        segments = []
        for idx, seg in enumerate(speech_timestamps):
            segment = {
                'id': idx,
                'start': float(seg['start']),
                'end': float(seg['end']),
                'duration': float(seg['end'] - seg['start']),
                'confidence': 1.0  # Silero doesn't provide confidence scores
            }
            segments.append(segment)
        
        return segments
    
    def filter_short_segments(
        self,
        segments: List[Dict],
        min_duration: float = 0.5
    ) -> List[Dict]:
        """
        Remove segments shorter than threshold
        
        Args:
            segments: List of segment dictionaries
            min_duration: Minimum duration in seconds
        
        Returns:
            Filtered segments
        """
        filtered = [
            seg for seg in segments 
            if seg['duration'] >= min_duration
        ]
        
        removed = len(segments) - len(filtered)
        if removed > 0:
            print(f"   Filtered {removed} short segments (< {min_duration}s)")
        
        return filtered
    
    def merge_close_segments(
        self,
        segments: List[Dict],
        max_gap: float = 0.3
    ) -> List[Dict]:
        """
        Merge segments that are close together
        
        Args:
            segments: List of segment dictionaries
            max_gap: Maximum gap in seconds to merge
        
        Returns:
            Merged segments
        """
        if not segments:
            return segments
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = sorted_segments[0].copy()
        
        for next_seg in sorted_segments[1:]:
            gap = next_seg['start'] - current['end']
            
            if gap <= max_gap:
                # Merge segments
                current['end'] = next_seg['end']
                current['duration'] = current['end'] - current['start']
            else:
                # Save current and start new
                merged.append(current)
                current = next_seg.copy()
        
        # Add last segment
        merged.append(current)
        
        # Reindex
        for idx, seg in enumerate(merged):
            seg['id'] = idx
        
        original_count = len(segments)
        merged_count = len(merged)
        if merged_count < original_count:
            print(f"   Merged {original_count} → {merged_count} segments")
        
        return merged
    
    def process_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        filter_short: bool = True,
        merge_close: bool = True,
        min_duration: float = 0.5,
        max_gap: float = 0.3
    ) -> List[Dict]:
        """
        Complete VAD processing pipeline
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            filter_short: Remove short segments
            merge_close: Merge close segments
            min_duration: Minimum segment duration
            max_gap: Maximum gap for merging
        
        Returns:
            Processed speech segments
        """
        
        print("\n🎤 Running Voice Activity Detection...")
        start_time = time.time()
        
        # Detect speech
        segments = self.detect_speech(waveform, sample_rate)
        print(f"   Found {len(segments)} initial speech segments")
        
        # Filter short segments
        if filter_short and segments:
            segments = self.filter_short_segments(segments, min_duration)
        
        # Merge close segments
        if merge_close and segments:
            segments = self.merge_close_segments(segments, max_gap)
        
        elapsed = time.time() - start_time
        
        if segments:
            total_speech_duration = sum(s['duration'] for s in segments)
            audio_duration = len(waveform) / sample_rate
            speech_percentage = (total_speech_duration / audio_duration) * 100
            
            print(f"   ✅ VAD complete in {elapsed:.2f}s")
            print(f"   Final: {len(segments)} segments")
            print(f"   Speech: {total_speech_duration:.1f}s ({speech_percentage:.1f}%)")
        else:
            print(f"   ⚠️ No speech detected in audio")
        
        return segments
    
    def get_segment_statistics(self, segments: List[Dict]) -> Dict:
        """
        Calculate statistics about detected segments
        
        Returns:
            Dictionary with statistics
        """
        if not segments:
            return {
                'total_segments': 0,
                'total_speech_duration': 0.0,
                'avg_segment_duration': 0.0,
                'min_segment_duration': 0.0,
                'max_segment_duration': 0.0
            }
        
        durations = [s['duration'] for s in segments]
        
        return {
            'total_segments': len(segments),
            'total_speech_duration': sum(durations),
            'avg_segment_duration': np.mean(durations),
            'min_segment_duration': min(durations),
            'max_segment_duration': max(durations),
            'median_segment_duration': np.median(durations)
        }
    
    def visualize_segments(
        self,
        segments: List[Dict],
        audio_duration: float,
        max_width: int = 80
    ) -> str:
        """
        Create ASCII visualization of speech segments
        
        Args:
            segments: List of segments
            audio_duration: Total audio duration
            max_width: Width of visualization
        
        Returns:
            ASCII string visualization
        """
        if not segments:
            return "No speech segments detected"
        
        # Create timeline
        timeline = ['-'] * max_width
        
        for seg in segments:
            start_pos = int((seg['start'] / audio_duration) * max_width)
            end_pos = int((seg['end'] / audio_duration) * max_width)
            
            # Ensure within bounds
            start_pos = max(0, min(start_pos, max_width - 1))
            end_pos = max(0, min(end_pos, max_width))
            
            # Mark segment
            for i in range(start_pos, end_pos):
                timeline[i] = '█'
        
        # Create visualization
        vis = ''.join(timeline)
        
        # Add time markers
        time_markers = f"0s{' ' * (max_width - len(str(int(audio_duration))) - 3)}{audio_duration:.1f}s"
        
        return f"{vis}\n{time_markers}"


# Global instance
vad_processor = VADProcessor()
