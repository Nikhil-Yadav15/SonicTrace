# core/transcriber.py

"""
Speech Transcription using OpenAI Whisper
Converts speech to text with timestamps
"""

import whisper
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import time
from config import Config

class Transcriber:
    """Transcribe speech to text using Whisper"""
    
    def __init__(self):
        """Initialize Whisper model"""
        self.model = None
        self.device = Config.get_device()
        self.model_size = Config.WHISPER_MODEL_SIZE
        self.language = 'en'  # English only
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        print(f"Loading Whisper {self.model_size} model...")
        start_time = time.time()
        
        try:
            # Find model path
            model_cache = Config.MODELS_DIR / 'whisper'
            model_cache.mkdir(exist_ok=True)
            
            # Load model
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(model_cache)
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Whisper {self.model_size} loaded in {elapsed:.2f}s on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
        return_word_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe a single audio segment
        
        Args:
            audio_segment: Audio waveform
            sample_rate: Sample rate
            return_word_timestamps: Return word-level timestamps
        
        Returns:
            Dictionary with transcription and metadata
        """
        
        # Ensure float32 and correct range
        if audio_segment.dtype != np.float32:
            audio_segment = audio_segment.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        if np.abs(audio_segment).max() > 1.0:
            audio_segment = audio_segment / np.abs(audio_segment).max()
        
        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_segment = librosa.resample(
                audio_segment,
                orig_sr=sample_rate,
                target_sr=16000
            )
        
        # Transcribe
        try:
            result = self.model.transcribe(
                audio_segment,
                language=self.language,
                task='transcribe',
                word_timestamps=return_word_timestamps,
                fp16=False  # Use fp32 for CPU compatibility
            )
            
            # Extract transcription
            text = result['text'].strip()
            
            # Calculate confidence (average of segment probabilities)
            segments = result.get('segments', [])
            if segments:
                confidences = [seg.get('no_speech_prob', 0) for seg in segments]
                avg_confidence = 1.0 - np.mean(confidences)  # Invert no_speech_prob
            else:
                avg_confidence = 0.0
            
            transcription = {
                'text': text,
                'confidence': float(avg_confidence),
                'language': result.get('language', 'en')
            }
            
            # Add word timestamps if requested
            if return_word_timestamps and segments:
                words = []
                for seg in segments:
                    if 'words' in seg:
                        words.extend(seg['words'])
                transcription['words'] = words
            
            return transcription
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def transcribe_segments_batch(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        sample_rate: int = 16000,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Transcribe multiple segments
        
        Args:
            audio: Full audio waveform
            segments: List of segment dictionaries
            sample_rate: Sample rate
            show_progress: Show progress
        
        Returns:
            Segments with transcriptions added
        """
        
        if show_progress:
            print(f"\n🎙️ Transcribing {len(segments)} segments...")
        
        transcribed_segments = []
        
        for idx, segment in enumerate(segments):
            if show_progress and idx % 5 == 0:
                print(f"   Transcribing segment {idx+1}/{len(segments)}...", end='\r')
            
            # Extract segment audio
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            audio_segment = audio[start_sample:end_sample]
            
            # Skip very short segments
            if len(audio_segment) < sample_rate * 0.3:  # Less than 300ms
                seg = segment.copy()
                seg['text'] = ''
                seg['transcription_confidence'] = 0.0
                transcribed_segments.append(seg)
                continue
            
            # Transcribe
            try:
                transcription = self.transcribe_segment(
                    audio_segment,
                    sample_rate
                )
                
                # Add to segment
                seg = segment.copy()
                seg['text'] = transcription['text']
                seg['transcription_confidence'] = transcription['confidence']
                
                transcribed_segments.append(seg)
                
            except Exception as e:
                if show_progress:
                    print(f"\n   ⚠️ Failed to transcribe segment {idx}: {e}")
                
                # Add empty transcription
                seg = segment.copy()
                seg['text'] = ''
                seg['transcription_confidence'] = 0.0
                transcribed_segments.append(seg)
        
        if show_progress:
            print(f"   ✅ Transcribed {len(transcribed_segments)} segments" + " " * 20)
            
            # Count non-empty transcriptions
            non_empty = sum(1 for seg in transcribed_segments if seg.get('text', '').strip())
            print(f"   Non-empty: {non_empty}/{len(transcribed_segments)}")
        
        return transcribed_segments
    
    def get_full_transcription(
        self,
        segments: List[Dict],
        include_timestamps: bool = False,
        include_speaker: bool = False,
        include_emotion: bool = False
    ) -> str:
        """
        Get full transcription from segments
        
        Args:
            segments: List of segments with transcriptions
            include_timestamps: Include timestamps in output
            include_speaker: Include speaker labels
            include_emotion: Include emotion labels
        
        Returns:
            Formatted transcription string
        """
        lines = []
        
        for seg in segments:
            text = seg.get('text', '').strip()
            
            if not text:
                continue
            
            # Build line
            parts = []
            
            # Timestamp
            if include_timestamps:
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                parts.append(f"[{self._format_timestamp(start)} -> {self._format_timestamp(end)}]")
            
            # Speaker
            if include_speaker and 'speaker' in seg:
                parts.append(f"Speaker {seg['speaker']}:")
            
            # Emotion
            if include_emotion and 'emotion' in seg:
                emotion = seg['emotion']
                parts.append(f"({emotion})")
            
            # Text
            parts.append(text)
            
            # Combine
            line = ' '.join(parts)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def get_transcription_statistics(
        self,
        segments: List[Dict]
    ) -> Dict:
        """
        Calculate transcription statistics
        
        Returns:
            Dictionary with stats
        """
        if not segments:
            return {
                'total_segments': 0,
                'transcribed_segments': 0,
                'total_words': 0,
                'avg_confidence': 0.0
            }
        
        transcribed = [seg for seg in segments if seg.get('text', '').strip()]
        
        total_words = sum(len(seg['text'].split()) for seg in transcribed)
        
        confidences = [seg.get('transcription_confidence', 0) for seg in transcribed]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_segments': len(segments),
            'transcribed_segments': len(transcribed),
            'empty_segments': len(segments) - len(transcribed),
            'total_words': total_words,
            'avg_words_per_segment': total_words / len(transcribed) if transcribed else 0,
            'avg_confidence': float(avg_confidence)
        }
    
    def search_transcription(
        self,
        segments: List[Dict],
        query: str,
        case_sensitive: bool = False
    ) -> List[Dict]:
        """
        Search for text in transcriptions
        
        Args:
            segments: List of segments with transcriptions
            query: Search query
            case_sensitive: Case-sensitive search
        
        Returns:
            List of matching segments
        """
        if not case_sensitive:
            query = query.lower()
        
        matches = []
        
        for seg in segments:
            text = seg.get('text', '')
            
            if not case_sensitive:
                text = text.lower()
            
            if query in text:
                matches.append(seg)
        
        return matches
    
    def export_transcription(
        self,
        segments: List[Dict],
        format: str = 'txt',
        filepath: Optional[str] = None
    ) -> str:
        """
        Export transcription to file
        
        Args:
            segments: Segments with transcriptions
            format: 'txt', 'srt', 'vtt', or 'json'
            filepath: Output file path (optional)
        
        Returns:
            Formatted string
        """
        if format == 'txt':
            output = self.get_full_transcription(segments, include_timestamps=True)
        
        elif format == 'srt':
            output = self._export_srt(segments)
        
        elif format == 'vtt':
            output = self._export_vtt(segments)
        
        elif format == 'json':
            import json
            output = json.dumps([
                {
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg.get('text', ''),
                    'speaker': seg.get('speaker', 'Unknown'),
                    'emotion': seg.get('emotion', 'neutral')
                }
                for seg in segments if seg.get('text', '').strip()
            ], indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Save to file if path provided
        if filepath:
            Path(filepath).write_text(output, encoding='utf-8')
            print(f"✅ Exported to {filepath}")
        
        return output
    
    def _export_srt(self, segments: List[Dict]) -> str:
        """Export as SRT subtitle format"""
        lines = []
        idx = 1
        
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue
            
            start = self._format_srt_timestamp(seg['start'])
            end = self._format_srt_timestamp(seg['end'])
            
            lines.append(f"{idx}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
            
            idx += 1
        
        return '\n'.join(lines)
    
    def _export_vtt(self, segments: List[Dict]) -> str:
        """Export as WebVTT format"""
        lines = ["WEBVTT", ""]
        
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue
            
            start = self._format_vtt_timestamp(seg['start'])
            end = self._format_vtt_timestamp(seg['end'])
            
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_vtt_timestamp(self, seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# Global instance
transcriber = Transcriber()
