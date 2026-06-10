# core/audio_processor.py

"""
Main Audio Processing Pipeline
Integrates all components: VAD, Overlap, Diarization, Emotion, Transcription
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import json
from datetime import datetime

from config import Config
from utils.audio_loader import audio_loader
from core.vad_processor import vad_processor
from core.overlap_detector import overlap_detector
from core.speaker_embeddings import speaker_embedder
from core.speaker_clustering import speaker_clusterer
from core.emotion_recognizer import emotion_recognizer
from core.transcriber import transcriber

class AudioProcessor:
    """Complete audio analysis pipeline"""
    
    def __init__(self):
        """Initialize audio processor"""
        self.sample_rate = Config.SAMPLE_RATE

        print("✅ Audio Processor initialized (models load on first use)")
    
    def process(
        self,
        audio_path: str,
        enable_overlap: bool = True,
        enable_diarization: bool = True,
        enable_emotion: bool = True,
        enable_transcription: bool = True,
        n_speakers: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Process audio file end-to-end
        
        Args:
            audio_path: Path to audio file
            enable_overlap: Enable overlap detection
            enable_diarization: Enable speaker diarization
            enable_emotion: Enable emotion recognition
            enable_transcription: Enable speech transcription
            n_speakers: Number of speakers (auto-detect if None)
            show_progress: Show progress messages
        
        Returns:
            Complete analysis results
        """
        
        start_time = time.time()
        
        if show_progress:
            print("\n" + "="*70)
            print("🎤 SONICTRACE - AUDIO ANALYSIS PIPELINE")
            print("="*70)
            print(f"File: {audio_path}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)
        
        # Step 1: Load and validate audio
        if show_progress:
            print("\n📁 STEP 1: Loading Audio")
            print("-" * 70)
        
        audio_info = self._load_audio(audio_path, show_progress)
        
        # Step 2: Voice Activity Detection
        if show_progress:
            print("\n🎤 STEP 2: Voice Activity Detection")
            print("-" * 70)
        
        vad_results = self._detect_speech(audio_info, show_progress)
        
        if not vad_results['segments']:
            if show_progress:
                print("\n⚠️ No speech detected in audio")
            
            return self._create_empty_result(audio_info, start_time)
        
        # Step 3: Overlap Detection
        if enable_overlap:
            if show_progress:
                print("\n🔀 STEP 3: Overlap Detection")
                print("-" * 70)
            
            overlap_results = self._detect_overlaps(
                audio_info,
                vad_results['segments'],
                show_progress
            )
        else:
            overlap_results = {
                'single_speaker': vad_results['segments'],
                'overlap': []
            }
            if show_progress:
                print("\n⏭️ STEP 3: Overlap Detection (Skipped)")

        # Mark overlap state on every segment up front
        for seg in overlap_results['single_speaker']:
            seg['is_overlap'] = False
        for seg in overlap_results['overlap']:
            seg['is_overlap'] = True

        # Step 4: Speaker Diarization (single-speaker segments only;
        # overlap segments by definition contain multiple voices)
        if enable_diarization:
            if show_progress:
                print("\n👥 STEP 4: Speaker Diarization")
                print("-" * 70)

            diarization_results = self._diarize_speakers(
                audio_info,
                overlap_results['single_speaker'],
                n_speakers,
                show_progress
            )
        else:
            diarization_results = {
                'segments': overlap_results['single_speaker'],
                'n_speakers': 1
            }
            if show_progress:
                print("\n⏭️ STEP 4: Speaker Diarization (Skipped)")

        # From here on, process ALL segments (labeled + overlap) together so
        # overlap regions also receive emotion labels and transcription.
        all_segments = diarization_results['segments'] + overlap_results['overlap']
        all_segments.sort(key=lambda x: x['start'])

        # Step 5: Emotion Recognition
        if enable_emotion:
            if show_progress:
                print("\n🎭 STEP 5: Emotion Recognition")
                print("-" * 70)

            emotion_results = self._recognize_emotions(
                audio_info,
                all_segments,
                show_progress
            )
        else:
            emotion_results = {'segments': all_segments}
            if show_progress:
                print("\n⏭️ STEP 5: Emotion Recognition (Skipped)")
        
        # Step 6: Speech Transcription
        if enable_transcription:
            if show_progress:
                print("\n🎙️ STEP 6: Speech Transcription")
                print("-" * 70)
            
            transcription_results = self._transcribe_speech(
                audio_info,
                emotion_results['segments'],
                show_progress
            )
        else:
            transcription_results = {'segments': emotion_results['segments']}
            if show_progress:
                print("\n⏭️ STEP 6: Transcription (Skipped)")
        
        # Step 7: Aggregate Results
        if show_progress:
            print("\n📊 STEP 7: Aggregating Results")
            print("-" * 70)
        
        results = self._aggregate_results(
            audio_info,
            vad_results,
            overlap_results,
            diarization_results,
            emotion_results,
            transcription_results,
            start_time,
            show_progress
        )
        
        if show_progress:
            self._print_summary(results)
        
        return results
    
    def _load_audio(self, audio_path: str, show_progress: bool) -> Dict:
        """Load and validate audio file"""
        
        # Validate file
        validation = audio_loader.validate_audio_file(audio_path)
        
        if not validation['valid']:
            raise ValueError(f"Invalid audio file: {validation['error']}")
        
        # Load audio
        waveform, sample_rate = audio_loader.load(audio_path)
        
        duration = len(waveform) / sample_rate
        
        if show_progress:
            print(f"   ✅ Loaded: {Path(audio_path).name}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Channels: mono")
        
        return {
            'path': str(Path(audio_path).resolve()),
            'filename': Path(audio_path).name,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'duration': duration,
            'size_mb': validation['info'].get('size_mb', 0)
        }
    
    def _detect_speech(self, audio_info: Dict, show_progress: bool) -> Dict:
        """Run Voice Activity Detection"""
        
        segments = vad_processor.process_audio(
            audio_info['waveform'],
            audio_info['sample_rate'],
            filter_short=Config.VAD_FILTER_SHORT_SEGMENTS,
            merge_close=Config.VAD_MERGE_CLOSE_SEGMENTS,
            min_duration=Config.VAD_MIN_SEGMENT_DURATION,
            max_gap=Config.VAD_MAX_MERGE_GAP
        )
        
        stats = vad_processor.get_segment_statistics(segments)
        
        if show_progress and segments:
            print(f"   ✅ Detected {len(segments)} speech segments")
            print(f"   Total speech: {stats['total_speech_duration']:.2f}s")
        
        return {
            'segments': segments,
            'statistics': stats
        }
    
    def _detect_overlaps(
        self,
        audio_info: Dict,
        segments: List[Dict],
        show_progress: bool
    ) -> Dict:
        """Detect overlapping speakers"""
        
        single_segments, overlap_segments = overlap_detector.classify_segments(
            audio_info['waveform'],
            segments,
            audio_info['sample_rate'],
            show_progress=show_progress
        )
        
        overlap_stats = overlap_detector.get_overlap_statistics(overlap_segments)
        
        return {
            'single_speaker': single_segments,
            'overlap': overlap_segments,
            'statistics': overlap_stats
        }
    
    def _diarize_speakers(
        self,
        audio_info: Dict,
        segments: List[Dict],
        n_speakers: Optional[int],
        show_progress: bool
    ) -> Dict:
        """Perform speaker diarization"""
        
        if not segments:
            return {'segments': [], 'n_speakers': 0}
        
        # Extract embeddings
        embeddings = speaker_embedder.extract_embeddings_batch(
            audio_info['waveform'],
            segments,
            audio_info['sample_rate'],
            show_progress=show_progress
        )
        
        # Cluster speakers
        labels = speaker_clusterer.cluster_speakers(
            embeddings,
            n_speakers=n_speakers
        )
        
        # Assign speaker labels ("1", "2", ... displayed as Speaker 1, Speaker 2)
        labeled_segments = speaker_clusterer.assign_speaker_labels(
            segments,
            labels,
            label_format='number'
        )
        
        # Get statistics
        speaker_stats = speaker_clusterer.get_speaker_statistics(labeled_segments)
        
        return {
            'segments': labeled_segments,
            'n_speakers': speaker_stats['total_speakers'],
            'statistics': speaker_stats
        }
    
    def _recognize_emotions(
        self,
        audio_info: Dict,
        segments: List[Dict],
        show_progress: bool
    ) -> Dict:
        """Recognize emotions in segments"""
        
        if not segments:
            return {'segments': []}
        
        # Recognize emotions
        emotion_segments = emotion_recognizer.recognize_emotions_batch(
            audio_info['waveform'],
            segments,
            audio_info['sample_rate'],
            show_progress=show_progress
        )
        
        # Get statistics
        emotion_stats = emotion_recognizer.get_emotion_statistics(emotion_segments)
        
        return {
            'segments': emotion_segments,
            'statistics': emotion_stats
        }
    
    def _transcribe_speech(
        self,
        audio_info: Dict,
        segments: List[Dict],
        show_progress: bool
    ) -> Dict:
        """Transcribe speech segments"""
        
        if not segments:
            return {'segments': []}
        
        # Transcribe
        transcribed_segments = transcriber.transcribe_segments_batch(
            audio_info['waveform'],
            segments,
            audio_info['sample_rate'],
            show_progress=show_progress
        )
        
        # Get statistics
        transcription_stats = transcriber.get_transcription_statistics(transcribed_segments)
        
        return {
            'segments': transcribed_segments,
            'statistics': transcription_stats
        }
    
    def _aggregate_results(
        self,
        audio_info: Dict,
        vad_results: Dict,
        overlap_results: Dict,
        diarization_results: Dict,
        emotion_results: Dict,
        transcription_results: Dict,
        start_time: float,
        show_progress: bool
    ) -> Dict:
        """Aggregate all results"""
        
        processing_time = time.time() - start_time
        
        # Transcription output already contains every segment
        # (single-speaker + overlap), fully annotated
        all_segments = sorted(transcription_results['segments'], key=lambda x: x['start'])

        # Reindex
        for idx, seg in enumerate(all_segments):
            seg['id'] = idx

        n_overlap = sum(1 for seg in all_segments if seg.get('is_overlap'))

        results = {
            'metadata': {
                'filename': audio_info['filename'],
                'filepath': audio_info['path'],
                'duration': audio_info['duration'],
                'sample_rate': audio_info['sample_rate'],
                'size_mb': audio_info['size_mb'],
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_segments': len(all_segments),
                'speech_segments': len(all_segments) - n_overlap,
                'overlap_segments': n_overlap,
                'n_speakers': diarization_results.get('n_speakers', 0),
                'total_words': transcription_results.get('statistics', {}).get('total_words', 0)
            },
            'segments': all_segments,
            'statistics': {
                'vad': vad_results.get('statistics', {}),
                'overlap': overlap_results.get('statistics', {}),
                'speakers': diarization_results.get('statistics', {}),
                'emotions': emotion_results.get('statistics', {}),
                'transcription': transcription_results.get('statistics', {})
            }
        }
        
        if show_progress:
            print(f"   ✅ Results aggregated")
            print(f"   Total segments: {len(all_segments)}")
        
        return results
    
    def _create_empty_result(self, audio_info: Dict, start_time: float) -> Dict:
        """Create empty result for no speech detected"""
        
        return {
            'metadata': {
                'filename': audio_info['filename'],
                'filepath': audio_info['path'],
                'duration': audio_info['duration'],
                'sample_rate': audio_info['sample_rate'],
                'size_mb': audio_info['size_mb'],
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_segments': 0,
                'speech_segments': 0,
                'overlap_segments': 0,
                'n_speakers': 0,
                'total_words': 0
            },
            'segments': [],
            'statistics': {}
        }
    
    def _print_summary(self, results: Dict):
        """Print processing summary"""
        
        print("\n" + "="*70)
        print("📊 PROCESSING COMPLETE")
        print("="*70)
        
        summary = results['summary']
        metadata = results['metadata']
        
        print(f"\n⏱️ Processing Time: {metadata['processing_time']:.2f}s")
        print(f"📝 Total Segments: {summary['total_segments']}")
        print(f"👥 Speakers Detected: {summary['n_speakers']}")
        print(f"💬 Total Words: {summary['total_words']}")
        
        # Speaker breakdown
        if 'speakers' in results['statistics']:
            speakers = results['statistics']['speakers'].get('speakers', {})
            if speakers:
                print(f"\n👥 Speaker Breakdown:")
                for speaker, stats in sorted(speakers.items()):
                    print(f"   Speaker {speaker}: {stats['segments']} segments, "
                          f"{stats['total_duration']:.1f}s ({stats['percentage']:.1f}%)")
        
        # Emotion breakdown
        if 'emotions' in results['statistics']:
            emotions = results['statistics']['emotions'].get('emotions', {})
            if emotions:
                print(f"\n🎭 Emotion Distribution:")
                for emotion, stats in sorted(
                    emotions.items(),
                    key=lambda x: x[1]['percentage'],
                    reverse=True
                )[:3]:  # Top 3
                    print(f"   {emotion}: {stats['count']} segments ({stats['percentage']:.1f}%)")
        
        print("\n" + "="*70 + "\n")
    
    def save_results(
        self,
        results: Dict,
        output_path: str,
        format: str = 'json'
    ):
        """
        Save results to file
        
        Args:
            results: Processing results
            output_path: Output file path
            format: 'json', 'txt', 'srt', or 'vtt'
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Save full results as JSON (numpy types are not serializable)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._to_serializable(results), f, indent=2, ensure_ascii=False)
        
        elif format in ['txt', 'srt', 'vtt']:
            # Save transcription only
            segments = results['segments']
            content = transcriber.export_transcription(segments, format=format)
            output_path.write_text(content, encoding='utf-8')
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        print(f"✅ Results saved to: {output_path}")

    @classmethod
    def _to_serializable(cls, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: cls._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._to_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


# Global instance
audio_processor = AudioProcessor()
