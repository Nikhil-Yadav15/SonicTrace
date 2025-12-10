# core/__init__.py

"""
Core processing modules
"""

from .vad_processor import vad_processor, VADProcessor
from .overlap_detector import overlap_detector, OverlapDetector
from .speaker_embeddings import speaker_embedder, SpeakerEmbedder
from .speaker_clustering import speaker_clusterer, SpeakerClusterer
from .emotion_recognizer import emotion_recognizer, EmotionRecognizer
from .transcriber import transcriber, Transcriber
from .audio_processor import audio_processor, AudioProcessor

__all__ = [
    'vad_processor', 'VADProcessor',
    'overlap_detector', 'OverlapDetector',
    'speaker_embedder', 'SpeakerEmbedder',
    'speaker_clusterer', 'SpeakerClusterer',
    'emotion_recognizer', 'EmotionRecognizer',
    'transcriber', 'Transcriber',
    'audio_processor', 'AudioProcessor'
]
