# core/emotion_recognizer.py

"""
Emotion Recognition using Wav2Vec2
Detects emotions in speech segments

Note: the ehcalabres emotion checkpoint was fine-tuned with a custom
classification head (dense -> tanh -> out_proj over mean-pooled hidden
states). Loading it with the stock Wav2Vec2ForSequenceClassification class
silently leaves the head randomly initialized, so we define the matching
architecture here.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from pathlib import Path
from typing import List, Dict, Tuple
import time
from config import Config


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec2 classification - matches the ehcalabres checkpoint
    (weight names: classifier.dense.* and classifier.output.*)"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, 'final_dropout', 0.1))
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.output(x)


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 with mean-pooled classification head (ehcalabres architecture)"""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooled = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled)


class EmotionRecognizer:
    """Recognize emotions in speech using Wav2Vec2"""

    def __init__(self):
        """Initialize recognizer (model loads lazily on first use)"""
        self.model = None
        self.processor = None
        self.device = Config.get_device()
        self.sample_rate = 16000

        # Overridden by the model config after load (ehcalabres = RAVDESS:
        # angry, calm, disgust, fearful, happy, neutral, sad, surprised)
        self.emotion_labels = [
            'angry', 'calm', 'disgust', 'fearful',
            'happy', 'neutral', 'sad', 'surprised'
        ]

    def _ensure_loaded(self):
        """Load Wav2Vec2 emotion recognition model on first use"""
        if self.model is not None:
            return

        print("Loading Wav2Vec2 emotion model...")
        start_time = time.time()

        try:
            # Prefer a local copy; otherwise download from the Hub (public)
            model_source = self._find_model_path()
            local_only = model_source is not None

            if model_source is None:
                model_source = Config.EMOTION_MODEL_NAME
                print(f"   Local copy not found, downloading {model_source}...")
            else:
                print(f"   Loading from: {model_source}")

            kwargs = {
                'local_files_only': local_only,
                'cache_dir': str(Config.MODELS_DIR / 'hf_cache'),
            }

            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(model_source), **kwargs
            )
            self.model = Wav2Vec2ForSpeechClassification.from_pretrained(
                str(model_source), **kwargs
            )

            self.model.to(self.device)
            self.model.eval()

            # Get emotion labels from model config
            if getattr(self.model.config, 'id2label', None):
                self.emotion_labels = [
                    self.model.config.id2label[i]
                    for i in range(len(self.model.config.id2label))
                ]

            elapsed = time.time() - start_time
            print(f"✅ Emotion model loaded in {elapsed:.2f}s on {self.device}")
            print(f"   Emotions: {', '.join(self.emotion_labels)}")

        except Exception as e:
            self.model = None
            self.processor = None
            raise RuntimeError(f"Failed to load emotion model: {e}")

    def _find_model_path(self):
        """Find a locally downloaded Wav2Vec2 emotion model"""
        model_dirs = [
            Config.MODELS_DIR / 'ehcalabres_wav2vec2-lg-xlsr-en-speech-emotion-recognition',
            Config.MODELS_DIR / 'ehcalabres_wav2vec2-lg-xlsr-en-speech-emotion-recognition_files',
            Config.MODELS_DIR / 'wav2vec2_emotion',
        ]

        for model_dir in model_dirs:
            if model_dir.exists():
                # Check for config.json (required file)
                if (model_dir / 'config.json').exists():
                    return model_dir

                # Check subdirectories
                for subdir in model_dir.rglob('config.json'):
                    potential_path = subdir.parent
                    # Verify it has model file too
                    if (potential_path / 'pytorch_model.bin').exists() or \
                       (potential_path / 'model.safetensors').exists():
                        return potential_path

        return None

    def recognize_emotion(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = None,
        return_all_scores: bool = False
    ) -> Dict:
        """
        Recognize emotion in audio segment

        Args:
            audio_segment: Audio waveform
            sample_rate: Sample rate (must be 16kHz)
            return_all_scores: Return scores for all emotions

        Returns:
            Dictionary with emotion and confidence
        """
        self._ensure_loaded()

        sr = sample_rate or self.sample_rate

        if sr != 16000:
            raise ValueError("Wav2Vec2 requires 16kHz audio")

        # Minimum duration check
        min_duration = 0.5  # seconds
        if len(audio_segment) < sr * min_duration:
            # Pad if too short
            target_length = int(sr * min_duration)
            audio_segment = np.pad(
                audio_segment,
                (0, target_length - len(audio_segment)),
                mode='constant'
            )

        # Process audio
        try:
            inputs = self.processor(
                audio_segment.astype(np.float32),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                logits = self.model(**inputs)

            # Get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]

            # Get predicted emotion
            predicted_idx = int(np.argmax(probabilities))
            predicted_emotion = self.emotion_labels[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            result = {
                'emotion': predicted_emotion,
                'confidence': confidence
            }

            # Add all scores if requested
            if return_all_scores:
                result['all_scores'] = {
                    emotion: float(prob)
                    for emotion, prob in zip(self.emotion_labels, probabilities)
                }

            return result

        except Exception as e:
            raise RuntimeError(f"Emotion recognition failed: {e}")

    def recognize_emotions_batch(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        sample_rate: int = None,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Recognize emotions for multiple segments

        Args:
            audio: Full audio waveform
            segments: List of segment dictionaries
            sample_rate: Sample rate
            show_progress: Show progress

        Returns:
            Segments with emotion labels added
        """
        self._ensure_loaded()

        sr = sample_rate or self.sample_rate

        if show_progress:
            print(f"\n🎭 Recognizing emotions for {len(segments)} segments...")

        emotion_segments = []

        for idx, segment in enumerate(segments):
            if show_progress and idx % 10 == 0:
                print(f"   Processing segment {idx+1}/{len(segments)}...", end='\r')

            # Extract segment audio
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)

            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            audio_segment = audio[start_sample:end_sample]

            # Recognize emotion
            try:
                emotion_result = self.recognize_emotion(
                    audio_segment,
                    sr,
                    return_all_scores=True
                )

                # Add to segment
                seg = segment.copy()
                seg['emotion'] = emotion_result['emotion']
                seg['emotion_confidence'] = emotion_result['confidence']
                seg['emotion_scores'] = emotion_result['all_scores']

                emotion_segments.append(seg)

            except Exception as e:
                if show_progress:
                    print(f"\n   ⚠️ Failed to recognize emotion for segment {idx}: {e}")

                # Add default values
                seg = segment.copy()
                seg['emotion'] = 'unknown'
                seg['emotion_confidence'] = 0.0
                seg['emotion_scores'] = {}
                emotion_segments.append(seg)

        if show_progress:
            print(f"   ✅ Recognized emotions for {len(emotion_segments)} segments" + " " * 20)

            # Print emotion distribution
            emotion_counts = {}
            for seg in emotion_segments:
                emotion = seg.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            print(f"\n   Emotion distribution:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(emotion_segments)) * 100
                print(f"     {emotion}: {count} ({percentage:.1f}%)")

        return emotion_segments

    def get_emotion_statistics(
        self,
        segments: List[Dict]
    ) -> Dict:
        """
        Calculate emotion statistics

        Returns:
            Dictionary with emotion stats
        """
        if not segments:
            return {
                'total_segments': 0,
                'emotions': {}
            }

        # Count emotions
        emotion_counts = {}
        emotion_durations = {}
        emotion_confidences = {}

        for seg in segments:
            emotion = seg.get('emotion', 'unknown')
            duration = seg.get('duration', 0)
            confidence = seg.get('emotion_confidence', 0)

            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            emotion_durations[emotion] = emotion_durations.get(emotion, 0) + duration

            if emotion not in emotion_confidences:
                emotion_confidences[emotion] = []
            emotion_confidences[emotion].append(confidence)

        # Calculate percentages
        total_segments = len(segments)
        total_duration = sum(emotion_durations.values())

        emotions_stats = {}
        for emotion in emotion_counts:
            emotions_stats[emotion] = {
                'count': emotion_counts[emotion],
                'percentage': (emotion_counts[emotion] / total_segments) * 100,
                'duration': emotion_durations[emotion],
                'duration_percentage': (emotion_durations[emotion] / total_duration * 100) if total_duration > 0 else 0,
                'avg_confidence': float(np.mean(emotion_confidences[emotion]))
            }

        return {
            'total_segments': total_segments,
            'total_duration': total_duration,
            'emotions': emotions_stats
        }

    def get_dominant_emotion(
        self,
        segments: List[Dict],
        by: str = 'duration'
    ) -> Tuple[str, float]:
        """
        Get dominant emotion

        Args:
            segments: List of segments with emotions
            by: 'duration' or 'count'

        Returns:
            (emotion, percentage)
        """
        stats = self.get_emotion_statistics(segments)

        if not stats['emotions']:
            return 'unknown', 0.0

        # Find dominant emotion
        if by == 'duration':
            dominant = max(
                stats['emotions'].items(),
                key=lambda x: x[1]['duration']
            )
            return dominant[0], dominant[1]['duration_percentage']
        else:
            dominant = max(
                stats['emotions'].items(),
                key=lambda x: x[1]['count']
            )
            return dominant[0], dominant[1]['percentage']

    def visualize_emotion_timeline(
        self,
        segments: List[Dict],
        max_width: int = 80
    ) -> str:
        """
        Create ASCII visualization of emotion timeline

        Returns:
            ASCII string visualization
        """
        if not segments:
            return "No segments with emotions"

        # Simple ASCII mapping
        emotion_ascii = {
            'happy': 'H',
            'sad': 'S',
            'angry': 'A',
            'fear': 'F',
            'fearful': 'F',
            'surprise': '!',
            'surprised': '!',
            'disgust': 'D',
            'calm': 'C',
            'neutral': 'N',
            'unknown': '?'
        }

        # Get total duration
        total_duration = max(seg['end'] for seg in segments)
        if total_duration <= 0:
            total_duration = 1.0

        # Create timeline
        timeline = ['-'] * max_width

        for seg in segments:
            start_pos = int((seg['start'] / total_duration) * max_width)
            end_pos = int((seg['end'] / total_duration) * max_width)

            # Ensure within bounds
            start_pos = max(0, min(start_pos, max_width - 1))
            end_pos = max(0, min(end_pos, max_width))

            # Get emotion character
            emotion = seg.get('emotion', 'unknown')
            char = emotion_ascii.get(emotion, '?')

            # Fill timeline
            for i in range(start_pos, end_pos):
                timeline[i] = char

        # Create visualization
        vis = ''.join(timeline)

        # Add legend
        legend_items = []
        for emotion in set(seg.get('emotion', 'unknown') for seg in segments):
            char = emotion_ascii.get(emotion, '?')
            legend_items.append(f"{char}={emotion}")

        legend = ", ".join(legend_items)

        # Add time markers
        time_markers = f"0s{' ' * (max_width - len(str(int(total_duration))) - 3)}{total_duration:.1f}s"

        return f"{vis}\n{time_markers}\n{legend}"


# Global instance
emotion_recognizer = EmotionRecognizer()
