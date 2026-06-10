# core/speaker_embeddings.py

"""
Speaker Embedding Extraction using PyAnnote
Extracts speaker voice characteristics for diarization

Model resolution order:
  1. Local copy of pyannote/embedding (models/pyannote_embedding)
  2. pyannote/embedding from HuggingFace (gated - needs HF_TOKEN)
  3. pyannote/wespeaker-voxceleb-resnet34-LM from HuggingFace (public)
"""

import torch
import torch.serialization
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import time
from config import Config

class SpeakerEmbedder:
    """Extract speaker embeddings using PyAnnote"""

    # Public model used when the gated pyannote/embedding is unavailable
    PUBLIC_FALLBACK_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"

    def __init__(self):
        """Initialize embedder (model loads lazily on first use)"""
        self.model = None
        self.inference = None
        self.model_name = None
        self.device = Config.get_device()
        self.sample_rate = 16000
        self.embedding_dim = None  # determined after model load

    def _ensure_loaded(self):
        """Load embedding model on first use"""
        if self.inference is not None:
            return

        print("Loading speaker embedding model...")
        start_time = time.time()

        from pyannote.audio import Model, Inference

        # PyTorch 2.6+ ships weights_only=True by default which breaks
        # pyannote checkpoints (pickled Lightning callbacks). Allow them.
        try:
            import pytorch_lightning
            torch.serialization.add_safe_globals([
                pytorch_lightning.callbacks.early_stopping.EarlyStopping,
                pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint,
            ])
        except Exception:
            pass

        original_torch_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)

        candidates = []

        local_path = self._find_model_path()
        if local_path:
            candidates.append(('local pyannote/embedding', str(local_path)))
        if Config.HF_TOKEN:
            candidates.append(('pyannote/embedding (HF)', 'pyannote/embedding'))
        candidates.append((f'{self.PUBLIC_FALLBACK_REPO} (HF, public)', self.PUBLIC_FALLBACK_REPO))

        errors = []
        torch.load = patched_load
        try:
            for name, source in candidates:
                try:
                    model = Model.from_pretrained(
                        source,
                        use_auth_token=Config.HF_TOKEN,
                        cache_dir=str(Config.MODELS_DIR / 'hf_cache'),
                    )
                    if model is None:
                        raise RuntimeError("Model.from_pretrained returned None "
                                           "(model gated or token missing)")
                    self.model = model
                    self.model_name = name
                    break
                except Exception as e:
                    errors.append(f"{name}: {e}")
        finally:
            torch.load = original_torch_load

        if self.model is None:
            raise RuntimeError(
                "Failed to load any speaker embedding model:\n" +
                "\n".join(f"  - {err}" for err in errors)
            )

        self.model.to(self.device)
        self.model.eval()

        # Inference with window="whole" handles framing/dimensions correctly
        self.inference = Inference(self.model, window="whole", device=self.device)

        # Determine embedding dimension with a short dummy probe
        probe = np.zeros(self.sample_rate, dtype=np.float32)
        probe[::50] = 0.01
        self.embedding_dim = len(self._run_inference(probe))

        elapsed = time.time() - start_time
        print(f"✅ Speaker embedding model loaded in {elapsed:.2f}s on {self.device}")
        print(f"   Model: {self.model_name} ({self.embedding_dim}-d embeddings)")

    def _find_model_path(self):
        """Find a locally downloaded pyannote embedding model"""
        search_dirs = [
            Config.MODELS_DIR / 'pyannote_embedding',
            Config.MODELS_DIR / 'pyannote_embedding_files',
            Path.home() / '.cache' / 'torch' / 'pyannote',
        ]

        for directory in search_dirs:
            if directory.exists():
                model_files = list(directory.rglob('pytorch_model.bin'))
                if model_files:
                    return model_files[0].parent

        return None

    def _run_inference(self, audio_segment: np.ndarray) -> np.ndarray:
        """Run the embedding model on a mono float32 waveform"""
        waveform = torch.from_numpy(audio_segment.astype(np.float32))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (channel, samples)

        embedding = self.inference({
            'waveform': waveform,
            'sample_rate': self.sample_rate
        })

        return np.asarray(embedding).flatten()

    def extract_embedding(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = None
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio segment

        Args:
            audio_segment: Audio waveform
            sample_rate: Sample rate (must be 16kHz)

        Returns:
            Embedding vector
        """
        self._ensure_loaded()

        sr = sample_rate or self.sample_rate

        if sr != 16000:
            raise ValueError("PyAnnote requires 16kHz audio")

        # Pad very short segments so the model gets enough context
        min_duration = 0.5  # seconds
        if len(audio_segment) < sr * min_duration:
            target_length = int(sr * min_duration)
            audio_segment = np.pad(
                audio_segment,
                (0, target_length - len(audio_segment)),
                mode='constant'
            )

        with torch.no_grad():
            try:
                return self._run_inference(audio_segment)
            except Exception as e:
                raise RuntimeError(f"Embedding extraction failed: {e}")

    def extract_embeddings_batch(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        sample_rate: int = None,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple segments

        Args:
            audio: Full audio waveform
            segments: List of segment dictionaries
            sample_rate: Sample rate
            show_progress: Show progress

        Returns:
            List of embedding vectors
        """
        self._ensure_loaded()

        sr = sample_rate or self.sample_rate

        if show_progress:
            print(f"\n🎯 Extracting speaker embeddings for {len(segments)} segments...")

        embeddings = []
        failed = 0

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

            try:
                embedding = self.extract_embedding(audio_segment, sr)
                embeddings.append(embedding)
            except Exception as e:
                failed += 1
                if show_progress:
                    print(f"\n   ⚠️ Failed to extract embedding for segment {idx}: {e}")
                # Cosine-safe fallback: reuse the previous embedding if any,
                # otherwise a tiny random vector (a zero vector breaks cosine
                # distance and crashes clustering).
                if embeddings:
                    embeddings.append(embeddings[-1].copy())
                else:
                    rng = np.random.default_rng(idx)
                    embeddings.append(rng.normal(0, 1e-3, self.embedding_dim))

        if show_progress:
            print(f"   ✅ Extracted {len(embeddings)} embeddings"
                  + (f" ({failed} fallback)" if failed else "") + " " * 20)

        return embeddings

    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Convert to 0-1 range
        similarity = (similarity + 1) / 2

        return float(similarity)

    def calculate_similarity_matrix(
        self,
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate pairwise similarity matrix

        Returns:
            NxN similarity matrix
        """
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = self.calculate_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix


# Global instance
speaker_embedder = SpeakerEmbedder()
