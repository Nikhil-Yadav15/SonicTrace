# core/speaker_embeddings.py

"""
Speaker Embedding Extraction using PyAnnote
Extracts speaker voice characteristics for diarization
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
    
    def __init__(self):
        """Initialize PyAnnote embedding model"""
        self.model = None
        self.device = Config.get_device()
        self.sample_rate = 16000
        self.embedding_dim = 512  # PyAnnote embedding dimension
        
        self._load_model()
    
    def _load_model(self):
        """Load PyAnnote embedding model"""
        print("Loading PyAnnote embedding model...")
        start_time = time.time()
        
        try:
            from pyannote.audio import Model
            import pytorch_lightning
            
            # Add safe globals for PyTorch 2.6+ compatibility
            try:
                torch.serialization.add_safe_globals([
                    pytorch_lightning.callbacks.early_stopping.EarlyStopping,
                    pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint,
                ])
            except:
                pass  # Older PyTorch versions don't have add_safe_globals
            
            # Find model path
            model_path = self._find_model_path()
            
            if not model_path:
                raise RuntimeError("PyAnnote embedding model not found")
            
            # Load model with weights_only=False for PyTorch Lightning models
            import functools
            original_torch_load = torch.load
            
            # Monkey-patch torch.load temporarily to use weights_only=False
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = patched_load
            
            try:
                # Load model
                self.model = Model.from_pretrained(
                    str(model_path),
                    use_auth_token=Config.HF_TOKEN
                )
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            elapsed = time.time() - start_time
            print(f"✅ PyAnnote embedding model loaded in {elapsed:.2f}s on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyAnnote embedding model: {e}")
    
    def _find_model_path(self):
        """Find PyAnnote embedding model path"""
        # Try models directory
        models_dir = Config.MODELS_DIR / 'pyannote_embedding'
        
        if models_dir.exists():
            # Look for model files
            model_files = list(models_dir.rglob('pytorch_model.bin'))
            if model_files:
                return model_files[0].parent
        
        # Try the direct files directory
        files_dir = Config.MODELS_DIR / 'pyannote_embedding_files'
        if files_dir.exists():
            model_files = list(files_dir.rglob('pytorch_model.bin'))
            if model_files:
                return model_files[0].parent
        
        # Try cache directory
        cache_dirs = [
            Path.home() / '.cache' / 'torch' / 'pyannote',
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                model_files = list(cache_dir.rglob('pytorch_model.bin'))
                if model_files:
                    return model_files[0].parent
        
        return None
    
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
            Embedding vector (512-d)
        """
        sr = sample_rate or self.sample_rate
        
        if sr != 16000:
            raise ValueError("PyAnnote requires 16kHz audio")
        
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
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_segment).float()
        
        # Add batch dimension
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Move to device
        audio_tensor = audio_tensor.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            try:
                # PyAnnote model expects (batch, samples)
                embedding = self.model(audio_tensor)
                
                # Convert to numpy
                embedding = embedding.cpu().numpy()
                
                # Flatten if needed
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                return embedding
                
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
        sr = sample_rate or self.sample_rate
        
        if show_progress:
            print(f"\n🎯 Extracting speaker embeddings for {len(segments)} segments...")
        
        embeddings = []
        
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
            
            # Extract embedding
            try:
                embedding = self.extract_embedding(audio_segment, sr)
                embeddings.append(embedding)
            except Exception as e:
                if show_progress:
                    print(f"\n   ⚠️ Failed to extract embedding for segment {idx}: {e}")
                # Use zero embedding as fallback
                embeddings.append(np.zeros(self.embedding_dim))
        
        if show_progress:
            print(f"   ✅ Extracted {len(embeddings)} embeddings" + " " * 20)
        
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
