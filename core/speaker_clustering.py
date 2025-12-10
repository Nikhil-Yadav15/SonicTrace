# core/speaker_clustering.py

"""
Speaker Clustering - Automatically cluster speech segments by speaker
Auto-detects number of speakers and assigns labels
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
from config import Config

class SpeakerClusterer:
    """Cluster speech segments by speaker"""
    
    def __init__(self):
        """Initialize speaker clusterer"""
        self.min_speakers = Config.MIN_SPEAKERS
        self.max_speakers = Config.MAX_SPEAKERS
        self.clustering_method = Config.CLUSTERING_METHOD
        
        print("✅ Speaker Clusterer initialized")
    
    def auto_detect_speakers(
        self,
        embeddings: List[np.ndarray],
        min_speakers: int = None,
        max_speakers: int = None
    ) -> int:
        """
        Automatically detect optimal number of speakers
        
        Args:
            embeddings: List of speaker embeddings
            min_speakers: Minimum number of speakers to try
            max_speakers: Maximum number of speakers to try
        
        Returns:
            Optimal number of speakers
        """
        min_spk = min_speakers or self.min_speakers
        max_spk = max_speakers or self.max_speakers
        
        n_segments = len(embeddings)
        
        # Ensure max speakers doesn't exceed segments
        max_spk = min(max_spk, n_segments)
        
        if n_segments < 2:
            return 1
        
        # Convert embeddings to array
        X = np.array(embeddings)
        
        # Try different numbers of speakers
        best_n_speakers = min_spk
        best_score = -1
        
        print(f"\n🔍 Auto-detecting number of speakers ({min_spk}-{max_spk})...")
        
        for n_speakers in range(min_spk, max_spk + 1):
            try:
                # Cluster
                clustering = AgglomerativeClustering(
                    n_clusters=n_speakers,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(X)
                
                # Calculate silhouette score (quality metric)
                if n_speakers > 1 and len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels, metric='cosine')
                else:
                    score = 0
                
                print(f"   {n_speakers} speakers: score = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_n_speakers = n_speakers
                    
            except Exception as e:
                print(f"   {n_speakers} speakers: failed ({e})")
                continue
        
        print(f"   ✅ Detected {best_n_speakers} speakers (score: {best_score:.3f})")
        
        return best_n_speakers
    
    def cluster_speakers(
        self,
        embeddings: List[np.ndarray],
        n_speakers: Optional[int] = None,
        method: str = None
    ) -> np.ndarray:
        """
        Cluster embeddings into speakers
        
        Args:
            embeddings: List of speaker embeddings
            n_speakers: Number of speakers (auto-detect if None)
            method: Clustering method ('agglomerative' or 'dbscan')
        
        Returns:
            Array of speaker labels (0, 1, 2, ...)
        """
        method = method or self.clustering_method
        
        # Convert to array
        X = np.array(embeddings)
        
        if len(X) == 0:
            return np.array([])
        
        # Auto-detect number of speakers if not specified
        if n_speakers is None:
            n_speakers = self.auto_detect_speakers(embeddings)
        
        print(f"\n👥 Clustering {len(embeddings)} segments into {n_speakers} speakers...")
        
        # Cluster using specified method
        if method == 'agglomerative':
            labels = self._cluster_agglomerative(X, n_speakers)
        elif method == 'dbscan':
            labels = self._cluster_dbscan(X)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Print cluster distribution
        unique_labels = np.unique(labels)
        print(f"   Found {len(unique_labels)} clusters:")
        for label in unique_labels:
            count = np.sum(labels == label)
            percentage = (count / len(labels)) * 100
            print(f"   Speaker {label}: {count} segments ({percentage:.1f}%)")
        
        return labels
    
    def _cluster_agglomerative(
        self,
        X: np.ndarray,
        n_speakers: int
    ) -> np.ndarray:
        """Agglomerative hierarchical clustering"""
        
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(X)
        
        return labels
    
    def _cluster_dbscan(self, X: np.ndarray) -> np.ndarray:
        """DBSCAN clustering (density-based)"""
        
        # DBSCAN with cosine distance
        # eps is the maximum distance between samples
        clustering = DBSCAN(
            eps=0.3,
            min_samples=2,
            metric='cosine'
        )
        
        labels = clustering.fit_predict(X)
        
        # DBSCAN uses -1 for noise, remap to positive integers
        if -1 in labels:
            labels = labels + 1
        
        return labels
    
    def assign_speaker_labels(
        self,
        segments: List[Dict],
        cluster_labels: np.ndarray,
        label_format: str = 'letter'
    ) -> List[Dict]:
        """
        Assign speaker labels to segments
        
        Args:
            segments: List of segment dictionaries
            cluster_labels: Cluster labels from clustering
            label_format: 'letter' (A, B, C) or 'number' (0, 1, 2)
        
        Returns:
            Segments with speaker labels added
        """
        labeled_segments = []
        
        for segment, label in zip(segments, cluster_labels):
            seg = segment.copy()
            
            # Convert label to desired format
            if label_format == 'letter':
                speaker_label = chr(65 + int(label))  # 65 = 'A'
            else:
                speaker_label = str(label)
            
            seg['speaker'] = speaker_label
            seg['speaker_id'] = int(label)
            
            labeled_segments.append(seg)
        
        return labeled_segments
    
    def get_speaker_statistics(
        self,
        segments: List[Dict]
    ) -> Dict:
        """
        Calculate speaker statistics
        
        Returns:
            Dictionary with speaker stats
        """
        if not segments:
            return {'total_speakers': 0}
        
        # Get unique speakers
        speakers = set(seg.get('speaker', 'Unknown') for seg in segments)
        
        # Calculate per-speaker stats
        speaker_stats = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s.get('speaker') == speaker]
            
            total_duration = sum(s['duration'] for s in speaker_segments)
            
            speaker_stats[speaker] = {
                'segments': len(speaker_segments),
                'total_duration': total_duration,
                'avg_duration': total_duration / len(speaker_segments),
                'percentage': 0.0  # Will calculate below
            }
        
        # Calculate percentages
        total_duration = sum(stats['total_duration'] for stats in speaker_stats.values())
        
        for speaker in speaker_stats:
            if total_duration > 0:
                speaker_stats[speaker]['percentage'] = (
                    speaker_stats[speaker]['total_duration'] / total_duration * 100
                )
        
        return {
            'total_speakers': len(speakers),
            'speakers': speaker_stats,
            'total_duration': total_duration
        }
    
    def refine_clusters(
        self,
        embeddings: List[np.ndarray],
        labels: np.ndarray,
        min_similarity: float = 0.7
    ) -> np.ndarray:
        """
        Refine cluster assignments based on similarity threshold
        
        Args:
            embeddings: Speaker embeddings
            labels: Initial cluster labels
            min_similarity: Minimum similarity to assign to cluster
        
        Returns:
            Refined labels
        """
        refined_labels = labels.copy()
        
        # Calculate cluster centroids
        unique_labels = np.unique(labels)
        centroids = {}
        
        for label in unique_labels:
            cluster_embeddings = [emb for emb, lbl in zip(embeddings, labels) if lbl == label]
            if cluster_embeddings:
                centroids[label] = np.mean(cluster_embeddings, axis=0)
        
        # Reassign segments with low similarity
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            if label in centroids:
                # Calculate similarity to assigned cluster
                centroid = centroids[label]
                norm1 = np.linalg.norm(embedding)
                norm2 = np.linalg.norm(centroid)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(embedding, centroid) / (norm1 * norm2)
                    similarity = (similarity + 1) / 2  # Convert to 0-1
                    
                    # If too dissimilar, try other clusters
                    if similarity < min_similarity:
                        best_label = label
                        best_sim = similarity
                        
                        for other_label, other_centroid in centroids.items():
                            if other_label != label:
                                norm_other = np.linalg.norm(other_centroid)
                                if norm_other > 0:
                                    other_sim = np.dot(embedding, other_centroid) / (norm1 * norm_other)
                                    other_sim = (other_sim + 1) / 2
                                    
                                    if other_sim > best_sim:
                                        best_sim = other_sim
                                        best_label = other_label
                        
                        refined_labels[idx] = best_label
        
        return refined_labels


# Global instance
speaker_clusterer = SpeakerClusterer()
