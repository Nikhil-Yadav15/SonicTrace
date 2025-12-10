# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
class Config:
    """Application configuration"""
    
    # ===== Paths =====
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    RESULTS_DIR = DATA_DIR / "results"
    MODELS_DIR = Path(os.getenv("MODEL_CACHE_DIR", "./models"))
    
    # ===== FFmpeg =====
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", None)
    
    # ===== Audio Settings =====
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    SUPPORTED_FORMATS = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'webm', 'mp4', 'aac']
    
    # ===== Processing Settings =====
    DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda
    NUM_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
    
    # ===== Model Settings =====
    WHISPER_MODEL_SIZE = "small"  # tiny, base, small, medium, large
    
    # ===== VAD Settings =====
    VAD_MIN_SPEECH_DURATION_MS = 500  # Minimum speech segment (500ms)
    VAD_MIN_SILENCE_DURATION_MS = 300  # Minimum silence between segments (300ms)
    VAD_SPEECH_PAD_MS = 30  # Padding around speech segments (30ms)
    
    # Post-processing
    VAD_FILTER_SHORT_SEGMENTS = True
    VAD_MERGE_CLOSE_SEGMENTS = True
    VAD_MIN_SEGMENT_DURATION = 0.5  # seconds
    VAD_MAX_MERGE_GAP = 0.3  # seconds
    
    # ===== Overlap Detection Settings =====
    OVERLAP_PITCH_THRESHOLD = 2  # Need at least 2 simultaneous pitches
    OVERLAP_ENERGY_THRESHOLD = 0.25  # Higher threshold
    OVERLAP_DETECTION_METHOD = 'auto'  # 'auto', 'spectral', 'pitch', 'energy'
    
    # ===== Speaker Clustering Settings =====
    MIN_SPEAKERS = 2
    MAX_SPEAKERS = 10
    CLUSTERING_METHOD = 'agglomerative'  # 'agglomerative' or 'dbscan'
    CLUSTERING_MIN_SIMILARITY = 0.7
    
    # ===== Emotion Recognition Settings =====
    EMOTION_MIN_CONFIDENCE = 0.3  # Minimum confidence to trust emotion
    EMOTION_MODEL_NAME = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'
    
    # ===== Transcription Settings =====
    WHISPER_MODEL_SIZE = 'small'  # 'tiny', 'base', 'small', 'medium', 'large'
    TRANSCRIPTION_LANGUAGE = 'en'  # English only
    TRANSCRIPTION_MIN_CONFIDENCE = 0.5
    
    # ===== Pipeline Settings =====
    ENABLE_OVERLAP_DETECTION = True
    ENABLE_SPEAKER_DIARIZATION = True
    ENABLE_EMOTION_RECOGNITION = True
    ENABLE_TRANSCRIPTION = True
    
    # Output
    DEFAULT_OUTPUT_FORMAT = 'json'
    RESULTS_DIR = BASE_DIR / 'data' / 'results'
    
    # ===== HuggingFace =====
    HF_TOKEN = os.getenv("HF_TOKEN", None)
    
    @classmethod
    def init_dirs(cls):
        """Create necessary directories"""
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directories initialized:")
        print(f"   - Uploads: {cls.UPLOAD_DIR}")
        print(f"   - Results: {cls.RESULTS_DIR}")
        print(f"   - Models: {cls.MODELS_DIR}")
    
    @classmethod
    def get_device(cls):
        """Get torch device"""
        import torch
        
        if cls.DEVICE == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(cls.DEVICE)
    
    @classmethod
    def validate_audio_file(cls, file_path):
        """Quick audio file validation"""
        from pathlib import Path
        
        path = Path(file_path)
        
        # Check existence
        if not path.exists():
            return False, "File not found"
        
        # Check format
        ext = path.suffix.lower().lstrip('.')
        if ext not in cls.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {ext}"
        
        # Check size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > cls.MAX_FILE_SIZE_MB:
            return False, f"File too large: {size_mb:.1f} MB"
        
        return True, None
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        import torch
        
        print("\n" + "="*50)
        print("🔧 SONICTRACE CONFIGURATION")
        print("="*50)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Models Directory: {cls.MODELS_DIR}")
        print(f"Sample Rate: {cls.SAMPLE_RATE} Hz")
        print(f"Max File Size: {cls.MAX_FILE_SIZE_MB} MB")
        print(f"Device: {cls.get_device()}")
        print(f"Workers: {cls.NUM_WORKERS}")
        print(f"Whisper Model: {cls.WHISPER_MODEL_SIZE}")
        print(f"FFmpeg: {cls.FFMPEG_PATH or 'Not configured'}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        print("="*50 + "\n")


# Initialize directories on import
Config.init_dirs()
