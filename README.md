# 🎤 SonicTrace

**AI-powered audio analysis platform for speaker diarization, emotion detection, and speech transcription**

SonicTrace is a comprehensive audio analysis tool that combines state-of-the-art deep learning models to provide detailed insights from audio recordings. Perfect for analyzing meetings, interviews, podcasts, and any multi-speaker conversations.

## ✨ Features

- **🎯 Voice Activity Detection (VAD)**: Automatically detect and segment speech regions using Silero VAD
- **👥 Speaker Diarization**: Identify and separate different speakers using PyAnnote embeddings
- **🔀 Overlap Detection**: Detect when multiple speakers talk simultaneously
- **😊 Emotion Recognition**: Analyze emotional tone in speech using Wav2Vec2
- **📝 Speech Transcription**: Convert speech to text with timestamps using OpenAI Whisper
- **📊 Interactive Dashboard**: Beautiful Streamlit web interface with visualizations
- **🎨 Rich Visualizations**: Timeline views, emotion distributions, and speaker statistics

## 🚀 Quick Start

### Prerequisites

- Python 3.9 - 3.12
- FFmpeg (for audio processing)
- CUDA-capable GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd SonicTrace
```

2. **Install FFmpeg**
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

3. **Install Python dependencies**
```bash
pip install -e .
```

4. **Set up models**
```bash
python setup_models.py
```

This will download:
- Whisper speech recognition model (small)
- PyAnnote speaker embedding model
- Wav2Vec2 emotion recognition model

5. **Create environment file** (optional)
```bash
# Create .env file with custom settings
MODEL_CACHE_DIR=./models
DEVICE=auto
SAMPLE_RATE=16000
```

## 💻 Usage

### Web Interface (Recommended)

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload audio files (MP3, WAV, M4A, FLAC, OGG, WebM, MP4, AAC)
- Toggle analysis components (diarization, emotion, transcription, overlap)
- View interactive timelines and visualizations
- Export results as JSON
- Download comprehensive analysis reports

### Command Line

```python
from core.audio_processor import audio_processor

# Process an audio file
results = audio_processor.process(
    audio_path="path/to/audio.mp3",
    enable_overlap=True,
    enable_diarization=True,
    enable_emotion=True,
    enable_transcription=True,
    n_speakers=None,  # Auto-detect
    show_progress=True
)

# Access results
print(f"Detected {results['num_speakers']} speakers")
for segment in results['segments']:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
          f"Speaker {segment['speaker']}: {segment['text']} "
          f"(Emotion: {segment['emotion']})")
```

## 📁 Project Structure

```
SonicTrace/
├── app.py                      # Streamlit web interface
├── config.py                   # Configuration settings
├── main.py                     # Entry point
├── pyproject.toml              # Project dependencies
├── setup_models.py             # Model download script
├── setup_ffmpeg.py             # FFmpeg setup helper
├── core/                       # Core processing modules
│   ├── audio_processor.py      # Main processing pipeline
│   ├── vad_processor.py        # Voice Activity Detection
│   ├── speaker_embeddings.py   # Speaker embedding extraction
│   ├── speaker_clustering.py   # Speaker clustering/diarization
│   ├── overlap_detector.py     # Overlap detection
│   ├── emotion_recognizer.py   # Emotion recognition
│   └── transcriber.py          # Speech transcription
├── utils/                      # Utility modules
│   ├── audio_loader.py         # Audio file loading
│   ├── audio_preprocessor.py   # Audio preprocessing
│   ├── ffmpeg_handler.py       # FFmpeg operations
│   └── model_manager.py        # Model management
├── data/                       # Data directories
│   ├── uploads/                # Uploaded audio files
│   └── results/                # Analysis results
├── models/                     # Downloaded ML models
└── test_*.py                   # Test scripts
```

## 🔧 Configuration

Edit `config.py` or create a `.env` file to customize:

```python
# Audio Settings
SAMPLE_RATE = 16000              # Audio sample rate
MAX_FILE_SIZE_MB = 500           # Maximum upload size

# Processing
DEVICE = "auto"                  # auto, cpu, cuda
WHISPER_MODEL_SIZE = "small"     # tiny, base, small, medium, large

# VAD Settings
VAD_MIN_SPEECH_DURATION_MS = 500
VAD_MIN_SILENCE_DURATION_MS = 300
VAD_MIN_SEGMENT_DURATION = 0.5

# Overlap Detection
OVERLAP_PITCH_THRESHOLD = 2
OVERLAP_ENERGY_THRESHOLD = 0.25
```

## 🧪 Testing

Run individual component tests:

```bash
# Test audio loading
python test_audio_loading.py

# Test VAD
python test_vad.py

# Test speaker diarization
python test_speaker_diarization.py

# Test emotion recognition
python test_emotion.py

# Test transcription
python test_transcription.py

# Test overlap detection
python test_overlap.py

# Test full pipeline
python test_pipeline.py
```

## 📊 Output Format

The analysis returns a structured JSON with:

```json
{
  "audio_file": "path/to/audio.mp3",
  "duration": 120.5,
  "num_speakers": 3,
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": 0,
      "text": "Hello, how are you?",
      "emotion": "happy",
      "emotion_scores": {
        "happy": 0.85,
        "neutral": 0.10,
        "sad": 0.05
      },
      "is_overlap": false
    }
  ],
  "speakers": {
    "0": {
      "total_duration": 45.2,
      "segments": 23,
      "emotions": {...}
    }
  },
  "overlaps": [...],
  "statistics": {...}
}
```

## 🎯 Use Cases

- **Meeting Analysis**: Automatically transcribe and analyze team meetings
- **Interview Processing**: Extract insights from job interviews or surveys
- **Podcast Production**: Generate transcripts and speaker labels for podcasts
- **Customer Service**: Analyze call center recordings for quality assurance
- **Research**: Study conversation dynamics and emotional patterns
- **Media Production**: Process audio for subtitling and captioning

## 🛠️ Technologies

- **[OpenAI Whisper](https://github.com/openai/whisper)**: State-of-the-art speech recognition
- **[PyAnnote Audio](https://github.com/pyannote/pyannote-audio)**: Speaker diarization and embeddings
- **[Wav2Vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html)**: Emotion recognition
- **[Silero VAD](https://github.com/snakers4/silero-vad)**: Voice activity detection
- **[Streamlit](https://streamlit.io/)**: Interactive web interface
- **[Plotly](https://plotly.com/)**: Data visualization
- **[LibROSA](https://librosa.org/)**: Audio processing
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

## ⚙️ System Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 8GB
- Storage: 5GB (for models)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 10GB

## 📝 License

This project uses the following models and libraries:
- OpenAI Whisper: MIT License
- PyAnnote Audio: MIT License
- Silero VAD: MIT License
- Wav2Vec2 models: Various (check model cards)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Support

For questions, issues, or feature requests, please open an issue on GitHub.

## 🙏 Acknowledgments

Built with amazing open-source AI models from:
- OpenAI (Whisper)
- PyAnnote Team
- Hugging Face Community
- Silero Team

---

**Made with ❤️ for audio analysis**
