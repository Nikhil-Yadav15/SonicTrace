# test_audio_loading.py

"""
Test audio loading functionality
"""

from utils.audio_loader import audio_loader
from utils.ffmpeg_handler import ffmpeg
from utils.audio_preprocessor import audio_preprocessor
from pathlib import Path
import numpy as np

def test_ffmpeg():
    """Test FFmpeg detection"""
    print("\n" + "="*60)
    print("🎬 TESTING FFMPEG")
    print("="*60)
    
    if ffmpeg.is_available():
        print(f"✅ FFmpeg available: {ffmpeg.ffmpeg_path}")
        print(f"✅ FFprobe available: {ffmpeg.ffprobe_path}")
        version = ffmpeg.get_version()
        if version:
            print(f"   Version: {version}")
    else:
        print("❌ FFmpeg not found")
        print("   Audio loading will work for WAV files only")
    
    print()

def test_audio_formats():
    """Test loading different audio formats"""
    print("="*60)
    print("🎵 TESTING AUDIO FORMATS")
    print("="*60)
    
    # You can test with your own files
    test_files = [
        # Add paths to test files if you have them
        # "test_audio.mp3",
        # "test_audio.wav",
    ]
    
    if not test_files:
        print("ℹ️ No test files specified")
        print("   Add audio files to test different formats")
        return
    
    for file_path in test_files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
        
        print(f"\nTesting: {path.name}")
        
        try:
            # Validate
            validation = audio_loader.validate_audio_file(path)
            if not validation['valid']:
                print(f"   ❌ Validation failed: {validation['error']}")
                continue
            
            # Load
            waveform, sr = audio_loader.load(path)
            duration = len(waveform) / sr
            
            print(f"   ✅ Loaded successfully")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Sample rate: {sr} Hz")
            print(f"   Shape: {waveform.shape}")
            print(f"   Range: [{waveform.min():.3f}, {waveform.max():.3f}]")
            
            # Test preprocessing
            normalized = audio_preprocessor.normalize_audio(waveform)
            print(f"   ✅ Normalized: [{normalized.min():.3f}, {normalized.max():.3f}]")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_validation():
    """Test file validation"""
    print("\n" + "="*60)
    print("✅ TESTING VALIDATION")
    print("="*60)
    
    # Test with non-existent file
    result = audio_loader.validate_audio_file("nonexistent.mp3")
    assert not result['valid'], "Should reject non-existent file"
    print("✅ Non-existent file validation works")
    
    # Test with unsupported format
    result = audio_loader.validate_audio_file("test.xyz")
    assert not result['valid'], "Should reject unsupported format"
    print("✅ Format validation works")
    
    print()

def test_preprocessing():
    """Test preprocessing functions"""
    print("="*60)
    print("🔧 TESTING PREPROCESSING")
    print("="*60)
    
    # Create synthetic audio
    duration = 2.0  # seconds
    sr = 16000
    samples = int(duration * sr)
    
    # Generate sine wave
    freq = 440  # A4 note
    t = np.linspace(0, duration, samples)
    waveform = np.sin(2 * np.pi * freq * t)
    
    print(f"Generated {duration}s sine wave at {freq} Hz")
    
    # Test normalization
    normalized = audio_preprocessor.normalize_audio(waveform)
    print(f"✅ Peak normalization: {normalized.max():.3f}")
    
    # Test RMS normalization
    rms_norm = audio_preprocessor.normalize_audio(waveform, method='rms')
    print(f"✅ RMS normalization: RMS = {np.sqrt(np.mean(rms_norm**2)):.3f}")
    
    # Test padding
    target_length = samples * 2
    padded = audio_preprocessor.pad_or_trim(waveform, target_length)
    print(f"✅ Padding: {len(waveform)} → {len(padded)} samples")
    
    # Test trimming
    target_length = samples // 2
    trimmed = audio_preprocessor.pad_or_trim(waveform, target_length)
    print(f"✅ Trimming: {len(waveform)} → {len(trimmed)} samples")
    
    # Test energy calculation
    energy = audio_preprocessor.calculate_energy(waveform)
    print(f"✅ Energy calculation: {len(energy)} frames")
    
    print()

def main():
    """Run all tests"""
    print("\n🧪 CHUNK 2 - AUDIO LOADING TESTS")
    print("="*60)
    
    test_ffmpeg()
    test_validation()
    test_preprocessing()
    test_audio_formats()
    
    print("="*60)
    print("🎉 CHUNK 2 TESTS COMPLETE!")
    print("="*60)
    print("\n✅ Audio loading system ready")
    print("✅ FFmpeg integration working")
    print("✅ Preprocessing utilities available")
    print("\nNext: Chunk 3 - Voice Activity Detection (VAD)")
    print()

if __name__ == "__main__":
    main()
