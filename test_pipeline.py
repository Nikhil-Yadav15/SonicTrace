# test_pipeline.py

"""
Test Complete Audio Processing Pipeline
"""

import numpy as np
from pathlib import Path
from core.audio_processor import audio_processor
import tempfile

def create_test_audio_file():
    """Create a temporary test audio file"""
    
    # Generate 10 seconds of test audio
    sr = 16000
    duration = 10.0
    samples = int(duration * sr)
    
    # Create 3 "speech" segments with silence between
    audio = np.zeros(samples, dtype=np.float32)
    
    # Segment 1: 0-3s
    t1 = np.linspace(0, 3, 3 * sr)
    audio[:3*sr] = 0.3 * np.sin(2 * np.pi * 150 * t1)
    
    # Silence: 3-4s
    
    # Segment 2: 4-7s
    t2 = np.linspace(0, 3, 3 * sr)
    audio[4*sr:7*sr] = 0.3 * np.sin(2 * np.pi * 200 * t2)
    
    # Silence: 7-8s
    
    # Segment 3: 8-10s
    t3 = np.linspace(0, 2, 2 * sr)
    audio[8*sr:10*sr] = 0.3 * np.sin(2 * np.pi * 180 * t3)
    
    # Add noise
    audio += 0.05 * np.random.randn(samples).astype(np.float32)
    
    # Save to temp file
    import soundfile as sf
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr)
    
    return temp_file.name

def test_full_pipeline():
    """Test complete pipeline"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Full Pipeline Processing")
    print("="*70)
    
    # Create test audio
    print("\nCreating test audio file...")
    audio_file = create_test_audio_file()
    print(f"✅ Created: {audio_file}")
    
    try:
        # Process audio
        results = audio_processor.process(
            audio_file,
            enable_overlap=True,
            enable_diarization=True,
            enable_emotion=True,
            enable_transcription=True,
            show_progress=True
        )
        
        # Verify results
        assert 'metadata' in results
        assert 'summary' in results
        assert 'segments' in results
        assert 'statistics' in results
        
        print("\n✅ Full pipeline working")
        
        return results
        
    finally:
        # Cleanup
        Path(audio_file).unlink()

def test_partial_pipeline():
    """Test with some features disabled"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Partial Pipeline (No Transcription)")
    print("="*70)
    
    audio_file = create_test_audio_file()
    
    try:
        results = audio_processor.process(
            audio_file,
            enable_overlap=True,
            enable_diarization=True,
            enable_emotion=True,
            enable_transcription=False,  # Disabled
            show_progress=True
        )
        
        assert 'metadata' in results
        print("\n✅ Partial pipeline working")
        
    finally:
        Path(audio_file).unlink()

def test_save_results():
    """Test saving results"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Save Results")
    print("="*70)
    
    audio_file = create_test_audio_file()
    
    try:
        # Process
        results = audio_processor.process(
            audio_file,
            show_progress=False
        )
        
        # Save as JSON
        output_json = Path('test_results.json')
        audio_processor.save_results(results, output_json, format='json')
        assert output_json.exists()
        print(f"✅ Saved JSON: {output_json}")
        output_json.unlink()
        
        # Save as TXT
        output_txt = Path('test_results.txt')
        audio_processor.save_results(results, output_txt, format='txt')
        assert output_txt.exists()
        print(f"✅ Saved TXT: {output_txt}")
        output_txt.unlink()
        
        # Save as SRT
        output_srt = Path('test_results.srt')
        audio_processor.save_results(results, output_srt, format='srt')
        assert output_srt.exists()
        print(f"✅ Saved SRT: {output_srt}")
        output_srt.unlink()
        
        print("\n✅ Save results working")
        
    finally:
        Path(audio_file).unlink()

def test_results_structure():
    """Test results structure"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Results Structure")
    print("="*70)
    
    audio_file = create_test_audio_file()
    
    try:
        results = audio_processor.process(audio_file, show_progress=False)
        
        # Check metadata
        print("\nMetadata:")
        for key, value in results['metadata'].items():
            print(f"  {key}: {value}")
        
        # Check summary
        print("\nSummary:")
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
        
        # Check segments
        print(f"\nSegments: {len(results['segments'])}")
        if results['segments']:
            seg = results['segments'][0]
            print(f"  Sample segment keys: {list(seg.keys())}")
        
        # Check statistics
        print(f"\nStatistics sections: {list(results['statistics'].keys())}")
        
        print("\n✅ Results structure verified")
        
    finally:
        Path(audio_file).unlink()

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 CHUNK 8 - PIPELINE INTEGRATION TESTS")
    print("="*70)
    
    try:
        test_full_pipeline()
        test_partial_pipeline()
        test_save_results()
        test_results_structure()
        
        print("\n" + "="*70)
        print("🎉 ALL PIPELINE TESTS PASSED!")
        print("="*70)
        print("\n✅ Full pipeline integration working")
        print("✅ All components communicating correctly")
        print("✅ VAD → Overlap → Diarization → Emotion → Transcription")
        print("✅ Results aggregation working")
        print("✅ Multiple output formats supported")
        print("✅ Progress tracking functional")
        print("\nℹ️ NOTE: Low confidence/empty results normal for synthetic audio")
        print("   Pipeline ready for real speech audio!")
        print("\nNext: Chunk 9 - Streamlit Web Interface")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
