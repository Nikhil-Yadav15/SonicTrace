# main.py

"""
SonicTrace command-line interface

Usage:
    python main.py path/to/audio.wav
    python main.py audio.mp3 -o results/ --formats json srt txt
    python main.py audio.wav --speakers 3 --no-emotion
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        prog="sonictrace",
        description="Speaker diarization with overlap detection, "
                    "emotion recognition and transcription.",
    )
    parser.add_argument("audio", help="Path to the audio file to analyze")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory for result files (default: data/results)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["json"],
        choices=["json", "txt", "srt", "vtt"],
        help="Output formats to write (default: json)",
    )
    parser.add_argument(
        "--speakers", type=int, default=None,
        help="Number of speakers (omit to auto-detect)",
    )
    parser.add_argument("--no-overlap", action="store_true",
                        help="Skip overlap detection")
    parser.add_argument("--no-diarization", action="store_true",
                        help="Skip speaker diarization")
    parser.add_argument("--no-emotion", action="store_true",
                        help="Skip emotion recognition")
    parser.add_argument("--no-transcription", action="store_true",
                        help="Skip transcription")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")
    return parser.parse_args()


def main():
    args = parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    from config import Config
    from core.audio_processor import audio_processor

    output_dir = Path(args.output_dir) if args.output_dir else Config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = audio_processor.process(
        str(audio_path),
        enable_overlap=not args.no_overlap,
        enable_diarization=not args.no_diarization,
        enable_emotion=not args.no_emotion,
        enable_transcription=not args.no_transcription,
        n_speakers=args.speakers,
        show_progress=not args.quiet,
    )

    base = audio_path.stem
    for fmt in args.formats:
        out_path = output_dir / f"{base}_sonictrace.{fmt}"
        audio_processor.save_results(results, out_path, format=fmt)

    summary = results["summary"]
    print(f"\nDone. {summary['n_speakers']} speaker(s), "
          f"{summary['total_segments']} segment(s), "
          f"{summary['total_words']} word(s) transcribed.")


if __name__ == "__main__":
    main()
