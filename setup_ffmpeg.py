# setup_ffmpeg.py

"""
FFmpeg detection and setup helper

Usage:
    uv run setup_ffmpeg.py
"""

import platform
import subprocess
from pathlib import Path
from config import Config

def check_ffmpeg():
    """Check FFmpeg installation"""
    
    print("\n🎬 FFmpeg Setup Check")
    print("="*60)
    
    # Check configured path
    if Config.FFMPEG_PATH:
        ffmpeg_path = Path(Config.FFMPEG_PATH)
        if ffmpeg_path.exists():
            print(f"✅ FFmpeg found at: {ffmpeg_path}")
            print_ffmpeg_version(str(ffmpeg_path))
            return True
        else:
            print(f"❌ Configured path not found: {ffmpeg_path}")
    
    # Check system PATH
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=3
        )
        if result.returncode == 0:
            print("✅ FFmpeg found in system PATH")
            print_ffmpeg_version('ffmpeg')
            return True
    except:
        pass
    
    # Not found
    print("❌ FFmpeg not found")
    print_installation_instructions()
    return False

def print_ffmpeg_version(ffmpeg_cmd):
    """Print FFmpeg version info"""
    try:
        result = subprocess.run(
            [ffmpeg_cmd, '-version'],
            capture_output=True,
            text=True,
            timeout=3
        )
        version_line = result.stdout.split('\n')[0]
        print(f"   Version: {version_line}")
    except:
        pass

def print_installation_instructions():
    """Print OS-specific installation instructions"""
    
    print("\n📝 Installation Instructions:")
    print("="*60)
    
    system = platform.system()
    
    if system == "Windows":
        print("""
Windows:

Option 1 - Download & Extract:
1. Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract to C:\\ffmpeg
3. Update .env file:
   FFMPEG_PATH=C:\\ffmpeg\\bin\\ffmpeg.exe

Option 2 - Add to PATH:
1. Download and extract as above
2. Add C:\\ffmpeg\\bin to system PATH
3. Restart terminal

Option 3 - Use Chocolatey:
   choco install ffmpeg
        """)
    
    elif system == "Darwin":  # macOS
        print("""
macOS:

Using Homebrew:
   brew install ffmpeg

Using MacPorts:
   sudo port install ffmpeg
        """)
    
    else:  # Linux
        print("""
Linux:

Ubuntu/Debian:
   sudo apt update
   sudo apt install ffmpeg

Fedora:
   sudo dnf install ffmpeg

Arch:
   sudo pacman -S ffmpeg
        """)
    
    print("="*60)

def main():
    success = check_ffmpeg()
    
    if not success:
        print("\n⚠️ FFmpeg is optional but recommended for best compatibility")
        print("   The app will work without it for WAV files")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
