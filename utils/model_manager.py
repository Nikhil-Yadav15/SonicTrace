# utils/model_manager.py (COMPLETE WINDOWS-COMPATIBLE VERSION)

import os
import json
import torch
import whisper
from pathlib import Path
from huggingface_hub import snapshot_download, HfFolder
import time
import warnings
from config import Config

# Disable symlink warnings globally
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')

class ModelManager:
    """Smart model manager with Windows compatibility"""
    
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else Config.MODELS_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.cache_dir / "download_status.json"
        self.load_status()
        
        # Model registry
        self.models = {
            'silero_vad': {
                'name': 'Silero VAD',
                'size_mb': 75,
                'type': 'torch_hub',
                'repo': 'snakers4/silero-vad',
                'model': 'silero_vad',
                'required': True,
                'needs_auth': False
            },
            'pyannote_embedding': {
                'name': 'PyAnnote Embedding',
                'size_mb': 500,
                'type': 'huggingface',
                'repo': 'pyannote/embedding',
                'required': True,
                'needs_auth': True,
                'license_url': 'https://huggingface.co/pyannote/embedding'
            },
            'wav2vec2_emotion': {
                'name': 'Wav2Vec2 Emotion',
                'size_mb': 1200,
                'type': 'huggingface',
                'repo': 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
                'required': True,
                'needs_auth': False
            },
            'whisper_small': {
                'name': f'Whisper {Config.WHISPER_MODEL_SIZE}',
                'size_mb': self._get_whisper_size(Config.WHISPER_MODEL_SIZE),
                'type': 'whisper',
                'model_name': Config.WHISPER_MODEL_SIZE,
                'required': True,
                'needs_auth': False
            }
        }
    
    def _get_whisper_size(self, model_name):
        """Get Whisper model size"""
        sizes = {
            'tiny': 75,
            'base': 145,
            'small': 488,
            'medium': 1500,
            'large': 3000
        }
        return sizes.get(model_name, 488)
    
    def load_status(self):
        """Load download status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                self.download_status = json.load(f)
        else:
            self.download_status = {}
    
    def save_status(self):
        """Save download status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.download_status, f, indent=2)
    
    def is_model_downloaded(self, model_key):
        """Check if model is downloaded"""
        return self.download_status.get(model_key, {}).get('downloaded', False)
    
    def get_hf_token(self):
        """Get HuggingFace token from config or CLI"""
        if Config.HF_TOKEN:
            return Config.HF_TOKEN
        
        try:
            token = HfFolder.get_token()
            if token:
                return token
        except:
            pass
        
        return None
    
    def check_authentication(self):
        """Check if HuggingFace authentication is properly configured"""
        token = self.get_hf_token()
        
        needs_auth_models = [
            info for info in self.models.values() 
            if info.get('needs_auth', False) and info['required']
        ]
        
        if not needs_auth_models:
            return True
        
        if not token:
            print("\n" + "="*70)
            print("🔐 HUGGINGFACE AUTHENTICATION REQUIRED")
            print("="*70)
            print("\nAdd HF_TOKEN to .env file")
            print("="*70 + "\n")
            return False
        
        try:
            from huggingface_hub import whoami
            user_info = whoami(token=token)
            print(f"✅ Authenticated as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Token verification failed: {e}")
            return False
    
    def get_download_info(self):
        """Get information about what needs to be downloaded"""
        to_download = []
        total_size = 0
        
        for key, info in self.models.items():
            if not self.is_model_downloaded(key) and info['required']:
                to_download.append(info['name'])
                total_size += info['size_mb']
        
        return to_download, total_size
    
    def download_all(self, progress_callback=None):
        """Download all required models"""
        
        if not self.check_authentication():
            return False
        
        to_download, total_size = self.get_download_info()
        
        if not to_download:
            print("✅ All models already downloaded!")
            return True
        
        print("\n" + "="*70)
        print("📦 MODEL DOWNLOAD")
        print("="*70)
        print(f"Models: {', '.join(to_download)}")
        print(f"Size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        print(f"Time: {self._estimate_time(total_size)}")
        print("="*70 + "\n")
        
        response = input("Start download? (y/n): ").lower()
        if response != 'y':
            print("❌ Download cancelled.")
            return False
        
        print("\n🚀 Downloading...\n")
        
        downloaded_count = 0
        for key, info in self.models.items():
            if self.is_model_downloaded(key) or not info['required']:
                continue
            
            downloaded_count += 1
            print(f"[{downloaded_count}/{len(to_download)}] {info['name']} ({info['size_mb']} MB)")
            
            try:
                start_time = time.time()
                self._download_model(key, info)
                elapsed = time.time() - start_time
                
                self.download_status[key] = {
                    'downloaded': True,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'size_mb': info['size_mb'],
                    'download_time': f"{elapsed:.1f}s"
                }
                self.save_status()
                
                print(f"   ✅ Done in {elapsed:.1f}s\n")
                
                if progress_callback:
                    progress_callback(key, downloaded_count, len(to_download))
                
            except Exception as e:
                print(f"   ❌ Failed: {e}\n")
                return False
        
        print("="*70)
        print("🎉 ALL MODELS DOWNLOADED!")
        print("="*70 + "\n")
        
        return True
    
    def _download_model(self, key, info):
        """Download a specific model"""
        
        if info['type'] == 'torch_hub':
            self._download_torch_hub(info)
        elif info['type'] == 'huggingface':
            self._download_huggingface(info)
        elif info['type'] == 'whisper':
            self._download_whisper(info)
    
    def _download_torch_hub(self, info):
        """Download Torch Hub model"""
        torch.hub.load(
            repo_or_dir=info['repo'],
            model=info['model'],
            force_reload=False,
            skip_validation=True,
            verbose=False
        )
    
    def _download_huggingface(self, info):
        """Download HuggingFace model (Windows-compatible, no symlinks)"""
        
        # Create local directory for this model
        local_path = self.cache_dir / info['repo'].replace('/', '_')
        local_path.mkdir(parents=True, exist_ok=True)
        
        token = self.get_hf_token()
        
        # Download directly to local directory WITHOUT symlinks
        snapshot_download(
            repo_id=info['repo'],
            local_dir=str(local_path),
            local_dir_use_symlinks=False,  # CRITICAL: No symlinks on Windows
            token=token,
        )
    
    def _download_whisper(self, info):
        """Download Whisper model"""
        
        whisper_cache = self.cache_dir / 'whisper'
        whisper_cache.mkdir(exist_ok=True)
        
        whisper.load_model(
            info['model_name'],
            download_root=str(whisper_cache)
        )
    
    def _estimate_time(self, size_mb):
        """Estimate download time"""
        mbps = 50
        seconds = (size_mb * 8) / mbps
        
        if seconds < 60:
            return f"~{int(seconds)} seconds"
        else:
            minutes = seconds / 60
            return f"~{int(minutes)} minutes"
    
    def verify_downloads(self):
        """Verify all required models are downloaded"""
        missing = []
        
        for key, info in self.models.items():
            if info['required'] and not self.is_model_downloaded(key):
                missing.append(info['name'])
        
        if missing:
            print(f"⚠️ Missing: {', '.join(missing)}")
            print("Run: uv run python setup_models.py")
            return False
        
        print("✅ All models downloaded")
        return True
    
    def get_cache_size(self):
        """Get total cache size in GB"""
        total_size = 0
        
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        
        return total_size / (1024**3)
    
    def print_status(self):
        """Print download status"""
        print("\n" + "="*70)
        print("📊 MODEL STATUS")
        print("="*70)
        
        for key, info in self.models.items():
            status = "✅" if self.is_model_downloaded(key) else "❌"
            auth = " 🔐" if info.get('needs_auth') else ""
            print(f"{status}{auth} {info['name']}: {info['size_mb']} MB")
        
        cache_size = self.get_cache_size()
        print(f"\nCache: {cache_size:.2f} GB")
        print("="*70 + "\n")
