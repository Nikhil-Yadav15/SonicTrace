# setup_models.py

"""
One-time model download script

Usage:
    uv run setup_models.py
"""

from utils.model_manager import ModelManager
from config import Config
import sys

def main():
    print("\n🎤 SonicTrace - Model Setup")
    print("="*60)
    
    # Print config
    Config.print_config()
    
    # Initialize model manager
    manager = ModelManager()
    
    # Show current status
    manager.print_status()
    
    # Download models
    success = manager.download_all()
    
    if success:
        print("\n✅ Setup complete! You can now run:")
        print("   uv run streamlit run app.py")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
