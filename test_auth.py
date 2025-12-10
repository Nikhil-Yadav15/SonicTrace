# test_auth.py

from config import Config
from utils.model_manager import ModelManager

print("Testing HuggingFace Authentication...\n")

manager = ModelManager()

# Check if token is configured
token = manager.get_hf_token()

if token:
    print(f"✅ Token found: {token[:10]}...{token[-5:]}")
    
    # Verify token
    if manager.check_authentication():
        print("✅ Authentication successful!")
    else:
        print("❌ Authentication failed!")
else:
    print("❌ No token found!")
    print("\nAdd to .env file:")
    print("HF_TOKEN=hf_your_token_here")
