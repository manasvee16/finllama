from huggingface_hub import login
import os

# Set your token via environment variable HF_TOKEN or use setup_huggingface_auth.py
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
    print("Hugging Face login successful!")
else:
    print("HF_TOKEN environment variable not set. Please set it or run setup_huggingface_auth.py")
