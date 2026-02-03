#!/usr/bin/env python
"""
Quick setup and verification script for ECG-ReGen
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'wfdb': 'WFDB (ECG data reader)',
        'faiss': 'FAISS (similarity search)',
        'nltk': 'NLTK',
        'rouge_score': 'Rouge Score',
        'bert_score': 'BERTScore'
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_data():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    required_files = [
        'data/ptbxl_train_proper.jsonl',
        'data/ptbxl_val_proper.jsonl',
        'data/ptbxl_test_proper.jsonl',
        'data/ptbxl/ptbxl_database.csv'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing.append(file_path)
    
    if missing:
        print(f"\nWarning: Some data files are missing")
        return False
    
    print("\n✓ All data files found!")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        return False


def verify_gpu():
    """Check if GPU is available"""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("✗ No GPU available (will use CPU - training will be slower)")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def main():
    print("=" * 60)
    print("ECG-ReGen Setup and Verification")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠ Please install missing dependencies first")
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Check GPU
    verify_gpu()
    
    # Check data
    data_ok = check_data()
    
    print("\n" + "=" * 60)
    if deps_ok and data_ok:
        print("✓ Setup complete! Ready to train.")
        print("\nTo start training:")
        print("  python main.py --mode train")
        print("\nTo evaluate:")
        print("  python main.py --mode evaluate --checkpoint outputs/best_model.pt")
    else:
        print("⚠ Setup incomplete. Please resolve issues above.")
    print("=" * 60)


if __name__ == '__main__':
    main()
