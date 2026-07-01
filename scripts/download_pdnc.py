import os
import sys
import hashlib
from src.utils.config import get_data_dir

DATA_DIR = get_data_dir()
EXPECTED_FILES = [
    # These are illustrative files that should be present.
    # Adjust based on the actual PDNC structure.
    "coref.json",
    "quotes.json",
    "texts/"
]

def print_error(msg):
    print(f"\033[91mERROR: {msg}\033[0m", file=sys.stderr)

def print_success(msg):
    print(f"\033[92mSUCCESS: {msg}\033[0m")

def print_instructions():
    print(f"""
Please download the PDNC dataset manually and place it in the following directory:
    speaker-attribution-research/{DATA_DIR}/

Ensure the following structure exists:
    {DATA_DIR}/
    ├── coref.json
    ├── quotes.json
    └── texts/
        ├── ...

If you need access to PDNC, please refer to their official documentation.
    """)

def check_structure():
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print_error(f"Dataset directory '{DATA_DIR}' does not exist.")
        print_instructions()
        sys.exit(1)
        
    missing = []
    for item in EXPECTED_FILES:
        target = DATA_DIR / item
        if item.endswith('/'):
            if not target.is_dir():
                missing.append(item)
        else:
            if not target.is_file():
                missing.append(item)
                
    if missing:
        print_error(f"The following required files/directories are missing in '{DATA_DIR}':")
        for m in missing:
            print(f"  - {m}")
        print_instructions()
        sys.exit(1)
        
    print_success("Dataset directory structure is valid.")

if __name__ == "__main__":
    print("Checking PDNC dataset...")
    check_structure()
    print_success("PDNC dataset is ready for use.")
