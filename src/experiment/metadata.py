import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any
from src.utils.config import get_data_dir

def get_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"

def get_python_version() -> str:
    return sys.version.split()[0]

def get_dataset_fingerprint() -> str:
    """Computes a basic SHA-256 hash representing the dataset files and their sizes."""
    data_dir = get_data_dir()
    if not data_dir.exists():
        return "dataset_not_found"
        
    hasher = hashlib.sha256()
    # Ensure stable ordering
    for filepath in sorted(data_dir.rglob('*')):
        if filepath.is_file():
            hasher.update(str(filepath.relative_to(data_dir)).encode('utf-8'))
            hasher.update(str(filepath.stat().st_size).encode('utf-8'))
            
    return hasher.hexdigest()

def gather_metadata() -> Dict[str, Any]:
    return {
        "git_commit": get_git_commit(),
        "python_version": get_python_version(),
        "dataset_name": "PDNC",
        "dataset_hash": get_dataset_fingerprint()
    }
