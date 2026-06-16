"""
Clean Repository for Release
============================
Removes temporary Python artifacts, logs, cache directories, and any untracked files
not needed for the public release.
"""

import os
import shutil
from pathlib import Path

def clean_repo():
    root = Path(__file__).resolve().parent.parent
    print(f"Cleaning repository at {root}...")
    
    # 1. Directories to remove
    dirs_to_remove = [
        "__pycache__",
        ".ipynb_checkpoints",
        ".pytest_cache",
        "build",
        "dist",
        "*.egg-info"
    ]
    
    # 2. File extensions to remove
    exts_to_remove = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    # Find and remove directories
    for d in dirs_to_remove:
        # Check matching directories
        for p in root.rglob(d):
            if p.is_dir():
                print(f"Removing directory: {p.relative_to(root)}")
                try:
                    shutil.rmtree(p)
                except Exception as e:
                    print(f"Error removing {p}: {e}")
                    
    # Find and remove files
    for pattern in exts_to_remove:
        for p in root.rglob(pattern):
            if p.is_file():
                print(f"Removing file: {p.relative_to(root)}")
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Error removing {p}: {e}")
                    
    print("Clean completed successfully!")

if __name__ == "__main__":
    clean_repo()
