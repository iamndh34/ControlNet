#!/usr/bin/env python3
"""
Download Vietnamese Cultural VQA dataset from HuggingFace.
Works on servers without GUI.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_dataset(repo_id="Dangindev/viet-cultural-vqa", local_dir="./viet-cultural-vqa"):
    """
    Download dataset from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save files
    """
    print(f"Downloading {repo_id} to {local_dir}...")

    # Download dataset
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print(f"\nDownload completed!")
    print(f"Files saved to: {local_dir}")

    # List downloaded files
    local_path = Path(local_dir)
    image_files = list(local_path.rglob("*.jpg")) + list(local_path.rglob("*.png"))
    print(f"Total images: {len(image_files)}")

if __name__ == "__main__":
    download_dataset()
