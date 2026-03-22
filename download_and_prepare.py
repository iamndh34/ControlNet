#!/usr/bin/env python3
"""
Simple download and prepare script for Vietnamese Cultural dataset.
Uses snapshot_download which handles LFS properly.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from huggingface_hub import snapshot_download


def download_dataset(repo_id="Dangindev/viet-cultural-vqa", output_dir="./hf_dataset"):
    """Download entire dataset using snapshot_download."""
    print(f"Downloading {repo_id}...")

    # Download only image files
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        allow_patterns="images/**/*.jpg"
    )

    print(f"Downloaded to: {output_dir}")
    return output_dir


def prepare_training_data(
    input_dir="./hf_dataset",
    output_dir="./training/vietnamese_folk_art",
    num_samples=None,
    edge_method='canny'
):
    """Prepare training data with edge maps."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))

    if num_samples:
        image_files = image_files[:num_samples]

    print(f"Found {len(image_files)} images")
    print("Processing...")

    prompts = []

    for idx, img_path in enumerate(tqdm(image_files)):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))

            # Generate edge map
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = 255 - edges

            # Save
            filename = f"{idx:06d}.png"
            cv2.imwrite(str(target_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / filename), edges)

            # Get category from path
            path_parts = str(img_path).split('/')
            if len(path_parts) >= 2:
                category_part = path_parts[-2]  # Second to last
            else:
                category_part = "unknown"

            # Map to prompt
            category_map = {
                "kien_truc": "Vietnamese traditional temple architecture",
                "am_thuc": "Traditional Vietnamese cuisine dish",
                "phong_canh": "Vietnamese natural landscape",
                "trang_phuc": "Vietnamese traditional clothing",
                "doi_song": "Vietnamese daily life scene",
                "van_hoa": "Vietnamese folk culture art",
                "le_hoi": "Vietnamese traditional festival",
                "tro_choi": "Vietnamese folk game scene",
                "the_thao": "Vietnamese traditional sport",
                "thu_cong": "Vietnamese handicraft art",
                "nhac_cu": "Vietnamese traditional instrument",
                "giao_thong": "Vietnamese traditional transportation"
            }

            # Try to match category
            prompt = "Vietnamese traditional folk art"
            for key, val in category_map.items():
                if key in str(img_path):
                    prompt = val
                    break

            prompts.append({
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": prompt
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save prompts
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted!")
    print(f"  Total samples: {len(prompts)}")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Prompts: {prompt_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_download", action="store_true", help="Skip download if already downloaded")
    parser.add_argument("--download_dir", type=str, default="./hf_dataset")
    parser.add_argument("--output_dir", type=str, default="./training/vietnamese_folk_art")
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    download_dir = args.download_dir

    # Download if needed
    if not args.skip_download:
        download_dir = download_dataset(output_dir=args.download_dir)

    # Prepare training data
    prepare_training_data(
        input_dir=download_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
