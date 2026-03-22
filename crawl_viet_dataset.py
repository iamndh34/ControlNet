#!/usr/bin/env python3
"""
Crawl Vietnamese Cultural VQA dataset from HuggingFace.
Downloads images directly using requests to bypass LFS issues.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi, login
from PIL import Image
import io


def get_dataset_files(repo_id="Dangindev/viet-cultural-vqa"):
    """List all files in the dataset repository."""
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")

    # Filter for image files and data files
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    data_files = [f for f in files if f.endswith('.json') or f.endswith('.parquet')]

    return image_files, data_files


def download_file(repo_id, file_path, output_dir, token=None):
    """Download a single file from HuggingFace."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{file_path}"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    output_path = Path(output_dir) / file_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(response.content)

    return output_path


def download_images_from_split(
    repo_id="Dangindev/viet-cultural-vqa",
    split="train",
    output_dir="./viet_cultural_data",
    num_samples=None,
    token=None
):
    """
    Download images from a specific split using the dataset API.

    This bypasses LFS by loading the dataset and extracting images.
    """
    print(f"Loading {split} split from {repo_id}...")

    try:
        from datasets import load_dataset
        import shutil

        # Load dataset
        dataset = load_dataset(repo_id, split=split, trust_remote_code=True)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"Processing {len(dataset)} samples...")

        output_path = Path(output_dir) / split
        images_dir = output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        prompts = []

        for idx, item in enumerate(tqdm(dataset)):
            try:
                # Get image
                image = item.get('image')
                if image is None:
                    continue

                # Save image
                image_filename = f"{idx:06d}.png"
                image_path = images_dir / image_filename
                image.save(image_path)

                # Get category info
                category = item.get('category', 'unknown')
                keyword = item.get('keyword', '')

                # Generate prompt
                if isinstance(category, int):
                    categories = ['kien_truc', 'am_thuc', 'phong_canh', 'trang_phuc',
                                'doi_song', 'van_hoa', 'le_hoi', 'thu_cong']
                    category = categories[category % len(categories)] if category < len(categories) else 'van_hoa'

                prompt_map = {
                    'kien_truc': f"Vietnamese traditional {keyword} architecture",
                    'am_thuc': f"Traditional Vietnamese {keyword} cuisine",
                    'phong_canh': f"Vietnamese {keyword} landscape",
                    'trang_phuc': f"Vietnamese traditional {keyword} clothing",
                    'doi_song': f"Vietnamese {keyword} daily life scene",
                    'van_hoa': f"Vietnamese traditional {keyword} cultural art",
                    'le_hoi': f"Vietnamese {keyword} festival celebration",
                    'thu_cong': f"Vietnamese traditional {keyword} handicraft"
                }

                prompt = prompt_map.get(category, f"Vietnamese traditional {keyword} folk art")

                prompts.append({
                    "source": f"source/{image_filename}",
                    "target": f"target/{image_filename}",
                    "prompt": prompt
                })

            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue

        # Save prompts
        import json
        with open(output_path / "prompt.json", 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

        print(f"\nCompleted!")
        print(f"  Images: {images_dir}")
        print(f"  Prompts: {output_path / 'prompt.json'}")
        print(f"  Total samples: {len(prompts)}")

        return str(output_path)

    except Exception as e:
        print(f"Error: {e}")
        return None


def create_edge_maps(input_dir, output_dir="./training/vietnamese_folk_art", edge_method='canny'):
    """Create edge maps from downloaded images."""
    import cv2
    import numpy as np

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg"))

    print(f"Found {len(image_files)} images")
    print("Creating edge maps...")

    for img_path in tqdm(image_files):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))

            # Generate edge map
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if edge_method == 'canny':
                edges = cv2.Canny(gray, 50, 150)
                edges = 255 - edges
            else:
                edges = cv2.Canny(gray, 50, 150)
                edges = 255 - edges

            # Save
            filename = img_path.name
            cv2.imwrite(str(target_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / filename), edges)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"Edge maps created in {source_dir}")
    print(f"Target images in {target_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl Vietnamese Cultural VQA dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./viet_cultural_data")
    parser.add_argument("--create_edges", action="store_true")
    parser.add_argument("--edge_output", type=str, default="./training/vietnamese_folk_art")

    args = parser.parse_args()

    # Download dataset
    data_dir = download_images_from_split(
        split=args.split,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

    # Create edge maps if requested
    if args.create_edges and data_dir:
        images_dir = Path(data_dir) / "images"
        create_edge_maps(images_dir, args.edge_output)
