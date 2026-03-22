#!/usr/bin/env python3
"""
Simple crawler for Vietnamese Cultural VQA dataset.
Downloads images directly without using the broken dataset loader.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import cv2
import numpy as np


def get_image_urls_from_repo():
    """
    Get image URLs from HuggingFace repository.
    Since the dataset loader is broken, we'll use the API to list files.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "Dangindev/viet-cultural-vqa"

    print(f"Listing files from {repo_id}...")

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")

        # Filter for image files in the images directory
        image_files = [f for f in files if f.startswith('images/') and f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        print(f"Found {len(image_files)} image files")
        return image_files, repo_id

    except Exception as e:
        print(f"Error listing files: {e}")
        return [], None


def download_images_direct(image_files, repo_id, output_dir="./viet_cultural_images", num_samples=None):
    """
    Download images directly using raw URLs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_samples:
        image_files = image_files[:num_samples]

    print(f"Downloading {len(image_files)} images...")

    prompts = []
    downloaded = 0

    for idx, file_path in enumerate(tqdm(image_files)):
        try:
            # Construct raw URL
            url = f"https://huggingface.co/datasets/{repo_id}/raw/main/{file_path}"

            # Download image
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Load and process image
            img = Image.open(io.BytesIO(response.content))

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to 512x512
            img = img.resize((512, 512), Image.Resampling.LANCZOS)

            # Generate filename
            filename = f"{idx:06d}.png"

            # Save target image
            target_path = output_path / filename
            img.save(target_path)

            # Extract category from path
            path_parts = file_path.split('/')
            if len(path_parts) >= 3:
                category = path_parts[1]  # Second level directory
            else:
                category = "unknown"

            # Generate prompt based on category
            category_prompts = {
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

            prompt = category_prompts.get(category, "Vietnamese traditional folk art")

            prompts.append({
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": prompt
            })

            downloaded += 1

        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            continue

    # Save prompts
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\nDownloaded {downloaded}/{len(image_files)} images")
    print(f"Saved to: {output_path}")
    print(f"Prompts saved to: {prompt_file}")

    return str(output_path)


def create_edge_maps_from_images(
    image_dir="./viet_cultural_images",
    output_dir="./training/vietnamese_folk_art",
    edge_method='canny'
):
    """
    Create edge maps from downloaded images.
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompt_file = image_path / "prompt.json"
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
    else:
        prompts = []

    # Get all images
    image_files = sorted(image_path.glob("*.png")) + sorted(image_path.glob("*.jpg"))

    print(f"Found {len(image_files)} images")
    print("Creating edge maps...")

    for img_path in tqdm(image_files):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    # Copy prompts
    output_prompt = output_path / "prompt.json"
    with open(output_prompt, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\nEdge maps created:")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Prompts: {output_prompt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl Vietnamese Cultural VQA dataset")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of images to download")
    parser.add_argument("--output_dir", type=str, default="./viet_cultural_images")
    parser.add_argument("--create_edges", action="store_true", help="Create edge maps after download")
    parser.add_argument("--edge_output", type=str, default="./training/vietnamese_folk_art")
    parser.add_argument("--edge_method", type=str, default="canny", choices=["canny", "sobel"])

    args = parser.parse_args()

    # Get image files
    image_files, repo_id = get_image_urls_from_repo()

    if not image_files:
        print("No images found. Please check the repository.")
        exit(1)

    # Download images
    image_dir = download_images_direct(
        image_files=image_files,
        repo_id=repo_id,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

    # Create edge maps if requested
    if args.create_edges:
        create_edge_maps_from_images(
            image_dir=image_dir,
            output_dir=args.edge_output,
            edge_method=args.edge_method
        )
