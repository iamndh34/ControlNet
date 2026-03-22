#!/usr/bin/env python3
"""
Crawl images from Google Images by parsing <img> tags.
"""

import os
import json
import cv2
import time
import re
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlencode, quote
import requests
from PIL import Image
import io
from bs4 import BeautifulSoup
import base64


def get_google_images(query, num_images=100):
    """
    Get images from Google Images by parsing <img> tags.

    Args:
        query: Search query
        num_images: Maximum number of images to fetch

    Returns:
        List of tuples (image_data, source_url)
    """
    urls = []

    # Google Images search URL
    base_url = "https://www.google.com/search"

    params = {
        "q": query,
        "udm": "2",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    images = []

    try:
        print(f"Fetching: {query}")
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all img tags
        img_tags = soup.find_all('img')

        print(f"Found {len(img_tags)} <img> tags")

        for idx, img in enumerate(img_tags):
            if len(images) >= num_images:
                break

            src = img.get('src', '')
            data_src = img.get('data-src', '')
            data_iurl = img.get('data-iurl', '')

            # Try different attributes where Google stores image URLs
            img_url = data_iurl or data_src or src

            # Skip empty, base64, or tiny images
            if not img_url or img_url.startswith('data:'):
                continue

            # Skip Google's own logos/icons
            if any(x in img_url.lower() for x in ['logo', 'icon', 'favicon', 'google']):
                continue

            # Download the image
            try:
                img_response = requests.get(img_url, headers=headers, timeout=10)
                img_response.raise_for_status()

                pil_img = Image.open(io.BytesIO(img_response.content))

                # Skip very small images
                if pil_img.width < 200 or pil_img.height < 200:
                    continue

                # Convert to RGB
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                images.append((pil_img, img_url))

            except Exception as e:
                continue

    except Exception as e:
        print(f"Error: {e}")

    return images


def crawl_and_prepare(
    queries=None,
    output_dir="./crawled_folk_art",
    training_dir="./training/vietnamese_folk_art",
    images_per_query=50,
    create_edges=True
):
    """
    Crawl images from Google Images and prepare training data.
    """
    if queries is None:
        queries = [
            "tranh đông hồ",
            "tranh dân gian đông hồ",
            "dong ho folk painting",
            "vietnamese folk art",
            "tranh kính hồng",
            "tranh dân gian việt nam"
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    total_images = 0

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")

        images = get_google_images(query, num_images=images_per_query)

        print(f"Downloaded: {len(images)} images")

        for idx, (img, url) in enumerate(images):
            try:
                # Resize to 512x512
                img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)

                # Convert to numpy for OpenCV
                img_array = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

                # Save image
                filename = f"{total_images:06d}.png"
                img_path = output_path / filename
                cv2.imwrite(str(img_path), img_array)

                # Generate prompt
                prompt = f"Vietnamese {query} traditional folk art painting"

                all_prompts.append({
                    "source": f"source/{filename}",
                    "target": f"target/{filename}",
                    "prompt": prompt
                })

                total_images += 1

            except Exception as e:
                print(f"Error saving image: {e}")
                continue

        # Rate limiting between queries
        time.sleep(2)

    print(f"\n{'='*50}")
    print(f"TOTAL IMAGES: {total_images}")
    print(f"{'='*50}")

    # Create edge maps if requested
    if create_edges and total_images > 0:
        print("\nCreating edge maps...")
        create_training_data(
            image_dir=output_dir,
            output_dir=training_dir,
            prompts=all_prompts
        )


def create_training_data(image_dir, output_dir, prompts):
    """
    Create training data with edge maps.
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_path.glob("*.png")) + sorted(image_path.glob("*.jpg"))

    print(f"Processing {len(image_files)} images...")

    final_prompts = []

    for img_path in tqdm(image_files):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate edge map
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = 255 - edges  # Invert

            # Save
            filename = img_path.name
            cv2.imwrite(str(target_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / filename), edges)

            # Find corresponding prompt
            for prompt in prompts:
                if filename in prompt["target"]:
                    final_prompts.append(prompt)
                    break

        except Exception as e:
            continue

    # Save prompts
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(final_prompts, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"COMPLETED!")
    print(f"{'='*50}")
    print(f"  Total samples: {len(final_prompts)}")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Prompts: {prompt_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Crawl from Google Images using <img> tags")
    parser.add_argument("--queries", type=str, nargs='+',
                        help="Search queries (default: folk art queries)")
    parser.add_argument("--output_dir", type=str, default="./crawled_folk_art")
    parser.add_argument("--training_dir", type=str, default="./training/vietnamese_folk_art")
    parser.add_argument("--images_per_query", type=int, default=50)
    parser.add_argument("--no_edges", action="store_true",
                        help="Don't create edge maps")

    args = parser.parse_args()

    crawl_and_prepare(
        queries=args.queries,
        output_dir=args.output_dir,
        training_dir=args.training_dir,
        images_per_query=args.images_per_query,
        create_edges=not args.no_edges
    )
