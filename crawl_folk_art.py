#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crawl tranh Dong Ho and tranh Lang Trong from Bing Images.
"""

import os
import json
import cv2
import time
import random
from pathlib import Path
from tqdm import tqdm
import requests
from PIL import Image
import io
import numpy as np
import re


def get_random_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
        "Referer": "https://www.bing.com/"
    }


def get_bing_images(query, num_images=300):
    """Get images from Bing Images."""
    images = []
    seen_urls = set()
    offset = 0

    print(f"  Searching: '{query}'")

    while len(images) < num_images and offset < 1000:
        url = "https://www.bing.com/images/async"

        params = {
            "q": query,
            "async": "content",
            "first": offset,
            "count": 35,
        }

        try:
            headers = get_random_headers()
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                html = response.text
                matches = re.findall(r'"murl":"([^"]+)"', html)

                if not matches:
                    break

                for img_url in matches:
                    if len(images) >= num_images:
                        break

                    img_url = img_url.replace('\\u003d', '=').replace('\\u0026', '&')

                    if img_url in seen_urls:
                        continue

                    seen_urls.add(img_url)

                    try:
                        img_response = requests.get(img_url, headers=headers, timeout=15, stream=True)
                        img_response.raise_for_status()

                        img = Image.open(io.BytesIO(img_response.content))

                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        if img.width >= 300 and img.height >= 300:
                            images.append(img)
                            time.sleep(random.uniform(0.2, 0.5))

                    except:
                        continue

                print(f"    Downloaded: {len(images)}/{num_images}")
                offset += 35
                time.sleep(random.uniform(1, 2))

            else:
                break

        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(3)
            continue

    return images


def crawl_folk_art(
    output_dir="./crawled_images",
    training_dir="./training/vietnamese_folk_art",
    images_per_type=300,
    create_edges=True
):
    """Crawl tranh Dong Ho and tranh Lang Trong."""

    queries = [
        "Tranh đông Hồ",
        "Tranh Làng Trống"
    ]

    prompts_map = {
        "Tranh đông Hồ": "Vietnamese Dong Ho traditional folk art painting",
        "Tranh Làng Trống": "Vietnamese traditional folk art painting from Lang Trong village"
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    total_images = 0

    for query in queries:
        print(f"\n{'='*60}")
        print(f"CRAWLING: {query}")
        print(f"{'='*60}")

        images = get_bing_images(query, num_images=images_per_type)

        print(f"  Downloaded: {len(images)} images")

        for img in images:
            try:
                img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
                img_array = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

                filename = f"{total_images:06d}.png"
                img_path = output_path / filename
                cv2.imwrite(str(img_path), img_array)

                all_prompts.append({
                    "source": f"source/{filename}",
                    "target": f"target/{filename}",
                    "prompt": prompts_map[query]
                })

                total_images += 1

            except:
                continue

        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"{'='*60}")
    print(f"  Total images: {total_images}")
    print(f"{'='*60}")

    # Save metadata
    metadata_file = output_path / "prompts.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)

    # Create edge maps
    if create_edges and total_images > 0:
        print("\nCreating edge maps...")
        create_training_data(output_dir, training_dir, all_prompts)


def create_training_data(image_dir, output_dir, prompts):
    """Create training data with edge maps."""
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_path.glob("*.png")) + sorted(image_path.glob("*.jpg"))

    print(f"  Processing {len(image_files)} images...")

    for img_path in tqdm(image_files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = 255 - edges

            filename = img_path.name
            cv2.imwrite(str(target_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / filename), edges)

        except:
            continue

    # Save prompts
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA READY!")
    print(f"{'='*60}")
    print(f"  Samples: {len(prompts)}")
    print(f"  Source (edge maps): {source_dir}")
    print(f"  Target (original): {target_dir}")
    print(f"  Prompts: {prompt_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl tranh Dong Ho and tranh Lang Trong")
    parser.add_argument("--output_dir", type=str, default="./crawled_images")
    parser.add_argument("--training_dir", type=str, default="./training/vietnamese_folk_art")
    parser.add_argument("--images_per_type", type=int, default=300)
    parser.add_argument("--no_edges", action="store_true")

    args = parser.parse_args()

    crawl_folk_art(
        output_dir=args.output_dir,
        training_dir=args.training_dir,
        images_per_type=args.images_per_type,
        create_edges=not args.no_edges
    )
