#!/usr/bin/env python3
"""
Unlimited Google Images crawler with advanced features.
Supports scrolling, multiple user agents, retries, and rate limit handling.
"""

import os
import json
import cv2
import time
import random
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlencode, quote
import requests
from PIL import Image
import io
from bs4 import BeautifulSoup
import numpy as np


# User agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]


def get_random_headers():
    """Get random headers with rotating user agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }


def download_image_with_retry(url, max_retries=3, timeout=15):
    """Download image with retry mechanism."""
    for attempt in range(max_retries):
        try:
            headers = get_random_headers()
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            # Check if image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return None

            img = Image.open(io.BytesIO(response.content))

            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            return img

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
            continue

    return None


def get_google_images_unlimited(query, num_images=1000, start=0):
    """
    Get unlimited images from Google Images.

    Args:
        query: Search query
        num_images: Target number of images
        start: Starting index (for pagination)

    Returns:
        List of downloaded PIL Images
    """
    images = []
    seen_urls = set()

    # Google Images supports pagination with 'start' parameter
    # Each page shows ~100 images
    page = 0
    max_pages = 50  # Up to 5000 images

    while len(images) < num_images and page < max_pages:
        start_idx = page * 100

        base_url = "https://www.google.com/search"
        params = {
            "q": query,
            "udm": "2",
            "start": start_idx,
        }

        try:
            headers = get_random_headers()
            response = requests.get(base_url, params=params, headers=headers, timeout=30)

            if response.status_code == 429:
                print(f"  Rate limited! Waiting 30 seconds...")
                time.sleep(30)
                continue

            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all img tags and script tags with image data
            img_tags = soup.find_all('img')
            scripts = soup.find_all('script')

            # Extract image URLs from various sources
            image_urls = []

            for img in img_tags:
                src = img.get('src', '')
                data_src = img.get('data-src', '')
                data_iurl = img.get('data-iurl', '')

                for url in [data_iurl, data_src, src]:
                    if url and url.startswith('http') and url not in seen_urls:
                        image_urls.append(url)
                        seen_urls.add(url)

            # Also check script tags for embedded image data
            for script in scripts:
                if script.string:
                    # Look for image URLs in JavaScript
                    import re
                    urls = re.findall(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)', script.string)
                    for url in urls:
                        if url not in seen_urls:
                            image_urls.append(url)
                            seen_urls.add(url)

            print(f"  Page {page+1}: Found {len(image_urls)} new URLs")

            # Download images
            downloaded_this_page = 0

            for url in tqdm(image_urls, desc=f"  Downloading page {page+1}"):
                if len(images) >= num_images:
                    break

                # Skip Google's own images
                if any(x in url.lower() for x in ['logo', 'icon', 'favicon', 'google.com', 'gstatic.com']):
                    continue

                img = download_image_with_retry(url)

                if img is not None:
                    # Check size
                    if img.width >= 300 and img.height >= 300:
                        images.append(img)
                        downloaded_this_page += 1

                        # Random delay to avoid rate limiting
                        time.sleep(random.uniform(0.3, 0.8))

            print(f"  Downloaded: {downloaded_this_page} images (Total: {len(images)})")

            # If no new images, we've reached the end
            if downloaded_this_page == 0:
                print(f"  No more images found!")
                break

            page += 1

            # Longer delay between pages
            time.sleep(random.uniform(2, 5))

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            time.sleep(5)
            page += 1
            continue

    return images


def crawl_unlimited(
    queries=None,
    output_dir="./crawled_images",
    training_dir="./training/vietnamese_folk_art",
    images_per_query=500,
    create_edges=True
):
    """
    Crawl unlimited images from Google Images.
    """
    if queries is None:
        queries = [
            "tranh đông hồ",
            "tranh dân gian đông hồ",
            "tranh kính hồng",
            "vietnamese folk art painting",
            "tranh dân gian việt nam",
            "vietnamese traditional painting",
            "tranh làng nghề việt nam",
            "tranh thủy mặc việt nam",
            "vietnamese watercolor painting",
            "tranh múa rối nước"
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    total_images = 0

    for query_idx, query in enumerate(queries):
        print(f"\n{'='*60}")
        print(f"Query {query_idx+1}/{len(queries)}: {query}")
        print(f"{'='*60}")

        images = get_google_images_unlimited(query, num_images=images_per_query)

        print(f"✓ Downloaded: {len(images)} images")

        for img in images:
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
                continue

        print(f"  Saved {len(images)} images")
        print(f"  Total so far: {total_images}")

        # Delay between queries
        if query_idx < len(queries) - 1:
            delay = random.uniform(5, 10)
            print(f"  Waiting {delay:.1f}s before next query...")
            time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"CRAWLING COMPLETED!")
    print(f"{'='*60}")
    print(f"  Total images: {total_images}")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*60}")

    # Save metadata
    metadata_file = output_path / "prompts.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)

    # Create edge maps if requested
    if create_edges and total_images > 0:
        print("\nCreating edge maps...")
        create_training_data(
            image_dir=output_dir,
            output_dir=training_dir,
            prompts=all_prompts
        )


def create_training_data(image_dir, output_dir, prompts):
    """Create training data with edge maps."""
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
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate edge map
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = 255 - edges

            filename = img_path.name
            cv2.imwrite(str(target_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / filename), edges)

            for prompt in prompts:
                if filename in prompt.get("target", ""):
                    final_prompts.append(prompt)
                    break

        except Exception as e:
            continue

    # Save prompts
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(final_prompts, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA READY!")
    print(f"{'='*60}")
    print(f"  Total samples: {len(final_prompts)}")
    print(f"  Source (edge maps): {source_dir}")
    print(f"  Target (original): {target_dir}")
    print(f"  Prompts: {prompt_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unlimited Google Images crawler")
    parser.add_argument("--queries", type=str, nargs='+',
                        help="Search queries")
    parser.add_argument("--output_dir", type=str, default="./crawled_images")
    parser.add_argument("--training_dir", type=str, default="./training/vietnamese_folk_art")
    parser.add_argument("--images_per_query", type=int, default=500,
                        help="Number of images per query (default: 500, max: ~5000)")
    parser.add_argument("--no_edges", action="store_true",
                        help="Don't create edge maps")

    args = parser.parse_args()

    crawl_unlimited(
        queries=args.queries,
        output_dir=args.output_dir,
        training_dir=args.training_dir,
        images_per_query=args.images_per_query,
        create_edges=not args.no_edges
    )
