#!/usr/bin/env python3
"""
Crawl Vietnamese folk art images from Google Images.
Supports multiple queries and automatic downloading.
"""

import os
import json
import cv2
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlencode, quote
import requests
from PIL import Image
import io
from bs4 import BeautifulSoup


def get_google_image_urls(query, num_images=100):
    """
    Get image URLs from Google Images search.

    Args:
        query: Search query
        num_images: Number of images to fetch

    Returns:
        List of image URLs
    """
    urls = []

    # Google Images search URL
    base_url = "https://www.google.com/search"

    params = {
        "q": query,
        "udm": "2",  # Image search
        "sxsrf": "ANbL-n7UGv-_qvkuxJYbYBke0Du8cRqBgQ:1774201399980"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image tags
        img_tags = soup.find_all('img')

        for img in img_tags:
            if 'src' in img.attrs:
                src = img['src']
                # Filter out small icons and base64 data
                if src.startswith('http') and 'logo' not in src.lower():
                    urls.append(src)
                    if len(urls) >= num_images:
                        break

    except Exception as e:
        print(f"Error searching for '{query}': {e}")

    return urls


def download_image(url, timeout=10):
    """
    Download image from URL.

    Returns:
        PIL Image or None
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    except Exception as e:
        return None


def crawl_folk_art_images(
    queries=None,
    output_dir="./folk_art_images",
    images_per_query=50,
    min_size=256
):
    """
    Crawl Vietnamese folk art images from Google Images.

    Args:
        queries: List of search queries (default: folk art queries)
        output_dir: Directory to save images
        images_per_query: Number of images to download per query
        min_size: Minimum image size (width and height)
    """
    if queries is None:
        queries = [
            "tranh đông hồ vietnam",
            "tranh dân gian đông hồ",
            "tranh kim hoàng",
            "tranh kính hồng tranh dân gian vietnam",
            "vietnamese folk art painting",
            "vietnamese traditional art",
            "tranh dân gian viet nam",
            "tranh làng nghề vietnam",
            "vietnamese water puppet",
            "tranh múa rối nước"
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_images = []
    total_downloaded = 0

    for query in queries:
        print(f"\nSearching: '{query}'")

        # Get image URLs
        urls = get_google_image_urls(query, num_images=images_per_query * 2)

        print(f"  Found {len(urls)} URLs, downloading...")

        downloaded = 0

        for url in tqdm(urls, desc=f"  Downloading"):
            if downloaded >= images_per_query:
                break

            img = download_image(url)

            if img is None:
                continue

            # Check size
            if img.width < min_size or img.height < min_size:
                continue

            # Generate unique filename
            img_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{query.replace(' ', '_')[:20]}_{img_hash}.jpg"
            filename = filename.replace('/', '_')

            # Save image
            img_path = output_path / filename
            img.save(img_path, quality=90)

            # Generate prompt from query
            prompt = f"Vietnamese {query}"

            all_images.append({
                "path": str(img_path),
                "prompt": prompt,
                "query": query
            })

            downloaded += 1
            total_downloaded += 1

            # Rate limiting
            time.sleep(0.5)

        print(f"  Downloaded {downloaded} images")

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_images, f, ensure_ascii=False, indent=2)

    print(f"\n" + "="*50)
    print(f"COMPLETED!")
    print(f"="*50)
    print(f"  Total images: {total_downloaded}")
    print(f"  Saved to: {output_path}")
    print(f"  Metadata: {metadata_file}")
    print(f"="*50)

    return str(output_path)


def prepare_training_data_from_crawled(
    input_dir="./folk_art_images",
    output_dir="./training/vietnamese_folk_art",
    edge_method='canny'
):
    """
    Prepare training data from crawled images.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata if exists
    metadata_file = input_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = []

    # Get all images
    image_files = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))

    print(f"Found {len(image_files)} images")
    print("Creating edge maps...")

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

            if edge_method == 'canny':
                edges = cv2.Canny(gray, 50, 150)
                edges = 255 - edges
            else:
                edges = cv2.Canny(gray, 50, 150)
                edges = 255 - edges

            # Save
            filename = f"{idx:06d}.png"
            cv2.imwrite(str(target_dir / filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(source_dir / filename), edges)

            # Get prompt from metadata or use default
            prompt = "Vietnamese traditional folk art painting"
            if metadata:
                for meta in metadata:
                    if Path(meta["path"]).name == img_path.name:
                        prompt = meta["prompt"]
                        break

            prompts.append({
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": prompt
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save prompts
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\n" + "="*50)
    print(f"COMPLETED!")
    print(f"="*50)
    print(f"  Total samples: {len(prompts)}")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Prompts: {prompt_file}")
    print(f"="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crawl Vietnamese folk art from Google Images")
    parser.add_argument("--queries", type=str, nargs='+',
                        help="Custom search queries (default: built-in folk art queries)")
    parser.add_argument("--output_dir", type=str, default="./folk_art_images")
    parser.add_argument("--images_per_query", type=int, default=50)
    parser.add_argument("--min_size", type=int, default=256)
    parser.add_argument("--prepare", action="store_true",
                        help="Prepare training data after crawling")
    parser.add_argument("--training_output", type=str, default="./training/vietnamese_folk_art")

    args = parser.parse_args()

    # Crawl images
    image_dir = crawl_folk_art_images(
        queries=args.queries,
        output_dir=args.output_dir,
        images_per_query=args.images_per_query,
        min_size=args.min_size
    )

    # Prepare training data if requested
    if args.prepare:
        prepare_training_data_from_crawled(
            input_dir=image_dir,
            output_dir=args.training_output
        )
