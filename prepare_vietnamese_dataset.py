#!/usr/bin/env python3
"""
Script to prepare Vietnamese Folk Art dataset for ControlNet training.
This script will:
1. Download dataset from Hugging Face
2. Generate scribble/edge maps from source images
3. Create prompt.json with Vietnamese cultural prompts
4. Organize data in the format required by ControlNet
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Try importing datasets, fallback to manual download if needed
try:
    from datasets import load_dataset
    DATETS_AVAILABLE = True
except ImportError:
    DATETS_AVAILABLE = False

# Vietnamese cultural prompts by category
VIETNAMESE_PROMPTS = {
    "kien_truc": [
        "Vietnamese traditional temple architecture",
        "Ancient Buddhist pagoda in Vietnam",
        "Traditional Vietnamese wooden structure",
        "Historical Vietnamese temple with curved roof",
        "Vietnamese heritage temple architecture"
    ],
    "am_thuc": [
        "Traditional Vietnamese cuisine dish",
        "Authentic Vietnamese food presentation",
        "Vietnamese traditional cooking",
        "Classic Vietnamese dish arrangement",
        "Vietnamese culinary heritage"
    ],
    "phong_canh": [
        "Vietnamese natural landscape",
        "Traditional Vietnamese countryside scene",
        "Beautiful Vietnamese scenery",
        "Vietnamese rural landscape",
        "Traditional Vietnamese natural view"
    ],
    "trang_phuc": [
        "Vietnamese traditional clothing",
        "Ao dai traditional Vietnamese dress",
        "Vietnamese ethnic costume",
        "Traditional Vietnamese attire",
        "Vietnamese cultural clothing"
    ],
    "doi_song": [
        "Vietnamese traditional daily life",
        "Vietnamese street scene",
        "Traditional Vietnamese market",
        "Vietnamese cultural lifestyle",
        "Everyday life in Vietnam"
    ],
    "van_hoa": [
        "Vietnamese folk culture performance",
        "Traditional Vietnamese art form",
        "Vietnamese cultural heritage",
        "Vietnamese traditional performance",
        "Vietnamese folk art"
    ],
    "le_hoi": [
        "Vietnamese traditional festival",
        "Vietnamese cultural celebration",
        "Traditional Vietnamese ceremony",
        "Vietnamese festival scene",
        "Vietnamese cultural festivity"
    ],
    "thu_cong": [
        "Vietnamese traditional handicraft",
        "Vietnamese artisan craft",
        "Traditional Vietnamese pottery",
        "Vietnamese lacquerware art",
        "Vietnamese traditional craft work"
    ]
}


def generate_scribble(image, method='canny'):
    """
    Generate scribble/edge map from image.

    Args:
        image: Input image (RGB numpy array)
        method: 'canny', 'hed', or 'sobel'

    Returns:
        Edge map as grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        # Invert so edges are white on black
        edges = 255 - edges

    elif method == 'sobel':
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        # Threshold and invert
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        edges = 255 - edges

    elif method == 'hed':
        # Try to use HED detector if available
        try:
            from annotator.hed import HEDDetector
            hed = HEDDetector()
            edges = hed(image)
            edges = (edges * 255).astype(np.uint8)
        except ImportError:
            print("HED detector not available, falling back to Canny")
            edges = cv2.Canny(gray, 50, 150)
            edges = 255 - edges

    else:
        raise ValueError(f"Unknown method: {method}")

    return edges


def create_dataset_from_huggingface(
    output_dir="./training/vietnamese_folk_art",
    num_samples=None,
    edge_method='canny',
    image_size=512
):
    """
    Create ControlNet training dataset from Vietnamese Cultural VQA dataset.

    Args:
        output_dir: Output directory for training data
        num_samples: Number of samples to process (None = all)
        edge_method: Method for edge detection ('canny', 'sobel', 'hed')
        image_size: Size to resize images to
    """
    # Create output directories
    output_path = Path(output_dir)
    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from Hugging Face...")
    print("Note: If you get a 'Dataset scripts are no longer supported' error,")
    print("      downgrade datasets: pip install 'datasets[vision]<2.20.0'")

    dataset = load_dataset("Dangindev/viet-cultural-vqa", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Processing {len(dataset)} images...")

    prompts = []
    category_idx = 0

    for idx, item in enumerate(tqdm(dataset)):
        try:
            # Get image
            image = item['image']
            if image is None:
                continue

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize image
            image = np.array(image)
            image = cv2.resize(image, (image_size, image_size))

            # Generate edge map
            edge_map = generate_scribble(image, method=edge_method)

            # Get filename
            filename = f"{idx:06d}.png"

            # Save target image
            target_path = target_dir / filename
            cv2.imwrite(str(target_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Save source (edge map)
            source_path = source_dir / filename
            cv2.imwrite(str(source_path), edge_map)

            # Get prompt based on category
            category = item.get('category', 'van_hoa')
            if isinstance(category, int):
                # Map category index to name
                category_names = list(VIETNAMESE_PROMPTS.keys())
                category = category_names[category % len(category_names)]

            # Select prompt from category
            category_prompts = VIETNAMESE_PROMPTS.get(category, VIETNAMESE_PROMPTS['van_hoa'])
            prompt = category_prompts[idx % len(category_prompts)]

            # Add to prompts list
            prompts.append({
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": prompt
            })

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

    # Save prompts.json
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(prompts)}")
    print(f"Output directory: {output_path}")
    print(f"Source images: {source_dir}")
    print(f"Target images: {target_dir}")
    print(f"Prompt file: {prompt_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Vietnamese Folk Art dataset for ControlNet")
    parser.add_argument("--output_dir", type=str, default="./training/vietnamese_folk_art",
                        help="Output directory for training data")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--edge_method", type=str, default="canny", choices=["canny", "sobel", "hed"],
                        help="Edge detection method")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size to resize images to")

    args = parser.parse_args()

    create_dataset_from_huggingface(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        edge_method=args.edge_method,
        image_size=args.image_size
    )
