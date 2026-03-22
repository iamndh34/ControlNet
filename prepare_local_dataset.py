#!/usr/bin/env python3
"""
Simple script to create edge maps from local images.
Use this if you have images downloaded locally.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse


def generate_scribble(image, method='canny'):
    """Generate scribble/edge map from image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if method == 'canny':
        edges = cv2.Canny(gray, 50, 150)
        edges = 255 - edges
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        edges = 255 - edges
    else:
        raise ValueError(f"Unknown method: {method}")

    return edges


def create_dataset_from_images(
    input_dir,
    output_dir="./training/vietnamese_folk_art",
    prompt="Vietnamese traditional folk art",
    edge_method='canny',
    image_size=512
):
    """
    Create ControlNet training dataset from local images.

    Args:
        input_dir: Directory containing input images
        output_dir: Output directory for training data
        prompt: Prompt text for all images
        edge_method: Method for edge detection ('canny', 'sobel')
        image_size: Size to resize images to
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    source_dir = output_path / "source"
    target_dir = output_path / "target"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(input_path.glob(ext)))
        image_files.extend(list(input_path.rglob(ext)))

    print(f"Found {len(image_files)} images in {input_dir}")

    prompts = []

    for idx, img_path in enumerate(tqdm(image_files)):
        try:
            # Load image
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = np.array(image)
            image = cv2.resize(image, (image_size, image_size))

            # Generate edge map
            edge_map = generate_scribble(image, method=edge_method)

            # Filename
            filename = f"{idx:06d}.png"

            # Save target image
            target_path = target_dir / filename
            cv2.imwrite(str(target_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Save source (edge map)
            source_path = source_dir / filename
            cv2.imwrite(str(source_path), edge_map)

            prompts.append({
                "source": f"source/{filename}",
                "target": f"target/{filename}",
                "prompt": prompt
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save prompts.json
    prompt_file = output_path / "prompt.json"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\nDataset created successfully!")
    print(f"Total samples: {len(prompts)}")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ControlNet dataset from local images")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./training/vietnamese_folk_art",
                        help="Output directory for training data")
    parser.add_argument("--prompt", type=str, default="Vietnamese traditional folk art",
                        help="Prompt text for all images")
    parser.add_argument("--edge_method", type=str, default="canny", choices=["canny", "sobel"],
                        help="Edge detection method")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size to resize images to")

    args = parser.parse_args()

    create_dataset_from_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        edge_method=args.edge_method,
        image_size=args.image_size
    )
