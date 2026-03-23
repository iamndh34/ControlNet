import albumentations as A
import cv2
import os
import numpy as np
from pathlib import Path

# Paths
INPUT_DIR = "/home/haind/Desktop/ControlNet/TranhDongHo"
OUTPUT_DIR = "/home/haind/Desktop/ControlNet/TranhDongHo_augmented"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Augmentation pipeline - conserving artistic style
transform = A.Compose([
    # Geometric transformations
    A.Rotate(limit=15, p=0.8, border_mode=cv2.BORDER_REFLECT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.7),

    # Color and lighting (conservative for art preservation)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.6),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

    # Blur and noise (subtle)
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.3),

    A.OneOf([
        A.GaussNoise(var_limit=(10, 30), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
    ], p=0.3),

    # Perspective (very subtle)
    A.OneOf([
        A.Perspective(scale=(0.02, 0.05), p=1.0),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.02, 0.02), p=1.0),
    ], p=0.2),
])

# Get all images
input_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))])
print(f"Found {len(input_files)} input images")

# Target: ~17000 images from 170 originals = 100 per image
AUGMENTS_PER_IMAGE = 100
counter = 1

for img_file in input_files:
    img_path = os.path.join(INPUT_DIR, img_file)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {img_file}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original (1)
    output_path = os.path.join(OUTPUT_DIR, f"dongho_{counter:05d}.png")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    counter += 1

    # Generate augmentations
    for i in range(AUGMENTS_PER_IMAGE):
        augmented = transform(image=image)
        aug_image = augmented['image']

        output_path = os.path.join(OUTPUT_DIR, f"dongho_{counter:05d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        counter += 1

    print(f"Processed {img_file} -> {AUGMENTS_PER_IMAGE + 1} images (total: {counter-1})")

print(f"\nCompleted! Total images: {counter-1}")
print(f"Output directory: {OUTPUT_DIR}")
