"""
Preview script to visualize the prepared dataset.
This helps verify that the dataset is prepared correctly before training.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from viet_folk_art_dataset import VietFolkArtDataset


def preview_dataset(data_root="./training/vietnamese_folk_art", num_samples=4):
    """
    Preview the dataset by showing source (edge) and target (original) images.

    Args:
        data_root: Path to training dataset
        num_samples: Number of samples to display
    """
    # Load dataset
    dataset = VietFolkArtDataset(data_root)
    print(f"Dataset size: {len(dataset)}")

    # Create figure
    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        sample = dataset[idx]

        # Get images
        hint = sample['hint']  # Edge map [0, 1]
        jpg = sample['jpg']    # Target [-1, 1]
        txt = sample['txt']    # Prompt

        # Denormalize for display
        hint_disp = (hint * 255).astype(np.uint8)
        jpg_disp = ((jpg + 1) * 127.5).astype(np.uint8)

        # Display
        axes[idx, 0].imshow(hint_disp, cmap='gray')
        axes[idx, 0].set_title(f'Input (Edge)')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(jpg_disp)
        axes[idx, 1].set_title(f'Target (Original)')
        axes[idx, 1].axis('off')

        axes[idx, 2].text(0.5, 0.5, txt, ha='center', va='center',
                          wrap=True, fontsize=10)
        axes[idx, 2].set_title('Prompt')
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_preview.png', dpi=150, bbox_inches='tight')
    print(f"Preview saved to dataset_preview.png")
    plt.show()


def check_dataset_statistics(data_root="./training/vietnamese_folk_art"):
    """
    Check dataset statistics.
    """
    dataset = VietFolkArtDataset(data_root)

    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")

    # Check first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Prompt: {sample['txt']}")
        print(f"  Target shape: {sample['jpg'].shape}")
        print(f"  Target range: [{sample['jpg'].min():.3f}, {sample['jpg'].max():.3f}]")
        print(f"  Hint shape: {sample['hint'].shape}")
        print(f"  Hint range: [{sample['hint'].min():.3f}, {sample['hint'].max():.3f}]")

    # Check file sizes
    data_path = Path(data_root)
    source_files = list((data_path / "source").glob("*.png"))
    target_files = list((data_path / "target").glob("*.png"))

    print(f"\nFiles:")
    print(f"  Source files: {len(source_files)}")
    print(f"  Target files: {len(target_files)}")

    if source_files:
        total_size = sum(f.stat().st_size for f in source_files)
        print(f"  Source total size: {total_size / 1024 / 1024:.2f} MB")

    if target_files:
        total_size = sum(f.stat().st_size for f in target_files)
        print(f"  Target total size: {total_size / 1024 / 1024:.2f} MB")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview Vietnamese Folk Art dataset")
    parser.add_argument("--data_root", type=str, default="./training/vietnamese_folk_art",
                        help="Path to training dataset")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of samples to display")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only show statistics without preview")

    args = parser.parse_args()

    check_dataset_statistics(args.data_root)

    if not args.stats_only:
        preview_dataset(args.data_root, args.num_samples)
