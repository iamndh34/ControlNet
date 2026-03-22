"""
Custom Dataset class for Vietnamese Folk Art ControlNet training.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class VietFolkArtDataset(Dataset):
    """
    Dataset class for Vietnamese Folk Art ControlNet training.

    Expected directory structure:
    vietnamese_folk_art/
    ├── prompt.json
    ├── source/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── target/
        ├── 000000.png
        ├── 000001.png
        └── ...

    prompt.json format:
    [
        {
            "source": "source/000000.png",
            "target": "target/000000.png",
            "prompt": "Vietnamese traditional temple architecture"
        },
        ...
    ]
    """

    def __init__(self, data_root="./training/vietnamese_folk_art"):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing prompt.json, source/, and target/
        """
        self.data_root = Path(data_root)
        self.data = []

        # Load prompts
        prompt_file = self.data_root / "prompt.json"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples from {data_root}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single training sample.

        Returns:
            dict with keys:
                - jpg: target image normalized to [-1, 1], shape (H, W, 3)
                - txt: prompt text
                - hint: source edge map normalized to [0, 1], shape (H, W, 3)
        """
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # Load images
        source_path = self.data_root / source_filename
        target_path = self.data_root / target_filename

        source = cv2.imread(str(source_path))
        target = cv2.imread(str(target_path))

        if source is None:
            raise FileNotFoundError(f"Source image not found: {source_path}")
        if target is None:
            raise FileNotFoundError(f"Target image not found: {target_path}")

        # Convert BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source to [0, 1]
        source = source.astype(np.float32) / 255.0

        # Normalize target to [-1, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


def test_dataset():
    """Test the dataset loading."""
    print("Testing dataset loading...")

    dataset = VietFolkArtDataset("./training/vietnamese_folk_art")
    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Prompt: {sample['txt']}")
        print(f"Target shape: {sample['jpg'].shape}, range: [{sample['jpg'].min():.2f}, {sample['jpg'].max():.2f}]")
        print(f"Hint shape: {sample['hint'].shape}, range: [{sample['hint'].min():.2f}, {sample['hint'].max():.2f}]")


if __name__ == "__main__":
    test_dataset()
