"""
Training script for Vietnamese Folk Art ControlNet.

This script trains a ControlNet to generate Vietnamese folk art images
from scribble/edge maps.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from viet_folk_art_dataset import VietFolkArtDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def train_controlnet(
    model_path="./models/control_sd15_ini.ckpt",
    config_path="./models/cldm_v15.yaml",
    data_root="./training/vietnamese_folk_art",
    output_dir="./logs/vietnamese_folk_art",
    batch_size=4,
    num_workers=4,
    learning_rate=1e-5,
    sd_locked=True,
    only_mid_control=False,
    logger_freq=300,
    max_epochs=10,
    accumulate_grad_batches=1,
    precision=32,
    gpus=1
):
    """
    Train ControlNet on Vietnamese Folk Art dataset.

    Args:
        model_path: Path to initial ControlNet model (SD + ControlNet)
        config_path: Path to model config YAML
        data_root: Path to training dataset
        output_dir: Path to save logs and checkpoints
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        learning_rate: Learning rate
        sd_locked: Whether to lock Stable Diffusion weights
        only_mid_control: Whether to only control middle layers
        logger_freq: Frequency of image logging
        max_epochs: Maximum number of training epochs
        accumulate_grad_batches: Gradient accumulation steps
        precision: Mixed precision (16 or 32)
        gpus: Number of GPUs to use
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configs
    print(f"Loading model from {model_path}")
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(model_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Dataset and DataLoader
    print(f"Loading dataset from {data_root}")
    dataset = VietFolkArtDataset(data_root)
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    # Logger
    logger = ImageLogger(batch_frequency=logger_freq)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="controlnet-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=logger_freq,
        save_last=True
    )

    # Trainer
    print("Starting training...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  SD locked: {sd_locked}")
    print(f"  Only mid control: {only_mid_control}")
    print(f"  Gradient accumulation: {accumulate_grad_batches}")
    print(f"  Precision: {precision}")

    trainer = pl.Trainer(
        gpus=gpus,
        precision=precision,
        callbacks=[logger, checkpoint_callback],
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        default_root_dir=output_dir,
        log_every_n_steps=50,
        val_check_interval=logger_freq
    )

    # Train!
    trainer.fit(model, dataloader)

    print(f"Training completed! Model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ControlNet on Vietnamese Folk Art")
    parser.add_argument("--model_path", type=str, default="./models/control_sd15_ini.ckpt",
                        help="Path to initial ControlNet model")
    parser.add_argument("--config_path", type=str, default="./models/cldm_v15.yaml",
                        help="Path to model config")
    parser.add_argument("--data_root", type=str, default="./training/vietnamese_folk_art",
                        help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="./logs/vietnamese_folk_art",
                        help="Path to save logs and checkpoints")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--sd_locked", action="store_true", default=True,
                        help="Lock Stable Diffusion weights")
    parser.add_argument("--no_sd_locked", action="store_false", dest="sd_locked",
                        help="Unlock Stable Diffusion weights")
    parser.add_argument("--only_mid_control", action="store_true", default=False,
                        help="Only control middle layers")
    parser.add_argument("--logger_freq", type=int, default=300,
                        help="Image logging frequency")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32],
                        help="Mixed precision (16 or 32)")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs")

    args = parser.parse_args()

    train_controlnet(
        model_path=args.model_path,
        config_path=args.config_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        sd_locked=args.sd_locked,
        only_mid_control=args.only_mid_control,
        logger_freq=args.logger_freq,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        gpus=args.gpus
    )
