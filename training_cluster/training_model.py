#!/usr/bin/env python3
"""
Single-file training runner for pytorch3dunet.

Features:
- YAML loader with automatic device selection (respects explicit CPU)
- Optional deterministic seeding via `manual_seed` in YAML
- Optional quick HDF5 inspection before training
- Clean logging and CLI (override config path from command line)

Usage:
  python training_main.py --config /path/to/training.yaml
  python training_main.py --config /path/to/training.yaml --inspect /path/to/sample.h5
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, Any, Tuple, Optional

import h5py
import torch
import yaml
from pytorch3dunet.unet3d.config import copy_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("TrainingSetup")


# -----------------------------
# Optional Napari preview
# -----------------------------
def show_napari(image):
    """Lazy napari import so headless runs don't require a GUI."""
    import napari
    with napari.gui_qt():
        _ = napari.view_image(image, title="Preview")


# -----------------------------
# Config / device / seed
# -----------------------------
def load_config(config_path: str) -> Tuple[Dict[str, Any], str]:
    """Load YAML and set a sane 'device' default (CUDA if available)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    device = config.get("device", None)
    if device == "cpu":
        logger.warning("CPU mode forced in config; training/prediction will be slow.")
        config["device"] = "cpu"
    else:
        if torch.cuda.is_available():
            config["device"] = "cuda"
        else:
            logger.warning("CUDA not available; falling back to CPU.")
            config["device"] = "cpu"

    return config, config_path


def apply_determinism_if_requested(cfg: Dict[str, Any]) -> None:
    """Enable deterministic behavior if 'manual_seed' key exists in YAML."""
    manual_seed: Optional[int] = cfg.get("manual_seed", None)
    if manual_seed is None:
        return

    logger.info(f"Seeding RNG for all devices with {manual_seed}")
    logger.warning("CuDNN deterministic mode enabled (may slow down training).")

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # better determinism


# -----------------------------
# HDF5 inspection (optional)
# -----------------------------
def inspect_hdf5_file(hdf5_path: str, raw_internal_path: str = "raw", label_internal_path: str = "label") -> None:
    """Print dataset keys and basic shapes/dtypes for a training HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        print(f"[HDF5] {hdf5_path}")
        print("  keys:", list(f.keys()))
        if raw_internal_path in f:
            d = f[raw_internal_path]
            print(f"  raw:   shape={d.shape} dtype={d.dtype}")
        if label_internal_path in f:
            d = f[label_internal_path]
            print(f"  label: shape={d.shape} dtype={d.dtype}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3D U-Net with pytorch3dunet.")
    p.add_argument(
        "--config",
        default="/home/grunwaldlab/development/rna_scope/data_prep_and_training/training_green_yellow/training_green_yellow.yaml",
        help="Path to YAML config (default points to training_green_yellow.yaml).",
    )
    p.add_argument(
        "--inspect",
        default=None,
        help="Optional: path to an HDF5 file to inspect before training.",
    )
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Load config and set device
    config, config_path = load_config(args.config)
    logger.info("===== Loaded Config =====")
    logger.info(config)

    # Optional: quick check of a sample HDF5 before training
    if args.inspect:
        inspect_hdf5_file(
            args.inspect,
            raw_internal_path=config.get("raw_internal_path", "raw"),
            label_internal_path=config.get("label_internal_path", "label"),
        )

    # Deterministic run if requested
    apply_determinism_if_requested(config)

    # Build trainer & persist resolved config
    trainer = create_trainer(config)
    copy_config(config, config_path)

    # Train
    trainer.fit()


if __name__ == "__main__":
    main()
