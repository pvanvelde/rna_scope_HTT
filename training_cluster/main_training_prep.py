#!/usr/bin/env python3
"""
Runner for the training data preparation using config_training_prep and utils_training_prep.
"""

from pathlib import Path
from typing import List

import config_training_prep as C
import utils_training_prep as U


def main():
    # Discover & pair
    all_tifs = U.list_tifs(C.FOLDER, image_ext=C.IMAGE_EXT)
    raw_files, label_files = U.pair_raw_and_labels(all_tifs, label_suffix=C.LABEL_SUFFIX)

    # Split
    tr_raw, va_raw, tr_lab, va_lab = U.split_data(raw_files, label_files, val_split=C.VAL_SPLIT)

    # Write HDF5s (train/val)
    U.create_hdf5_per_image(
        tr_raw, tr_lab, C.OUT_TRAIN,
        process_labels_flag=C.PROCESS_LABELS,
        min_label_size=C.MIN_LABEL_SIZE,
        smooth_sigma=C.SMOOTH_SIGMA,
        resize_to=C.RESIZE_TO,
    )
    U.create_hdf5_per_image(
        va_raw, va_lab, C.OUT_VAL,
        process_labels_flag=C.PROCESS_LABELS,
        min_label_size=C.MIN_LABEL_SIZE,
        smooth_sigma=C.SMOOTH_SIGMA,
        resize_to=C.RESIZE_TO,
    )

    # Optional quick inspection
    if C.INSPECT_N and C.INSPECT_N > 0:
        train_h5 = sorted([p for p in Path(C.OUT_TRAIN).iterdir() if p.suffix == ".h5"])
        if not train_h5:
            print("[WARN] No HDF5 files found in training output; skipping inspection.")
        else:
            n = min(C.INSPECT_N, len(train_h5))
            U.inspect_hdf5_labels(train_h5[:n], max_images_to_display=n)


if __name__ == "__main__":
    main()
