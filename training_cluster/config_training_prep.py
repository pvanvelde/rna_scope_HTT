"""
Configuration for the TIFF→HDF5 training data prep.
Edit paths and options below to match your environment.
"""

from pathlib import Path
from typing import Optional, Tuple

# ---- I/O paths ----
FOLDER: Path = Path("/media/grunwaldlab/Curcial P3 Plus 4TB/slide_scanner/data_for_training/tifffile/annotated_mousehtt_q111_june2025")
OUT_TRAIN: Path = Path("/media/grunwaldlab/Curcial P3 Plus 4TB/slide_scanner/data_for_training/h5files/annotated_mousehtt_q111_june2025/training_set_test")
OUT_VAL: Path   = Path("/media/grunwaldlab/Curcial P3 Plus 4TB/slide_scanner/data_for_training/h5files/annotated_mousehtt_q111_june2025/validation_set_test")

# ---- file pairing ----
IMAGE_EXT: str = ".tif"
LABEL_SUFFIX: str = "_label.tif"   # label file must be "<stem><LABEL_SUFFIX>"

# ---- split ----
VAL_SPLIT: float = 0.10            # 10% for validation

# ---- label processing (enable for nucleus-like labels, etc.) ----
PROCESS_LABELS: bool = False       # set True to apply cleanup
SMOOTH_SIGMA: float = 1.0          # Gaussian sigma for smoothing (if PROCESS_LABELS)
MIN_LABEL_SIZE: int = 50           # remove components smaller than this (pixels)

# ---- optional resizing (Z, Y, X). Use None to keep native size ----
RESIZE_TO: Optional[Tuple[int, int, int]] = None  # e.g., (16, 512, 512)

# ---- quick visualization after writing (0 = skip) ----
INSPECT_N: int = 10                 # e.g., 16 to open first 16 in Napari
