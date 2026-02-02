from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Optional

import h5py
import numpy as np
import tifffile as tiff
from scipy.ndimage import binary_fill_holes
from skimage import measure
from skimage.filters import gaussian
from skimage.transform import resize
from tqdm import tqdm


# -----------------------------
# Napari helpers (lazy import)
# -----------------------------
def show_napari(image: np.ndarray, title: str = "Image") -> None:
    import napari
    with napari.gui_qt():
        _ = napari.view_image(image, title=title)


def show_napari_pluslabel(raw_images: np.ndarray, label_images: np.ndarray) -> None:
    import napari
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw_images, name="Raw", blending="additive", colormap="gray")
        v.add_labels(label_images, name="Labels")


# -----------------------------
# I/O & pairing
# -----------------------------
def list_tifs(folder: Path, image_ext: str = ".tif") -> List[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == image_ext])


def pair_raw_and_labels(
    all_tifs: Sequence[Path],
    label_suffix: str = "_label.tif",
) -> Tuple[List[Path], List[Path]]:
    """
    Pair raw images with label images by filename stem:
    raw: <stem>.tif
    lab: <stem><label_suffix>
    """
    # Raw candidates: files that do NOT end with the label suffix
    raw_map = {p.stem: p for p in all_tifs if not p.name.endswith(label_suffix)}

    # Label candidates: map back to the raw stem
    # e.g. if suffix is "_label.tif", label file stem is "<stem>_label"
    label_suffix_without_ext = label_suffix.replace(".tif", "")
    lab_map = {}
    for p in all_tifs:
        if p.name.endswith(label_suffix):
            # turn "<stem>_label" back into "<stem>"
            raw_stem = p.stem.replace(label_suffix_without_ext, "")
            lab_map[raw_stem] = p

    raw_files, label_files, missing = [], [], []
    for stem, raw_path in raw_map.items():
        lab_path = lab_map.get(stem, None)
        if lab_path is None:
            missing.append(stem)
            continue
        raw_files.append(raw_path)
        label_files.append(lab_path)

    if missing:
        print(f"[WARN] Missing labels for {len(missing)} raw files (first 5): {missing[:5]}")
    if not raw_files:
        raise RuntimeError("No paired raw/label files found. Check LABEL_SUFFIX and filenames.")

    return raw_files, label_files


# -----------------------------
# Loading utilities
# -----------------------------
def ensure_zyx(arr: np.ndarray) -> np.ndarray:
    """Ensure array is (Z,Y,X); if 2D, add leading Z=1 axis."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array; got shape {arr.shape}")
    return arr


def load_hdf5_images(hdf5_files: Sequence[Path], max_images: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw + label arrays from a list of HDF5 files.
    Returns (raw, label) stacked (N,Z,Y,X).
    """
    raws, labs = [], []
    count = len(hdf5_files) if max_images is None else min(len(hdf5_files), max_images)
    for p in tqdm(hdf5_files[:count], desc="Loading HDF5 files"):
        try:
            with h5py.File(p, "r") as f:
                raw = ensure_zyx(f["raw"][:])
                lab = ensure_zyx(f["label"][:])
            raws.append(raw)
            labs.append(lab)
        except Exception as e:
            print(f"[ERROR] Failed reading {p}: {e}")

    if not raws:
        raise RuntimeError("No HDF5 images loaded.")
    return np.concatenate(raws, axis=0), np.concatenate(labs, axis=0)


def load_images_for_visualization(
    raw_files: Sequence[Path], label_files: Sequence[Path], max_images: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Read paired TIFFs and return stacked arrays (N,Z,Y,X)."""
    raws, labs = [], []
    num = len(raw_files) if max_images is None else min(len(raw_files), max_images)
    for r, l in zip(raw_files[:num], label_files[:num]):
        raw = ensure_zyx(tiff.imread(str(r)))
        lab = ensure_zyx(tiff.imread(str(l)))
        raws.append(raw)
        labs.append(lab)
    return np.concatenate(raws, axis=0), np.concatenate(labs, axis=0)


# -----------------------------
# Label processing
# -----------------------------
def process_labels(
    label_image: np.ndarray,
    min_size: int = 100,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Clean label image by:
      • fill holes
      • optional Gaussian smooth + rebinarize
      • remove small connected components
    Preserves original label IDs (excludes background=0).
    """
    lab = ensure_zyx(label_image)
    out = np.zeros_like(lab, dtype=lab.dtype)

    for z in range(lab.shape[0]):
        plane = lab[z]
        unique = np.unique(plane)
        unique = unique[unique != 0]

        for lbl in unique:
            mask = (plane == lbl)
            mask = binary_fill_holes(mask)

            if smooth_sigma and smooth_sigma > 0:
                sm = gaussian(mask.astype(float), sigma=smooth_sigma)
                mask = sm > 0.5
                mask = binary_fill_holes(mask)

            # Split disconnected pieces; keep >= min_size
            lab_conn = measure.label(mask, connectivity=1)
            for region in measure.regionprops(lab_conn):
                if region.area < min_size:
                    continue
                coords = tuple(region.coords.T)
                out[z][coords] = lbl

    return out


# -----------------------------
# HDF5 writing
# -----------------------------
def create_hdf5_per_image(
    image_files: Sequence[Path],
    label_files: Sequence[Path],
    output_folder: Path,
    *,
    process_labels_flag: bool = False,
    min_label_size: int = 100,
    smooth_sigma: float = 1.0,
    dtype_raw=np.uint16,
    dtype_label=np.int32,
    resize_to: Optional[Tuple[int, int, int]] = None,  # (Z,Y,X)
) -> None:
    """
    Create one HDF5 per image/label pair with datasets: 'raw', 'label'.
    Optionally resize BOTH raw and label stacks to the same shape.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for raw_path, lab_path in tqdm(list(zip(image_files, label_files)), desc="Writing HDF5", total=len(image_files)):
        try:
            raw = ensure_zyx(tiff.imread(str(raw_path)))
            lab = ensure_zyx(tiff.imread(str(lab_path)))
            print(f"[INFO] {raw_path.name}: raw {raw.shape}, label {lab.shape}")

            if process_labels_flag:
                lab = process_labels(lab, min_size=min_label_size, smooth_sigma=smooth_sigma)

            if resize_to is not None:
                # Raw: linear interpolation; Label: nearest (order=0)
                raw = resize(raw, resize_to, anti_aliasing=True, preserve_range=True).astype(dtype_raw)
                lab = resize(lab, resize_to, order=0, anti_aliasing=False, preserve_range=True).astype(dtype_label)

            out_path = output_folder / f"{raw_path.stem}.h5"
            with h5py.File(out_path, "w") as f:
                f.create_dataset("raw", data=raw.astype(dtype_raw))
                f.create_dataset("label", data=lab.astype(dtype_label))

        except Exception as e:
            print(f"[ERROR] {raw_path.name} / {lab_path.name}: {e}")


# -----------------------------
# Split & inspection
# -----------------------------
def split_data(
    image_files: Sequence[Path],
    label_files: Sequence[Path],
    val_split: float = 0.10,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """
    Sequential split (first (1-val) for train, last val for validation).
    """
    assert len(image_files) == len(label_files), "Mismatch raw/label counts before splitting."
    n = len(image_files)
    n_val = int(round(n * val_split))
    n_train = n - n_val

    train_imgs = list(image_files[:n_train])
    val_imgs = list(image_files[n_train:])
    train_labs = list(label_files[:n_train])
    val_labs = list(label_files[n_train:])
    return train_imgs, val_imgs, train_labs, val_labs


def inspect_hdf5_labels(hdf5_files: Sequence[Path], max_images_to_display: Optional[int] = None) -> None:
    raw, lab = load_hdf5_images(hdf5_files, max_images=max_images_to_display)
    show_napari_pluslabel(raw, lab)
