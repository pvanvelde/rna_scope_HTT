"""
Utilities for the RNAscope pipeline:
- safe helpers
- string normalization
- MIP handling
- Napari viewers
- SVG exporters for crisp MIP + labels + vector points
"""

from __future__ import annotations
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import napari


# -----------------------------
# Safe helpers & naming
# -----------------------------
def safe_array(value, dtype=np.float32, shape=0):
    """Return empty array if value is None; else the value."""
    return np.empty(shape, dtype=dtype) if value is None else value

def safe_scalar(value, default=0):
    """Return default if value is None; else the value."""
    return default if value is None else value

def standardize_folder_name(name: str) -> str:
    """Remove leading 'slide', punctuation/spaces, lowercase."""
    name = re.sub(r"^slide\s*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9]+", "", name)
    return name.lower()

def mip2d(img: np.ndarray) -> np.ndarray:
    """Return a 2D MIP along axis 0 if img is 3D; otherwise return as-is."""
    return img if img.ndim == 2 else np.max(img, axis=0)


# -----------------------------
# Napari viewers
# -----------------------------
def show_napari(raw_image2d, title="Image"):
    with napari.gui_qt():
        v = napari.Viewer(title=title)
        v.add_image(raw_image2d, name="Raw Image", colormap='gray', blending='additive')

def show_napari_pluslabel(raw2d, label2d):
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw2d, name="Raw MIP")
        v.add_labels(label2d, name="Labels MIP")

def show_napari_pluslabel_plusimage(raw2d, label2d, image2):
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw2d, name="Raw MIP")
        v.add_labels(label2d, name="Labels MIP")
        v.add_image(image2, name="Overlay", blending='additive', opacity=0.7)

def show_napari_spots(image2d, spots, second_spots=None, mode_name='spots'):
    with napari.gui_qt():
        v = napari.Viewer(title=f"Detected Spots - {mode_name}")
        v.add_image(image2d, name="Image", colormap='gray', blending='additive')
        if spots is not None and spots.size > 0:
            v.add_points(spots, face_color='transparent', edge_color='red', size=8, name="Spots")
        if second_spots is not None and second_spots.size > 0:
            v.add_points(second_spots, face_color='transparent', edge_color='green', size=8, name="Filtered Spots")


# -----------------------------
# SVG export (crisp MIP + labels + vector points)
# -----------------------------
def _prep_label_cmap(label2d, alpha=0.35, cmap_name="tab20"):
    """Make discrete colormap with index 0 transparent, others alpha=alpha."""
    if label2d is None:
        return None
    max_lab = int(np.nanmax(label2d)) if label2d.size else 0
    max_lab = max(1, max_lab)
    base = cm.get_cmap(cmap_name, max_lab + 1)
    colors = base(np.arange(max_lab + 1))
    colors[0, :] = (0, 0, 0, 0)
    colors[1:, 3] = alpha
    return ListedColormap(colors)

def export_svg_mip_labels_points(
    raw2d: np.ndarray,
    points2d: np.ndarray | None,
    out_path: str,
    *,
    labels2d: np.ndarray | None = None,
    labels_alpha: float = 0.35,
    labels_cmap_name: str = "tab20",
    vmin=None,
    vmax=None,
    ppi: int = 400,
    show_axes: bool = False,
    point_size: float = 16,
    point_edgecolor: str = "white",
    point_facecolor: str = "none",
    point_linewidth: float = 1.0,
    overlay_labels: bool = True,
):
    """
    Export an SVG with:
      - raw2d (raster, no interpolation),
      - optional labels2d (raster, discrete cmap),
      - points2d (vector scatter).
    """
    if raw2d is None or raw2d.ndim != 2:
        raise ValueError("raw2d must be a 2D ndarray")

    H, W = raw2d.shape

    # vmin/vmax without auto-stretch
    if vmin is None or vmax is None:
        if np.issubdtype(raw2d.dtype, np.integer):
            info = np.iinfo(raw2d.dtype)
            vmin_ = info.min if vmin is None else vmin
            vmax_ = info.max if vmax is None else vmax
        else:
            rmin, rmax = float(np.nanmin(raw2d)), float(np.nanmax(raw2d))
            if 0.0 <= rmin <= 1.0 and 0.0 <= rmax <= 1.0:
                vmin_ = 0.0 if vmin is None else vmin
                vmax_ = 1.0 if vmax is None else vmax
            else:
                vmin_ = rmin if vmin is None else vmin
                vmax_ = rmax if vmax is None else vmax
    else:
        vmin_, vmax_ = vmin, vmax

    fig_w_in, fig_h_in = W / ppi, H / ppi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_subplot(111)

    # Raw MIP background (no interpolation)
    ax.imshow(
        raw2d, cmap="gray", vmin=vmin_, vmax=vmax_,
        origin="upper", extent=(0, W, H, 0),
        interpolation="none", rasterized=True, zorder=0
    )

    # Labels overlay
    if overlay_labels and labels2d is not None:
        lab_cmap = _prep_label_cmap(labels2d, alpha=labels_alpha, cmap_name=labels_cmap_name)
        ax.imshow(
            labels2d, cmap=lab_cmap, vmin=0, vmax=np.nanmax(labels2d) if labels2d.size else 1,
            origin="upper", extent=(0, W, H, 0),
            interpolation="none", rasterized=True, zorder=5
        )

    # Axes & limits
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal", adjustable="box")
    if not show_axes:
        ax.axis("off")

    # Points (vector)
    if points2d is not None and points2d.size > 0:
        pts = np.asarray(points2d, dtype=float)
        if pts.shape[1] != 2:
            raise ValueError("points2d must be (N,2) in (y,x)")
        y = pts[:, 0]
        x = pts[:, 1]
        m = np.isfinite(x) & np.isfinite(y) & (0 <= x) & (x <= W - 1) & (0 <= y) & (y <= H - 1)
        if np.any(m):
            ax.scatter(
                x[m], y[m],
                s=point_size,
                edgecolors=point_edgecolor,
                facecolors=point_facecolor,
                linewidths=point_linewidth,
                zorder=10,
            )

    fig.savefig(out_path, format="svg", dpi=ppi)  # not tight -> preserves pixel grid
    plt.close(fig)


# -----------------------------
# High-level viewer + svg combo
# -----------------------------
def show_napari_pluslabel_plusspotsv2(
    raw_images, label_images, spots_all,
    filt_indices, filt_indices_sig, pfa_values, filter_on_break,
    show_final=True, pfa_thresh=0.05,
    save_svgs=False, output_dir=".", base_name="spots",
    svg_include_axes=False, svg_ppi=400, svg_vmin=None, svg_vmax=None,
    svg_overlay_labels=True, svg_labels_alpha=0.35, svg_labels_cmap="tab20",
):
    """
    Show raw MIP + labels + multiple spot layers in Napari.
    Optionally export per-layer SVGs with crisp MIP & labels.
    """
    raw2d   = mip2d(raw_images)
    label2d = mip2d(label_images) if label_images is not None else None

    # points to (y,x)
    spots_all = np.asarray(spots_all)
    if spots_all.ndim == 2 and spots_all.shape[1] == 3:  # (z,y,x) -> (y,x)
        pts_all = spots_all[:, 1:]
    else:
        pts_all = spots_all

    filt_indices      = np.asarray(filt_indices, dtype=bool)
    filt_indices_sig  = np.asarray(filt_indices_sig, dtype=bool)
    filter_on_break   = np.asarray(filter_on_break, dtype=bool) if filter_on_break is not None else np.zeros(len(pts_all), bool)
    pfa_values        = np.asarray(pfa_values)

    both_filt  = filt_indices & filt_indices_sig
    pfa_ok     = np.all(pfa_values <= pfa_thresh, axis=1) if pfa_values.size else np.zeros(len(pts_all), bool)
    final_mask = both_filt & pfa_ok & filter_on_break

    pts_both  = pts_all[both_filt]      if pts_all.size else np.empty((0,2))
    pts_pfa   = pts_all[pfa_ok]         if pts_all.size else np.empty((0,2))
    pts_break = pts_all[filter_on_break]if pts_all.size else np.empty((0,2))
    pts_final = pts_all[final_mask]     if (show_final and pts_all.size) else np.empty((0,2))

    # Exports
    if save_svgs:
        os.makedirs(output_dir, exist_ok=True)
        export_svg_mip_labels_points(raw2d, pts_all,  os.path.join(output_dir, f"{base_name}__all.svg"),
                                     labels2d=label2d, labels_alpha=svg_labels_alpha, labels_cmap_name=svg_labels_cmap,
                                     vmin=svg_vmin, vmax=svg_vmax, ppi=svg_ppi, show_axes=svg_include_axes,
                                     point_edgecolor="white", overlay_labels=svg_overlay_labels)
        export_svg_mip_labels_points(raw2d, pts_both, os.path.join(output_dir, f"{base_name}__filt_and_sig.svg"),
                                     labels2d=label2d, labels_alpha=svg_labels_alpha, labels_cmap_name=svg_labels_cmap,
                                     vmin=svg_vmin, vmax=svg_vmax, ppi=svg_ppi, show_axes=svg_include_axes,
                                     point_edgecolor="orange", overlay_labels=svg_overlay_labels)
        export_svg_mip_labels_points(raw2d, pts_pfa,  os.path.join(output_dir, f"{base_name}__pfa.svg"),
                                     labels2d=label2d, labels_alpha=svg_labels_alpha, labels_cmap_name=svg_labels_cmap,
                                     vmin=svg_vmin, vmax=svg_vmax, ppi=svg_ppi, show_axes=svg_include_axes,
                                     point_edgecolor="cyan", overlay_labels=svg_overlay_labels)
        export_svg_mip_labels_points(raw2d, pts_break,os.path.join(output_dir, f"{base_name}__break.svg"),
                                     labels2d=label2d, labels_alpha=svg_labels_alpha, labels_cmap_name=svg_labels_cmap,
                                     vmin=svg_vmin, vmax=svg_vmax, ppi=svg_ppi, show_axes=svg_include_axes,
                                     point_edgecolor="magenta", overlay_labels=svg_overlay_labels)
        if show_final:
            export_svg_mip_labels_points(raw2d, pts_final, os.path.join(output_dir, f"{base_name}__final.svg"),
                                         labels2d=label2d, labels_alpha=svg_labels_alpha, labels_cmap_name=svg_labels_cmap,
                                         vmin=svg_vmin, vmax=svg_vmax, ppi=svg_ppi, show_axes=svg_include_axes,
                                         point_edgecolor="green", overlay_labels=svg_overlay_labels)

    # Napari
    with napari.gui_qt():
        v = napari.Viewer()
        v.add_image(raw2d, name="Raw MIP")
        if label2d is not None:
            v.add_labels(label2d, name="Labels MIP")

        def _add(arr, name, color):
            if arr is not None and arr.size:
                v.add_points(arr, name=name, face_color='transparent', edge_color=color, size=8)

        _add(pts_all,  "Spots: all",               "white")
        _add(pts_both, "Spots: filt & sig",        "yellow")
        _add(pts_pfa,  f"Spots: pfa≤{pfa_thresh}", "cyan")
        _add(pts_break,"Spots: filter_on_break",   "magenta")
        if show_final:
            _add(pts_final, "Spots: FINAL (all)",  "lime")

        for layer in v.layers:
            if isinstance(layer, napari.layers.Points):
                layer.blending = 'translucent'
