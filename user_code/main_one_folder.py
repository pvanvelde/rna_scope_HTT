#!/usr/bin/env python
"""
RNAscope single-folder pipeline

Usage
-----
python runscript_onefolder.py  <folder_path>  <metadata_xlsx>
"""

# --- keep one clean import section ------------------------------------------
import os, re, argparse
import numpy as np
import pandas as pd
import torch
import napari
import traceback
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

torch.set_default_dtype(torch.float32)  # optional if you relied on it
from rna_scope_backbone import RNAscope
# Custom modules
from rna_scope_backbone.RNAscope import RNAscopeclass
from rna_scope_backbone.RNAscopeDataManager import RNAscopeDataManager
from config import cfg, data_manager_cfg, rootfolder, data_file, color_config, detection_cfg, fit_cfg
from utils import (
    safe_array, safe_scalar, standardize_folder_name, mip2d,
    show_napari_pluslabel_plusspotsv2,  # optional visualization/export
)


# =============================================================================
# Core Processing Function
# =============================================================================
def process_file(file_path, RS, data_manager,
                 color_config, detection_cfg, fit_cfg,
                 file_id, trailing_digits, main_cfg, meta_data_sample,
                 fit_bg_per_slice=True, sampling=(0.5, 0.1625, 0.1625)):
    """
    Process a single .npz (or .tif) file for all colors defined in color_config.
    """
    h5_data = {}
    darp_flag = any(cfg_color["mode_name"].lower() == "darp" for cfg_color in color_config.values())
    color_darp = "red" if darp_flag else None

    # Ensure DAPI labels exist BEFORE other channels need them
    dapi_labels = None
    dapi_sizes = None
    com_dapi_darp = None
    darp_image = None
    num_cells = None

    # Pass 1: do DAPI first if present, to get dapi_labels
    ordered_items = list(color_config.items())
    ordered_items.sort(key=lambda kv: 0 if kv[1]["mode_name"].lower() == "dapi" else 1)

    for color, cfg_color in ordered_items:
        ch_index = cfg_color["channel_index"]
        mode_name = cfg_color["mode_name"]
        do_detect_fit = cfg_color["detect_and_fit"]
        do_detect_labels = cfg_color["detect_labels"]
        min_size = cfg_color["min_size"]
        max_size = cfg_color["max_size"]
        break_sigma = cfg_color["break_sigma"]

        # 1) Load image
        image_data, channel_name = RS.load_image(file_path, channel_to_load=ch_index)
        if image_data is None:
            print(f"Warning: Could not load {color} channel from {file_path}")
            h5_data[color] = {}
            continue

        # 2) Labels
        label_mask = label_image = label_sizes = None
        if (min_size is not None) and (max_size is not None) and do_detect_labels:
            label_mask, label_image, label_sizes = RS.generate_label(
                min_size=min_size, max_size=max_size, mode=color
            )

        # 3) Spot detection & fitting
        if do_detect_fit:
            (mu_fil, smp_fil, final_params, traces, mip_image,
             filtered_coords, z_starts, filt_indices, pfa_values, params_raw) = RS.detect_and_fit(
                detection_cfg, fit_cfg, image_data, cfg_color,
                label_mask=label_mask, batch_size=5000,
                fit_bg_per_slice=fit_bg_per_slice
            )

            distance_spots_to_closest_dapi = None
            if dapi_labels is not None:
                distance_spots_to_closest_dapi = RS.compute_distance_to_closest_dapi(
                    dapi_labels, z_starts, filt_indices, final_params, filtered_coords, sampling
                )

            # Sigma-fitting pass (conditioned on previous filters)
            (mu_fil_sig, smp_fil_sig, final_params_sig, traces_sig, mip_image_sig,
             filtered_coords_sig, z_starts_sig, filt_indices_sig, pfa_values_sig, params_raw_sig) = RS.detect_and_fit(
                detection_cfg, fit_cfg, image_data, cfg_color,
                label_mask=label_mask, batch_size=5000,
                fit_bg_per_slice=fit_bg_per_slice, fit_sigma=filt_indices
            )

            # Optional break filter
            filter_on_break = None
            num_clusters_before_pruning = None
            num_clusters_after_pruning = None
            if break_sigma is not None and label_mask is not None and final_params_sig is not None:
                filter_on_break = (
                    (params_raw_sig[:, -3] < break_sigma[0]) &
                    (params_raw_sig[:, -2] < break_sigma[1]) &
                    (params_raw_sig[:, -1] < break_sigma[2])
                )
                final_filter = (filt_indices) & (filt_indices_sig) & (np.all(pfa_values <= 0.05, axis=1)) & (filter_on_break)
                # Track number of clusters before pruning
                num_clusters_before_pruning = len(np.unique(label_mask)) - 1  # -1 to exclude background (0)
                # Remove labels touching final-filtered spots
                _, pruned_mip, pruned = RS.remove_labels_touching_spots(
                    filtered_coords_sig[final_filter, :], label_mask, dilation_radius=0
                )
                label_mask = pruned  # use pruned mask for downstream
                # Track number of clusters after pruning
                num_clusters_after_pruning = len(np.unique(label_mask)) - 1  # -1 to exclude background (0)
                print(f"  {mode_name}: Pruned {num_clusters_before_pruning - num_clusters_after_pruning} clusters "
                      f"({num_clusters_before_pruning} → {num_clusters_after_pruning})")

        else:
            mu_fil = smp_fil = final_params = traces = filtered_coords = params_raw = None
            z_starts = filt_indices = pfa_values = None
            mu_fil_sig = smp_fil_sig = final_params_sig = traces_sig = filtered_coords_sig = params_raw_sig = None
            z_starts_sig = filt_indices_sig = pfa_values_sig = None
            mip_image = mip2d(image_data)
            distance_spots_to_closest_dapi = None
            filter_on_break = None
            final_filter = None
            num_clusters_before_pruning = None
            num_clusters_after_pruning = None

        # 4) Cluster intensities & distances (needs labels)
        distance_clusters_to_closest_dapi = None
        cluster_label_sizes = None  # Will store sizes from analyze_label_intensitiesv2 (post-pruning)
        cluster_cvs = None  # Will store CVs from analyze_label_intensitiesv2 (post-pruning)
        # Note: num_clusters_before_pruning and num_clusters_after_pruning are set in pruning section above

        if label_mask is not None:
            converted_image = RS.convert_to_photons(image_data)
            # analyze_label_intensitiesv2 now returns (intensities, com_array, label_sizes, label_cvs)
            # where label_sizes matches the indices of intensities (post-pruning for green/orange)
            cluster_intensities, com_array, cluster_label_sizes, cluster_cvs = RS.analyze_label_intensitiesv2(label_mask, converted_image)

            if mode_name.lower() == "dapi":
                dapi_labels = label_mask
                num_cells = len(np.unique(dapi_labels)) - 1
            else:
                # For non-DAPI channels (green/orange), compute distance to DAPI cells
                # Signed distance: negative = inside DAPI, positive = outside DAPI
                if dapi_labels is not None and com_array is not None and len(com_array) > 0:
                    distance_clusters_to_closest_dapi = RS.compute_distance_to_closest_dapi(
                        dapi_labels, None, None, None, com_array, sampling
                    )

            # Validate that distance array length matches cluster_intensities
            if distance_clusters_to_closest_dapi is not None and len(distance_clusters_to_closest_dapi) != len(cluster_intensities):
                raise ValueError(
                    f"Length mismatch: distance_clusters_to_closest_dapi ({len(distance_clusters_to_closest_dapi)}) "
                    f"!= cluster_intensities ({len(cluster_intensities)}) for {mode_name} channel"
                )

            # Validate that cluster_label_sizes matches cluster_intensities
            if cluster_label_sizes is not None and len(cluster_label_sizes) != len(cluster_intensities):
                raise ValueError(
                    f"Length mismatch: cluster_label_sizes ({len(cluster_label_sizes)}) "
                    f"!= cluster_intensities ({len(cluster_intensities)}) for {mode_name} channel"
                )

            # Validate that cluster_cvs matches cluster_intensities
            if cluster_cvs is not None and len(cluster_cvs) != len(cluster_intensities):
                raise ValueError(
                    f"Length mismatch: cluster_cvs ({len(cluster_cvs)}) "
                    f"!= cluster_intensities ({len(cluster_intensities)}) for {mode_name} channel"
                )
        else:
            cluster_intensities, com_array, cluster_label_sizes, cluster_cvs = np.array([]), np.array([]), np.array([]), np.array([])

        # 5) DARP extras
        if darp_flag:
            if mode_name.lower() == "darp":
                darp_image = image_data
            if mode_name.lower() == "dapi":
                com_dapi_darp = com_array
                dapi_sizes = label_sizes

        # 6) Spots dictionaries
        spots_dict = {
            "x": final_params[:, 0] if final_params is not None else np.array([]),
            "y": final_params[:, 1] if final_params is not None else np.array([]),
            "z": final_params[:, 2] if final_params is not None else np.array([]),
            "photons": final_params[:, 3] if final_params is not None else np.array([]),
            "bg": final_params[:, 4::] if final_params is not None else np.array([]),
            "params_raw": params_raw if params_raw is not None else np.array([]),
            "filtered_coords": filtered_coords if filtered_coords is not None else np.array([]),
            "z_starts": z_starts if z_starts is not None else np.array([]),
            "filter_indices": filt_indices if filt_indices is not None else np.array([]),
            "pfa_values": safe_array(pfa_values),
            "dist_to_dapi_um": safe_array(distance_spots_to_closest_dapi),
            "filter_on_break": filter_on_break if filter_on_break is not None else np.array([]),
            "final_filter": final_filter if final_filter is not None else np.array([]),
        }

        spots_dict_sigma_var = {
            "x": final_params_sig[:, 0] if final_params_sig is not None else np.array([]),
            "y": final_params_sig[:, 1] if final_params_sig is not None else np.array([]),
            "z": final_params_sig[:, 2] if final_params_sig is not None else np.array([]),
            "photons": final_params_sig[:, 3] if final_params_sig is not None else np.array([]),
            "bg": final_params_sig[:, 4:-3] if final_params_sig is not None else np.array([]),
            "params_raw": params_raw_sig if params_raw_sig is not None else np.array([]),
            "sigma": final_params_sig[:, -3::] if final_params_sig is not None else np.array([]),
            "filtered_coords": filtered_coords_sig if filtered_coords_sig is not None else np.array([]),
            "z_starts": z_starts_sig if z_starts_sig is not None else np.array([]),
            "filter_indices": filt_indices_sig if filt_indices_sig is not None else np.array([]),
            "pfa_values": safe_array(pfa_values_sig),
            "filter_on_break": filter_on_break if filter_on_break is not None else np.array([]),
            "final_filter": final_filter if final_filter is not None else np.array([]),
        }

        h5_data[color] = {
            "cluster_intensities": cluster_intensities,
            "cluster_distance_dapi_um": distance_clusters_to_closest_dapi,
            "label_coms": com_array,
            # Use cluster_label_sizes from analyze_label_intensitiesv2 (post-pruning, matches cluster_intensities indices)
            "label_sizes": cluster_label_sizes if cluster_label_sizes is not None else np.array([]),
            "cluster_cvs": cluster_cvs if cluster_cvs is not None else np.array([]),
            "num_cells": num_cells,
            "spots": spots_dict,
            "spots_sigma_var": spots_dict_sigma_var,
            "metadata": {
                "channel_index": ch_index,
                "mode_name": mode_name,
                "min_size": safe_scalar(min_size, 0),
                "max_size": safe_scalar(max_size, 0),
                "num_labels": safe_scalar(len(np.unique(label_mask)) - 1 if label_mask is not None else 0, 0),
                "channel name": channel_name,
                # Pruning statistics
                "num_clusters_before_pruning": safe_scalar(num_clusters_before_pruning, 0),
                "num_clusters_after_pruning": safe_scalar(num_clusters_after_pruning, 0),
            },
        }

    # DARP post
    if darp_flag and (dapi_labels is not None) and (darp_image is not None):
        _, Darp_mean = RS.process_labelsindapi_intensityelse(dapi_labels, darp_image)
        if Darp_mean is not None:
            h5_data[color_darp]['darp_signal'] = np.stack(safe_array(Darp_mean))
        else:
            h5_data[color_darp]['darp_signal'] = np.nan
        h5_data[color_darp]['darp_com'] = com_dapi_darp
        h5_data[color_darp]['size_dapi'] = dapi_sizes

    # General metadata
    h5_data["general_metadata"] = {
        "file_path": file_path or "",
        "FOVnumber": trailing_digits,
        "color_info": color_config,
        "fit_cfg": fit_cfg,
        "detection_cfg": detection_cfg,
        "main_cfg": main_cfg,
    }
    h5_data["metadata_sample"] = meta_data_sample.to_dict(orient='list')

    data_manager.add_file_data(file_id=file_id, file_path=file_path, h5_data=h5_data)
    print(f"Finished processing: {file_id}")


def process_region(region_folder: str,
                   slide_basename: str,
                   meta_df: pd.DataFrame,
                   data_manager: RNAscopeDataManager):
    """Process every .npz file in one region folder."""
    root_basename = os.path.basename(os.path.dirname(os.path.dirname(region_folder)))
    region_basename = os.path.basename(region_folder)
    slide_std = standardize_folder_name(slide_basename)
    region_std = standardize_folder_name(region_basename)

    meta_rows = meta_df[
        (meta_df["slide_name_std"] == slide_std) &
        (meta_df["region_std"] == region_std)
        ]
    if meta_rows.empty:
        print(f"⚠️  no metadata for slide '{slide_basename}', "
              f"region '{region_basename}' – skipping region.")
        return

    npz_files = sorted(f for f in os.listdir(region_folder) if f.endswith(".npz"))
    if not npz_files:
        print(f"⚠️  no .npz files in {region_folder}")
        return

    for idx_total, file_name in enumerate(npz_files, start=1):
        file_path = os.path.join(region_folder, file_name)
        trailing_digits = "".join(c for c in reversed(file_name) if c.isdigit())[::-1]

        RS = RNAscopeclass(cfg)
        RS.compute_gain(images_max=1)
        RS.image_file = file_path
        root_tag = standardize_folder_name(root_basename)
        slide_tag = standardize_folder_name(slide_basename)
        region_tag = standardize_folder_name(region_basename)
        fov_tag = trailing_digits.zfill(3)
        idx_tag = f"n{idx_total:04d}"
        file_id = f"{root_tag}--{slide_tag}--{region_tag}--{fov_tag}--{idx_tag}"

        print(f"[{slide_tag}/{region_tag}] {file_name} → {file_id}")
        try:
            process_file(
                file_path=file_path,
                RS=RS,
                data_manager=data_manager,
                color_config=color_config,
                detection_cfg=detection_cfg,
                fit_cfg=fit_cfg,
                file_id=file_id,
                trailing_digits=trailing_digits,
                main_cfg=cfg,
                meta_data_sample=meta_rows,
            )
        except Exception as e:
            print(f"❌ {file_path} failed: {e}")
            traceback.print_exc()  # ← prints the full stack trace


def main(slide_folder: str, metadata_xlsx: str):
    if not os.path.isdir(slide_folder):
        raise FileNotFoundError(slide_folder)
    if not os.path.isfile(metadata_xlsx):
        raise FileNotFoundError(metadata_xlsx)
    # ── data-manager outputs (per slide) ─────────────────────────────────────
    slide_name = os.path.basename(os.path.normpath(slide_folder))
    output_h5 = os.path.join(slide_folder, f"{slide_name}_results.h5")
    output_csv = os.path.join(slide_folder, f"{slide_name}_summary.csv")

    data_manager_cfg = {
        "master_h5_path": output_h5,
        "summary_csv": output_csv,
        "data_aq": True,
    }
    meta_df = pd.read_excel(metadata_xlsx)
    meta_df["slide_name_std"] = meta_df["Slide name"].apply(standardize_folder_name)
    meta_df["region_std"] = meta_df["Region"].apply(standardize_folder_name)

    data_manager = RNAscopeDataManager(**data_manager_cfg)

    region_dirs = [d for d in os.listdir(slide_folder)
                   if os.path.isdir(os.path.join(slide_folder, d))]

    if not region_dirs:
        print(f"⚠️  No region sub-folders found in {slide_folder}")
        return

    for region_dir in region_dirs:
        region_path = os.path.join(slide_folder, region_dir)
        process_region(region_path,
                       slide_basename=os.path.basename(slide_folder),
                       meta_df=meta_df,
                       data_manager=data_manager)

    print("✓ Slide complete – results saved to", output_h5)


# =============================================================================
# Command-line interface
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process one slide directory (all its region sub-folders)."
    )
    parser.add_argument("slide_folder",
                        help="Path to the slide folder whose immediate "
                             "sub-directories are region folders with .npz files.")
    parser.add_argument("metadata_xlsx",
                        help="Path to the metadata Excel workbook.")
    args = parser.parse_args()

    main(args.slide_folder, args.metadata_xlsx)