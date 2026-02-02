#!/usr/bin/env python
"""
RNAscope Image Processing Pipeline (main module)
"""

# =============================================================================
# Imports
# =============================================================================
import os
import numpy as np
import pandas as pd
import torch

from rna_scope_backbone.RNAscope import RNAscopeclass
from rna_scope_backbone.RNAscopeDataManager import RNAscopeDataManager

# local modules
from config import cfg, data_manager_cfg, rootfolder, data_file, color_config, detection_cfg, fit_cfg
from utils import (
    safe_array, safe_scalar, standardize_folder_name, mip2d,
    show_napari_pluslabel_plusspotsv2,  # optional visualization/export
)

# =============================================================================
# Torch Configuration
# =============================================================================
torch.set_default_dtype(torch.float32)

# =============================================================================
# Core Processing
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
            if break_sigma is not None and label_mask is not None and final_params_sig is not None:
                filter_on_break = (
                    (params_raw_sig[:, -3] < break_sigma[0]) &
                    (params_raw_sig[:, -2] < break_sigma[1]) &
                    (params_raw_sig[:, -1] < break_sigma[2])
                )
                final_filter = (filt_indices) & (filt_indices_sig) & (np.all(pfa_values <= 0.05, axis=1)) & (filter_on_break)
                # Remove labels touching final-filtered spots
                _, pruned_mip, pruned = RS.remove_labels_touching_spots(
                    filtered_coords_sig[final_filter, :], label_mask, dilation_radius=0
                )
                label_mask = pruned  # use pruned mask for downstream

        else:
            mu_fil = smp_fil = final_params = traces = filtered_coords = params_raw = None
            z_starts = filt_indices = pfa_values = None
            mu_fil_sig = smp_fil_sig = final_params_sig = traces_sig = filtered_coords_sig = params_raw_sig = None
            z_starts_sig = filt_indices_sig = pfa_values_sig = None
            mip_image = mip2d(image_data)
            distance_spots_to_closest_dapi = None
            filter_on_break = None
            final_filter = None

        # 4) Cluster intensities & distances (needs labels)
        if label_mask is not None:
            converted_image = RS.convert_to_photons(image_data)
            cluster_intensities, com_array = RS.analyze_label_intensitiesv2(label_mask, converted_image)

            if mode_name.lower() == "dapi":
                dapi_labels = label_mask
                num_cells = len(np.unique(dapi_labels)) - 1
                if num_cells and num_cells > 0:
                    distance_clusters_to_closest_dapi = RS.compute_distance_to_closest_dapi(
                        dapi_labels, None, None, None, com_array, sampling
                    )
                else:
                    distance_clusters_to_closest_dapi = None
            else:  # For non-DAPI channels
                if dapi_labels is not None and len(com_array) > 0:
                    distance_clusters_to_closest_dapi = RS.compute_distance_to_closest_dapi(
                        dapi_labels, None, None, None, com_array, sampling
                    )
                else:
                    distance_clusters_to_closest_dapi = None

        else:
            cluster_intensities, com_array = np.array([]), np.array([])
            distance_clusters_to_closest_dapi = None

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

        # Check consistency of cluster-related arrays
        cluster_arrays = {
            "cluster_intensities": cluster_intensities,
            "cluster_distance_dapi_um": distance_clusters_to_closest_dapi,
            "label_coms": com_array,
            "label_sizes": label_sizes,
        }
        lengths = []
        for name, arr in cluster_arrays.items():
            if arr is None:
                lengths.append((name, None))
            elif hasattr(arr, '__len__'):
                lengths.append((name, len(arr)))
            else:
                lengths.append((name, 1))  # scalar

        # Check if all are None/empty or all have the same length
        non_empty = [(name, l) for name, l in lengths if l is not None and l > 0]
        if non_empty:
            unique_lengths = set(l for _, l in non_empty)
            if len(unique_lengths) > 1:
                print(f"WARNING [{color}]: Cluster arrays have inconsistent lengths: {lengths}")

        h5_data[color] = {
            "cluster_intensities": cluster_intensities,
            "cluster_distance_dapi_um": distance_clusters_to_closest_dapi,
            "label_coms": com_array,
            "label_sizes": label_sizes,
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


# =============================================================================
# Main
# =============================================================================
def main():
    with torch.no_grad():
        # 1) Data manager
        data_manager = RNAscopeDataManager(**data_manager_cfg)

        # 2) Load metadata
        df = pd.read_excel(data_file)
        df["slide_name_std"] = df["Slide name"].apply(standardize_folder_name)
        df["region_std"] = df["Region"].apply(standardize_folder_name)

        # Slides present on disk
        slide_folders = [d for d in os.listdir(rootfolder) if os.path.isdir(os.path.join(rootfolder, d))]
        idx_total = 0

        for slide_folder in slide_folders:
            slide_folder_std = standardize_folder_name(slide_folder)
            matching_slide_rows = df[df["slide_name_std"] == slide_folder_std]
            if matching_slide_rows.empty:
                print(f"WARNING: No metadata for slide '{slide_folder}', skipping.")
                continue

            slide_path = os.path.join(rootfolder, slide_folder)
            region_folders = [d for d in os.listdir(slide_path) if os.path.isdir(os.path.join(slide_path, d))]

            for region_folder in region_folders:
                region_folder_std = standardize_folder_name(region_folder)
                matching_region_rows = matching_slide_rows[matching_slide_rows["region_std"] == region_folder_std]
                if matching_region_rows.empty:
                    print(f"WARNING: No metadata for region '{region_folder}' in slide '{slide_folder}'.")
                    continue

                region_path = os.path.join(slide_path, region_folder)
                npz_files = [os.path.join(region_path, f) for f in os.listdir(region_path) if f.endswith(".npz")]

                for file_path in npz_files:
                    idx_total += 1
                    # if idx_total > 10:  # optional short-circuit
                    #     break

                    base = os.path.basename(file_path)
                    root_name, _ = os.path.splitext(base)
                    trailing_digits = "".join(reversed([c for c in reversed(root_name) if c.isdigit()]))

                    RS = RNAscopeclass(cfg)
                    RS.compute_gain(images_max=1)
                    RS.image_file = file_path
                    file_id = f"file_{idx_total}"

                    print(f"Processing {idx_total}: {file_path}")
                    print(matching_region_rows["Probe-Set"])

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
                        meta_data_sample=matching_region_rows,
                    )

    print("All done!")


if __name__ == "__main__":
    main()
