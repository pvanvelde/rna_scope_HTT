import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def compute_thresholds(
    df_extracted,
    slide_field=None,
    desired_channels=None,
    negative_control_field='Negative Control',
    experimental_field='ExperimentalQ111 - 488mHT - 548mHTa - 647Darp',
    quantile_negative_control=0.95,
    max_pfa=1.0,
    plot=True,
    n_bootstrap=1000,
    use_region = True,
    cluster_threshold_val=None,
    use_final_filter = True,

):
    """
    Compute thresholds and bootstrapped threshold arrays for negative control data.

    Parameters:
        df_extracted (pd.DataFrame): DataFrame with extracted data.
        slide_field (str or None): Column name for slide identifier. If None, process all rows as a single group.
        desired_channels (list or None): List of channel names to process. If None, process all channels.
        negative_control_field (str): String used to filter negative control probe-sets.
        quantile_negative_control (float): Quantile to compute (e.g., 0.95 for the 95th percentile).
        max_pfa (float): Maximum value to filter the 'spots.pfa_values'.
        plot (bool): If True, plot the CDF and threshold line for each group.
        n_bootstrap (int): Number of bootstrap iterations.

    Returns:
        thresholds (dict): Mean threshold values for each group.
        thresholds_cluster (dict): Mean threshold values for cluster data for each group.
        error_thresholds (dict): Bootstrapped threshold arrays for photons.
        error_thresholds_cluster (dict): Bootstrapped threshold arrays for cluster data.
    """
    thresholds = {}
    thresholds_cluster = {}
    error_thresholds = {}
    error_thresholds_cluster = {}
    number_of_points = {}
    age = {}
    def bootstrap_thresholds(data, quantile, n_bootstrap=n_bootstrap):
        bootstrapped_thresholds = []
        n = len(data)
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            threshold_val = np.quantile(sample, quantile)
            bootstrapped_thresholds.append(threshold_val)
        return np.array(bootstrapped_thresholds)

    # Determine slides to process: if slide_field is provided, get unique slides;
    # otherwise, process the entire DataFrame as one group (with key None).
    if slide_field is not None:
        unique_slides = df_extracted[slide_field].unique()
    else:
        unique_slides = [None]

    for slide in unique_slides:
        if slide is not None:
            slide_df = df_extracted[df_extracted[slide_field] == slide].copy()
        else:
            slide_df = df_extracted.copy()

        slide_df['final_filter'] = pd.Series([None] * len(slide_df), dtype='object')

        # Loop over each row to add a new column for filtering based on 'spots.pfa_values'
        for idx, row in slide_df.iterrows():

            if use_final_filter:
                arr = row['spots.final_filter']
            else:

                arr = row['spots.pfa_values']

            if arr.size == 0:

                slide_df.at[idx, 'final_filter'] = []
            else:
                if use_final_filter:
                    mask = row['spots.final_filter']
                else:



                    mask = arr < max_pfa  # True where element is below max_pfa
                if mask.ndim == 1:
                    slide_df.at[idx, 'final_filter'] = mask
                else:
                    slide_df.at[idx, 'final_filter'] = np.any(mask, axis=1)


        # Process channels; if desired_channels is not provided, process all channels in the DataFrame.
        if desired_channels is None:
            channels_to_process = slide_df['channel'].unique()
        else:
            channels_to_process = desired_channels
        print(slide)
        test = slide_df[
            slide_df['metadata_sample_Probe-Set']
            .str.lower()
            .str.contains(negative_control_field.lower(), na=False)
        ]
        print(np.size(test))
        for channel in channels_to_process:
            channel_df = slide_df[slide_df['channel'] == channel]
            # Filter negative control rows (case-insensitive matching).

            neg_control_df_all = channel_df[
                channel_df['metadata_sample_Probe-Set']
                .str.lower()
                .str.contains(negative_control_field.lower(), na=False)
            ]

            # Determine unique regions if use_region is True
            if use_region and 'metadata_sample_Slice_Region' in neg_control_df_all.columns:
                unique_regions = neg_control_df_all['metadata_sample_Slice_Region'].unique()
            else:
                unique_regions = [None]

            for region in unique_regions:
                if use_region and region is not None:
                    neg_control_df = neg_control_df_all[neg_control_df_all['metadata_sample_Slice_Region'] == region]
                else:
                    neg_control_df = neg_control_df_all


                # Collect arrays for negative control rows.
                photons_list = []
                filter_list = []
                cluster_intensities_list = []
                for idx, row in neg_control_df.iterrows():
                    photons_array = row['spots.params_raw'][:,3]
                    filter_mask = row['final_filter']
                    cluster_int = row['cluster_intensities']
                    photons_list.append(photons_array)
                    filter_list.append(filter_mask)
                    cluster_intensities_list.append(cluster_int)

                # Process photon data.
                if photons_list:
                    concatenated_photons = np.concatenate(photons_list, axis=0)
                    # plt.hist(concatenated_photons, bins=100)
                    # plt.show()
                    filter_list = [np.atleast_1d(np.array(x)) for x in filter_list]
                    non_empty_filters = [f for f in filter_list if f.size > 0]

                    concatenated_filter = np.concatenate(non_empty_filters, axis=0).astype(bool)
                    filtered_photons = concatenated_photons[concatenated_filter]

                    if filtered_photons.size > 0:
                        threshold_value_arr = bootstrap_thresholds(filtered_photons, quantile_negative_control)
                        thresholds[(slide, channel, region)] = np.mean(threshold_value_arr)
                        age[(slide, channel, region)] =neg_control_df['metadata_sample_Age'].values[0]
                        error_thresholds[(slide, channel, region)] = threshold_value_arr
                        number_of_points[(slide, channel, region)] = len(filtered_photons)
                        if plot:
                            sorted_vals = np.sort(filtered_photons)
                            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                            plt.figure(figsize=(6, 4))
                            plt.plot(sorted_vals, cdf, label='CDF')
                            plt.axvline(
                                np.mean(threshold_value_arr),
                                color='red',
                                linestyle='--',
                                label=f'{quantile_negative_control * 100:.1f}th Percentile'
                            )
                            plt.xlabel('Spots Photons')
                            plt.ylabel('Cumulative Distribution')
                            title_str = f"Slide: {slide}" if slide is not None else "All Slides"
                            title_str += f", Channel: {channel}"
                            if region is not None:
                                title_str += f", Region: {region}"
                            plt.title(title_str)
                            plt.legend()
                            plt.grid(True)
                            plt.show()
                    else:
                        print(f"No valid filtered photons for Slide: {slide}, Channel: {channel}, Region: {region}")

                # Process cluster intensity data.
                if cluster_intensities_list:
                    concatenated_cluster = np.concatenate(cluster_intensities_list, axis=0)
                    if concatenated_cluster.size > 0:
                        threshold_value_arr = bootstrap_thresholds(concatenated_cluster, quantile_negative_control)
                        thresholds_cluster[(slide, channel, region)]  =np.mean(threshold_value_arr)
                        error_thresholds_cluster[(slide, channel, region)] =threshold_value_arr
                        if cluster_threshold_val is not None:
                            thresholds_cluster[(slide, channel, region)]=cluster_threshold_val
                        if plot:
                            sorted_vals = np.sort(concatenated_cluster)
                            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                            plt.figure(figsize=(6, 4))
                            plt.plot(sorted_vals, cdf, label='CDF')
                            plt.axvline(
                                np.mean(threshold_value_arr),
                                color='red',
                                linestyle='--',
                                label=f'{quantile_negative_control * 100:.1f}th Percentile'
                            )
                            plt.xlabel('Cluster Intensites [photons]')
                            plt.ylabel('Cumulative Distribution')
                            title_str = f"Slide: {slide}" if slide is not None else "All Slides"
                            title_str += f", Channel: {channel}"
                            if region is not None:
                                title_str += f", Region: {region}"
                            plt.title(title_str)
                            plt.legend()
                            plt.grid(True)
                            plt.show()

                    else:
                        print(f"No valid cluster data for Slide: {slide}, Channel: {channel}, Region: {region}")
                if use_region:
                    print(f"Region used for  Slide: {slide}, Channel: {channel}, Region: {region}")

    return thresholds, thresholds_cluster, error_thresholds, error_thresholds_cluster,number_of_points,age


def concatenate_fields(
        df_extracted,
        slide_field=None,
        desired_channels=None,
        fields_to_extract=None,
        probe_set_field='metadata_sample_Probe-Set'
):
    """
    Concatenate the values of specified fields from the DataFrame, grouping by slide, channel, region,
    and then by each unique probe_set (specified by probe_set_field). If slide_field is None, the entire
    DataFrame is processed as a single group. If desired_channels is None, all channels in the DataFrame will be processed.

    Parameters:
        df_extracted (pd.DataFrame): DataFrame with extracted data.
        slide_field (str or None): Column name for slide identifier. If None, process all rows as one group.
        desired_channels (list or None): List of channels to process. If None, process all unique channels.
        fields_to_extract (list): List of field names (strings) to extract and concatenate.
        probe_set_field (str): Column name that contains the Probe-Set information.

    Returns:
        concatenated_data (dict): Dictionary whose keys are tuples of the form
            (slide, channel, region, probe_set) and whose values are dictionaries mapping each field in
            fields_to_extract to a concatenated NumPy array.
    """
    concatenated_data = {}

    # Determine slides to process.
    if slide_field is not None:
        unique_slides = df_extracted[slide_field].unique()
    else:
        unique_slides = [None]

    for slide in unique_slides:
        if slide is not None:
            slide_df = df_extracted[df_extracted[slide_field] == slide].copy()
        else:
            slide_df = df_extracted.copy()

        # Determine channels to process.
        if desired_channels is None:
            channels_to_process = slide_df['channel'].unique()
        else:
            channels_to_process = desired_channels

        for channel in channels_to_process:
            channel_df = slide_df[slide_df['channel'] == channel]

            # Determine unique regions if available.
            if 'metadata_sample_Slice_Region' in channel_df.columns:
                unique_regions = channel_df['metadata_sample_Slice_Region'].unique()
            else:
                unique_regions = [None]

            for region in unique_regions:
                if region is not None:
                    group_df = channel_df[channel_df['metadata_sample_Slice_Region'] == region]
                else:
                    group_df = channel_df

                # Group by every unique probe set.
                unique_probe_sets = group_df[probe_set_field].unique()
                for probe_set in unique_probe_sets:
                    probe_set_df = group_df[group_df[probe_set_field] == probe_set]

                    # Initialize a dictionary to collect data for each field.
                    group_data = {field: [] for field in fields_to_extract}

                    for idx, row in probe_set_df.iterrows():
                        for field in fields_to_extract:
                            group_data[field].append(row[field])

                    for field in fields_to_extract:
                        # ── gather all non-None values for this field ─────────────────────────
                        values = [v for v in group_data[field] if v is not None]

                        if not values:  # nothing to merge
                            group_data[field] = np.array([])  # or keep as empty list
                            continue

                        # ── CASE 1: every value is array-like → concatenate ───────────────────
                        if all(hasattr(v, "size") for v in values):
                            non_empty_arrays = [v for v in values if v.size > 0]
                            if non_empty_arrays:
                                try:
                                    group_data[field] = np.concatenate(non_empty_arrays, axis=0)
                                except Exception as e:  # mismatched shapes, etc.
                                    print(f"[WARN] concat failed for {field}: {e}")
                                    group_data[field] = non_empty_arrays  # keep list instead
                            else:
                                group_data[field] = np.array([])

                        # ── CASE 2: at least one scalar → treat as metadata/scalar field ──────
                        else:
                            # ► numeric scalars: store a single representative value
                            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                                group_data[field] = float(np.median(values))  # or mean, first, etc.

                            # ► strings or mixed types: keep the list (or pick the first)
                            else:
                                group_data[field] = values[0]  # or `values` to keep them all
                    # Use a 4-tuple as the key: (slide, channel, region, probe_set)
                    concatenated_data[(slide, channel, region, probe_set)] = group_data

    return concatenated_data


def concatenated_data_to_df(concatenated_data):
    """
    Convert the nested dictionary of concatenated data into a DataFrame.

    Parameters:
        concatenated_data (dict): Dictionary with keys as tuples
            (slide, channel, region, probe_set) and values as dictionaries of concatenated fields.

    Returns:
        pd.DataFrame: DataFrame with columns 'slide', 'channel', 'region', 'probe_set',
            and one column for each field from the nested dictionaries.
    """
    rows = []
    for key, data_dict in concatenated_data.items():
        slide, channel, region, probe_set = key
        row = {
            'slide': slide,
            'channel': channel,
            'region': region,
            'probe_set': probe_set
        }
        row.update(data_dict)
        rows.append(row)
    return pd.DataFrame(rows)

def recursively_load_dict(h5_group):
    """Recursively load an HDF5 group into a Python dictionary."""
    output = {}
    for key, item in h5_group.items():
        if isinstance(item, h5py.Dataset):
            output[key] = item[()]
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict(item)
    return output

def get_nested_value(data, key_path):
    """
    Given a dictionary `data` and a dot-separated key_path (e.g., "spots.simga"),
    returns the nested value, or None if any key along the path is missing.
    """
    parts = key_path.split('.')
    for part in parts:
        if isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return None
    return data


def _decode_bytes(obj):
    """
    Recursively convert bytes → str.
    If the decoded result is a list/tuple/set/array of length 1,
    unwrap it to the single element.
    """
    # ── bytes ─────────────────────────────────────────────────────────────
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj            # leave undecoded bytes unchanged

    # ── NumPy arrays ──────────────────────────────────────────────────────
    elif isinstance(obj, np.ndarray):
        decoded = [_decode_bytes(x) for x in obj.tolist()]
        return decoded[0] if len(decoded) == 1 else decoded

    # ── Python containers ─────────────────────────────────────────────────
    elif isinstance(obj, (list, tuple, set)):
        decoded = [_decode_bytes(x) for x in obj]
        if len(decoded) == 1:
            return decoded[0]
        return type(obj)(decoded)   # rebuild same container type

    elif isinstance(obj, dict):
        return {k: _decode_bytes(v) for k, v in obj.items()}

    # ── everything else ──────────────────────────────────────────────────
    else:
        return obj





def compute_binned_fit_and_stats(x, y, weights):
    """
    Compute a weighted linear fit (slope and intercept) on the binned data (x, y)
    using weights (number of observations per bin). Also, compute a weighted R²
    (here used as the "Neyman–Pearson parameter") for how linear the data are.
    """
    valid = weights > 0
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan  # Not enough points for a reliable fit.
    x_valid = x[valid]
    y_valid = y[valid]
    w_valid = weights[valid]
    # Compute weighted linear regression; np.polyfit supports weights.
    slope, intercept = np.polyfit(x_valid, y_valid, 1, w=w_valid)
    y_fit = slope * x_valid + intercept
    weighted_mean = np.average(y_valid, weights=w_valid)
    ss_res = np.sum(w_valid * (y_valid - y_fit) ** 2)
    ss_tot = np.sum(w_valid * (y_valid - weighted_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return slope, intercept, r2

def extract_dataframe(
    data_dict,
    field_keys=None,
    channels=None,
    explode_fields=None,
    include_file_metadata_sample=False,
    file_metadata_sample_key="metadata_sample",
    file_metadata_prefix="metadata_sample_",

):
    """
       Extract a DataFrame from a nested HDF5 data dictionary.
       """
    rows = []

    for file_id, file_data in data_dict.items():
        # ── file-level metadata (decode bytes here only) ───────────────────
        file_metadata = {}
        if include_file_metadata_sample and file_metadata_sample_key in file_data:
            sample_data = file_data[file_metadata_sample_key]
            if isinstance(sample_data, dict):
                for key, value in sample_data.items():
                    file_metadata[file_metadata_prefix + key] = _decode_bytes(value)

        for channel, channel_data in file_data.items():
            if channel in [file_metadata_sample_key, "general_metadata"]:
                continue
            if channels is not None and channel not in channels:
                continue
            if not isinstance(channel_data, dict):
                continue

            row = {"file_id": file_id, "channel": channel}
            if field_keys is not None:
                for key in field_keys:
                    if '.' in key:
                        value = get_nested_value(channel_data, key)
                    else:
                        value = channel_data.get(key, None)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except Exception:
                            pass
                    row[key] = value
            else:
                for key, value in channel_data.items():
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except Exception:
                            pass
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, bytes):
                                try:
                                    subvalue = subvalue.decode('utf-8')
                                except Exception:
                                    pass
                            row[f"{key}_{subkey}"] = subvalue
                    else:
                        row[key] = value

            row.update(file_metadata)
            rows.append(row)
    # ── build and explode DataFrame ────────────────────────────────────
    df = pd.DataFrame(rows)
    # ── CLEAN ALL COLUMN NAMES ─────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()  # remove leading/trailing spaces
        .str.replace(r'\s+', '_', regex=True)  # replace internal spaces with underscores
    )
    if explode_fields:
        for field in explode_fields:
            if field in df.columns:
                df = df.explode(field)

    # ── drop any remaining list‐valued slide names ───────────────────────
    slide_col = f"{file_metadata_prefix}slide_name_std"
    if slide_col in df.columns:
        list_mask = df[slide_col].apply(lambda v: isinstance(v, (list, np.ndarray)))
        if list_mask.any():
            count = list_mask.sum()
            print(f"⚠️ Warning: dropping {count} rows where '{slide_col}' is list-valued")
            #print(df.loc[list_mask, ['file_id', 'channel', slide_col]].to_string(index=False))
            # drop them and reset index
            df = df.loc[~list_mask].reset_index(drop=True)

    # ── ensure slide‐names only clash across different folders ──────────

    # ── only warn when a slide name really lives in >1 folder ─────────────
    slide_col = f"{file_metadata_prefix}slide_name_std"
    if slide_col in df.columns:
        # 1) extract folder prefix
        df["_folder"] = df["file_id"].str.split("--", n=1).str[0]

        # 2) find which slide names span >1 distinct folder
        folder_counts = df.groupby(slide_col)["_folder"].nunique()
        multi = set(folder_counts[folder_counts > 1].index)

        # 3) build a slide→folder→index map, in order of first appearance
        slide_folder_idx = {}
        for slide in multi:
            # `.unique()` preserves the order rows appear in df
            folders = df.loc[df[slide_col] == slide, "_folder"].unique().tolist()
            slide_folder_idx[slide] = {f: i + 1 for i, f in enumerate(folders)}

        # 4) only rename those “multi-folder” slides
        def _rename(row):
            name = row[slide_col]
            if name in slide_folder_idx:
                idx = slide_folder_idx[name][row["_folder"]]
                return name if idx == 1 else f"{name}_{idx}"
            return name

        df[slide_col] = df.apply(_rename, axis=1)

        # 5) clean up helper column
        df.drop(columns=["_folder"], inplace=True)
    return df


def merge_region_names(df, merge_rules):
    """Merge region names according to specified rules"""
    df = df.copy()

    # Create mapping from original names to merged names
    region_mapping = {}
    for merged_name, patterns in merge_rules.items():
        for pattern in patterns:
            region_mapping[pattern] = merged_name

    # Apply mapping to both region columns
    for col in ['metadata_sample_region_std', 'metadata_sample_Slice_Region']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: region_mapping.get(x, x))

    return df


# Define merge rules (same as before)
merge_regions = {
    "Striatum": [
        "Striatum - lower left",
        "Striatum - lower right",
        "Striatum - upper left",
        "Striatum - upper right",
        "Striatum - undefined",
    ],
    "Cortex": [
        "Cortex - Piriform area",
        "Cortex - Primary and secondary motor areas",
        "Cortex - Primary somatosensory (mouth, upper limb)",
        "Cortex - Supplemental/primary somatosensory (nose)",
        "Cortex - Visceral/gustatory/agranular areas",
        "Cortex - undefined",
    ]
}

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def threshold_after_first_peak(data, nbins=200, smoothing_sigma=2.0):
    """
    Compute a cutoff just after the first peak in `data`’s distribution,
    or, if there is no clear valley afterwards (i.e. truly unimodal),
    return the maximum data value so that all points lie 'left' of the threshold.
    """
    # 1) build & smooth histogram
    counts, bin_edges = np.histogram(data, bins=nbins)
    smooth = gaussian_filter1d(counts.astype(float), sigma=smoothing_sigma)

    # 2) find the first (leftmost) peak
    peaks, _ = find_peaks(smooth)
    if peaks.size == 0:
        raise RuntimeError("No peaks found in histogram.")
    first_peak = peaks[0]

    # 3) find valleys (i.e. peaks in –smooth) after that peak
    valleys, _ = find_peaks(-smooth)
    post_valleys = valleys[valleys > first_peak]

    # 4) if we found a valley, use it; otherwise use max(data)
    if post_valleys.size > 0:
        idx = post_valleys[0]
        return bin_edges[idx + 1]
    else:
        # unimodal fallback: threshold at the max so that all data <= thresh
        return np.max(data)

def compute_grouped_spots(
        df_extracted: pd.DataFrame,
        voxels_per_cell: float,
        max_pfa: float = 0.05,
        slide_col: str = "metadata_sample_slide_name_std",
        region_col: str = "metadata_sample_region_std",
        slice_region_col: str = "metadata_sample_Slice_Region",
        channel_col: str = "channel",
        probe_set_col: str = "metadata_sample_Probe-Set",
        mouse_model_col: str = "metadata_sample_Mouse_Model",
        age_col: str = "metadata_sample_Age",
        label_sizes_col: str = "label_sizes",
        pfa_col: str = "spots.pfa_values",
        photons_col: str = "spots.photons",
        cluster_col: str = "cluster_intensities",
        blue_channel: str = "blue",
        merge_regions: dict = None,  # New parameter for region merging
        use_sigma_var_params: bool = False,  # Use spots_sigma_var.params_raw instead of spots.photons
        use_final_filter: bool = False  # Use spots.final_filter to filter photons
) -> (pd.DataFrame, dict):
    """
    Compute group-level spot counts and normalized spots per cell,
    using blue-channel area for normalization.

    Args:
        df_extracted: DataFrame with extracted data
        voxels_per_cell: Volume of single cell in voxel units
        max_pfa: Maximum PFA value for filtering (only used if use_final_filter=False)
        use_sigma_var_params: If True, use spots_sigma_var.params_raw[:,3] for photons
                             instead of spots.photons (default: False)
        use_final_filter: If True, use spots.final_filter to filter photons before
                         counting (default: False). When True with use_sigma_var_params,
                         skips PFA filtering as it's already applied.
        merge_regions: Dictionary of {new_region_name: [pattern1, pattern2]}
                      to merge regions containing these patterns

    Returns:
      - agg: DataFrame with one row per group, containing:
          * n_spots_group
          * norm_spots_group
          * n_spots_cluster
      - blue_area_lookup: dict mapping (slide, region) -> area
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df_extracted.copy()

    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    # 0) Apply region merging if specified
    if merge_regions is not None:
        def merge_region_name(original_name):
            if not isinstance(original_name, str):
                return original_name
            for new_name, patterns in merge_regions.items():
                if any(pattern.lower() in original_name.lower() for pattern in patterns):
                    return new_name
            return original_name

        df[slice_region_col] = df[slice_region_col].apply(merge_region_name)
        df[region_col] = df[region_col].apply(merge_region_name)

    # Rest of the function remains the same...
    # 1) Compute blue-channel total cell area per (slide, region)
    blue_df = df[df[channel_col] == blue_channel]
    blue_area = (
                    blue_df
                    .groupby([slide_col, region_col])[label_sizes_col]
                    .apply(lambda lst: sum(np.sum(arr) for arr in lst))
                ) / voxels_per_cell
    blue_area_lookup = blue_area.to_dict()

    # 2) Define grouping keys
    group_keys = [
        slide_col,
        region_col,
        slice_region_col,
        channel_col,
        probe_set_col,
        mouse_model_col,
        age_col,
        'metadata_sample_Level',
        'metadata_sample_mouse_ID',
        'metadata_sample_Brain_Atlas_coordinates',
    ]

    # 3) Aggregation function
    def _agg(group):
        # Extract photons based on configuration
        if use_sigma_var_params and 'spots_sigma_var.params_raw' in group.columns:
            # Use spots_sigma_var.params_raw with spots.final_filter
            photons_list = []

            for idx, row in group.iterrows():
                sigma_var_params = row.get('spots_sigma_var.params_raw', None)

                # Skip if no data
                if sigma_var_params is None:
                    continue

                # Convert to numpy array if needed
                if not isinstance(sigma_var_params, np.ndarray):
                    try:
                        sigma_var_params = np.asarray(sigma_var_params)
                    except:
                        continue

                # Check shape
                if sigma_var_params.ndim < 2 or sigma_var_params.shape[1] <= 3:
                    continue

                if use_final_filter and 'spots.final_filter' in group.columns:
                    final_filter = row.get('spots.final_filter', None)
                    if final_filter is not None:
                        try:
                            final_filter = np.asarray(final_filter).astype(bool)
                            if final_filter.sum() > 0:  # If any spots pass filter
                                photons_filtered = sigma_var_params[final_filter, 3]
                                if photons_filtered.size > 0:
                                    photons_list.append(photons_filtered)
                        except (IndexError, TypeError, ValueError) as e:
                            # Fallback: use all photons
                            photons_list.append(sigma_var_params[:, 3])
                else:
                    # Use all photons without filtering
                    photons_list.append(sigma_var_params[:, 3])

            photons = np.concatenate(photons_list) if photons_list else np.array([])
        else:
            # Original behavior: use spots.photons directly
            photons = np.concatenate([np.asarray(x) for x in group[photons_col]])

        # cluster counts
        try:
            clusters = np.concatenate([np.asarray(x) for x in group[cluster_col]])
        except Exception:
            clusters = np.array([])
        try:
            darp_signal = np.concatenate([np.asarray(x) for x in group['darp_signal']])
            darp_th =  90 #threshold_after_first_peak(darp_signal, smoothing_sigma=5)
            fraction_darp_positive = len(darp_signal[darp_signal > darp_th]) / len(darp_signal)
            # counts=plt.hist(darp_signal,bins=50, density=True)
            # plt.vlines(darp_th,0,max(counts[0]))
            # plt.title(group['metadata_sample_Slice_Region'].values[0])
            # plt.show()
            kaka = 0
        except Exception:
            darp_signal = np.array([])
            darp_th =0
            fraction_darp_positive=0




        # apply PFA filter (only if not already using final_filter)
        if use_final_filter and use_sigma_var_params:
            # Photons already filtered by spots.final_filter, only apply threshold
            keep_pfa = np.ones(len(photons), dtype=bool)
        else:
            # Apply PFA filter as before
            pfa_list = [sub for lst in group[pfa_col] for sub in lst]
            if pfa_list:
                pfa_arr = np.vstack(pfa_list)
                keep_pfa = np.all(pfa_arr <= max_pfa, axis=1)
            else:
                keep_pfa = np.array([], dtype=bool)

        # apply photon threshold
        thr = group["threshold"].iloc[0]
        id = group['file_id'].iloc[0]
        keep_phot = photons > thr
        mask = keep_phot & keep_pfa


        n_spots = np.count_nonzero(mask)
        # clusters-to-spots conversion
        if clusters.size > 0 and mask.sum() > 0:
            sum_clusters = np.sum(clusters)
            n_clusters = len(clusters)
            num_spots_cluster = sum_clusters / np.mean(photons[mask])
        else:
            num_spots_cluster = 0.0
            n_clusters = 0
        # normalization
        slide, region = group.name[0], group.name[1]
        total_area = blue_area_lookup.get((slide, region), np.nan)
        norm = (n_spots + num_spots_cluster) / total_area if total_area > 0 else np.nan
        n_clusters_per_cell = n_clusters / total_area if total_area > 0 else np.nan
        n_spots_per_cell = n_spots/ total_area if total_area > 0 else np.nan
        return pd.Series({
            "n_spots_group": n_spots,
            "norm_spots_group": norm,
            "n_spots_cluster": n_spots + num_spots_cluster,
            "n_clusters_per_cell":n_clusters_per_cell,
            "n_spots_per_cell": n_spots_per_cell,
            "fraction_darp_positive": fraction_darp_positive,
            "darp_th": darp_th,
            'id':id,

        })

    # 4) Build aggregated DataFrame
    agg = (
        df
        .groupby(group_keys, as_index=False)
        .apply(_agg)
        .reset_index(drop=True)
    )

    return agg, blue_area_lookup