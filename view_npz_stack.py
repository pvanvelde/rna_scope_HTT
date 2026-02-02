#!/usr/bin/env python3
"""
Simple napari viewer for NPZ stacks created by utils_data_prep.py

Usage:
    # View a single file
    python view_npz_stack.py /path/to/your/file.npz

    # View a folder (concatenates all NPZ files)
    python view_npz_stack.py /path/to/folder/

    # Use max projection along z-axis
    python view_npz_stack.py /path/to/folder/ --max-project

    # View multiple files
    python view_npz_stack.py file1.npz file2.npz file3.npz

Or from Python:
    from view_npz_stack import view_npz, view_folder_concatenated
    view_npz('/path/to/your/file.npz')
    view_folder_concatenated('/path/to/folder/', max_project=True)
"""

import sys
import argparse
import os
import numpy as np
import blosc
import napari
from pathlib import Path
from typing import List, Tuple
import glob

# No external dependencies needed for scale information


def load_npz_stack(npz_path):
    """
    Load and decompress NPZ file created by utils_data_prep.py

    Args:
        npz_path: Path to NPZ file

    Returns:
        tuple: (array with shape (channels, z, y, x), metadata dict)
    """
    with np.load(npz_path, allow_pickle=True) as data:
        compressed_data = data['compressed_data']
        metadata = data['metadata'].item()

    # Decompress using blosc
    decompressed = blosc.decompress(compressed_data)

    # Reconstruct array
    dtype = np.dtype(metadata['dtype'])
    shape = tuple(metadata['shape'])  # (channels, z, y, x)
    arr = np.frombuffer(decompressed, dtype=dtype).reshape(shape)

    return arr, metadata


def view_npz(npz_path, channel_colors=None, max_project=False, pixel_size_nm=162.5):
    """
    Open NPZ file in napari as a multi-channel z-stack.

    Args:
        npz_path: Path to NPZ file
        channel_colors: Optional list of colormap names for each channel.
                       Default uses different colors for each channel.
        max_project: If True, show max projection along z-axis instead of full stack
        pixel_size_nm: Pixel size in nanometers (default: 162.5 nm)
    """
    print(f"Loading: {npz_path}")
    arr, metadata = load_npz_stack(npz_path)

    n_channels, n_z, height, width = arr.shape
    channel_names = metadata.get('channels', [f'Channel_{i}' for i in range(n_channels)])

    print(f"Shape: {arr.shape} (channels={n_channels}, z={n_z}, y={height}, x={width})")
    print(f"Dtype: {arr.dtype}")
    print(f"Channels: {channel_names}")
    print(f"Pixel size: {pixel_size_nm} nm")

    # Apply max projection if requested
    if max_project:
        print("Applying max projection along z-axis...")
        arr = arr.max(axis=1)  # Max over z-axis: (c, z, y, x) -> (c, y, x)
        print(f"Projected shape: {arr.shape} (channels={n_channels}, y={height}, x={width})")

    # Default colormaps for common channel types (blue, green, orange, red)
    default_colormaps = ['blue', 'green', 'yellow', 'red', 'magenta', 'cyan']
    if channel_colors is None:
        channel_colors = default_colormaps[:n_channels]

    # Calculate scale: convert nm to µm for napari
    pixel_size_um = pixel_size_nm / 1000.0  # Convert nm to µm
    if max_project:
        scale = (pixel_size_um, pixel_size_um)  # (y, x) in µm
    else:
        scale = (pixel_size_um, pixel_size_um, pixel_size_um)  # (z, y, x) in µm

    # Create napari viewer
    title = f"NPZ Stack: {os.path.basename(npz_path)}"
    if max_project:
        title += " (Max Projection)"
    viewer = napari.Viewer(title=title)

    # Add each channel as a separate layer
    for i, (channel_name, colormap) in enumerate(zip(channel_names, channel_colors)):
        layer = viewer.add_image(
            arr[i],  # (z, y, x) or (y, x) for this channel
            name=channel_name,
            colormap=colormap,
            blending='additive',
            scale=scale,
            contrast_limits=[arr[i].min(), arr[i].max()]
        )
        # Set contrast limits to auto-adjust
        layer.reset_contrast_limits()

    print("\nNapari viewer opened. Close the window to exit.")
    napari.run()


def get_npz_files_from_folder(folder_path: str) -> List[str]:
    """
    Get all NPZ files from a folder, sorted by name.

    Args:
        folder_path: Path to folder

    Returns:
        List of NPZ file paths
    """
    npz_files = sorted(glob.glob(os.path.join(folder_path, '*.npz')))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {folder_path}")
    return npz_files


def view_folder_concatenated(folder_path: str, max_project=False, axis='fov', pixel_size_nm=162.5):
    """
    Load all NPZ files from a folder and concatenate them for viewing.

    Args:
        folder_path: Path to folder containing NPZ files
        max_project: If True, apply max projection along z-axis
        axis: Concatenation axis - 'fov' concatenates along a new FOV axis,
              'z' stacks all z-slices together
        pixel_size_nm: Pixel size in nanometers (default: 162.5 nm)
    """
    npz_files = get_npz_files_from_folder(folder_path)
    print(f"Found {len(npz_files)} NPZ files in {folder_path}")

    # Load all files
    arrays = []
    metadatas = []
    for npz_file in npz_files:
        print(f"Loading: {os.path.basename(npz_file)}")
        arr, metadata = load_npz_stack(npz_file)
        arrays.append(arr)
        metadatas.append(metadata)

    # Get channel names from first file
    channel_names = metadatas[0].get('channels', [f'Channel_{i}' for i in range(arrays[0].shape[0])])
    n_channels = arrays[0].shape[0]

    # Check that all files have the same shape (except z dimension can vary)
    shapes = [arr.shape for arr in arrays]
    if not all(s[0] == n_channels for s in shapes):
        raise ValueError(f"All files must have the same number of channels. Found: {shapes}")

    # Calculate scale: convert nm to µm for napari
    pixel_size_um = pixel_size_nm / 1000.0  # Convert nm to µm

    if axis == 'fov':
        # Concatenate along a new axis: (fov, channels, z, y, x)
        print("Concatenating along FOV axis...")
        concatenated = np.stack(arrays, axis=0)
        print(f"Concatenated shape: {concatenated.shape} (fov, channels, z, y, x)")

        # Apply max projection if requested
        if max_project:
            print("Applying max projection along z-axis...")
            concatenated = concatenated.max(axis=2)  # Max over z: (fov, c, z, y, x) -> (fov, c, y, x)
            print(f"Projected shape: {concatenated.shape} (fov, channels, y, x)")
            scale = (1, pixel_size_um, pixel_size_um)  # (fov, y, x) - FOV has no physical scale
        else:
            scale = (1, pixel_size_um, pixel_size_um, pixel_size_um)  # (fov, z, y, x)

        # Create viewer
        title = f"Folder: {os.path.basename(folder_path)} ({len(npz_files)} FOVs)"
        if max_project:
            title += " (Max Projection)"
        viewer = napari.Viewer(title=title)

        # Default colormaps (blue, green, orange, red)
        default_colormaps = ['blue', 'green', 'yellow', 'red', 'magenta', 'cyan']

        # Add each channel
        for i, channel_name in enumerate(channel_names):
            colormap = default_colormaps[i % len(default_colormaps)]
            layer = viewer.add_image(
                concatenated[:, i],  # (fov, z, y, x) or (fov, y, x)
                name=channel_name,
                colormap=colormap,
                blending='additive',
                scale=scale,
                contrast_limits=[concatenated[:, i].min(), concatenated[:, i].max()]
            )
            # Set contrast limits to auto-adjust
            layer.reset_contrast_limits()

    elif axis == 'z':
        # Concatenate along z-axis: (channels, z_total, y, x)
        print("Concatenating along z-axis...")
        # Need to concatenate along z for each channel
        concatenated_channels = []
        for ch_idx in range(n_channels):
            ch_stacks = [arr[ch_idx] for arr in arrays]  # List of (z, y, x) arrays
            concatenated_ch = np.concatenate(ch_stacks, axis=0)  # (z_total, y, x)
            concatenated_channels.append(concatenated_ch)
        concatenated = np.stack(concatenated_channels, axis=0)  # (c, z_total, y, x)
        print(f"Concatenated shape: {concatenated.shape} (channels, z_total, y, x)")

        # Apply max projection if requested
        if max_project:
            print("Applying max projection along z-axis...")
            concatenated = concatenated.max(axis=1)  # (c, z, y, x) -> (c, y, x)
            print(f"Projected shape: {concatenated.shape} (channels, y, x)")
            scale = (pixel_size_um, pixel_size_um)  # (y, x)
        else:
            scale = (pixel_size_um, pixel_size_um, pixel_size_um)  # (z, y, x)

        # Create viewer
        title = f"Folder: {os.path.basename(folder_path)} (Z-concatenated)"
        if max_project:
            title += " (Max Projection)"
        viewer = napari.Viewer(title=title)

        # Default colormaps (blue, green, orange, red)
        default_colormaps = ['blue', 'green', 'yellow', 'red', 'magenta', 'cyan']

        # Add each channel
        for i, channel_name in enumerate(channel_names):
            colormap = default_colormaps[i % len(default_colormaps)]
            layer = viewer.add_image(
                concatenated[i],  # (z, y, x) or (y, x)
                name=channel_name,
                colormap=colormap,
                blending='additive',
                scale=scale,
                contrast_limits=[concatenated[i].min(), concatenated[i].max()]
            )
            # Set contrast limits to auto-adjust
            layer.reset_contrast_limits()

    print("\nNapari viewer opened. Close the window to exit.")
    napari.run()


def view_multiple_npz(npz_paths, max_project=False):
    """
    Open multiple NPZ files in napari, each as a separate set of layers.

    Args:
        npz_paths: List of paths to NPZ files
        max_project: If True, apply max projection along z-axis
    """
    viewer = napari.Viewer(title="NPZ Stack Viewer")

    default_colormaps = ['blue', 'green', 'yellow', 'red', 'magenta', 'cyan']

    for npz_path in npz_paths:
        print(f"\nLoading: {npz_path}")
        arr, metadata = load_npz_stack(npz_path)

        n_channels = arr.shape[0]
        channel_names = metadata.get('channels', [f'Channel_{i}' for i in range(n_channels)])
        base_name = Path(npz_path).stem

        print(f"  Shape: {arr.shape}")
        print(f"  Channels: {channel_names}")

        # Apply max projection if requested
        if max_project:
            print(f"  Applying max projection...")
            arr = arr.max(axis=1)  # (c, z, y, x) -> (c, y, x)

        # Add each channel
        for i, channel_name in enumerate(channel_names):
            layer_name = f"{base_name}_{channel_name}"
            colormap = default_colormaps[i % len(default_colormaps)]

            layer = viewer.add_image(
                arr[i],
                name=layer_name,
                colormap=colormap,
                blending='additive',
                visible=(npz_path == npz_paths[0]),  # Only show first file by default
                contrast_limits=[arr[i].min(), arr[i].max()]
            )
            # Set contrast limits to auto-adjust
            layer.reset_contrast_limits()

    print("\nNapari viewer opened. Close the window to exit.")
    napari.run()


def view_multiple_folders_concatenated(folder_paths: List[str], max_project=False, axis='fov'):
    """
    Load all NPZ files from multiple folders and concatenate them for viewing.

    Args:
        folder_paths: List of paths to folders containing NPZ files
        max_project: If True, apply max projection along z-axis
        axis: Concatenation axis - 'fov' concatenates along a new FOV axis,
              'z' stacks all z-slices together
    """
    # Collect all NPZ files from all folders
    all_npz_files = []
    for folder_path in folder_paths:
        npz_files = get_npz_files_from_folder(folder_path)
        print(f"Found {len(npz_files)} NPZ files in {folder_path}")
        all_npz_files.extend(npz_files)

    print(f"\nTotal: {len(all_npz_files)} NPZ files from {len(folder_paths)} folders")

    # Load all files
    arrays = []
    metadatas = []
    for npz_file in all_npz_files:
        print(f"Loading: {os.path.basename(npz_file)}")
        arr, metadata = load_npz_stack(npz_file)
        arrays.append(arr)
        metadatas.append(metadata)

    # Get channel names from first file
    channel_names = metadatas[0].get('channels', [f'Channel_{i}' for i in range(arrays[0].shape[0])])
    n_channels = arrays[0].shape[0]

    # Check that all files have the same shape (except z dimension can vary)
    shapes = [arr.shape for arr in arrays]
    if not all(s[0] == n_channels for s in shapes):
        raise ValueError(f"All files must have the same number of channels. Found: {shapes}")

    if axis == 'fov':
        # Concatenate along a new axis: (fov, channels, z, y, x)
        print("Concatenating along FOV axis...")
        concatenated = np.stack(arrays, axis=0)
        print(f"Concatenated shape: {concatenated.shape} (fov, channels, z, y, x)")

        # Apply max projection if requested
        if max_project:
            print("Applying max projection along z-axis...")
            concatenated = concatenated.max(axis=2)  # Max over z: (fov, c, z, y, x) -> (fov, c, y, x)
            print(f"Projected shape: {concatenated.shape} (fov, channels, y, x)")

        # Create viewer
        title = f"Multiple Folders ({len(folder_paths)} folders, {len(all_npz_files)} FOVs)"
        if max_project:
            title += " (Max Projection)"
        viewer = napari.Viewer(title=title)

        # Default colormaps (blue, green, orange, red)
        default_colormaps = ['blue', 'green', 'yellow', 'red', 'magenta', 'cyan']

        # Add each channel
        for i, channel_name in enumerate(channel_names):
            colormap = default_colormaps[i % len(default_colormaps)]
            layer = viewer.add_image(
                concatenated[:, i],  # (fov, z, y, x) or (fov, y, x)
                name=channel_name,
                colormap=colormap,
                blending='additive',
                contrast_limits=[concatenated[:, i].min(), concatenated[:, i].max()]
            )
            # Set contrast limits to auto-adjust
            layer.reset_contrast_limits()

    elif axis == 'z':
        # Concatenate along z-axis: (channels, z_total, y, x)
        print("Concatenating along z-axis...")
        # Need to concatenate along z for each channel
        concatenated_channels = []
        for ch_idx in range(n_channels):
            ch_stacks = [arr[ch_idx] for arr in arrays]  # List of (z, y, x) arrays
            concatenated_ch = np.concatenate(ch_stacks, axis=0)  # (z_total, y, x)
            concatenated_channels.append(concatenated_ch)
        concatenated = np.stack(concatenated_channels, axis=0)  # (c, z_total, y, x)
        print(f"Concatenated shape: {concatenated.shape} (channels, z_total, y, x)")

        # Apply max projection if requested
        if max_project:
            print("Applying max projection along z-axis...")
            concatenated = concatenated.max(axis=1)  # (c, z, y, x) -> (c, y, x)
            print(f"Projected shape: {concatenated.shape} (channels, y, x)")

        # Create viewer
        title = f"Multiple Folders (Z-concatenated, {len(folder_paths)} folders)"
        if max_project:
            title += " (Max Projection)"
        viewer = napari.Viewer(title=title)

        # Default colormaps (blue, green, orange, red)
        default_colormaps = ['blue', 'green', 'yellow', 'red', 'magenta', 'cyan']

        # Add each channel
        for i, channel_name in enumerate(channel_names):
            colormap = default_colormaps[i % len(default_colormaps)]
            layer = viewer.add_image(
                concatenated[i],  # (z, y, x) or (y, x)
                name=channel_name,
                colormap=colormap,
                blending='additive',
                contrast_limits=[concatenated[i].min(), concatenated[i].max()]
            )
            # Set contrast limits to auto-adjust
            layer.reset_contrast_limits()

    print("\nNapari viewer opened. Close the window to exit.")
    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="View NPZ stacks (from utils_data_prep.py) in napari",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a single NPZ file
  python view_npz_stack.py data.npz

  # View a folder (concatenates all NPZ files along FOV axis)
  python view_npz_stack.py /path/to/folder/

  # View multiple folders (concatenates all NPZ files from all folders)
  python view_npz_stack.py /path/to/folder1/ /path/to/folder2/ /path/to/folder3/

  # View folder with max projection along z-axis
  python view_npz_stack.py /path/to/folder/ --max-project

  # View folder concatenated along z-axis (stacks all z-slices)
  python view_npz_stack.py /path/to/folder/ --concat-axis z

  # View multiple specific files
  python view_npz_stack.py file1.npz file2.npz file3.npz

  # View multiple files with max projection
  python view_npz_stack.py file1.npz file2.npz --max-project
        """
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Path(s) to NPZ file(s) or folder(s) to open'
    )
    parser.add_argument(
        '--max-project', '-m',
        action='store_true',
        help='Apply max projection along z-axis'
    )
    parser.add_argument(
        '--concat-axis',
        choices=['fov', 'z'],
        default='fov',
        help='Concatenation axis for folder mode (default: fov)'
    )

    args = parser.parse_args()

    # Separate paths into files and directories
    files = [p for p in args.paths if os.path.isfile(p)]
    dirs = [p for p in args.paths if os.path.isdir(p)]

    # Check that all paths exist
    for path in args.paths:
        if not os.path.exists(path):
            print(f"Error: Path not found: {path}")
            sys.exit(1)

    # Determine mode
    if len(dirs) > 0 and len(files) > 0:
        print("Error: Cannot mix files and directories. Provide either files or directories, not both.")
        sys.exit(1)

    if len(dirs) > 0:
        # Directory mode: concatenate all NPZ files from all directories
        if len(dirs) == 1:
            view_folder_concatenated(
                dirs[0],
                max_project=args.max_project,
                axis=args.concat_axis
            )
        else:
            view_multiple_folders_concatenated(
                dirs,
                max_project=args.max_project,
                axis=args.concat_axis
            )
    else:
        # File mode
        if len(files) == 1:
            view_npz(files[0], max_project=args.max_project)
        else:
            view_multiple_npz(files, max_project=args.max_project)


if __name__ == '__main__':
    main()
