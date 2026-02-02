"""
Utilities & the ImageLoader class for the data-prep pipeline.
"""

from __future__ import annotations
import os
import re
import glob
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import imagecodecs
import blosc
import h5py
from tqdm import tqdm
import napari

# -----------------------------
# Small viewer helper
# -----------------------------
def show_napari(raw_image: np.ndarray, title: str = "Image"):
    """
    Display a single image in Napari (grayscale).
    """
    with napari.gui_qt():
        viewer = napari.Viewer(title=title)
        viewer.add_image(raw_image, name="Raw Image", colormap='gray', blending='additive')


# -----------------------------
# Regex builders (two flavors)
# -----------------------------
def build_filename_regex(version: str = 'v2') -> re.Pattern:
    """
    Build a compiled regex used to parse filenames.
    version:
        'v1' → take slide name from the filename
        'v2' → take slide name from the grandparent folder; filename still parsed for region/FOV/channel/z
    """
    if version == 'v1':
        pattern = r"""
            Slide\s(?P<slide>[\w\s-]+)      # Slide identifier
            -Region\s(?P<region>\d+)        # Region number
            -FOV\s(?P<fov>\d+)              # FOV number
            -(?P<channel>[\w]+)_C           # Channel
            _(?P<z_sign>[+-])               # Sign of Z-plane index
            (?P<z_index>\d+)                # Z-plane index number
            _[+-]\d+\.\d+                   # Additional number (ignored)
            \.tif$                          # File extension
        """
    else:  # 'v2'
        pattern = r"""
            ^(?P<slide_name_from_file>[\w\s-]+)   # ignored in v2 (we replace with folder name)
            -Region\s(?P<region>\d+)              # Region number
            -FOV\s(?P<fov>\d+)                    # FOV number
            -(?P<channel>[\w]+)_C                 # Channel
            _(?P<z_sign>[+-])                     # Sign of Z-plane index
            (?P<z_index>\d+)                      # Z-plane index number
            _[+-]\d+\.\d+                         # Additional number (ignored)
            \.tif$                                # File extension
        """
    return re.compile(pattern, re.VERBOSE)


# -----------------------------
# ImageLoader
# -----------------------------
class ImageLoader:
    """
    Collects per-FOV Z-stacks per channel and saves compressed FOV tensors in a single file:
      shape: (channels, z, y, x)
    """

    def __init__(
        self,
        base_dir: str,
        channels: List[str],
        compressed_data_path: str | None = None,
        compression_method: str | None = None,
        parser_version: str = 'v2',
        handle_tiff_jpeg_xr: bool = True,
    ):
        self.base_dir = base_dir
        self.channels = channels
        self.image_filenames: List[str] = []
        self.parsed_filenames: List[Dict] = []
        self.zstack_groups: Dict[Tuple, List[Dict]] = {}
        self.compressed_data_path = compressed_data_path
        self.compression_method = compression_method
        self.parser_version = parser_version
        self._compiled_pattern = build_filename_regex(parser_version)
        self.handle_tiff_jpeg_xr = handle_tiff_jpeg_xr

    # ---------- discovery ----------
    def collect_image_filenames(self):
        """
        Recursively collect .tif files under base_dir.
        """
        self.image_filenames = glob.glob(os.path.join(self.base_dir, '**', '*.tif'), recursive=True)
        print(f"Collected {len(self.image_filenames)} .tif files under {self.base_dir}")

    # ---------- parsing ----------
    def parse_filenames(self):
        """
        Parse file names. For 'v2', slide name is taken from the grandparent folder.
        Keeps only files that match the required pattern (with Z-plane info).
        """
        parsed = []
        pat = self._compiled_pattern

        for filepath in self.image_filenames:
            filename = os.path.basename(filepath)
            m = pat.match(filename)
            if not m:
                # Silent skip to reduce noise; uncomment for debugging:
                # print(f"No match for filename: {filename}")
                continue

            # Default: use the name found in filename (v1).
            slide = m.groupdict().get('slide', None)

            if self.parser_version == 'v2':
                # In v2, we override slide with the grandparent dir
                # .../Slide_XYZ/Images/Slide_XYZ/Region.../file.tif
                grandparent = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
                slide = grandparent.strip()

            region = int(m.group('region'))
            fov = int(m.group('fov'))
            channel = m.group('channel').strip()
            z_sign = m.group('z_sign')
            z_index = int(m.group('z_index'))
            z_plane = int(f"{z_sign}{z_index}")

            parsed.append({
                'filename': filepath,
                'slide': slide,
                'region': region,
                'fov': fov,
                'channel': channel,
                'z_plane': z_plane,
            })

        self.parsed_filenames = parsed
        print(f"Parsed {len(self.parsed_filenames)} filenames with Z-plane info")

    # ---------- grouping (optional) ----------
    def group_files_for_zstack(self):
        """
        Group by (slide, region, fov, channel) and sort by z-plane.
        """
        groups = defaultdict(list)
        for d in self.parsed_filenames:
            key = (d['slide'], d['region'], d['fov'], d['channel'])
            groups[key].append(d)
        for key in groups:
            groups[key].sort(key=lambda x: x['z_plane'])
        self.zstack_groups = groups
        print(f"Created {len(self.zstack_groups)} groups for Z-stacks")

    # ---------- IO ----------
    def load_image(self, file_path: str) -> np.ndarray | None:
        """
        Load TIFF (including a JPEG XR 16-bit 2nd page variant) or NPZ (LZ4 payload).
        Returns a numpy array or None.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ('.tif', '.tiff'):
            with tifffile.TiffFile(file_path) as tif:
                pages = tif.pages
                if len(pages) == 1:
                    return tif.asarray()

                # Handle 16-bit compressed data on page 1 when present
                if self.handle_tiff_jpeg_xr:
                    second = pages[1]
                    with open(file_path, 'rb') as f:
                        chunks = []
                        for offset, nbytes in zip(second.dataoffsets, second.databytecounts):
                            f.seek(offset)
                            chunks.append(f.read(nbytes))
                        compressed_data = b''.join(chunks)
                    if not hasattr(imagecodecs, 'jpegxr_decode'):
                        raise RuntimeError("imagecodecs has no jpegxr_decode; install a build with JPEG XR support.")
                    decompressed = imagecodecs.jpegxr_decode(compressed_data)
                    h, w = second.imagelength, second.imagewidth
                    return np.frombuffer(decompressed, dtype=np.uint16).reshape((h, w))

                # Fallback: read as-is (multi-page)
                return tif.asarray()

        if ext == '.npz':
            with np.load(file_path, allow_pickle=True) as data:
                compressed_data = data['compressed_data']
                meta = data['metadata'].item()
            arr = np.frombuffer(blosc.decompress(compressed_data), dtype=np.dtype(meta['dtype']))
            return arr.reshape(tuple(meta['shape']))

        print(f"Unknown file extension for load_image: {ext}")
        return None

    # ---------- writing ----------
    def save_fov_compressed_images(self, compression_method: str | None = None):
        """
        Build per-FOV arrays with shape (channels, z, y, x) and save them with the chosen compression.
        """
        if compression_method is None:
            compression_method = self.compression_method or 'lz4'

        if not self.compressed_data_path:
            raise ValueError("compressed_data_path is not set.")

        os.makedirs(self.compressed_data_path, exist_ok=True)

        # unique (slide, region, fov)
        keys = []
        seen = set()
        for d in self.parsed_filenames:
            key = (d['slide'], d['region'], d['fov'])
            if key not in seen:
                seen.add(key)
                keys.append(key)

        for slide, region, fov in tqdm(keys, desc="Processing FOVs"):
            try:
                channel_arrays = []
                for ch in self.channels:
                    # collect entries for this (slide, region, fov, ch)
                    entries = [e for e in self.parsed_filenames
                               if e['slide'] == slide and e['region'] == region and e['fov'] == fov and e['channel'] == ch]
                    if not entries:
                        print(f"No data for channel {ch} in Slide:{slide} Region:{region} FOV:{fov}")
                        break

                    entries.sort(key=lambda x: x['z_plane'])
                    zimgs = [self.load_image(e['filename']) for e in entries]
                    if any(img is None for img in zimgs):
                        raise RuntimeError("One or more images failed to load.")
                    zstack = np.stack(zimgs, axis=0)  # (z, y, x)
                    channel_arrays.append(zstack)

                else:
                    # all channels present
                    fov_array = np.stack(channel_arrays, axis=0)  # (c, z, y, x)
                    base = f"{slide}_Region_{region}_FOV_{fov}"
                    out = os.path.join(self.compressed_data_path, base)

                    if compression_method == 'raw':
                        tifffile.imwrite(out + '.tif', fov_array)
                    elif compression_method == 'lz4':
                        payload = blosc.compress(fov_array.tobytes(), typesize=2, cname='lz4')
                        meta = {'shape': fov_array.shape, 'dtype': str(fov_array.dtype), 'channels': self.channels}
                        np.savez_compressed(out + '.npz', compressed_data=payload, metadata=meta)
                    elif compression_method == 'zstd':
                        # zstd via tifffile (note: this is per-page; tifffile handles the layout)
                        tifffile.imwrite(out + '.tif', fov_array, compression='zstd')
                    elif compression_method == 'gzip':
                        with h5py.File(out + '.h5', 'w') as f:
                            f.create_dataset('image', data=fov_array, compression='gzip')
                    else:
                        raise ValueError(f"Unsupported compression method: {compression_method}")

                    print(f"Saved: {out}")

            except Exception as e:
                print(f"Error processing Slide:{slide} Region:{region} FOV:{fov}")
                print(str(e))

    # ---------- convenience ----------
    def load_random_zstack(self) -> np.ndarray | None:
        """
        Load a random saved .tif/.npz/.h5 from compressed_data_path and return as np.ndarray.
        """
        files = [f for f in os.listdir(self.compressed_data_path)
                 if os.path.isfile(os.path.join(self.compressed_data_path, f))]
        if not files:
            print("No compressed files found.")
            return None
        choice = random.choice(files)
        path = os.path.join(self.compressed_data_path, choice)
        print(f"Loading random z-stack from {choice}")
        return self.load_image(path)
