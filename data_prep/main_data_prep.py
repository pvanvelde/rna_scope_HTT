# main_data_prep.py
import os
from config_data_prep import ROOT_DIRS, OUTPUT_DIRS, CHANNELS, DEFAULT_COMPRESSION, PARSER_VERSION, HANDLE_TIFF_JPEG_XR
from utils_data_prep import ImageLoader

if __name__ == "__main__":
    for root_dir, output_root in zip(ROOT_DIRS, OUTPUT_DIRS):
        basedirs, compressed_folders = [], []

        # find slide dirs
        slide_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('Slide')
        ]

        for slide_dir in slide_dirs:
            slide_name = os.path.basename(slide_dir)
            images_dir = os.path.join(slide_dir, 'Images', slide_name)
            if not os.path.exists(images_dir):
                print(f"Images directory not found for slide {slide_name}: {images_dir}")
                continue

            region_dirs = [
                os.path.join(images_dir, d)
                for d in os.listdir(images_dir)
                if os.path.isdir(os.path.join(images_dir, d)) and d.startswith('Region')
            ]

            for region_dir in region_dirs:
                region_name = os.path.basename(region_dir)
                basedirs.append(region_dir)
                compressed_folders.append(os.path.join(output_root, slide_name, region_name))

        assert len(basedirs) == len(compressed_folders), "base vs output length mismatch"

        for base_dir, compressed_data_root in zip(basedirs, compressed_folders):
            os.makedirs(compressed_data_root, exist_ok=True)

            loader = ImageLoader(
                base_dir=base_dir,
                channels=CHANNELS,
                compressed_data_path=compressed_data_root,
                compression_method=DEFAULT_COMPRESSION,
                parser_version=PARSER_VERSION,
                handle_tiff_jpeg_xr=HANDLE_TIFF_JPEG_XR,
            )

            loader.collect_image_filenames()
            loader.parse_filenames()         # uses the selected regex flavor
            loader.save_fov_compressed_images()
