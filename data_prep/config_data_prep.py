"""
Config for the data-prep (FOV export/compression) pipeline.
Adjust paths/channels/regex selection to your environment.
"""

# -----------------------------
# Input discovery
# -----------------------------
ROOT_DIRS = [
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/beads raw/umcms-scope_grunwald-beads-on-sl_2025-04-03_1939/Service 2025-03-14 15-30 F Pieter Beads/Images',
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/beads raw/umcms-scope_grunwald-beads-on-sl_2025-04-03_1939/Service 2025-04-02 19-21 F Pieter beads WF SL/Images',
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/YAC128 raw data/stainingno2/umcms-scope_grunwald-2025-03-18-10-49-annotated-zip_2025-04-13_0524/2025-03-18 10-49 annotated',
    '/media/grunwaldlab/SG Skyhawk AI 24TB/Q111 raw data/Q111_10slidesno2_june2025_new_illumination/2025-06-26 09-47 annotated new settings',
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/Q111 raw data/Q111_15slidesno1_june2025/2025-06-18 16-34 annotated/2025-06-18 16-34 annotated/2025-06-18 16-34',
]

# -----------------------------
# Output roots (one per ROOT_DIRS entry)
# -----------------------------
OUTPUT_DIRS =  [
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/beads raw/umcms-scope_grunwald-beads-on-sl_2025-04-03_1939/exported_confocal',
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/beads raw/umcms-scope_grunwald-beads-on-sl_2025-04-03_1939/exported_widefield',
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/YAC128 raw data/stainingno2/umcms-scope_grunwald-2025-03-18-10-49-annotated-zip_2025-04-13_0524/exported',
    '/media/grunwaldlab/SG Skyhawk AI 24TB/Q111 raw data/Q111_10slidesno2_june2025_new_illumination/exported_test',
    # '/media/grunwaldlab/SG Skyhawk AI 24TB/Q111 raw data/Q111_15slidesno1_june2025/exported',
]
# -----------------------------
# Channels to process (order matters)
# -----------------------------
CHANNELS = ['mDAPI', 'sFITC', 'sCY3', 'sCY5']

# -----------------------------
# Compression defaults
#    one of: 'raw' | 'lz4' | 'zstd' | 'gzip'
# -----------------------------
DEFAULT_COMPRESSION = 'lz4'

# -----------------------------
# Filename parsing
# Choose which regex flavor to use for parsing:
#   'v1' = use filename for slide name
#   'v2' = derive slide from grandparent folder (matches your newer exports)
# -----------------------------
PARSER_VERSION = 'v2'   # 'v1' or 'v2'

# -----------------------------
# TIFF / Codec flags
# -----------------------------
HANDLE_TIFF_JPEG_XR = True  # if True, try imagecodecs.jpegxr_decode for page-1 compressed data

# -----------------------------
# Napari defaults
# -----------------------------
NAPARI_WINDOW_TITLE = "Image"
