# RNAscope Pipeline Overview: Imaging → Data Prep → Training → Analysis

This document provides a high‑level overview of the RNAscope workflow, from imaging on a slide scanner through data preparation, model training, and analysis.  All paths are given relative to the root of this repository.

## 1. Imaging & Region Annotation

After you scan each slide, annotate every region and record acquisition settings and specimen metadata in a spreadsheet. **Do not change the column order or names** – downstream tools rely on this format.

### Required Spreadsheet Columns

- Probe‑Set  
- Condition  
- Age  
- Mouse Model  
- Sex  
- mouse ID  
- Level  
- Slide name  
- Date  
- Person  
- intensity dapi  
- Exposure time dapi  
- intensity FITC  
- Exposure time FITC  
- intensity CY3  
- Exposure time CY3  
- intensity CY5  
- Exposure time CY5  
- Brain_Atlas_coordinates  
- Tissue_notes  

### Template

A template with these columns is included in the repository (`RNA_scope_template_example.xlsx`). Use it as a starting point. If you need extra fields, append new columns to the right of the required set so existing parsers continue to work.

### How the Metadata Is Used

- **Data preparation:** The spreadsheet row corresponding to a FOV is used to locate files (via `Slide name`, `Region` and `FOV`) and to capture acquisition parameters.  
- **Analysis:** Results are joined back to the metadata to propagate fields such as `Condition`, `Mouse Model`, and `Level` into downstream QC, plots, and summaries.

## 2. Data Preparation

The `data_prep` folder contains scripts for converting slide‑scanner exports into per‑FOV tensors with consistent shapes.

**Directory:** `./data_prep/`

- `config_data_prep.py` – defines input/output folders, channels, parser options, and compression settings
- `main_data_prep.py` – entry point: scans slide/region directories, groups TIFFs by FOV & channel, writes compressed arrays
- `utils_data_prep.py` – ImageLoader class and helpers for TIFF/NPZ I/O, JPEG XR decoding, filename parsing
- `manual/` – notes and how‑tos for acquisition & export
- `run_rsync.sh` – optional script to sync raw or exported data to long‑term storage

### Usage

1. Edit `config_data_prep.py` to set:
   - `ROOT_DIRS` and `OUTPUT_DIRS` (the folders to read from and write to)
   - `CHANNELS` and `PARSER_VERSION` (`'v1'` to parse slide name from filenames or `'v2'` to infer from folder structure)
   - `DEFAULT_COMPRESSION` (recommended: `lz4`)

2. Run the preparer:

```bash
cd ./data_prep
python main_data_prep.py
```

The preparer will:

- Scan each slide/region and group TIFFs by FOV and channel.
- Build arrays with shape `(channels, z, y, x)`.
- Save them in the format you selected (e.g. `.npz` with LZ4 compression, `.tif` with zstd, `.h5` with gzip, or uncompressed TIFF).

## 3. Training (Training Cluster)

The `training_cluster` directory holds scripts and utilities for preparing training data and training models using [pytorch‑3dunet](https://github.com/wolny/pytorch-3dunet).

**Directory:** `./training_cluster/`

- `config_training_prep.py` – defines paths and options for TIFF→HDF5 conversion
- `utils_training_prep.py` – pairing, label cleanup, HDF5 writing, Napari previews
- `main_training_prep.py` – entry point: produces per‑image HDF5s for train/validation
- `training_model.py` – runs a training experiment; loads a YAML configuration
- `training_dapi/` – data and checkpoints for a DAPI experiment
- `training_green_yellow/` – data and checkpoints for a green/yellow experiment

### Preparing Training Data

1. Set the input/output paths and options in `config_training_prep.py`:
   - `FOLDER`, `OUT_TRAIN`, `OUT_VAL`
   - `VAL_SPLIT`, `PROCESS_LABELS`, `SMOOTH_SIGMA`, `MIN_LABEL_SIZE`
   - `RESIZE_TO` if you need to force a uniform 3D shape

2. Build the HDF5 sets:

```bash
cd ./training_cluster
python main_training_prep.py
```

Each TIFF/label pair is converted into one HDF5 file containing `/raw` and `/label` datasets. See the training README for details.

### Training a Model

Training is driven by YAML configs (e.g. `training_green_yellow.yaml`). These define the U‑Net architecture, loss, optimizer, schedules, and data loaders (pointing to your HDF5 train/val directories). To start training:

```bash
cd ./training_cluster
python training_model.py  # reads the YAML hardcoded inside training_model.py
# or explicitly specify:
# python training_main.py --config ../data_prep_and_training/training_green_yellow/training_green_yellow.yaml
```

Checkpoints and logs are written under the `training_*` directories specified by your YAML. See [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) for more information.

## 4. Analysis (User Code)

Once models are trained, use the scripts in `user_code` to run detection, fitting, filtering, visualization, and to aggregate results.

**Directory:** `./user_code/`

- `config.py` – central analysis configuration: paths to prediction YAMLs, output locations, and channel‑specific thresholds
- `main.py` – entry point: loads images, runs prediction via the backbone, detects and fits spots/labels, writes output
- `utils.py` – helpers for Napari viewers and SVG export

### What `config.py` Controls

- **Device:** set `cfg["dev"]` to `"cuda"` (GPU) or `"cpu"` (no GPU).
- **Prediction configs:** `cfg["config_paths"]` maps each channel to a *prediction* YAML. These YAMLs point to the model checkpoints produced during training.
- **Output locations:** `data_manager_cfg["master_h5_path"]` and `data_manager_cfg["summary_csv"]` determine where the aggregated results are written.
- **Input discovery:** set `rootfolder` to the folder containing your exported NPZ/TIFFs and `data_file` to your annotation spreadsheet.
- **Channel behaviour:** `color_config` specifies which channels should detect labels and/or fit spots and provides per‑channel priors (`sigma`, `break_sigma`, `min_size`, etc.).
- **Detection/fitting parameters:** adjust `detection_cfg` and `fit_cfg` to suit your data (e.g. `min_distance`, `zslices`, `iterations`).

### Running Analysis

1. Open `./user_code/config.py` and update:
   - `cfg["config_paths"]` to point at the prediction YAMLs for your trained models (relative paths are recommended, e.g. `./predict_yamls/green_predict.yaml`).
   - `rootfolder` and `data_file` to your imaging export and metadata spreadsheet.
   - `data_manager_cfg` paths if you wish to change the output locations.

2. Run the analysis:

```bash
cd ./user_code
python main.py
```

This will:

- Load the imaging data (with MIPs where appropriate) and run predictions using your trained models.
- Detect labels (e.g. nuclei) and fit spots on the specified channels.
- Apply multi‑stage filters to refine detections (signal significance, cluster break, etc.).
- Visualize results in Napari: raw MIP, labels, and multiple spot layers (all, filtered, PFA, break, final).
- Optionally export per‑layer SVGs with a raster MIP background, label overlay, and vector spots.
- Write merged results to your chosen HDF5 and CSV outputs.

### Additional Notes

- **Trained models:** The `training_cluster` produces checkpoints. Use the *prediction YAMLs* (not the training YAML) to load them during analysis; these YAMLs live in `predict_yamls/` or a similar folder.  
- **Spreadsheet format:** Always preserve column names and order. Downstream scripts expect the required set of columns.  
- **GPU vs CPU:** Set `cfg["dev"]="cuda"` if a GPU is available; otherwise, use `"cpu"`.  
- **Troubleshooting:** 
  - Empty spot layers? Ensure the channel order and model mapping (`cfg["channel_map"]`) match your imaging data.  
  - Missing labels? Ensure `detect_labels=True` for that channel and adjust `min_size/max_size`.  
  - White/blank images in Napari? Use the viewers in `utils.py` which set `interpolation="none"` and handle intensity scaling.  
  - Slow runs on CPU? Switch to GPU (if available) or reduce patch sizes / `zslices` in `fit_cfg`.

## Quick Start Cheat Sheet

```bash
# Build per‑FOV arrays (data prep)
cd ./data_prep
python main_data_prep.py

# Prepare training HDF5s (if training models)
cd ./training_cluster
python main_training_prep.py

# Train a model (using an example YAML)
python training_model.py

# Run analysis on processed data
cd ./user_code
python main.py
```

All paths in this document are relative to the repository root. Adjust them as needed if you move the code or data.

<!-- End of document -->