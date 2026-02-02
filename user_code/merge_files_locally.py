#!/usr/bin/env python3
"""
Merge local RNAscope 'merged_*.h5' and 'merged_*.csv' files that are all in ONE folder.

Usage
-----
python merge_local_folder_results.py /path/to/folder
python merge_local_folder_results.py /path/to/folder --out /path/to/folder/global_merged_20250923
python merge_local_folder_results.py /path/to/folder --overwrite  # overwrite colliding H5 groups
python merge_local_folder_results.py /path/to/folder --include-all  # include any *.h5/*.csv, not just merged_*
"""


# python merge_local_folder_results.py "/media/grunwaldlab/Crucial P3 Plus 4TB/files_rna_scope" \
#   --out "/media/grunwaldlab/Crucial P3 Plus 4TB/files_rna_scope/global_merge_q111_jun2025_local"
import argparse, warnings, datetime as _dt
from pathlib import Path
import h5py
import pandas as pd

def find_inputs(folder: Path, include_all: bool=False):
    if include_all:
        h5s  = sorted(p for p in folder.glob("*.h5")  if p.is_file())
        csvs = sorted(p for p in folder.glob("*.csv") if p.is_file())
    else:
        # This matches both `merged_2025...h5` and `merged_...(1).h5`
        h5s  = sorted(p for p in folder.glob("merged_*.h5")  if p.is_file())
        csvs = sorted(p for p in folder.glob("merged_*.csv") if p.is_file())
    return h5s, csvs


def copy_h5_append_all(src: Path, dst: Path, overwrite: bool=False):
    """
    Append all top-level groups from src.h5 into dst.h5.
    - If a group exists:
        overwrite=False → skip with warning
        overwrite=True  → delete dest group then copy
    """
    with h5py.File(src, "r") as s, h5py.File(dst, "a") as d:
        for k in s.keys():
            if k in d:
                if overwrite:
                    del d[k]
                else:
                    warnings.warn(f"{dst.name}: group '{k}' exists – skip (from {src.name})")
                    continue
            s.copy(s[k], d, name=k)

def merge_csvs(csv_files, out_csv: Path):
    dfs = []
    for cf in csv_files:
        try:
            df = pd.read_csv(cf)
            df["source_file"] = cf.name
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Skipping CSV {cf.name}: {e}")
    if not dfs:
        return False
    merged = pd.concat(dfs, ignore_index=True)
    # optional: drop perfect duplicates (across all columns)
    merged = merged.drop_duplicates()
    merged.to_csv(out_csv, index=False)
    return True

def main():
    ap = argparse.ArgumentParser(description="Merge local RNAscope merged_*.h5/.csv in one folder.")
    ap.add_argument("folder", help="Folder containing merged_*.h5 and merged_*.csv")
    ap.add_argument("--out", help="Output prefix (no extension). Defaults to <folder>/global_merged_YYYYMMDD_HHMMSS")
    ap.add_argument("--overwrite", action="store_true",
                    help="If groups with same name exist in H5, delete in destination and copy from source.")
    ap.add_argument("--include-all", action="store_true",
                    help="Include any *.h5/*.csv in the folder (not only names starting with 'merged_').")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    if args.out:
        out_prefix = Path(args.out)
        if out_prefix.suffix in (".h5", ".csv"):
            out_prefix = out_prefix.with_suffix("")  # enforce prefix
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_prefix = folder / f"global_merged_{ts}"

    out_h5  = out_prefix.with_suffix(".h5")
    out_csv = out_prefix.with_suffix(".csv")

    h5s, csvs = find_inputs(folder, include_all=args.include_all)
    print(f"Found {len(h5s)} H5 and {len(csvs)} CSV candidates in {folder}")

    if not h5s and not csvs:
        raise SystemExit("Nothing to merge.")

    # fresh outputs
    out_h5.unlink(missing_ok=True)
    out_csv.unlink(missing_ok=True)

    # H5 merge
    if h5s:
        print(f"→ Writing H5 → {out_h5}")
        for i, hf in enumerate(h5s, 1):
            print(f"  [{i}/{len(h5s)}] {hf.name}")
            copy_h5_append_all(hf, out_h5, overwrite=args.overwrite)
    else:
        print("No H5 files to merge.")

    # CSV merge
    if csvs:
        print(f"→ Writing CSV → {out_csv}")
        ok = merge_csvs(csvs, out_csv)
        if not ok:
            print("No valid CSV rows were written.")
    else:
        print("No CSV files to merge.")

    print("✓ Done.")
    print(f"H5 : {out_h5.exists() and out_h5 or '—'}")
    print(f"CSV: {out_csv.exists() and out_csv or '—'}")

if __name__ == "__main__":
    main()
