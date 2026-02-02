#!/usr/bin/env python
"""
Merge per-slide RNAscope outputs.

Examples
--------
# one merged file per root (default filenames)
python merge_rnascope_results.py /root/A /root/B

# timestamped filenames inside each root
python merge_rnascope_results.py /root/A /root/B --timestamp

# fully-custom prefix (e.g. called from a launcher)
python merge_rnascope_results.py /root/A --out-prefix /root/A/merged_20250714_120301

# global merge as well
python merge_rnascope_results.py /root/A /root/B --global-out /analysis/all_batches_20250714
"""
import sys, argparse, warnings, datetime as _dt
from pathlib import Path
import h5py, pandas as pd


# ───────────────────────── I/O HELPERS ───────────────────────────────────────
def copy_h5(src: Path, dst: Path):
    """Append groups from `src` into `dst`. Skip groups that already exist."""
    with h5py.File(src, "r") as s, h5py.File(dst, "a") as d:
        for k in s.keys():
            if k in d:
                warnings.warn(f"{dst.name}: group '{k}' exists – skip")
                continue
            s.copy(s[k], d, name=k)


def merge_one_root(root: Path, out_prefix: Path | None = None):
    """
    If *out_prefix* is given → build <prefix>.h5/.csv fresh.

    If *out_prefix* is None:
      - Prefer reusing newest merged_*.h5/.csv pair.
      - If only one of the pair exists, reuse what exists.
      - If neither exists and no slide-level inputs are found, return any existing merged_* as a fallback.
    """
    def _find_latest_merged_pair(r: Path):
        cand = sorted(r.glob("merged_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cand:
            return None, None
        h5 = cand[0]
        csv = h5.with_suffix(".csv")
        if not csv.exists():
            # try to find the newest merged_*.csv separately
            csv_cand = sorted(r.glob("merged_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            csv = csv_cand[0] if csv_cand else None
        return h5, csv

    # A) explicit prefix → build fresh
    if out_prefix is not None:
        out_h5  = Path(f"{out_prefix}.h5")
        out_csv = Path(f"{out_prefix}.csv")
    else:
        # Try to reuse newest merged_* first
        reuse_h5, reuse_csv = _find_latest_merged_pair(root)
        if reuse_h5 and reuse_csv and reuse_h5.exists() and reuse_csv.exists():
            print(f"  Reusing existing per-root merge: {reuse_h5.name}, {reuse_csv.name}")
            return reuse_h5, reuse_csv
        # Legacy fallback filenames if we need to (re)build
        out_h5  = root / "merged_results.h5"
        out_csv = root / "merged_summary.csv"

    # B) Build/rebuild from slide-level inputs (ignore previous merged_* files)
    h5_files  = [p for p in sorted(root.rglob("*_results.h5")) if not p.name.startswith("merged_")]
    csv_files = [p for p in sorted(root.rglob("*_summary.csv")) if not p.name.startswith("merged_")]

    if not h5_files and out_prefix is None:
        # Nothing to build from. Fall back to any existing merged_* even if pair incomplete.
        fb_h5, fb_csv = _find_latest_merged_pair(root)
        if fb_h5:
            print(f"  No slide-level inputs; falling back to existing: {fb_h5.name}{' + ' + fb_csv.name if fb_csv else ''}")
            return fb_h5, fb_csv
        print(f"  WARNING: no slide-level inputs and no merged_* found in {root}")
        return None, None

    # wipe previous outputs if rebuilding
    out_h5.unlink(missing_ok=True)
    out_csv.unlink(missing_ok=True)

    with h5py.File(out_h5, "w") as dest:
        for src_path in h5_files:
            with h5py.File(src_path, "r") as src:
                for name in src.keys():
                    if name in dest:
                        del dest[name]
                    src.copy(name, dest)

    dfs = []
    for cf in csv_files:
        df = pd.read_csv(cf)
        df["source_file"] = cf.name
        dfs.append(df)
    if dfs:
        pd.concat(dfs, ignore_index=True).to_csv(out_csv, index=False)

    return out_h5, (out_csv if out_csv.exists() else None)



def merge_global(global_path: Path, h5_files, csv_files):
    g_h5  = global_path.with_suffix(".h5")
    g_csv = global_path.with_suffix(".csv")

    if not h5_files and not csv_files:
        print("No per-root merges were found → global merge skipped.")
        return

    g_h5.unlink(missing_ok=True)
    print("Will merge these H5 files:", *[f"  • {p}" for p in h5_files], sep="\n")
    print("Will merge these CSV files:", *[f"  • {p}" for p in csv_files], sep="\n")

    for hf in h5_files:
        copy_h5(hf, g_h5)

    gdfs = []
    for cf in csv_files:
        try:
            gdfs.append(pd.read_csv(cf))
        except Exception as e:
            print(f"  WARNING: failed to read {cf}: {e}")

    if gdfs:
        pd.concat(gdfs, ignore_index=True).to_csv(g_csv, index=False)

# ─────────────────────────── CLI PARSING ─────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", help="batch root directories")
    p.add_argument("--global-out",
                   help="basename (no extension) for a full merged result")
    p.add_argument("--out-prefix",
                   help="basename (no extension) for the per-root output "
                        "(launcher uses this)")
    p.add_argument("--timestamp", action="store_true",
                   help="append _YYYYMMDD_HHMMSS to default filenames "
                        "inside each root")
    args = p.parse_args()

    if args.out_prefix and args.timestamp:
        sys.exit("Choose either --out-prefix or --timestamp, not both.")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else None

    all_h5, all_csv = [], []

    for r in args.roots:
        root = Path(r).resolve()
        print(f"[{root.name}] merging …")

        if args.out_prefix:               # custom prefix provided
            prefix = Path(args.out_prefix)
        elif ts:                          # timestamp flag
            prefix = root / f"merged_{ts}"
        else:                             # legacy filenames
            prefix = None

        h5f, csvf = merge_one_root(root, prefix)
        if h5f:  all_h5.append(h5f)
        if csvf: all_csv.append(csvf)

    if args.global_out:
        gp = Path(args.global_out).resolve()
        print("Creating global merge →", gp.with_suffix(".h5"))
        merge_global(gp, all_h5, all_csv)