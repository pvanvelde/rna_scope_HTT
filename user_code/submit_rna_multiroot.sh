#!/usr/bin/env bash
###############################################################################
# submit_rna_multiroot.sh  – hard-coded root/metadata pairs, no CLI args
###############################################################################

PAIRS=(
  # existing batch
  "/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno1_june2025  /pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno1_june2025/RNA_scope_template_q111.xlsx"

  # 10 slides – bad illumination
#   "/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno2_june2025_bad_illumination  /pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno2_june2025_bad_illumination/RNA_scope_template_q11110_slides_bad.xlsx"

  # 15 slides
  "/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_15slidesno1_june2025  /pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_15slidesno1_june2025/RNA_scope_template_q111_15_slides.xlsx"

  "/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno2_june2025_newillumination  /pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno2_june2025_newillumination/RNA_scope_template_q11110_slides_good.xlsx"
)

# ── slide-job resources ──────────────────────────────────────────────────────
QUEUE="gpu"
MEM=48000                 # MB RAM per slide job
WALL="720:00"             # hh:mm
CUDA_MOD="cuda/12.6.3"
GPU="num=1:mode=exclusive_process"

# ── merge-job resources ──────────────────────────────────────────────────────
MERGE_MEM=16000
MERGE_WALL="4:00"

SIF="/home/pieterfop.vanvelde-umw/rna_scope_cluster/rna_scope.sif"
RUNSCRIPT="$HOME/rna_scope_v3/rna_scope/user_code/main_one_folder.py"
MERGER="$HOME/rna_scope_v3/rna_scope/user_code/merge_rnascope_results.py"
LOG_DIR="$HOME/logs_rna_scope"
KEEP=1000

# ── single timestamp for *all* merge prefixes & log files ─────────────────────
NOW=$(date +'%Y%m%d_%H%M%S')

# define a timestamped global‐merge prefix
GLOBAL_OUT_BASE="/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/global_merge_sigma_from_exp"
GLOBAL_OUT="${GLOBAL_OUT_BASE}_${NOW}"

###############################################################################
set -euo pipefail
mkdir -p "$LOG_DIR"

# rotate old logs
n=$(ls -1 "$LOG_DIR"/*.out 2>/dev/null | wc -l || true)
if (( n > KEEP )); then
  for o in $(ls -1t "$LOG_DIR"/*.out | tail -n +"$((KEEP+1))"); do
    rm -f "$o" "${o%.out}.err"
  done
fi

# collect merge job IDs for final global merge
MERGE_JIDS=()
ROOTS=()

# ── iterate PAIRS ────────────────────────────────────────────────────────────
for pair in "${PAIRS[@]}"; do
  read -r ROOT META <<<"$pair"
  ROOT=$(readlink -f "$ROOT")
  META=$(readlink -f "$META")
  ROOTS+=("$ROOT")

  [[ -d "$ROOT" ]] || { echo "Root not found: $ROOT" >&2;  exit 1; }
  [[ -f "$META" ]] || { echo "Metadata not found: $META" >&2; exit 1; }

  echo "Scanning $ROOT   (metadata: $(basename "$META"))"
  jobids=()   # slide-job IDs for this root

  # ── submit one slide-job per subdirectory ────────────────────────────────
  while IFS= read -r -d '' SLIDE; do
    SLIDE_BN=$(basename "$SLIDE")
    OUT="$LOG_DIR/${SLIDE_BN}_${NOW}.out"
    ERR="$LOG_DIR/${SLIDE_BN}_${NOW}.err"

    echo "  → submitting slide: $SLIDE_BN"

    jid=$( bsub -J "rna_${SLIDE_BN}" \
                -q "$QUEUE" \
                -n 1 \
                -R "rusage[mem=${MEM}]" \
                -W "$WALL" \
                -o "$OUT" \
                -e "$ERR" \
                -gpu "$GPU" <<EOF | awk '{print $2}' | tr -d '<>'
#!/bin/bash
module load $CUDA_MOD
singularity exec --nv "$SIF" /opt/venv/bin/python \
    "$RUNSCRIPT" "$SLIDE" "$META"
EOF
    )
    jobids+=("$jid")
  done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print0)

  # ── per-root merge ────────────────────────────────────────────────────────
  if ((${#jobids[@]})); then
    dep=$(printf 'done(%s) && ' "${jobids[@]}")
    dep=${dep% && }

    PREFIX="${ROOT}/merged_${NOW}"
    MERGE_OUT="$LOG_DIR/merge_$(basename "$ROOT")_${NOW}.out"
    MERGE_ERR="${MERGE_OUT%.out}.err"

    echo "  → submitting per-root MERGE for $(basename "$ROOT")"

    mid=$( bsub -J "merge_$(basename "$ROOT")" \
                -q "$QUEUE" \
                -n 1 \
                -R "rusage[mem=${MERGE_MEM}]" \
                -W "$MERGE_WALL" \
                -w "$dep" \
                -o "$MERGE_OUT" \
                -e "$MERGE_ERR" <<EOF | awk '{print $2}' | tr -d '<>'
#!/bin/bash
module load $CUDA_MOD
singularity exec "$SIF" /opt/venv/bin/python \
    "$MERGER" "$ROOT" --out-prefix "$PREFIX"
EOF
    )
    MERGE_JIDS+=("$mid")
  fi
done

# ── final global merge (after all per-root merges) ───────────────────────────
if ((${#MERGE_JIDS[@]})); then
  dep=$(printf 'done(%s) && ' "${MERGE_JIDS[@]}")
  dep=${dep% && }

  G_OUT="$LOG_DIR/global_merge_${NOW}.out"
  G_ERR="${G_OUT%.out}.err"

  echo "Submitting GLOBAL MERGE job → ${GLOBAL_OUT}.{h5,csv}"

  bsub -J merge_global_q111_jun2025 \
       -q "$QUEUE" \
       -n 1 \
       -R "rusage[mem=${MERGE_MEM}]" \
       -W "$MERGE_WALL" \
       -w "$dep" \
       -o "$G_OUT" \
       -e "$G_ERR" <<EOF
#!/bin/bash
module load $CUDA_MOD
singularity exec "$SIF" /opt/venv/bin/python \
    "$MERGER" ${ROOTS[@]/#/"\""} ${ROOTS[@]/%/"\""} --global-out "$GLOBAL_OUT"
EOF
fi