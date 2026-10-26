#!/bin/bash
# Black-box ambiguity detection (rule-based perturbation), text QA in ABG-CoQA format.
#
# Usage:
#   bash scripts/run_blackbox.sh <MODEL> <INPUT_JSON> <OUT_DIR>
#   e.g. bash scripts/run_blackbox.sh Qwen/Qwen3-4B-Instruct-2507 data/abg_coqa_200_blackbox.json results/blackbox_qwen3_abg
#
# Env vars (optional):
#   PYTHON        python interpreter                          (default: python)
#   DEVICE        torch device for the NLI model              (default: cuda:0)
#   HF_CACHE_DIR  HuggingFace hub cache dir                   (default: HF default)
#   NUM_SAMPLES   sampled answers per question / variant      (default: 10)
#   MAX_TOKENS    max new tokens per answer                   (default: 256)
#   TP            vLLM tensor-parallel size                   (default: 1)
#   RULES         subset of rules, e.g. "A1 A2 B1"            (default: all 11)
#
# Steps whose output already exists are skipped, so the script can be re-run to resume.
set -euo pipefail

MODEL=${1:?MODEL required}
INPUT=${2:?INPUT_JSON required}
OUT=${3:?OUT_DIR required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
NUM_SAMPLES=${NUM_SAMPLES:-10}
MAX_TOKENS=${MAX_TOKENS:-256}
TP=${TP:-1}
BB=$ROOT/blackbox

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date)] model=$MODEL input=$INPUT out=$OUT"

skip() { [ -f "$1" ] && echo "[skip] $1 exists"; }

# ── 1. Rule-based perturbation → N answers per (original + variant) ──────────
skip "$OUT/results.json" || \
$PY $BB/run_perturbation.py --input_path "$INPUT" --output_path "$OUT/results.json" \
    --model "$MODEL" --temperature 0.5 --top_p 0.9 --num_samples $NUM_SAMPLES \
    --max_tokens $MAX_TOKENS --tensor_parallel_size $TP --max_examples -1 \
    ${RULES:+--rules $RULES}

# ── 2. NLI agreement → per-variant and cross-variant (pooled) UQ ─────────────
skip "$OUT/uq.json" || \
$PY $BB/compute_uq.py --input "$OUT/results.json" --output "$OUT/uq.json" --device $DEVICE

# ── 3. AUROC per rule + cross-variant (the paper's black-box score) ──────────
$PY $BB/analyze_uq.py --uq "$OUT/uq.json" --results "$OUT/results.json" \
    --output_dir "$OUT/analysis"
echo "[$(date)] done"
