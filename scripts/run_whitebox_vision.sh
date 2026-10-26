#!/bin/bash
# White-box ambiguity detection, visual QA (AQuA, VQ-FA).
#
# Usage:
#   bash scripts/run_whitebox_vision.sh <MODEL> <INPUT_JSON> <OUT_DIR>
#   e.g. bash scripts/run_whitebox_vision.sh Qwen/Qwen2.5-VL-3B-Instruct data/aqua_720_input.json results/qwenvl_aqua
#
# Relative `image_path` entries in INPUT_JSON resolve against $IMAGE_ROOT, or
# against the directory that contains INPUT_JSON when IMAGE_ROOT is unset.
#
# Env vars (optional): PYTHON, DEVICE, HF_CACHE_DIR, METHODS, RUN_BASELINE,
#   IMAGE_ROOT, N_SAMPLED (sampled answers on the original question besides the
#   greedy one; default 4 → 5 answers from the original question).
set -euo pipefail

MODEL=${1:?MODEL required}
INPUT=${2:?INPUT_JSON required}
OUT=${3:?OUT_DIR required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
METHODS=${METHODS:-"vanilla_gradient logit_lens diffmask"}
RUN_BASELINE=${RUN_BASELINE:-1}
N_SAMPLED=${N_SAMPLED:-4}
WB=$ROOT/whitebox/vision
PIPE=$ROOT/whitebox/pipeline

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date)] model=$MODEL input=$INPUT out=$OUT methods=($METHODS)"

skip() { [ -f "$1" ] && echo "[skip] $1 exists"; }

# ── 0. Baseline: 10 samples from the original (image, question), no perturbation
if [ "$RUN_BASELINE" = "1" ] && ! skip "$OUT/pipeline_b_baseline_n10.json"; then
    [ -f "$OUT/_gen_checkpoint_baseline_n10.json" ] || \
    $PY $PIPE/pipeline_b_perturb_and_uq_vl.py --baseline --baseline_n_samples 10 \
        --input_json "$INPUT" --model_path "$MODEL" --output_dir "$OUT" \
        --temperature 0.5 --max_new_tokens 128 --device $DEVICE --backend hf --gen_only
    $PY $PIPE/compute_uq_from_checkpoint.py --checkpoint "$OUT/_gen_checkpoint_baseline_n10.json" \
        --output "$OUT/pipeline_b_baseline_n10.json" --input_json "$INPUT" \
        --method baseline --eigv_variant zero_diag --device $DEVICE
fi

# ── 1. Token importance (white-box attribution, text span only) ──────────────
for METHOD in $METHODS; do
    case $METHOD in
    vanilla_gradient)
        skip "$OUT/vanilla_gradient.json" || \
        $PY $WB/gradient_attribution.py --model_path "$MODEL" --input_json "$INPUT" \
            --output_json "$OUT/vanilla_gradient.json" --method vanilla ;;
    logit_lens)
        skip "$OUT/logit_lens.json" || \
        $PY $WB/logit_lens_uncertainty.py --model_path "$MODEL" --input_json "$INPUT" \
            --output_json "$OUT/logit_lens.json" ;;
    diffmask)
        skip "$OUT/diffmask.json" || \
        $PY $WB/diffmask_uncertainty.py --model_path "$MODEL" --input_json "$INPUT" \
            --output_json "$OUT/diffmask.json" --n_steps 30 --n_generate 4 --sparsity_weight 0.5 ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
    esac
done

# ── 2. Top-3 words → WordNet perturbation → sampling → NLI UQ ────────────────
for METHOD in $METHODS; do
    skip "$OUT/pipeline_b_$METHOD.json" && continue
    [ -f "$OUT/_gen_checkpoint_$METHOD.json" ] || \
    $PY $PIPE/pipeline_b_perturb_and_uq_vl.py --attribution_json "$OUT/$METHOD.json" \
        --input_json "$INPUT" --model_path "$MODEL" --output_dir "$OUT" \
        --temperature 0.5 --max_new_tokens 128 --top_k 3 --n_variants 4 \
        --n_sampled $N_SAMPLED --method_name $METHOD --device $DEVICE --backend hf --gen_only
    $PY $PIPE/compute_uq_from_checkpoint.py --checkpoint "$OUT/_gen_checkpoint_$METHOD.json" \
        --output "$OUT/pipeline_b_$METHOD.json" --input_json "$INPUT" \
        --method $METHOD --eigv_variant zero_diag --device $DEVICE
done

# ── 3. AUROC ─────────────────────────────────────────────────────────────────
$PY $PIPE/compute_auroc.py "$OUT"/pipeline_b_*.json --output "$OUT/auroc.json"
echo "[$(date)] done"
