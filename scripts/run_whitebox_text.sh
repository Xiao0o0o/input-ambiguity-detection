#!/bin/bash
# White-box ambiguity detection, text QA (ABG-CoQA, AmbigQA-w-Ctx).
#
# Usage:
#   bash scripts/run_whitebox_text.sh <MODEL> <INPUT_JSON> <OUT_DIR>
#   e.g. bash scripts/run_whitebox_text.sh meta-llama/Llama-3.1-8B-Instruct data/abg_coqa_200_input.json results/llama_abg
#
# Env vars (optional):
#   PYTHON        python interpreter                       (default: python)
#   DEVICE        torch device for generation / NLI        (default: cuda:0)
#   HF_CACHE_DIR  HuggingFace cache dir for models          (default: HF default)
#   METHODS       attribution methods to run               (default: "vanilla_gradient logit_lens diffmask")
#   RUN_BASELINE  1 = also run the K=10 EigV(Agree) baseline (default: 1)
#
# Steps whose output already exists are skipped, so the script can be re-run to resume.
set -euo pipefail

MODEL=${1:?MODEL required}
INPUT=${2:?INPUT_JSON required}
OUT=${3:?OUT_DIR required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
METHODS=${METHODS:-"vanilla_gradient logit_lens diffmask"}
RUN_BASELINE=${RUN_BASELINE:-1}
WB=$ROOT/whitebox/text
PIPE=$ROOT/whitebox/pipeline

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "[$(date)] model=$MODEL input=$INPUT out=$OUT methods=($METHODS)"

skip() { [ -f "$1" ] && echo "[skip] $1 exists"; }

# ── 0. Baseline: 10 samples from the original question, no perturbation ──────
if [ "$RUN_BASELINE" = "1" ] && ! skip "$OUT/pipeline_b_baseline_n10.json"; then
    [ -f "$OUT/_gen_checkpoint_baseline_n10.json" ] || \
    $PY $PIPE/pipeline_b_perturb_and_uq.py --baseline --baseline_n_samples 10 \
        --input_json "$INPUT" --model_path "$MODEL" --output_dir "$OUT" \
        --temperature 0.5 --max_new_tokens 128 --scope question_only \
        --device $DEVICE --gen_only
    $PY $PIPE/compute_uq_from_checkpoint.py --checkpoint "$OUT/_gen_checkpoint_baseline_n10.json" \
        --output "$OUT/pipeline_b_baseline_n10.json" --input_json "$INPUT" \
        --method baseline --eigv_variant polygraph --device $DEVICE
fi

# ── 1. Token importance (white-box attribution) ──────────────────────────────
for METHOD in $METHODS; do
    case $METHOD in
    vanilla_gradient)
        skip "$OUT/vanilla_gradient.json" || \
        $PY $WB/gradient_attribution.py --model_path "$MODEL" --input_json "$INPUT" \
            --output_json "$OUT/vanilla_gradient.json" --method vanilla --top_k 10 ;;
    logit_lens)
        # Raw per-token logit-lens entropies; mapped to words in step 2.
        skip "$OUT/logit_lens_raw.json" || skip "$OUT/logit_lens.json" || \
        $PY $WB/logit_lens_uncertainty.py --model_path "$MODEL" --input_json "$INPUT" \
            --output_json "$OUT/logit_lens_raw.json" ;;
    diffmask)
        # >20B models: INT8 weights so the autoregressive mask objective fits in memory.
        INT8=0
        case "$MODEL" in *27b*|*27B*|*70b*|*70B*) INT8=1 ;; esac
        skip "$OUT/diffmask.json" || \
        FORCE_INT8=$INT8 $PY $WB/diffmask_uncertainty.py --model_path "$MODEL" --input_json "$INPUT" \
            --output_json "$OUT/diffmask.json" --n_steps 30 --n_generate 4 --sparsity_weight 0.5 ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
    esac
done

# ── 2. Normalise attribution files to top-level `word_scores` ────────────────
#    (logit_lens re-uses the word boundaries of vanilla_gradient.json)
$PY $WB/normalize_attributions.py --output_dir "$OUT"

# ── 3. Top-3 words → WordNet perturbation → sampling → NLI UQ ────────────────
for METHOD in $METHODS; do
    TAG=${METHOD}_top3_question_only
    skip "$OUT/pipeline_b_$TAG.json" && continue
    [ -f "$OUT/_gen_checkpoint_$TAG.json" ] || \
    $PY $PIPE/pipeline_b_perturb_and_uq.py --attribution_json "$OUT/$METHOD.json" \
        --input_json "$INPUT" --model_path "$MODEL" --output_dir "$OUT" \
        --temperature 0.5 --max_new_tokens 128 --scope question_only \
        --top_k 3 --n_variants 4 --n_sampled 4 --method_name $METHOD \
        --device $DEVICE --gen_only
    $PY $PIPE/compute_uq_from_checkpoint.py --checkpoint "$OUT/_gen_checkpoint_$TAG.json" \
        --output "$OUT/pipeline_b_$TAG.json" --input_json "$INPUT" \
        --method $METHOD --eigv_variant polygraph --device $DEVICE
done

# ── 4. AUROC ─────────────────────────────────────────────────────────────────
$PY $PIPE/compute_auroc.py "$OUT"/pipeline_b_*.json --output "$OUT/auroc.json"
echo "[$(date)] done"
