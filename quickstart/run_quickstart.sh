#!/bin/bash
# Quick start: Qwen2.5-VL-3B-Instruct × AQuA × Logit Lens (white-box).
# Every stage's output from the paper run is stored in intermediate/; this
# script resumes from any stage and writes fresh results to outputs/.
#
# Usage:
#   bash quickstart/run_quickstart.sh [STAGE]
#
#   STAGE        recomputes                                     needs
#   auroc        AUROC from the stored UQ scores (default)      CPU, seconds
#   uq           NLI UQ from the stored answers → AUROC         NLI model (GPU recommended)
#   generate     answers from the stored perturbations → ...    Qwen2.5-VL-3B on GPU
#   perturb      WordNet perturbations from the stored attribution → ...
#   attribution  logit-lens attribution from scratch → ...      Qwen2.5-VL-3B on GPU
#
# Env vars: PYTHON (default python), DEVICE (default cuda:0), HF_CACHE_DIR,
#           MODEL (default Qwen/Qwen2.5-VL-3B-Instruct)
set -euo pipefail

STAGE=${1:-auroc}
QS=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$QS/.." && pwd)
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
WB=$ROOT/whitebox/vision
PIPE=$ROOT/whitebox/pipeline
IN=$QS/intermediate
OUT=$QS/outputs
INPUT=$QS/data/aqua_720_input.json

case $STAGE in
    auroc|uq|generate|perturb|attribution) ;;
    *) echo "unknown stage '$STAGE' (auroc|uq|generate|perturb|attribution)"; exit 1 ;;
esac
mkdir -p "$OUT"

# logit_lens.json (79 MB) is stored gzipped; it is only read by the perturb stage.
ATTR=$IN/logit_lens.json.gz
PERT=$IN/_perturbations_logit_lens.json
GEN=$IN/_gen_checkpoint_logit_lens.json
UQ=$IN/pipeline_b_logit_lens.json
BASE_GEN=$IN/_gen_checkpoint_baseline_n10.json
BASE_UQ=$IN/pipeline_b_baseline_n10.json

# 1. Token importance: mean logit-lens entropy per input token → word_scores
if [ "$STAGE" = attribution ]; then
    $PY $WB/logit_lens_uncertainty.py --model_path "$MODEL" \
        --input_json "$INPUT" --output_json "$OUT/logit_lens.json"
    ATTR=$OUT/logit_lens.json; STAGE=perturb
fi

# 2. Top-3 question words → up to 4 WordNet-synonym variants
if [ "$STAGE" = perturb ]; then
    if [ "${ATTR%.gz}" != "$ATTR" ]; then
        gzip -dc "$ATTR" > "$OUT/logit_lens.json"; ATTR=$OUT/logit_lens.json
    fi
    $PY $PIPE/pipeline_b_perturb_and_uq_vl.py --attribution_json "$ATTR" \
        --input_json "$INPUT" --output_dir "$OUT" --method_name logit_lens \
        --top_k 3 --n_variants 4 --perturb_only
    PERT=$OUT/_perturbations_logit_lens.json; STAGE=generate
fi

# 3. Original: 1 greedy + 2 sampled (T=0.5); each variant: 1 greedy
if [ "$STAGE" = generate ]; then
    $PY $PIPE/pipeline_b_perturb_and_uq_vl.py --perturbations_json "$PERT" \
        --input_json "$INPUT" --model_path "$MODEL" --output_dir "$OUT" \
        --method_name logit_lens --n_sampled 2 --temperature 0.5 \
        --max_new_tokens 128 --backend hf --device $DEVICE --gen_only
    GEN=$OUT/_gen_checkpoint_logit_lens.json; STAGE=uq
fi

# 4. DeBERTa-large-MNLI agreement matrix over the pooled answers → EigV(Agree)
if [ "$STAGE" = uq ]; then
    $PY $PIPE/compute_uq_from_checkpoint.py --checkpoint "$GEN" \
        --output "$OUT/pipeline_b_logit_lens.json" --input_json "$INPUT" \
        --method logit_lens --eigv_variant zero_diag --device $DEVICE
    $PY $PIPE/compute_uq_from_checkpoint.py --checkpoint "$BASE_GEN" \
        --output "$OUT/pipeline_b_baseline_n10.json" --input_json "$INPUT" \
        --method baseline --eigv_variant zero_diag --device $DEVICE
    UQ=$OUT/pipeline_b_logit_lens.json; BASE_UQ=$OUT/pipeline_b_baseline_n10.json
fi

# 5. Ambiguity-detection AUROC (ours vs. the K=10 EigV(Agree) baseline)
$PY $PIPE/compute_auroc.py "$UQ" "$BASE_UQ" \
    --metrics EigV_entail EigV_disagree DegMat_entail --output "$OUT/auroc.json"
