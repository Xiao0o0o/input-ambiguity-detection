# Quick start: Qwen2.5-VL-3B × AQuA × Logit Lens

This folder holds the data (720 AQuA questions and their images) and every intermediate file from the paper run of the white-box detector with **Logit Lens** importance on **AQuA** with **Qwen2.5-VL-3B-Instruct**. That run gives AUROC **0.708** in Table 1 of the paper; the EigV(Agree) baseline gives 0.644. The code it runs is the repository's own [`whitebox/`](../whitebox/) code.

```bash
# from the repository root
pip install -r requirements.txt                  # or: torch transformers accelerate nltk scikit-learn numpy scipy pillow tqdm
bash quickstart/run_quickstart.sh                # AUROC from the stored UQ scores (CPU, seconds)
bash quickstart/run_quickstart.sh uq             # recompute NLI uncertainty from the stored answers
bash quickstart/run_quickstart.sh generate       # regenerate answers from the stored perturbations
bash quickstart/run_quickstart.sh perturb        # rebuild perturbations from the stored attribution
bash quickstart/run_quickstart.sh attribution    # everything from scratch
```

Each stage writes to `quickstart/outputs/` and then continues through the remaining stages. Optional env vars: `PYTHON`, `DEVICE` (default `cuda:0`; use `cpu` for the `uq` stage on a machine without a GPU), `HF_CACHE_DIR`, `MODEL`.

## Layout

```
quickstart/
├── run_quickstart.sh
├── data/
│   ├── aqua_720_input.json              # 720 questions (540 ambiguous / 180 unambiguous)
│   └── images/                          # 720 AQuA images
└── intermediate/                        # outputs of the paper run
    ├── logit_lens.json.gz               # 1. per-token logit-lens entropies + word_scores (79 MB unpacked)
    ├── _perturbations_logit_lens.json   # 2. top-3 words and perturbed prompts per question
    ├── _gen_checkpoint_logit_lens.json  # 3. pooled answers (original + perturbed)
    ├── pipeline_b_logit_lens.json       # 4. per-question EigV / DegMat / SemEntropy scores
    ├── auroc.json                       # 5. AUROC
    ├── _gen_checkpoint_baseline_n10.json   # baseline: 10 answers to the unperturbed question
    └── pipeline_b_baseline_n10.json        # baseline UQ scores
```

The stages call, from the repository root:

| stage | script |
|---|---|
| 1 attribution | `whitebox/vision/logit_lens_uncertainty.py` (+ `vl_attribution_utils.py`) |
| 2-3 perturb, generate | `whitebox/pipeline/pipeline_b_perturb_and_uq_vl.py` |
| 4 uq | `whitebox/pipeline/compute_uq_from_checkpoint.py` |
| 5 auroc | `whitebox/pipeline/compute_auroc.py` |

## What each stage does

| stage | input → output | setting |
|---|---|---|
| 1 attribution | image + prompt → `logit_lens.json` | Each layer's hidden state is projected through `lm_head`. A token's score is the entropy of that distribution averaged over layers, min-max normalised over the whole sequence, and summed over the word's sub-tokens (text span only). |
| 2 perturb | `logit_lens.json` → `_perturbations_*.json` | The top-3 scoring words inside the question (stop-words and template words skipped) are each replaced by WordNet synonyms with POS-matched inflection, giving up to 4 variants. Only the question span is edited; the image is unchanged. |
| 3 generate | `_perturbations_*.json` → `_gen_checkpoint_*.json` | Original question: 1 greedy + 2 sampled answers (T=0.5, top-p 0.9). Each variant: 1 greedy answer. max_new_tokens=128, HF backend. |
| 4 uq | `_gen_checkpoint_*.json` → `pipeline_b_*.json` | `microsoft/deberta-large-mnli` entailment on all ordered answer pairs. EigV(Agree) is the sum of max(0, 1−λ) over the eigenvalues of the normalised graph Laplacian (`--eigv_variant zero_diag`). |
| 5 auroc | `pipeline_b_*.json` → `auroc.json` | Binary AUROC with ambiguous as the positive class, `EigV_entail` as the score. |

## Expected output

`bash quickstart/run_quickstart.sh` prints:

```
== pipeline_b_logit_lens.json
  EigV_entail     AUROC=0.7083   (scores rounded to 6 dp: 0.7148)   n=720
  ...
== pipeline_b_baseline_n10.json
  EigV_entail     AUROC=0.6435   (scores rounded to 6 dp: 0.6520)   n=720
```

## Reproducibility notes

- **Stages 4-5 from stored answers** (`uq`): per-question scores match the stored ones to within 2e-6. EigV AUROC can still move by about ±0.002 (0.7096 on an L40S vs 0.7083 stored). The reason is that many questions get the same EigV in exact arithmetic (for example, when all answers are identical), so float noise decides how those ties are ordered. With scores rounded to 6 decimals, the stored and recomputed scores both give 0.7148.
- **Stage 2 from the stored attribution** (`perturb`): reproduces `_perturbations_logit_lens.json` exactly (720/720 items; only the absolute `image_path` differs).
- **Stage 3** (`generate`): greedy answers to the original and perturbed questions match the stored ones (checked on 3 questions). The 2 sampled answers per question are stochastic, so the final AUROC varies from run to run.
- **Stage 1 from scratch** (`attribution`): with the library versions in `../requirements.txt`, word lists are identical but normalised scores differ by up to about 0.05 from the stored file, which changes the top-3 words for some questions (2 of 3 checked). The stored `logit_lens.json.gz` is the file used for the paper.
- Stages 1 and 3 load Qwen2.5-VL-3B-Instruct (3.75B parameters) in fp16 on the GPU, about 7.5 GB of weights plus activations.
