# Quick start: Qwen2.5-VL-3B × AQuA × Logit Lens

This folder contains one complete paper run of the **white-box detector**: Logit Lens importance, Qwen2.5-VL-3B-Instruct, AQuA (720 questions). The output of every stage is stored. The run gives AUROC **0.708** in Table 1 of the paper, and the EigV(Agree) baseline gives **0.644**.

**The step-by-step guide (what each step does, its command, and its output) is in the [Quick start section of the main README](../README.md#quick-start).** This page lists the folder's contents and the one-command script.

## Layout

```
quickstart/
├── run_quickstart.sh                    # runs Steps 1-5 from any starting stage
├── data/
│   ├── aqua_720_input.json              # 720 questions (540 ambiguous / 180 unambiguous)
│   └── images/                          # 720 AQuA images
├── intermediate/                        # outputs of the paper run (never overwritten)
│   ├── logit_lens.json.gz               # Step 1: per-token Logit Lens entropies + word_scores (79 MB unpacked)
│   ├── _perturbations_logit_lens.json   # Step 2: top-3 words and perturbed prompts per question
│   ├── _gen_checkpoint_logit_lens.json  # Step 3: pooled answers (original + perturbed)
│   ├── _gen_checkpoint_baseline_n10.json   # Step 3 baseline: 10 answers to the unperturbed question
│   ├── pipeline_b_logit_lens.json       # Step 4: per-question EigV / DegMat / SemEntropy scores
│   ├── pipeline_b_baseline_n10.json     # Step 4 baseline scores
│   └── auroc.json                       # Step 5: AUROC
└── outputs/                             # created by your runs (git-ignored)
```

The steps run, from the repository root:

| step | script | needs |
|---|---|---|
| 1 token importance | `whitebox/vision/logit_lens_uncertainty.py` | GPU (Qwen2.5-VL-3B, ~7.5 GB fp16) |
| 2 perturbation | `whitebox/pipeline/pipeline_b_perturb_and_uq_vl.py --perturb_only` | CPU, under a minute |
| 3 sampling | `whitebox/pipeline/pipeline_b_perturb_and_uq_vl.py --gen_only` | GPU |
| 4 NLI uncertainty | `whitebox/pipeline/compute_uq_from_checkpoint.py` | GPU recommended (CPU: ~45 min per file) |
| 5 AUROC | `whitebox/pipeline/compute_auroc.py` | CPU, seconds |

## One command

```bash
bash quickstart/run_quickstart.sh [STAGE]
```

| STAGE | runs | starts from |
|---|---|---|
| `auroc` (default) | Step 5 | stored Step 4 scores |
| `uq` | Steps 4-5 | stored answers |
| `generate` | Steps 3-5 | stored perturbations (the baseline's answers stay the stored ones) |
| `perturb` | Steps 2-5 | stored attribution |
| `attribution` | Steps 1-5 | scratch |

Env vars: `PYTHON` (default `python`), `DEVICE` (default `cuda:0`; use `cpu` for the `uq` stage without a GPU), `HF_CACHE_DIR`, `MODEL` (default `Qwen/Qwen2.5-VL-3B-Instruct`). Outputs go to `quickstart/outputs/`.

## Reproducibility notes

- **Step 4 from the stored answers** (`uq`): per-question scores match the stored ones to within about 2e-6. EigV AUROC can still move slightly: 0.7096 on an L40S GPU and 0.7150 on a CPU (about 45 minutes), vs 0.7083 stored. Many questions get the same EigV in exact arithmetic (for example, when all answers are identical), so float noise decides how those ties are ordered. With scores rounded to 6 decimals, the stored and all recomputed scores give 0.7148.
- **Step 2 from the stored attribution** (`perturb`): reproduces `_perturbations_logit_lens.json` exactly (720/720 items).
- **Step 3** (`generate`): greedy answers to the original and perturbed questions match the stored ones (checked on 3 questions). The 2 sampled answers per question are random, so the final AUROC varies from run to run.
- **Step 1 from scratch** (`attribution`): with the library versions in `../requirements.txt`, word lists are identical but normalised scores differ from the stored file by up to about 0.05. This changes the top-3 words for some questions (2 of 3 checked). The stored `logit_lens.json.gz` is the file used for the paper.
