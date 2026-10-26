# White-box input-ambiguity detection

This is the code for the **white-box detector** in *Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs*. It runs in five steps:

1. **Token importance.** One of three model-internal attribution methods scores the input tokens: **Vanilla Gradient**, **DiffMask**, or **Logit Lens**.
2. **Perturbation.** The top-k (k=3) scoring words in the question are each replaced by WordNet synonyms. This is a label-preserving, semantics-blind edit that gives up to 4 variants.
3. **Sampling.** The model answers the original question 5 times (1 greedy + 4 sampled) and each variant once (greedy).
4. **Scoring.** `microsoft/deberta-large-mnli` builds an entailment agreement matrix over the pooled answers, and its graph-Laplacian eigenvalues give **EigV(Agree)**. A higher score means the question is more likely ambiguous.
5. **Evaluation.** Binary AUROC against the ambiguity labels.

All commands below are run from the repository root.

**New here? Start with [`quickstart/`](../quickstart/).** It is a self-contained folder with every intermediate file of one paper run (Qwen2.5-VL-3B × AQuA × Logit Lens, AUROC 0.708) and a script that resumes from any stage.

## Layout

```
whitebox/
├── text/                         # text LLMs (AutoModelForCausalLM)
│   ├── gradient_attribution.py   #   Vanilla Gradient
│   ├── logit_lens_uncertainty.py #   Logit Lens (per-token entropies)
│   ├── diffmask_uncertainty.py   #   DiffMask
│   ├── attribution_utils.py
│   └── normalize_attributions.py #   lifts VG word_scores; maps LL tokens → words via VG boundaries
├── vision/                       # VLMs (AutoModelForImageTextToText); scores the text span only
│   ├── gradient_attribution.py
│   ├── logit_lens_uncertainty.py
│   ├── diffmask_uncertainty.py
│   └── vl_attribution_utils.py
└── pipeline/
    ├── pipeline_b_perturb_and_uq.py     # text:   top-k words → WordNet perturbation → generation
    ├── pipeline_b_perturb_and_uq_vl.py  # vision: same, perturbing only the question span
    ├── compute_uq_from_checkpoint.py    # NLI matrix → EigV / DegMat / SemEntropy per question
    └── compute_auroc.py                 # AUROC (optionally on the intersection of valid items)

scripts/
├── run_whitebox_text.sh          # full pipeline for one (model, text dataset)
└── run_whitebox_vision.sh        # full pipeline for one (model, vision dataset)

data/
├── abg_coqa_200_input.json       # ABG-CoQA, 200 items (100 / 100)
├── ambigqa_with_ctx_500_input.json  # AmbigQA + Wikipedia context, 500 items (250 / 250)
├── aqua_720_input.json           # AQuA, 720 items (540 ambiguous / 180 unambiguous)
├── vqfa_720_input.json           # VQ-FocusAmbiguity, 720 items (422 / 298)
└── build/                        # scripts that built these files (+ images for AQuA / VQ-FA)
```

## Setup

```bash
pip install -r requirements.txt
export HF_CACHE_DIR=/path/to/hf_cache     # optional; HF default cache otherwise
export HF_TOKEN=...                       # needed for meta-llama/Llama-3.1-8B-Instruct and google/gemma-3-*
```

NLTK downloads `wordnet`, `omw-1.4` and the POS tagger on first use.

**Images.** The text datasets are complete in `data/`. The vision inputs store image paths relative to `data/` (`images/aqua/…`, `images/vqfa/…`). Create those images with:

```bash
python data/build/build_aqua_input.py                                   # downloads jihyoung/AQuA (HF) and writes data/images/aqua/
VQFA_ROOT=/path/to/vq_focus_ambiguity python data/build/build_vqfa_input.py   # needs the VQ-FocusAmbiguity release
```

Both scripts rewrite their input JSON as well. Alternatively, point `IMAGE_ROOT` at a directory that contains `images/aqua/` and `images/vqfa/`.

## Running

```bash
# Text QA
bash scripts/run_whitebox_text.sh Qwen/Qwen3-4B-Instruct-2507        data/abg_coqa_200_input.json       results/qwen3_abg
bash scripts/run_whitebox_text.sh meta-llama/Llama-3.1-8B-Instruct  data/ambigqa_with_ctx_500_input.json results/llama_ambigqa
bash scripts/run_whitebox_text.sh google/gemma-3-27b-it             data/abg_coqa_200_input.json       results/gemma27_abg

# Vision QA
bash scripts/run_whitebox_vision.sh Qwen/Qwen2.5-VL-3B-Instruct  data/aqua_720_input.json results/qwenvl_aqua
bash scripts/run_whitebox_vision.sh OpenGVLab/InternVL3-8B-hf    data/vqfa_720_input.json results/internvl_vqfa
bash scripts/run_whitebox_vision.sh google/gemma-3-27b-it        data/aqua_720_input.json results/gemma27_aqua
```

Useful env vars: `PYTHON`, `DEVICE` (default `cuda:0`), `METHODS` (default `"vanilla_gradient logit_lens diffmask"`), `RUN_BASELINE` (default 1; the K=10 EigV(Agree) baseline). Steps whose output already exists are skipped, so re-running a script resumes it. In the text pipeline, Logit Lens borrows word boundaries from `vanilla_gradient.json`, so run VG whenever LL is in `METHODS`.

Files written to `OUT_DIR`, one set per method:

| stage | text | vision |
|---|---|---|
| token importance | `<method>.json` (`logit_lens_raw.json` before normalisation) | `<method>.json` |
| perturbations | `_perturbations_<method>_top3_question_only.json` | `_perturbations_<method>.json` |
| answers | `_gen_checkpoint_<method>_top3_question_only.json` | `_gen_checkpoint_<method>.json` |
| UQ scores | `pipeline_b_<method>_top3_question_only.json` | `pipeline_b_<method>.json` |
| AUROC | `auroc.json` | `auroc.json` |

## Methods and settings

| | |
|---|---|
| Vanilla Gradient | Gradient of the cross-entropy of the greedy next token w.r.t. the input embeddings (fp32). The token score is the L2 norm, min-max normalised and summed over a word's sub-tokens. Model weights are frozen. |
| Logit Lens | Every layer's hidden state is projected through `lm_head`. The token score is the entropy of that distribution averaged over layers, then mapped to words. |
| DiffMask | A sigmoid mask over input embeddings, with a zero-embedding baseline. Loss = −(mean output entropy over 4 greedy decoding steps) + 0.5·Σ mask, optimised for 30 Adam steps (lr 0.1). The word score is 1 − mask. |
| Perturbation | top_k = 3 words, restricted to the question (`--scope question_only`; template and stop-words skipped); up to 4 WordNet variants with POS-matched inflection. |
| Generation | T = 0.5, top-p 0.9, max_new_tokens = 128. Original: 1 greedy + `n_sampled` (4); each variant: 1 greedy. |
| NLI | `microsoft/deberta-large-mnli`, entailment probability, symmetrised. |
| Baseline | 10 answers (1 greedy + 9 sampled) to the unperturbed question, scored the same way. |

**Two EigV variants** (`compute_uq_from_checkpoint.py --eigv_variant`):

- `polygraph`, used for the paper's **text** numbers. This is identical to lm_polygraph's `EigValLaplacian`/`DegMat`: the diagonal of the matrix holds the NLI self-entailment.
- `zero_diag`, used for the paper's **vision** numbers. The Laplacian is built on the off-diagonal entailment only.

`scripts/run_whitebox_text.sh` and `scripts/run_whitebox_vision.sh` select the variant that matches the paper. Neither needs lm_polygraph installed.

**Memory.** For models above 20B, the text DiffMask runs with INT8 weights (`FORCE_INT8=1`, set automatically for `*27b*`). Models above 10B (text) and InternVL / Gemma-3-27B (vision) use gradient checkpointing. On a 46 GB GPU, InternVL3-8B DiffMask runs out of memory for most AQuA / VQ-FA items.

## Notes on reproducing numbers

- **Ties in EigV.** Many questions receive the same EigV in exact arithmetic, for example when all pooled answers are identical. Float noise in the NLI forward pass then decides how those ties are ordered, so recomputing UQ on different hardware can shift AUROC by a few thousandths. `compute_auroc.py` also prints the AUROC after rounding scores to 6 decimals, which does not depend on that noise.
- **Sampling.** Sampled answers are stochastic, so re-running generation changes AUROC from run to run. Greedy answers matched the stored ones in our checks.
- **Library versions.** Recomputing Logit Lens attribution for Qwen2.5-VL-3B with the versions in `requirements.txt` gave the same word lists as the stored paper run, but normalised scores differed by up to about 0.05. This can change which words land in the top-3.
