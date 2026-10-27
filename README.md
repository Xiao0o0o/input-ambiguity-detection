# Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs

[![PyPI](https://img.shields.io/pypi/v/input-ambiguity-detection.svg)](https://pypi.org/project/input-ambiguity-detection/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Xiao0o0o/input-ambiguity-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Xiao0o0o/input-ambiguity-detection/actions/workflows/ci.yml)

Code and data for the AACL 2026 paper **"Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs"** by Tiejin Chen\*, Xiaoou Liu\*, Michael Clifford, Ahmadreza Moradipari, Yitao Chen, Kyungtae Han, Nejib Ammar, and Hua Wei (\*equal contribution).

## Overview

LLMs often answer an ambiguous question as if it had a single meaning. A common way to detect ambiguity is to have an LLM paraphrase or clarify the question and measure how much the answers disagree. But to rephrase a question, the model must first understand it, and understanding an ambiguous question means picking one reading. The rewrite can remove the very ambiguity it is meant to measure.

We instead perturb the input with **low-capacity, label-preserving, semantics-blind** edits. These are small surface changes that never choose an interpretation. We then measure how much the model's answers spread across the original and perturbed questions. Two detectors implement this:

| detector | access | how it perturbs | code |
|---|---|---|---|
| **White-box** | model internals | Vanilla Gradient, Logit Lens or DiffMask ranks the input tokens; the top-3 question words are replaced by WordNet synonyms | [`whitebox/`](whitebox/) |
| **Black-box** | text in, text out | a fixed library of 11 rules: POS-constrained WordNet synonyms, contraction expand/contract, register prefixes, wh-word templates | [`blackbox/`](blackbox/) |

Both pool the answers to the original and perturbed questions. An NLI model (`microsoft/deberta-large-mnli`) builds an agreement matrix over them, and **EigV(Agree)**, the eigenvalue dispersion of its graph Laplacian, is the ambiguity score. A higher score means a more likely ambiguous question.

## Repository layout

```
.
├── data/                         # the four evaluation sets (shared by both detectors)
│   ├── abg_coqa_200_input.json         # ABG-CoQA, 200 items (100 ambiguous / 100 not)
│   ├── abg_coqa_200_blackbox.json      # the same 200 items in the original ABG-CoQA format (black-box input)
│   ├── ambigqa_with_ctx_500_input.json # AmbigQA + Wikipedia context, 500 items (250 / 250)
│   ├── aqua_720_input.json             # AQuA, 720 items (540 / 180)
│   ├── vqfa_720_input.json             # VQ-FocusAmbiguity, 720 items (422 / 298)
│   └── build/                          # scripts that built these files (+ images for AQuA / VQ-FA)
├── whitebox/                     # white-box detector  → see whitebox/README.md
│   ├── text/   vision/                 # Vanilla Gradient, Logit Lens, DiffMask attribution
│   └── pipeline/                       # top-k WordNet perturbation, generation, NLI UQ, AUROC
├── blackbox/                     # black-box detector  → see blackbox/README.md
│   ├── perturbation_rules.py           # the 11-rule library
│   ├── run_perturbation.py             # rules → vLLM sampling
│   ├── compute_uq.py  analyze_uq.py    # NLI UQ (per variant / pooled) → AUROC + plots
│   └── run_llm_perturbation.py  run_paraphrase.py   # LLM-executed rules (ablation), LLM paraphrase baseline
├── scripts/                      # one command per (model, dataset)
│   ├── run_whitebox_text.sh  run_whitebox_vision.sh
│   └── run_blackbox.sh
├── quickstart/                   # every intermediate file of one paper run + a resumable script
├── input_ambiguity/              # package entry point: the `input-ambiguity` CLI
├── tests/
├── pyproject.toml                # PyPI package (installs whitebox/ and blackbox/ as input_ambiguity.*)
└── requirements.txt              # full environment for both detectors
```

## Installation

From PyPI:

```bash
pip install input-ambiguity-detection               # white-box detector, NLI scoring, AUROC
pip install "input-ambiguity-detection[blackbox]"   # + black-box detector (vLLM, lm-polygraph, spaCy)
pip install "input-ambiguity-detection[all]"        # + INT8 DiffMask (bitsandbytes) and data/build/* deps
```

From the source, with the data, quick start, and shell scripts (recommended to reproduce the paper):

```bash
git clone https://github.com/Xiao0o0o/input-ambiguity-detection.git
cd input-ambiguity-detection
pip install -r requirements.txt     # exact versions used for the release checks
pip install -e . --no-deps          # optional: also installs the `input-ambiguity` command
```

Then:

```bash
python -m spacy download en_core_web_sm   # POS tags for the black-box rules
export HF_CACHE_DIR=/path/to/hf_cache     # optional; HF default cache otherwise
export HF_TOKEN=...                       # for meta-llama/Llama-3.1-8B-Instruct and google/gemma-3-*
```

NLTK downloads WordNet and the POS tagger on first use (`make nltk-data` fetches them up front).

Every script is also available through the installed command, with the same arguments:

```bash
input-ambiguity --help                       # list of commands
input-ambiguity rules                        # black-box rule variants for a few sample questions
input-ambiguity bb-perturb --help            # = python blackbox/run_perturbation.py --help
input-ambiguity wb-vision-logit-lens --help  # = python whitebox/vision/logit_lens_uncertainty.py --help
```

## Quick start

The [`quickstart/`](quickstart/) folder contains one complete paper run of the **white-box detector**, with the output of every stage stored. You can check the paper's number right away, look at what each stage produces, or re-run any stage from the stored output of the stage before it.

| Setting | Dataset | Model | Detector | Included artifacts |
| --- | --- | --- | --- | --- |
| AQuA + Qwen2.5-VL + Logit Lens | AQuA, 720 visual questions (540 ambiguous / 180 unambiguous) and their images | Qwen2.5-VL-3B-Instruct | White-box, Logit Lens importance | Token importance, perturbed questions, sampled answers, per-question uncertainty, AUROC, and the EigV(Agree) baseline's answers and scores |

You don't need a GPU to see the reported numbers: Step 5 reads the stored scores and runs in seconds on a CPU. Only Steps 1 and 3 load Qwen2.5-VL-3B-Instruct (about 7.5 GB of fp16 weights on a GPU).

### Quick start files

| File | Produced by | Description |
| --- | --- | --- |
| `quickstart/data/aqua_720_input.json` | input | 720 AQuA questions: the full prompt, `question_only`, `image_path`, and the label `ambiguity` (`ambiguous` / `unambiguous`). |
| `quickstart/data/images/` | input | The 720 AQuA images. |
| `quickstart/intermediate/logit_lens.json.gz` | Step 1 | Per-token Logit Lens entropies and `word_scores` (one importance score per word) for every question. Gzipped (79 MB unpacked). |
| `quickstart/intermediate/_perturbations_logit_lens.json` | Step 2 | Per question: the top-3 words and up to 4 perturbed prompts. |
| `quickstart/intermediate/_gen_checkpoint_logit_lens.json` | Step 3 | Per question: the pooled answers to the original and perturbed prompts. |
| `quickstart/intermediate/_gen_checkpoint_baseline_n10.json` | Step 3 (baseline) | Per question: 10 answers to the unperturbed prompt. |
| `quickstart/intermediate/pipeline_b_logit_lens.json` | Step 4 | Per question: EigV(Agree) (`EigV_entail`) and the other UQ scores of our detector. |
| `quickstart/intermediate/pipeline_b_baseline_n10.json` | Step 4 (baseline) | The same scores for the EigV(Agree) baseline. |
| `quickstart/intermediate/auroc.json` | Step 5 | AUROC of both score files. |

### Option A: run each step with the scripts

Run all commands from the repository root after `pip install -r requirements.txt`. New outputs go to `quickstart/outputs/`; the stored files in `quickstart/intermediate/` are never overwritten. Every step can start from the stored output of the previous step, so you can skip any GPU step. To try a GPU step quickly, add `--limit 20` to Steps 1 and 3 to process only the first 20 questions.

```bash
mkdir -p quickstart/outputs
```

The first AQuA question is used below as a running example:

> *What color is the frisbee that the man in the white T-shirt is holding?* (label: unambiguous)

**Step 1: token importance (Logit Lens).** *GPU. Optional: the stored output is `quickstart/intermediate/logit_lens.json.gz`.*

Every layer's hidden state is projected through the model's `lm_head`. A token's score is the entropy of that next-token distribution averaged over layers, min-max normalised over the sequence. Sub-token scores are summed into one score per word; only the text span is scored, the image is not.

```bash
python whitebox/vision/logit_lens_uncertainty.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --input_json quickstart/data/aqua_720_input.json \
  --output_json quickstart/outputs/logit_lens.json
```

Output: one record per question, with a `word_scores` list of `{word, score, token_indices}`.

**Step 2: label-preserving perturbation.** *CPU, under a minute.*

Template and stop-words are skipped. The 3 highest-scoring words of the question are each replaced by a WordNet synonym with matching POS and inflection, giving up to 4 perturbed prompts. Only the question changes; the image stays the same. To start from the stored Step 1 output, unpack it first:

```bash
# skip this line if you ran Step 1 yourself
gzip -dc quickstart/intermediate/logit_lens.json.gz > quickstart/outputs/logit_lens.json

python whitebox/pipeline/pipeline_b_perturb_and_uq_vl.py \
  --attribution_json quickstart/outputs/logit_lens.json \
  --input_json quickstart/data/aqua_720_input.json \
  --output_dir quickstart/outputs \
  --method_name logit_lens --top_k 3 --n_variants 4 --perturb_only
```

Output: `quickstart/outputs/_perturbations_logit_lens.json`. For the running example, the top-3 words are `frisbee`, `T-shirt` and `white`, and the variants are

```
What color is the frisbee that the man in the white Jersey is holding?
What color is the frisbee that the man in the snowy T-shirt is holding?
What color is the frisbee that the man in the lily-white T-shirt is holding?
```

Over the 720 questions, 642 get 4 variants, 62 get 1-3, and 16 get none (those are scored on their original answers only). Started from the stored Step 1 output, this step reproduces `quickstart/intermediate/_perturbations_logit_lens.json` exactly.

**Step 3: sampling.** *GPU. Optional: the stored outputs are `quickstart/intermediate/_gen_checkpoint_*.json`.*

The model answers the original question 3 times (1 greedy + 2 sampled at T=0.5, top-p 0.9) and each variant once (greedy), with `max_new_tokens=128`. All answers go into one pool. The baseline instead answers the unperturbed question 10 times (1 greedy + 9 sampled).

```bash
# our detector: original + perturbed prompts
python whitebox/pipeline/pipeline_b_perturb_and_uq_vl.py \
  --perturbations_json quickstart/intermediate/_perturbations_logit_lens.json \
  --input_json quickstart/data/aqua_720_input.json \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir quickstart/outputs \
  --method_name logit_lens --n_sampled 2 --temperature 0.5 --max_new_tokens 128 \
  --backend hf --device cuda:0 --gen_only

# EigV(Agree) baseline: 10 answers to the unperturbed prompt
python whitebox/pipeline/pipeline_b_perturb_and_uq_vl.py \
  --baseline --baseline_n_samples 10 \
  --input_json quickstart/data/aqua_720_input.json \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir quickstart/outputs \
  --temperature 0.5 --max_new_tokens 128 --backend hf --device cuda:0 --gen_only
```

Use `--perturbations_json quickstart/outputs/_perturbations_logit_lens.json` to continue from your own Step 2. Output: `quickstart/outputs/_gen_checkpoint_logit_lens.json` and `_gen_checkpoint_baseline_n10.json`. The running example has 3 variants, so it gets 6 answers, e.g., *"The frisbee that the man in the white T-shirt is holding is white."*

**Step 4: uncertainty from answer agreement.** *GPU recommended; on a CPU it takes about 45 minutes per score file.*

`microsoft/deberta-large-mnli` scores entailment between every ordered pair of answers in a question's pool, giving an agreement matrix A. EigV(Agree) = 1 + Σ max(0, 1 − λ) over the eigenvalues λ of the normalised graph Laplacian of A, built on the off-diagonal entailments (`--eigv_variant zero_diag`, used for the paper's vision numbers; the text numbers use `polygraph`, see [`whitebox/README.md`](whitebox/README.md)). One semantic cluster of answers gives EigV = 2, and every additional cluster raises it. A higher value means the question is more likely ambiguous.

```bash
python whitebox/pipeline/compute_uq_from_checkpoint.py \
  --checkpoint quickstart/intermediate/_gen_checkpoint_logit_lens.json \
  --output quickstart/outputs/pipeline_b_logit_lens.json \
  --input_json quickstart/data/aqua_720_input.json \
  --method logit_lens --eigv_variant zero_diag --device cuda:0

python whitebox/pipeline/compute_uq_from_checkpoint.py \
  --checkpoint quickstart/intermediate/_gen_checkpoint_baseline_n10.json \
  --output quickstart/outputs/pipeline_b_baseline_n10.json \
  --input_json quickstart/data/aqua_720_input.json \
  --method baseline --eigv_variant zero_diag --device cuda:0
```

Use `--checkpoint quickstart/outputs/_gen_checkpoint_*.json` to continue from your own Step 3, and `--device cpu` without a GPU. Output: `per_question` records with `EigV_entail` (the score we use), `EigV_disagree`, `DegMat_entail`, `NumSemSets`, `SemEntropy` and the answers. The running example's 6 answers all agree, so its `EigV_entail` is 2.0, the minimum.

**Step 5: ambiguity-detection AUROC.** *CPU, seconds.*

Binary AUROC with *ambiguous* as the positive class and `EigV_entail` as the score:

```bash
python whitebox/pipeline/compute_auroc.py \
  quickstart/intermediate/pipeline_b_logit_lens.json \
  quickstart/intermediate/pipeline_b_baseline_n10.json \
  --metrics EigV_entail --output quickstart/outputs/auroc.json
```

```
== pipeline_b_logit_lens.json
  EigV_entail     AUROC=0.7083   (scores rounded to 6 dp: 0.7148)   n=720
== pipeline_b_baseline_n10.json
  EigV_entail     AUROC=0.6435   (scores rounded to 6 dp: 0.6520)   n=720
```

To score your own Step 4 output, pass `quickstart/outputs/pipeline_b_*.json` instead.

**All steps in one command.** `quickstart/run_quickstart.sh` chains the steps above. It starts at the given stage and continues to Step 5:

```bash
bash quickstart/run_quickstart.sh              # Step 5 only (= make quick-eval)
bash quickstart/run_quickstart.sh uq           # Steps 4-5 from the stored answers
bash quickstart/run_quickstart.sh generate     # Steps 3-5 from the stored perturbations (baseline answers stay the stored ones)
bash quickstart/run_quickstart.sh perturb      # Steps 2-5 from the stored attribution
bash quickstart/run_quickstart.sh attribution  # Steps 1-5 from scratch
```

### Option B: run each step with the `input-ambiguity` CLI

```bash
pip install input-ambiguity-detection
```

Step 5 on the scores shipped with the package needs nothing else, not even a clone:

```bash
input-ambiguity quickstart
```

The other steps read the quick-start data, so run them from a clone of this repository. Every script has a CLI command that takes the same arguments as in Option A:

```bash
# Step 1: token importance (GPU)
input-ambiguity wb-vision-logit-lens \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --input_json quickstart/data/aqua_720_input.json \
  --output_json quickstart/outputs/logit_lens.json

# Step 2: perturbation (CPU)
input-ambiguity wb-perturb-vl \
  --attribution_json quickstart/outputs/logit_lens.json \
  --input_json quickstart/data/aqua_720_input.json \
  --output_dir quickstart/outputs \
  --method_name logit_lens --top_k 3 --n_variants 4 --perturb_only

# Step 3: sampling (GPU)
input-ambiguity wb-perturb-vl \
  --perturbations_json quickstart/outputs/_perturbations_logit_lens.json \
  --input_json quickstart/data/aqua_720_input.json \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir quickstart/outputs \
  --method_name logit_lens --n_sampled 2 --temperature 0.5 --max_new_tokens 128 \
  --backend hf --device cuda:0 --gen_only

# Step 4: NLI uncertainty
input-ambiguity wb-uq \
  --checkpoint quickstart/outputs/_gen_checkpoint_logit_lens.json \
  --output quickstart/outputs/pipeline_b_logit_lens.json \
  --input_json quickstart/data/aqua_720_input.json \
  --method logit_lens --eigv_variant zero_diag --device cuda:0

# Step 5: AUROC
input-ambiguity auroc quickstart/outputs/pipeline_b_logit_lens.json \
  quickstart/intermediate/pipeline_b_baseline_n10.json --metrics EigV_entail
```

`input-ambiguity --help` lists every command.

### Expected results

| Setting | Method | Source | AUROC | Command |
| --- | --- | --- | --- | --- |
| AQuA + Qwen2.5-VL-3B | Ours (White-Box, Logit Lens) | Paper Table 1 | 0.708 | Paper reference. |
| AQuA + Qwen2.5-VL-3B | Ours (White-Box, Logit Lens) | Included quick-start artifact | 0.7083 | Step 5, or `input-ambiguity quickstart` |
| AQuA + Qwen2.5-VL-3B | EigV(Agree) baseline | Paper Table 1 | 0.644 | Paper reference. |
| AQuA + Qwen2.5-VL-3B | EigV(Agree) baseline | Included quick-start artifact | 0.6435 | Step 5, or `input-ambiguity quickstart` |

The included artifacts are the files used for the paper, so they match Table 1 up to display rounding. Re-running a step can move the result slightly:

- **Step 4:** per-question scores match the stored ones to within about 2e-6. AUROC can still shift slightly (0.7096 on an L40S GPU, 0.7150 on a CPU, vs 0.7083 stored), because many questions tie in exact arithmetic (for example, all answers identical) and float noise decides their order. With scores rounded to 6 decimals, stored and recomputed scores all give the same AUROC, 0.7148.
- **Step 3:** greedy answers match the stored ones; the 2 sampled answers per question are random, so AUROC varies from run to run.
- **Step 1:** with the library versions in `requirements.txt`, the word lists are identical but normalised scores can differ by up to about 0.05. This changes the top-3 words for some questions.

More detail is in [`quickstart/README.md`](quickstart/README.md).

### Black-box detector, step by step

There are no stored black-box outputs. The steps below are the full black-box pipeline on the 200-question ABG-CoQA set; Step 1 runs on a CPU, Steps 2-3 need a GPU (vLLM, DeBERTa).

```bash
# Step 1: rule-based variants only, no model (CPU, seconds)
#   200 questions -> 1,863 variants from the 11 rules (about 9 per question)
python blackbox/run_perturbation.py --dry_run \
  --input_path data/abg_coqa_200_blackbox.json --output_path results/blackbox_demo/variants.json

# Step 2: variants + original -> N sampled answers each (vLLM)
python blackbox/run_perturbation.py --input_path data/abg_coqa_200_blackbox.json \
  --output_path results/blackbox_demo/results.json --model Qwen/Qwen3-4B-Instruct-2507 --max_examples -1

# Step 3: NLI uncertainty per variant and pooled over all variants
python blackbox/compute_uq.py --input results/blackbox_demo/results.json \
  --output results/blackbox_demo/uq.json --device cuda:0

# Step 4: AUROC per rule and pooled (the CROSS column is the black-box detector)
python blackbox/analyze_uq.py --uq results/blackbox_demo/uq.json \
  --results results/blackbox_demo/results.json --output_dir results/blackbox_demo/analysis
```

The CLI equivalents are `input-ambiguity bb-perturb`, `bb-uq` and `bb-analyze`. `scripts/run_blackbox.sh` runs Steps 2-4 in one command. See [`blackbox/README.md`](blackbox/README.md) for the rule library and the outputs.

## Running the detectors

**White-box**: attribution → top-3 words → WordNet perturbation → sampling → NLI UQ → AUROC, for all three importance methods and the K=10 EigV(Agree) baseline:

```bash
bash scripts/run_whitebox_text.sh   Qwen/Qwen3-4B-Instruct-2507      data/abg_coqa_200_input.json        results/qwen3_abg
bash scripts/run_whitebox_text.sh   meta-llama/Llama-3.1-8B-Instruct data/ambigqa_with_ctx_500_input.json results/llama_ambigqa
bash scripts/run_whitebox_vision.sh Qwen/Qwen2.5-VL-3B-Instruct      data/aqua_720_input.json            results/qwenvl_aqua
bash scripts/run_whitebox_vision.sh OpenGVLab/InternVL3-8B-hf        data/vqfa_720_input.json            results/internvl_vqfa
```

The vision inputs need the AQuA / VQ-FA images: `python data/build/build_aqua_input.py` downloads AQuA, and `VQFA_ROOT=... python data/build/build_vqfa_input.py` builds VQ-FA from its release. Settings, output files, the two EigV variants, and memory notes are documented in [`whitebox/README.md`](whitebox/README.md).

**Black-box**: 11 rules → N samples per variant (vLLM) → NLI UQ per variant and pooled → AUROC per rule and pooled:

```bash
bash scripts/run_blackbox.sh Qwen/Qwen3-4B-Instruct-2507      data/abg_coqa_200_blackbox.json results/blackbox_qwen3_abg
bash scripts/run_blackbox.sh meta-llama/Llama-3.1-8B-Instruct data/abg_coqa_200_blackbox.json results/blackbox_llama_abg
```

The detector's score is the cross-variant (`CROSS`) `EigValLaplacian_NLI_entail` AUROC in `results/.../analysis/uq_statistics.json`; the per-rule AUROCs correspond to Figure 3. The LLM-executed-rules ablation (Table 4) and the LLM-paraphrase baseline are in [`blackbox/README.md`](blackbox/README.md).

## Data

| dataset | type | items (amb / unamb) | source |
|---|---|---|---|
| ABG-CoQA | conversational QA | 200 (100 / 100) | [Guo et al., 2021](https://github.com/MeiqiGuo/AKBC2021-Abg-CoQA) |
| AmbigQA-w-Ctx | open-domain QA + Wikipedia passage | 500 (250 / 250) | [Min et al., 2020](https://huggingface.co/datasets/sewon/ambig_qa) |
| AQuA | visual QA | 720 (540 / 180) | [Jang and Kim, 2026](https://huggingface.co/datasets/jihyoung/AQuA) |
| VQ-FA | visual QA (focus ambiguity) | 720 (422 / 298) | Chen et al., 2025 (VQ-FocusAmbiguity release) |

All question files are included in `data/`, and `data/build/` holds the scripts that created them. Please also follow the licenses of the original datasets.

## Hardware

The quick-start AUROC and the unit tests run on a CPU. Our experiments ran on a single NVIDIA A100 80 GB. Gemma-3-27B's DiffMask attribution uses INT8 weights through bitsandbytes, which `scripts/run_whitebox_text.sh` turns on automatically for 27B models.

## Tests

```bash
pip install -e ".[dev]"
make check        # compileall + pytest + quick-start AUROC
make dist-check   # build the PyPI sdist / wheel and run twine check
```

The tests cover the black-box rule library, the shared prompt/data helpers, the EigV implementations, and the quick-start AUROC against the paper's numbers.

## Citation

```bibtex
@inproceedings{chen2026detect,
  title     = {Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in {LLMs}},
  author    = {Chen, Tiejin and Liu, Xiaoou and Clifford, Michael and Moradipari, Ahmadreza and
               Chen, Yitao and Han, Kyungtae and Ammar, Nejib and Wei, Hua},
  booktitle = {Proceedings of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL)},
  year      = {2026}
}
```

## License

MIT License, see [LICENSE](LICENSE).
