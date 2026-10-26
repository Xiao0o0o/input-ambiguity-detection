# Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs

[![PyPI](https://img.shields.io/pypi/v/input-ambiguity-detection.svg)](https://pypi.org/project/input-ambiguity-detection/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Xiao0o0o/input-ambiguity-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Xiao0o0o/input-ambiguity-detection/actions/workflows/ci.yml)

Code and data for the AACL 2026 paper **"Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs"** by Tiejin Chen\*, Xiaoou Liu\*, Michael Clifford, Ahmadreza Moradipari, Yitao Chen, Kyungtae Han, Nejib Ammar and Hua Wei (\*equal contribution).

## Overview

LLMs often answer an ambiguous question as if it had a single meaning. A common way to detect ambiguity is to have an LLM paraphrase or clarify the question and measure how much the answers disagree. But to rephrase a question the model must first understand it, and understanding an ambiguous question means picking one reading. The rewrite can remove the very ambiguity it is meant to measure.

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

From source, with the data, quick start and shell scripts (recommended to reproduce the paper):

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

The quick start is one paper run: the white-box detector on AQuA with Qwen2.5-VL-3B-Instruct. Its per-question scores are stored, so the paper's AUROC can be checked on a CPU in seconds, in either of two ways.

**Option A: with the PyPI package** (no clone, no GPU, no model download; the scores ship with the package)

```bash
pip install input-ambiguity-detection
input-ambiguity quickstart
```

**Option B: with the code in this repository**

```bash
git clone https://github.com/Xiao0o0o/input-ambiguity-detection.git
cd input-ambiguity-detection
pip install -r requirements.txt
bash quickstart/run_quickstart.sh          # or: make quick-eval
```

| setting | |
|---|---|
| model | Qwen2.5-VL-3B-Instruct |
| dataset | AQuA, 720 visual questions (540 ambiguous / 180 unambiguous) |
| detector | white-box, Logit Lens importance, top-3 words → WordNet synonyms (up to 4 variants) |
| sampling | original: 1 greedy + 2 sampled (T=0.5, top-p 0.9); each variant: 1 greedy |
| score | EigV(Agree), `microsoft/deberta-large-mnli` |

| method | quick-start AUROC | paper (Table 1) |
|---|---|---|
| EigV(Agree) baseline (K=10 samples, no perturbation) | 0.6435 | 0.644 |
| **Ours (White-Box, Logit Lens)** | **0.7083** | **0.708** |

Both options give the numbers above, which match the paper. Option B also stores every intermediate file of the run (attribution, perturbations, answers), and the same script resumes from any earlier stage: `uq` (recompute NLI from the stored answers), `generate`, `perturb` or `attribution` (from scratch), e.g. `bash quickstart/run_quickstart.sh uq`. Expected deviations when re-running those stages are listed in [`quickstart/README.md`](quickstart/README.md).

To see what the black-box rules do to a question, without a model:

```bash
python blackbox/perturbation_rules.py
python blackbox/run_perturbation.py --dry_run --input_path data/abg_coqa_200_blackbox.json --output_path variants.json
```

## Running the detectors

**White-box**: attribution → top-3 words → WordNet perturbation → sampling → NLI UQ → AUROC, for all three importance methods and the K=10 EigV(Agree) baseline:

```bash
bash scripts/run_whitebox_text.sh   Qwen/Qwen3-4B-Instruct-2507      data/abg_coqa_200_input.json        results/qwen3_abg
bash scripts/run_whitebox_text.sh   meta-llama/Llama-3.1-8B-Instruct data/ambigqa_with_ctx_500_input.json results/llama_ambigqa
bash scripts/run_whitebox_vision.sh Qwen/Qwen2.5-VL-3B-Instruct      data/aqua_720_input.json            results/qwenvl_aqua
bash scripts/run_whitebox_vision.sh OpenGVLab/InternVL3-8B-hf        data/vqfa_720_input.json            results/internvl_vqfa
```

The vision inputs need the AQuA / VQ-FA images: `python data/build/build_aqua_input.py` downloads AQuA, and `VQFA_ROOT=... python data/build/build_vqfa_input.py` builds VQ-FA from its release. Settings, output files, the two EigV variants and memory notes are documented in [`whitebox/README.md`](whitebox/README.md).

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
