# Black-box input-ambiguity detection

This is the code for the **black-box detector** in *Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs*. The detector only queries the model for text; it uses no gradients, hidden states or token probabilities.

1. **Perturbation.** A fixed library of 11 semantics-blind rules ([`perturbation_rules.py`](perturbation_rules.py)) rewrites the question. Each rule fires on surface features only (POS tag, orthographic form, leading wh-word) and never consults a model reading of the question.
2. **Sampling.** The model answers the original question and every variant N times (T=0.5, top-p 0.9) with vLLM.
3. **Scoring.** `microsoft/deberta-large-mnli` builds an NLI agreement matrix. EigV(Agree) over the answers **pooled from the original question and all of its variants** (the *cross-variant* score) is the ambiguity score.
4. **Evaluation.** Binary AUROC against the ambiguity labels, per rule and for the pooled score.

All commands below are run from the repository root.

## Rule library

| id | rule | example (actual output) |
|---|---|---|
| A1 | Verb synonym (WordNet, re-inflected) | Who wrote the book? → Who penned the book ? |
| A2 | Noun synonym (number preserved) | When was the building completed? → When was the edifice completed ? |
| A3 | Adjective synonym (degree preserved) | What is the largest city? → What is the biggest city ? |
| A4 | Adverb synonym | How quickly does it grow? → How rapidly does it grow ? |
| A5 | Contraction expand | Who's the author? → Who is the author? |
| A6 | Contraction contract | Who is the author? → Who's the author? |
| B1 | Test-like prefix | Who won? → According to the passage, who won? |
| B2 | Conversation-like prefix | Who won? → Hey so, who won? |
| B3 | Instruction prefix | Who won? → Based on the context above, who won? |
| B4 | WH expansion | Who won? → Which person won? |
| B5 | WH rephrase | How did it happen? → By what means did it happen? |

Synonyms come from the first (most frequent) WordNet synset. Multi-word, very short, length-mismatched and rare candidates are dropped. A1-A4 re-join the spaCy tokens with spaces (hence `book ?`); this is the behaviour used for the paper's numbers. B4/B5 do not fire when the wh-word is a determiner (*What color was it?*); in that case only the *what → which* swap is applied. With the default `--max_variants_per_rule 2`, each rule contributes at most two variants per matching word. Try the rules on a few questions with:

```bash
python blackbox/perturbation_rules.py
```

## Files

| file | role |
|---|---|
| `perturbation_rules.py` | the rule library (`generate_all_variants`) |
| `run_perturbation.py` | **our black-box detector**: rules → vLLM sampling → `results.json` (`--dry_run` only writes the variants) |
| `compute_uq.py` | NLI matrices → per-variant and cross-variant UQ (EigV, DegMat, Eccentricity, NumSemSets, Semantic Entropy, lexical similarity) via lm-polygraph |
| `analyze_uq.py` | AUROC / Cohen's d per rule and for the pooled score, plots, `uq_statistics.json` |
| `run_llm_perturbation.py` | ablation (Table 4): the same 11 rules executed by an LLM rewriter instead of code |
| `run_paraphrase.py` | baseline: free LLM paraphrasing of the question |
| `llm_utils.py` | shared chat templates, QA prompt, vLLM loading and I/O |

`run_llm_perturbation.py` and `run_paraphrase.py` write the same `results.json` schema, so `compute_uq.py` and `analyze_uq.py` work on their outputs unchanged.

## Setup

```bash
pip install -r requirements.txt                      # from a clone, or: pip install "input-ambiguity-detection[blackbox]"
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
export HF_CACHE_DIR=/path/to/hf_cache     # optional; HF default cache otherwise
export HF_TOKEN=...                       # needed for meta-llama/Llama-3.1-8B-Instruct and google/gemma-3-*
```

spaCy (`en_core_web_sm`) provides the POS tags for rules A1-A4. Without it, `perturbation_rules.py` falls back to a much cruder suffix heuristic, and the variants will differ from the paper's.

## Data

The scripts read the original ABG-CoQA record format:

```json
{"data": [{"id": "...", "story": "...", "history_turns": [{"question": "...", "answer": "..."}],
           "target_turn": {"question": "...", "answer": "..."},
           "ambiguity": "ambiguous" | "non_ambiguous", "clarification_turn": {}, "source": "..."}]}
```

`data/abg_coqa_200_blackbox.json` is the 200-question ABG-CoQA subset used in the paper (the same questions as the white-box `data/abg_coqa_200_input.json`; ids are shared). It was built from the [official test split](https://github.com/MeiqiGuo/AKBC2021-Abg-CoQA) with:

```bash
python data/build/build_abg_coqa_blackbox.py
```

## Running

```bash
bash scripts/run_blackbox.sh Qwen/Qwen3-4B-Instruct-2507       data/abg_coqa_200_blackbox.json results/blackbox_qwen3_abg
bash scripts/run_blackbox.sh meta-llama/Llama-3.1-8B-Instruct data/abg_coqa_200_blackbox.json results/blackbox_llama_abg
bash scripts/run_blackbox.sh google/gemma-3-27b-it            data/abg_coqa_200_blackbox.json results/blackbox_gemma27_abg
```

Env vars: `PYTHON`, `DEVICE` (NLI model, default `cuda:0`), `NUM_SAMPLES` (default 10), `MAX_TOKENS` (default 256), `TP` (vLLM tensor parallelism), `RULES` (e.g. `"A1 A2 B1"`; default all 11). Steps whose output exists are skipped.

Files written to `OUT_DIR`:

| file | content |
|---|---|
| `results.json` | every variant of every question with its N answers |
| `uq.json` | per question: `per_variant` scores, `cross_variant` scores and their `gap` |
| `results_nli_*_K*.npz` | cached NLI matrices (re-used if `compute_uq.py` is re-run) |
| `analysis/uq_statistics.json` | AUROC, Cohen's d and Mann-Whitney p per (method, rule) |
| `analysis/*.png` | AUROC bars / heatmap, ROC curves, violin plots, effect-size heatmap |

The paper's black-box number is the **`CROSS`** column of **`EigValLaplacian_NLI_entail`** (EigV(Agree) over the pooled answers). The per-rule columns correspond to Figure 3.

The LLM-based ablations use the same input file:

```bash
# Table 4: the same rules, executed by an LLM rewriter
python blackbox/run_llm_perturbation.py --input_path data/abg_coqa_200_blackbox.json \
    --output_path results/llm_rules_qwen3_abg/results.json --model Qwen/Qwen3-4B-Instruct-2507 --max_examples -1
# LLM-paraphrase baseline
python blackbox/run_paraphrase.py --input_path data/abg_coqa_200_blackbox.json \
    --output_path results/paraphrase_qwen3_abg/results.json --model Qwen/Qwen3-4B-Instruct-2507 --max_examples -1
# then score them as above
python blackbox/compute_uq.py --input results/llm_rules_qwen3_abg/results.json --output results/llm_rules_qwen3_abg/uq.json
python blackbox/analyze_uq.py --uq results/llm_rules_qwen3_abg/uq.json \
    --results results/llm_rules_qwen3_abg/results.json --output_dir results/llm_rules_qwen3_abg/analysis
```
