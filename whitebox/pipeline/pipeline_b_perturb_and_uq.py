#!/usr/bin/env python3
"""
Pipeline B: Token Importance (Explanation) → Input Uncertainty (Importance as Method)

Steps:
  1. Load attribution results → extract top-k important WORD tokens
  2. Perturb those words using WordNet synonyms (rule-based)
  3. Generate answers for original + perturbed questions using LLM
  4. Pool ALL answers together → compute UQ (EigValLaplacian_entail, DegMat_entail)
  5. Evaluate AUROC for ambiguity prediction

Usage:
    python pipeline_b_perturb_and_uq.py \
        --attribution_json results/attention_rollout_Qwen2.5-7B-Instruct.json \
        --input_json abg_coqa_attribution_input.json \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --top_k 3 \
        --n_variants 4
"""
import argparse
import json
import os
import re
import pickle
import time
import gc
import sys
import types
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# --- Patch for lm_polygraph compatibility with newer transformers ---
# transformers >=4.50 removed beam_search module + several generation outputs;
# lm_polygraph still imports them. Inject stubs.
try:
    from transformers.generation.beam_search import BeamScorer  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _stub = types.ModuleType("transformers.generation.beam_search")
    class _BeamScorer: pass
    class _BeamSearchScorer(_BeamScorer): pass
    _stub.BeamScorer = _BeamScorer
    _stub.BeamSearchScorer = _BeamSearchScorer
    sys.modules["transformers.generation.beam_search"] = _stub

# Stub deprecated output classes if missing
import transformers.generation.utils as _gu
for _name in ("BeamSearchOutput", "BeamSearchDecoderOnlyOutput",
              "BeamSearchEncoderDecoderOutput", "GreedySearchOutput",
              "GreedySearchDecoderOnlyOutput", "SampleOutput",
              "SampleDecoderOnlyOutput"):
    if not hasattr(_gu, _name):
        setattr(_gu, _name, type(_name, (object,), {}))

# Stub removed top-level transformers symbols
import transformers as _tf
for _name in ("AutoModelForVision2Seq",):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, type(_name, (object,), {}))

import numpy as np
import torch
from tqdm import tqdm


def _llm_dtype(model_path: str):
    """Return bf16 for Gemma (fp16 causes inf/nan in softmax/sampling), else fp16."""
    name = (model_path or "").lower()
    if "gemma" in name:
        return torch.bfloat16
    return torch.float16


# ── WordNet synonym perturbation ─────────────────────────────────────────────

WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from",
    "he", "her", "him", "his", "how", "if", "in", "into", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "with",
}


def _treebank_to_wordnet_pos(tag: str) -> Optional[str]:
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return None


def _wordnet_synonyms(wn, word: str, wn_pos: Optional[str], max_syns: int = 5) -> List[str]:
    out, seen = [], set()
    base = word.lower()
    # Try with POS first, then fall back to all POS
    if wn_pos:
        synsets = wn.synsets(base, pos=wn_pos)
    else:
        synsets = []
    # If no synsets with given POS, try all POS
    if not synsets:
        synsets = wn.synsets(base)
    # Search across multiple synsets (up to 5), not just the first
    for syn in synsets[:5]:
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ").strip().lower()
            if " " in cand or cand == base or cand in seen:
                continue
            if not WORD_TOKEN_RE.fullmatch(cand):
                continue
            if cand in STOPWORDS:
                continue
            seen.add(cand)
            out.append(cand)
            if len(out) >= max_syns:
                return out
    return out


def _apply_pos_surface_form(lemma: str, tb_pos: str) -> str:
    w = lemma
    if tb_pos in {"VBD", "VBN"}:
        if w.endswith("e"):
            return w + "d"
        if len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
            return w[:-1] + "ied"
        return w + "ed"
    if tb_pos == "VBG":
        if w.endswith("ie"):
            return w[:-2] + "ying"
        if w.endswith("e") and len(w) > 2:
            return w[:-1] + "ing"
        return w + "ing"
    if tb_pos == "VBZ":
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        if w.endswith(("s", "x", "z", "ch", "sh", "o")):
            return w + "es"
        return w + "s"
    if tb_pos == "NNS":
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        if w.endswith(("s", "x", "z", "ch", "sh")):
            return w + "es"
        return w + "s"
    return w


def _match_token_case(src: str, tgt: str) -> str:
    if src.isupper():
        return tgt.upper()
    if src and src[0].isupper():
        return tgt.capitalize()
    return tgt


def _normalize_spaces(text: str) -> str:
    return re.sub(r"  +", " ", text).strip()


def get_wordnet_perturbations(
    question: str,
    target_words: List[str],
    n_variants: int = 4,
    max_synonyms_per_token: int = 5,
) -> List[Dict[str, Any]]:
    """Generate perturbations by replacing target words with WordNet synonyms.

    Args:
        question: The full question text.
        target_words: List of important words to perturb (from attribution).
        n_variants: Max number of perturbed variants to generate.
        max_synonyms_per_token: Max synonyms per word from WordNet.

    Returns:
        List of dicts with keys: perturbed_question, changed_word, replacement, tb_pos.
    """
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.tag import pos_tag

    # Ensure NLTK data available
    for resource in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
                     "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"taggers/{resource}" if "tagger" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    # Find all word matches in the question
    matches = list(WORD_TOKEN_RE.finditer(question))
    tokens = [m.group(0) for m in matches]
    tagged = pos_tag(tokens)

    # Build lookup: lowercase word → list of (match_obj, tb_pos)
    word_positions = {}
    for idx, m in enumerate(matches):
        lower = m.group(0).lower()
        if lower not in word_positions:
            word_positions[lower] = []
        word_positions[lower].append((m, tagged[idx][1]))

    perturbations = []
    seen = set()

    for target_word in target_words:
        target_lower = target_word.lower().strip()
        if target_lower not in word_positions:
            continue

        for match_obj, tb_pos in word_positions[target_lower]:
            wn_pos = _treebank_to_wordnet_pos(tb_pos)
            # Pass wn_pos (can be None), _wordnet_synonyms handles fallback
            syns = _wordnet_synonyms(wn, target_lower, wn_pos, max_synonyms_per_token)
            for syn in syns:
                if wn_pos is not None:
                    surfaced = _apply_pos_surface_form(syn, tb_pos)
                else:
                    surfaced = syn  # No POS-based surface form when POS unknown
                new_token = _match_token_case(match_obj.group(0), surfaced)
                new_q = _normalize_spaces(
                    question[:match_obj.start()] + new_token + question[match_obj.end():]
                )
                if new_q == question or new_q in seen:
                    continue
                seen.add(new_q)
                perturbations.append({
                    "perturbed_question": new_q,
                    "changed_word": match_obj.group(0),
                    "replacement": new_token,
                    "tb_pos": tb_pos,
                    "strategy": "wordnet",
                })
                if len(perturbations) >= n_variants:
                    return perturbations

    return perturbations


# ── LLM-based perturbation ──────────────────────────────────────────────────

def _extract_question_part(full_question: str) -> str:
    """Extract the actual question from a full prompt that may include passage + history.

    ABG-CoQA format: 'Read the following passage...\n\nPassage: ...\n\nQuestion: ...'
    Also handles multi-turn: '...Question: Q1\nAnswer: A1\nQuestion: Q2'
    Returns the last question text, or the full string if no pattern found.
    """
    parts = re.split(r'\n(?:Question|Q)\s*:\s*', full_question)
    if len(parts) > 1:
        return parts[-1].strip()
    lines = [l.strip() for l in full_question.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        if line.endswith("?"):
            return line
    return full_question


def _extract_local_context(full_text: str, target_word: str, n_context_sentences: int = 1) -> Tuple[str, int, int]:
    """Extract the sentence containing target_word plus n surrounding sentences.

    Returns (context_str, start_char_idx, end_char_idx) so the caller can
    locate the context window within the full text.
    """
    # Split into sentences (handle period, ?, !, and newline boundaries)
    sentence_spans = []
    for m in re.finditer(r'[^\n.!?]*[.!?\n]', full_text + '\n'):
        s = m.group().strip()
        if s:
            sentence_spans.append((m.start(), m.end(), s))

    if not sentence_spans:
        return full_text, 0, len(full_text)

    # Find which sentence contains the target word (case-insensitive)
    target_lower = target_word.lower()
    target_idx = -1
    for i, (start, end, sent) in enumerate(sentence_spans):
        if re.search(r'\b' + re.escape(target_lower) + r'\b', sent.lower()):
            target_idx = i
            break

    if target_idx == -1:
        # Word not found in any sentence; return a window around the middle
        target_idx = len(sentence_spans) // 2

    lo = max(0, target_idx - n_context_sentences)
    hi = min(len(sentence_spans), target_idx + n_context_sentences + 1)
    start_char = sentence_spans[lo][0]
    end_char = sentence_spans[hi - 1][1]
    context = full_text[start_char:end_char].strip()
    return context, start_char, end_char


def _llm_get_synonym(word: str, context: str, model, tokenizer, device: str,
                     n_synonyms: int = 4, max_new_tokens: int = 32) -> List[str]:
    """Ask the LLM to produce synonyms for a word given its context.

    Returns a list of synonym strings (single words or short phrases).
    """
    prompt = (
        f'Given this context: "{context}"\n\n'
        f'Provide {n_synonyms} synonyms or near-synonyms for the word "{word}" '
        f'that fit naturally in the same position. '
        f'Output ONLY the replacement words, one per line, numbered 1-{n_synonyms}. '
        f'Each replacement should be a single word or at most two words. '
        f'Do not output anything else.'
    )

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse numbered lines
    synonyms = []
    seen = set()
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^\d+[\.\)\:\-]\s*', '', line).strip()
        if not cleaned:
            continue
        # Remove surrounding quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        if not cleaned:
            continue
        # Must be short (1-2 words max)
        if len(cleaned.split()) > 3:
            continue
        # Skip if identical to original word
        if cleaned.lower() == word.lower():
            continue
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        synonyms.append(cleaned)
        if len(synonyms) >= n_synonyms:
            break

    return synonyms


def get_llm_perturbations_local_context(
    question: str,
    target_words: List[str],
    model,
    tokenizer,
    device: str,
    n_variants: int = 4,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """LLM perturbation Approach B: local context.

    For each target word, extract the sentence containing it plus surrounding
    sentences as context. Ask the LLM for synonyms given that context. Then
    programmatically replace the word in the full prompt (like WordNet does).
    """
    # Find all word matches in the question (same as WordNet approach)
    matches = list(WORD_TOKEN_RE.finditer(question))

    # Build lookup: lowercase word -> list of match objects
    word_positions = {}
    for m in matches:
        lower = m.group(0).lower()
        if lower not in word_positions:
            word_positions[lower] = []
        word_positions[lower].append(m)

    perturbations = []
    seen = set()

    for target_word in target_words:
        target_lower = target_word.lower().strip()
        if target_lower not in word_positions:
            continue

        # Get local context for LLM
        context, _, _ = _extract_local_context(question, target_word, n_context_sentences=1)

        # Get synonyms from LLM
        synonyms = _llm_get_synonym(
            target_word, context, model, tokenizer, device,
            n_synonyms=max(4, n_variants), max_new_tokens=max_new_tokens,
        )

        if not synonyms:
            continue

        # Apply each synonym via programmatic replacement (like WordNet)
        for match_obj in word_positions[target_lower]:
            for syn in synonyms:
                new_token = _match_token_case(match_obj.group(0), syn)
                new_q = _normalize_spaces(
                    question[:match_obj.start()] + new_token + question[match_obj.end():]
                )
                if new_q == question or new_q in seen:
                    continue
                seen.add(new_q)
                perturbations.append({
                    "perturbed_question": new_q,
                    "changed_word": match_obj.group(0),
                    "replacement": new_token,
                    "tb_pos": "N/A",
                    "strategy": "llm_local",
                })
                if len(perturbations) >= n_variants:
                    return perturbations

    return perturbations


def get_llm_perturbations_full_prompt(
    question: str,
    target_words: List[str],
    model,
    tokenizer,
    device: str,
    n_variants: int = 4,
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """LLM perturbation Approach A: full prompt context.

    Give the LLM the FULL prompt text and ask it to provide synonyms for each
    target word in context. Then programmatically replace the word in the full
    prompt (NOT sentence rewriting -- word-level replacement only).
    """
    # Find all word matches in the question
    matches = list(WORD_TOKEN_RE.finditer(question))

    word_positions = {}
    for m in matches:
        lower = m.group(0).lower()
        if lower not in word_positions:
            word_positions[lower] = []
        word_positions[lower].append(m)

    perturbations = []
    seen = set()

    for target_word in target_words:
        target_lower = target_word.lower().strip()
        if target_lower not in word_positions:
            continue

        # Ask LLM for synonyms with full prompt as context
        synonyms = _llm_get_synonym(
            target_word, question, model, tokenizer, device,
            n_synonyms=max(4, n_variants), max_new_tokens=max_new_tokens,
        )

        if not synonyms:
            continue

        for match_obj in word_positions[target_lower]:
            for syn in synonyms:
                new_token = _match_token_case(match_obj.group(0), syn)
                new_q = _normalize_spaces(
                    question[:match_obj.start()] + new_token + question[match_obj.end():]
                )
                if new_q == question or new_q in seen:
                    continue
                seen.add(new_q)
                perturbations.append({
                    "perturbed_question": new_q,
                    "changed_word": match_obj.group(0),
                    "replacement": new_token,
                    "tb_pos": "N/A",
                    "strategy": "llm_full",
                })
                if len(perturbations) >= n_variants:
                    return perturbations

    return perturbations


# ── Extract top-k important words from attribution results ───────────────────

def extract_top_k_words(
    attribution_result: Dict[str, Any],
    k: int = 3,
    question_text: str = "",
) -> List[str]:
    """Extract top-k important words from word_scores in attribution results.

    Filters out special tokens, punctuation, template words, and multi-line
    aggregation artifacts. Cleans trailing punctuation from word tokens.
    Only keeps words that actually appear in the question text (if provided).
    """
    import math

    word_scores = attribution_result.get("word_scores", [])
    if not word_scores:
        return []

    # Extended skip list: template words, stopwords, chat tokens
    SKIP_WORDS = {
        # Chat template tokens
        "<|im_start|>", "<|im_end|>", "system", "user", "assistant",
        # Prompt template words
        "read", "following", "passage", "answer", "question", "previous",
        # Common stopwords (not useful for perturbation)
        "the", "and", "a", "an", "of", "in", "to", "is", "are", "was",
        "were", "that", "this", "it", "for", "on", "with", "as", "at",
        "by", "from", "he", "she", "they", "we", "you", "his", "her",
        "its", "their", "my", "your", "our", "who", "what", "where",
        "when", "how", "which", "do", "does", "did", "not", "no", "yes",
        "or", "but", "if", "so", "be", "been", "being", "has", "had",
        "have", "will", "would", "could", "should", "may", "might", "can",
        "shall", "there", "here", "then", "than", "also", "just", "very",
        "too", "more", "most", "some", "any", "all", "each", "every",
        "about", "up", "out", "into", "over", "after", "before",
        "between", "through", "during", "without", "against",
        "", " ", "\n",
    }

    # Filter out entries with NaN scores, then sort by score descending
    valid_entries = []
    for entry in word_scores:
        score = entry.get("score", 0)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            continue
        valid_entries.append(entry)

    if not valid_entries:
        # All scores are NaN (e.g., integrated_gradients, predictive_entropy)
        # Fall back to original order (first few content words)
        valid_entries = word_scores

    sorted_words = sorted(valid_entries, key=lambda x: x.get("score", 0) if isinstance(x.get("score", 0), (int, float)) and not math.isnan(x.get("score", 0)) else 0, reverse=True)

    # Pre-compute lowercase question for matching
    q_lower = question_text.lower() if question_text else ""

    top_words = []
    seen_lower = set()
    for entry in sorted_words:
        word = entry.get("word", "").strip()
        if not word:
            continue

        # Skip multi-line tokens (subword aggregation artifacts)
        if "\n" in word:
            continue

        # Skip tokens containing special markers
        if "<|" in word or "|>" in word:
            continue

        # Clean: strip leading/trailing punctuation but keep internal ones (e.g., hyphenated words)
        clean = re.sub(r'^[^A-Za-z]+', '', word)
        clean = re.sub(r'[^A-Za-z]+$', '', clean)
        if not clean or len(clean) <= 1:
            continue

        if clean.lower() in SKIP_WORDS:
            continue

        # Skip if not a proper word token
        if not WORD_TOKEN_RE.search(clean):
            continue

        # Skip duplicates
        if clean.lower() in seen_lower:
            continue

        # If question text provided, verify the word appears in it
        if q_lower and clean.lower() not in q_lower:
            continue

        seen_lower.add(clean.lower())
        top_words.append(clean)
        if len(top_words) >= k:
            break

    return top_words


# ── LLM Generation ──────────────────────────────────────────────────────────

def generate_answers(
    model,
    tokenizer,
    questions: List[str],
    device: str,
    n_greedy: int = 1,
    n_sampled: int = 4,
    temperature: float = 0.5,
    max_new_tokens: int = 128,
) -> List[str]:
    """Generate answers for a list of questions.

    For each question: 1 greedy + n_sampled stochastic answers.
    Returns flat list of all answers.
    """
    all_answers = []

    for question in questions:
        messages = [{"role": "user", "content": question}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = question

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Greedy generation
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        # Decode only the new tokens
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if answer:
            all_answers.append(answer)

        # Sampled generations
        for _ in range(n_sampled):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if answer:
                all_answers.append(answer)

    return all_answers


# ── UQ Computation ──────────────────────────────────────────────────────────

def compute_uq_metrics(
    answers: List[str],
    deberta,
    sem_calc,
    eigv_est,
    degmat_est,
) -> Dict[str, float]:
    """Compute EigValLaplacian_entail and DegMat_entail from pooled answers."""
    if len(answers) < 2:
        return {"EigV_entail": float("nan"), "DegMat_entail": float("nan")}

    deps = {"sample_texts": [answers]}
    stats = sem_calc(deps, texts=[""], model=None, max_new_tokens=1)
    stats["sample_texts"] = [answers]

    eigv_val = eigv_est(stats)[0]
    degmat_val = degmat_est(stats)[0]

    return {
        "EigV_entail": float(eigv_val),
        "DegMat_entail": float(degmat_val),
    }


# ── AUROC Evaluation ─────────────────────────────────────────────────────────

def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC. labels: 1=ambiguous, 0=unambiguous. scores: UQ values."""
    from sklearn.metrics import roc_auc_score
    valid = ~np.isnan(scores)
    if valid.sum() < 5:
        return float("nan")
    return roc_auc_score(labels[valid], scores[valid])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline B: Perturbation + UQ + AUROC")
    parser.add_argument("--attribution_json", type=str, default=None,
                        help="Path to attribution results JSON (from run_all_attribution.py). "
                             "Required unless --baseline is set.")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to abg_coqa_attribution_input.json")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_CACHE_DIR"))
    parser.add_argument("--output_dir", type=str,
                        default="results")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top important words to perturb")
    parser.add_argument("--n_variants", type=int, default=4, help="Number of perturbed variants per question")
    parser.add_argument("--n_sampled", type=int, default=4, help="Number of sampled generations per question")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--nli_model", type=str, default="microsoft/deberta-large-mnli")
    parser.add_argument("--nli_batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method_name", type=str, default=None,
                        help="Name of the attribution method (auto-detected from filename if omitted)")
    parser.add_argument("--resume_from_gen", type=str, default=None,
                        help="Path to generation checkpoint JSON to skip LLM generation")
    parser.add_argument("--perturbation_strategy", type=str, default="wordnet",
                        choices=["wordnet", "llm_local", "llm_full"],
                        help="Perturbation strategy: 'wordnet' (rule-based), "
                             "'llm_local' (LLM with local context), "
                             "'llm_full' (LLM with full prompt context)")
    parser.add_argument("--scope", type=str, default="whole",
                        choices=["whole", "question_only"],
                        help="Scope for feature importance and perturbation: "
                             "'whole' = full prompt (passage+history+question), "
                             "'question_only' = only the last question part")
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline mode: skip attribution/perturbation, generate "
                             "multiple sampled answers for the original question only, "
                             "then compute UQ and AUROC. This is the proper baseline "
                             "(vanilla UQ without input perturbation).")
    parser.add_argument("--baseline_n_samples", type=int, default=10,
                        help="Number of sampled answers per question in baseline mode "
                             "(default: 10, i.e. 1 greedy + 9 sampled)")
    parser.add_argument("--gen_only", action="store_true",
                        help="Stop after saving the generation checkpoint (compute UQ "
                             "separately with compute_uq_from_checkpoint.py).")

    args = parser.parse_args()

    # Validate: attribution_json is required unless baseline mode
    if not args.baseline and args.attribution_json is None:
        parser.error("--attribution_json is required unless --baseline is set.")

    # Auto-detect method name
    if args.method_name is None:
        if args.baseline:
            args.method_name = "baseline"
        else:
            basename = os.path.basename(args.attribution_json)
            args.method_name = basename.split("_Qwen")[0]

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    print(f"[Pipeline B] method={args.method_name}, top_k={args.top_k}, "
          f"n_variants={args.n_variants}, strategy={args.perturbation_strategy}, "
          f"scope={args.scope}, device={args.device}")

    # ══════════════════════════════════════════════════════════════════════════
    # Baseline mode: no attribution, no perturbation, just multi-sample UQ
    # ══════════════════════════════════════════════════════════════════════════
    if args.baseline:
        print(f"[Baseline Mode] Generating {args.baseline_n_samples} answers per question "
              f"(1 greedy + {args.baseline_n_samples - 1} sampled), no perturbation.")

        # Load input data
        with open(args.input_json, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        if args.limit:
            keys = list(input_data.keys())[:args.limit]
            input_data = {k: input_data[k] for k in keys}
        print(f"   Loaded {len(input_data)} questions")

        # Load LLM
        print("[1/4] Loading LLM...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, cache_dir=args.cache_dir, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, cache_dir=args.cache_dir,
            dtype=_llm_dtype(args.model_path), trust_remote_code=True,
        ).to(args.device).eval()

        # Generate answers
        print("[2/4] Generating answers (original questions only, no perturbation)...")
        generation_data = {}
        for qid, entry in tqdm(input_data.items(), desc="Baseline generation"):
            original_q = entry.get("question", "")
            if not original_q:
                continue
            answers = generate_answers(
                model, tokenizer, [original_q], args.device,
                n_greedy=1, n_sampled=args.baseline_n_samples - 1,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            generation_data[qid] = {
                "all_answers": answers,
                "n_answers": len(answers),
                "ambiguity": entry.get("ambiguity", ""),
            }

        # Save generation checkpoint
        gen_checkpoint = os.path.join(
            args.output_dir,
            f"_gen_checkpoint_baseline_n{args.baseline_n_samples}.json",
        )
        with open(gen_checkpoint, "w", encoding="utf-8") as f:
            json.dump(generation_data, f, ensure_ascii=False)
        print(f"   [checkpoint] Saved to {gen_checkpoint}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if args.gen_only:
            print("[gen_only] done; run compute_uq_from_checkpoint.py for UQ.")
            return

        # Compute UQ
        print("[3/4] Loading NLI model and computing UQ metrics...")
        try:
            from lm_polygraph.utils.deberta import Deberta
            from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator
            from lm_polygraph.estimators import EigValLaplacian, DegMat
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            print(f"   [warn] lm_polygraph unavailable ({type(e).__name__}: {e}).")
            print(f"   [info] Generation checkpoint saved; run compute_uq_from_checkpoint.py to compute UQ.")
            return

        deberta = Deberta(
            deberta_path=args.nli_model, batch_size=args.nli_batch_size,
            device=args.device, hf_cache=args.cache_dir,
        )
        sem_calc = SemanticMatrixCalculator(nli_model=deberta)
        eigv_est = EigValLaplacian()
        degmat_est = DegMat()

        uq_results = {}
        for qid, gdata in tqdm(generation_data.items(), desc="Computing UQ"):
            answers = gdata["all_answers"]
            uq = compute_uq_metrics(answers, deberta, sem_calc, eigv_est, degmat_est)
            uq_results[qid] = {
                **uq,
                "n_answers": len(answers),
                "answers": answers,
                "ambiguity": gdata["ambiguity"],
            }

        del deberta, sem_calc
        gc.collect()
        torch.cuda.empty_cache()

        # Compute AUROC
        print("[4/4] Computing AUROC...")
        labels, eigv_scores, degmat_scores = [], [], []
        for qid, uq in uq_results.items():
            label = 1 if uq["ambiguity"] == "ambiguous" else 0
            labels.append(label)
            eigv_scores.append(uq["EigV_entail"])
            degmat_scores.append(uq["DegMat_entail"])

        labels = np.array(labels)
        eigv_scores = np.array(eigv_scores)
        degmat_scores = np.array(degmat_scores)

        auroc_eigv = compute_auroc(labels, eigv_scores)
        auroc_degmat = compute_auroc(labels, degmat_scores)

        print(f"\n{'='*60}")
        print(f"  Baseline AUROC (no perturbation, {args.baseline_n_samples} samples)")
        print(f"{'='*60}")
        print(f"  EigV_entail:   {auroc_eigv:.4f}")
        print(f"  DegMat_entail: {auroc_degmat:.4f}")
        print(f"  N samples:     {len(labels)} (amb={labels.sum()}, unamb={len(labels)-labels.sum()})")

        # Save results
        outfile = os.path.join(
            args.output_dir,
            f"pipeline_b_baseline_n{args.baseline_n_samples}.json",
        )
        output = {
            "config": {
                "method": "baseline",
                "description": "No attribution, no perturbation. "
                               "Multi-sample generation on original questions only.",
                "model": args.model_path,
                "n_samples": args.baseline_n_samples,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "n_questions": len(uq_results),
                "timestamp": datetime.now().isoformat(),
            },
            "auroc": {
                "EigV_entail": auroc_eigv,
                "DegMat_entail": auroc_degmat,
            },
            "per_question": uq_results,
        }
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2,
                      default=lambda o: float(o) if isinstance(o, (np.floating,)) else
                      int(o) if isinstance(o, (np.integer,)) else
                      o.tolist() if isinstance(o, np.ndarray) else str(o))
        print(f"[saved] {outfile}")
        print("[Baseline] Done.")
        return

    # ── Load attribution results ──
    print("[1/6] Loading attribution results...")
    with open(args.attribution_json, "r", encoding="utf-8") as f:
        attr_data = json.load(f)
    attr_results = attr_data.get("results", attr_data)

    # ── Load original input data ──
    with open(args.input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Apply limit
    if args.limit:
        keys = list(attr_results.keys())[:args.limit]
        attr_results = {k: attr_results[k] for k in keys}

    print(f"   Loaded {len(attr_results)} attribution results")

    # ── If LLM perturbation, load model early (needed for perturbation step) ──
    llm_model = None
    llm_tokenizer = None
    if args.perturbation_strategy in ("llm_local", "llm_full") and not args.resume_from_gen:
        print("[2a/6] Loading LLM for perturbation + generation...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        llm_tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, cache_dir=args.cache_dir, trust_remote_code=True,
        )
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        llm_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            dtype=_llm_dtype(args.model_path),
            trust_remote_code=True,
        ).to(args.device).eval()
        print(f"   Model loaded on {args.device}")

    # ── Step 1: Extract top-k words and generate perturbations ──
    print(f"[2/6] Extracting top-{args.top_k} words and generating perturbations ({args.perturbation_strategy})...")
    perturbation_data = {}
    n_no_perturbation = 0

    for qid, attr_res in tqdm(attr_results.items(), desc="Perturbation"):
        if "error" in attr_res:
            continue

        # Get original question first (needed for word filtering)
        original_q = input_data.get(qid, {}).get("question", "")
        if not original_q:
            continue

        # Determine scope: which text to use for word filtering and perturbation
        if args.scope == "question_only":
            q_part = _extract_question_part(original_q)
            # Filter top-k words to only those appearing in the question part
            top_words = extract_top_k_words(attr_res, k=args.top_k, question_text=q_part)
        else:
            # Whole prompt scope (default)
            top_words = extract_top_k_words(attr_res, k=args.top_k, question_text=original_q)

        if not top_words:
            n_no_perturbation += 1
            continue

        # Perturbation always operates on the full prompt text (original_q)
        # even in question_only scope -- the difference is which words are selected
        if args.perturbation_strategy == "llm_local" and llm_model is not None:
            perturbations = get_llm_perturbations_local_context(
                original_q, top_words,
                model=llm_model,
                tokenizer=llm_tokenizer,
                device=args.device,
                n_variants=args.n_variants,
                max_new_tokens=32,
            )
        elif args.perturbation_strategy == "llm_full" and llm_model is not None:
            perturbations = get_llm_perturbations_full_prompt(
                original_q, top_words,
                model=llm_model,
                tokenizer=llm_tokenizer,
                device=args.device,
                n_variants=args.n_variants,
                max_new_tokens=32,
            )
        else:
            perturbations = get_wordnet_perturbations(
                original_q, top_words,
                n_variants=args.n_variants,
                max_synonyms_per_token=5,
            )

        if not perturbations:
            n_no_perturbation += 1
            # Still include with just the original question
            perturbation_data[qid] = {
                "original_question": original_q,
                "top_words": top_words,
                "perturbations": [],
                "ambiguity": attr_res.get("ambiguity", input_data.get(qid, {}).get("ambiguity", "")),
            }
            continue

        perturbation_data[qid] = {
            "original_question": original_q,
            "top_words": top_words,
            "perturbations": perturbations,
            "ambiguity": attr_res.get("ambiguity", input_data.get(qid, {}).get("ambiguity", "")),
        }

    print(f"   {len(perturbation_data)} questions with perturbation data, "
          f"{n_no_perturbation} without valid perturbations")

    # ── Save perturbation intermediate results ──
    strategy_sfx = f"_{args.perturbation_strategy}" if args.perturbation_strategy != "wordnet" else ""
    scope_sfx = f"_{args.scope}" if args.scope != "whole" else ""
    pert_checkpoint = os.path.join(
        args.output_dir,
        f"_perturbations_{args.method_name}_top{args.top_k}{strategy_sfx}{scope_sfx}.json",
    )
    with open(pert_checkpoint, "w", encoding="utf-8") as f:
        json.dump(perturbation_data, f, ensure_ascii=False, indent=2)
    print(f"   [checkpoint] Saved perturbation data to {pert_checkpoint}")

    # ── Step 2: Load LLM and generate answers (or resume from checkpoint) ──
    if args.resume_from_gen:
        print(f"[3/6] Resuming from generation checkpoint: {args.resume_from_gen}")
        with open(args.resume_from_gen, "r", encoding="utf-8") as f:
            generation_data = json.load(f)
        print(f"   Loaded {len(generation_data)} questions from checkpoint")
    else:
        # Reuse model if already loaded for LLM perturbation, otherwise load fresh
        if llm_model is not None:
            print("[3/6] Reusing LLM (already loaded for perturbation)...")
            model = llm_model
            tokenizer = llm_tokenizer
        else:
            print("[3/6] Loading LLM for answer generation...")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, cache_dir=args.cache_dir, trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                cache_dir=args.cache_dir,
                dtype=_llm_dtype(args.model_path),
                trust_remote_code=True,
            ).to(args.device).eval()

        print(f"   Model loaded on {args.device}")

        generation_data = {}
        for qid, pdata in tqdm(perturbation_data.items(), desc="Generating answers"):
            original_q = pdata["original_question"]
            perturbed_qs = [p["perturbed_question"] for p in pdata["perturbations"]]

            # Strategy: For original question, generate 1 greedy + n_sampled stochastic.
            # For each perturbed question, generate only 1 greedy answer.
            # This keeps the pool size manageable while capturing perturbation-induced variation.
            original_answers = generate_answers(
                model, tokenizer, [original_q], args.device,
                n_greedy=1, n_sampled=args.n_sampled,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

            perturbed_answers = []
            if perturbed_qs:
                perturbed_answers = generate_answers(
                    model, tokenizer, perturbed_qs, args.device,
                    n_greedy=1, n_sampled=0,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

            all_answers = original_answers + perturbed_answers

            generation_data[qid] = {
                "all_answers": all_answers,
                "n_questions": 1 + len(perturbed_qs),
                "n_answers": len(all_answers),
                "ambiguity": pdata["ambiguity"],
                "top_words": pdata["top_words"],
                "n_perturbations": len(perturbed_qs),
            }

        # Free LLM memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"   Generated answers for {len(generation_data)} questions")

        # Save generation checkpoint (reuse strategy_sfx/scope_sfx from perturbation step)
        gen_checkpoint = os.path.join(
            args.output_dir,
            f"_gen_checkpoint_{args.method_name}_top{args.top_k}{strategy_sfx}{scope_sfx}.json",
        )
        with open(gen_checkpoint, "w", encoding="utf-8") as f:
            json.dump(generation_data, f, ensure_ascii=False)
        print(f"   [checkpoint] Saved generation data to {gen_checkpoint}")

    if args.gen_only:
        print("[gen_only] done; run compute_uq_from_checkpoint.py for UQ.")
        return

    # ── Step 3: Compute UQ metrics ──
    print("[4/6] Loading NLI model and computing UQ metrics...")
    try:
        from lm_polygraph.utils.deberta import Deberta
        from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator
        from lm_polygraph.estimators import EigValLaplacian, DegMat
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        print(f"   [warn] lm_polygraph unavailable ({type(e).__name__}: {e}).")
        print(f"   [info] Generation checkpoint saved; run compute_uq_from_checkpoint.py to compute UQ.")
        return

    deberta = Deberta(
        deberta_path=args.nli_model,
        batch_size=args.nli_batch_size,
        device=args.device,
        hf_cache=args.cache_dir,
    )
    sem_calc = SemanticMatrixCalculator(nli_model=deberta)
    eigv_est = EigValLaplacian()
    degmat_est = DegMat()

    uq_results = {}
    for qid, gdata in tqdm(generation_data.items(), desc="Computing UQ"):
        answers = gdata["all_answers"]
        uq = compute_uq_metrics(answers, deberta, sem_calc, eigv_est, degmat_est)
        uq_results[qid] = {
            **uq,
            "n_answers": len(answers),
            "answers": answers,
            "ambiguity": gdata["ambiguity"],
            "top_words": gdata["top_words"],
            "n_perturbations": gdata["n_perturbations"],
        }

    # Free NLI model
    del deberta, sem_calc
    gc.collect()
    torch.cuda.empty_cache()

    # ── Step 4: Compute AUROC ──
    print("[5/6] Computing AUROC...")
    labels = []
    eigv_scores = []
    degmat_scores = []

    for qid, uq in uq_results.items():
        amb = uq["ambiguity"]
        label = 1 if amb == "ambiguous" else 0
        labels.append(label)
        eigv_scores.append(uq["EigV_entail"])
        degmat_scores.append(uq["DegMat_entail"])

    labels = np.array(labels)
    eigv_scores = np.array(eigv_scores)
    degmat_scores = np.array(degmat_scores)

    auroc_eigv = compute_auroc(labels, eigv_scores)
    auroc_degmat = compute_auroc(labels, degmat_scores)

    # Also compute for random baseline
    np.random.seed(42)
    random_scores = np.random.rand(len(labels))
    auroc_random = compute_auroc(labels, random_scores)

    print(f"\n{'='*60}")
    print(f"  AUROC Results: {args.method_name} (top-{args.top_k})")
    print(f"{'='*60}")
    print(f"  EigV_entail:   {auroc_eigv:.4f}")
    print(f"  DegMat_entail: {auroc_degmat:.4f}")
    print(f"  Random:        {auroc_random:.4f}")
    print(f"  N samples:     {len(labels)} (amb={labels.sum()}, unamb={len(labels)-labels.sum()})")
    n_valid_eigv = (~np.isnan(eigv_scores)).sum()
    n_valid_degmat = (~np.isnan(degmat_scores)).sum()
    print(f"  Valid scores:  EigV={n_valid_eigv}, DegMat={n_valid_degmat}")

    # ── Step 5: Save results ──
    print("[6/6] Saving results...")
    output = {
        "config": {
            "method": args.method_name,
            "model": args.model_path,
            "top_k": args.top_k,
            "n_variants": args.n_variants,
            "n_sampled": args.n_sampled,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "perturbation_strategy": args.perturbation_strategy,
            "scope": args.scope,
            "n_questions": len(uq_results),
            "timestamp": datetime.now().isoformat(),
        },
        "auroc": {
            "EigV_entail": auroc_eigv,
            "DegMat_entail": auroc_degmat,
            "random_baseline": auroc_random,
        },
        "per_question": uq_results,
    }

    strategy_suffix = f"_{args.perturbation_strategy}" if args.perturbation_strategy != "wordnet" else ""
    scope_suffix = f"_{args.scope}" if args.scope != "whole" else ""
    outfile = os.path.join(
        args.output_dir,
        f"pipeline_b_{args.method_name}_top{args.top_k}{strategy_suffix}{scope_suffix}.json",
    )
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else
                  int(o) if isinstance(o, (np.integer,)) else
                  o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"[saved] {outfile}")
    print("[Pipeline B] Done.")


if __name__ == "__main__":
    main()
