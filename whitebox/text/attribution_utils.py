#!/usr/bin/env python3
"""
attribution_utils.py

Shared utilities for all internal attribution methods (Tasks 17a-17k).
Handles model loading, device selection, data I/O, token-word alignment,
and visualization helpers.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Device Selection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """Select best available device: XPU > CUDA > CPU."""
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if torch.xpu.is_available():
            print("[device] XPU (Intel)")
            return "xpu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        print("[device] CUDA")
        return "cuda"
    print("[device] CPU")
    return "cpu"


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_path: str,
    device: str,
    cache_dir: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer.

    If dtype is None, picks bf16 for Gemma (fp16 produces NaN in attention/LM heads)
    and fp16 otherwise.

    """
    if dtype is None:
        dtype = torch.bfloat16 if "gemma" in model_path.lower() else torch.float16
    print(f"[model] Loading {model_path} on {device} ({dtype})"
          + (f" attn_impl={attn_implementation}" if attn_implementation else ""))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    # Opt-in INT8 quantization via env var FORCE_INT8=1, for memory-heavy
    # attribution methods (diffmask) on very large models (>20B) where bf16
    # weights+activations don't fit even with freeze+grad_ckpt.
    use_int8 = os.environ.get("FORCE_INT8", "0") == "1"
    if use_int8:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": device}
        print(f"[model] Using INT8 quantization (FORCE_INT8=1)")
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
    else:
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    # Enable gradient checkpointing for very large models — reduces activation
    # memory ~3x at ~25% speed cost. Always good on >10B with grad-based attrib.
    if n_params > 10:
        try:
            model.gradient_checkpointing_enable()
            print(f"[model] gradient_checkpointing enabled (params={n_params:.2f}B)")
        except Exception as e:
            print(f"[model] gradient_checkpointing_enable failed: {e}")
    print(f"[model] Loaded. Parameters: {n_params:.2f}B")
    return model, tokenizer


# ── Token-Word Alignment ─────────────────────────────────────────────────────

def token_to_word_spans(
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    text: str,
) -> List[Dict[str, Any]]:
    """
    Map each token index to its character span and word in the original text.
    Returns list of dicts: [{token_idx, token_str, char_start, char_end, word}, ...]
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    spans = []
    offset = 0
    for idx, tok_str in enumerate(tokens):
        decoded = tokenizer.decode(input_ids.squeeze()[idx].item())
        # Find in text
        pos = text.find(decoded.strip(), offset) if decoded.strip() else -1
        if pos >= 0:
            spans.append({
                "token_idx": idx,
                "token_str": tok_str,
                "char_start": pos,
                "char_end": pos + len(decoded.strip()),
                "word": decoded.strip(),
            })
            offset = pos + len(decoded.strip())
        else:
            spans.append({
                "token_idx": idx,
                "token_str": tok_str,
                "char_start": -1,
                "char_end": -1,
                "word": decoded.strip(),
            })
    return spans


def aggregate_subword_scores(
    token_scores: np.ndarray,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
) -> List[Dict[str, Any]]:
    """
    Merge subword token scores into word-level scores by summing.
    Returns list of {word: str, score: float, token_indices: list}.
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    words = []
    current_word_tokens = []
    current_word_indices = []

    for idx, tok in enumerate(tokens):
        # Detect subword continuation (##, Ġ, ▁ patterns)
        is_continuation = (
            tok.startswith("##")
            or (not tok.startswith("▁") and not tok.startswith("Ġ")
                and idx > 0 and not tokens[idx - 1] in tokenizer.all_special_tokens
                and tok not in tokenizer.all_special_tokens
                and len(current_word_tokens) > 0
                and not tok.startswith("<"))
        )

        if is_continuation and len(current_word_tokens) > 0:
            current_word_tokens.append(tok)
            current_word_indices.append(idx)
        else:
            if current_word_tokens:
                word_str = tokenizer.decode(
                    input_ids.squeeze()[current_word_indices].tolist()
                ).strip()
                word_score = sum(token_scores[i] for i in current_word_indices)
                words.append({
                    "word": word_str,
                    "score": float(word_score),
                    "token_indices": current_word_indices,
                })
            current_word_tokens = [tok]
            current_word_indices = [idx]

    # Flush last word
    if current_word_tokens:
        word_str = tokenizer.decode(
            input_ids.squeeze()[current_word_indices].tolist()
        ).strip()
        word_score = sum(token_scores[i] for i in current_word_indices)
        words.append({
            "word": word_str,
            "score": float(word_score),
            "token_indices": current_word_indices,
        })

    return words


# ── Score Normalization ───────────────────────────────────────────────────────

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] via min-max scaling."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ── Data I/O ──────────────────────────────────────────────────────────────────

def load_input_data(
    input_json: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Load SAUP-format JSON data."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit and isinstance(data, dict):
        keys = list(data.keys())[:limit]
        data = {k: data[k] for k in keys}
    return data


def save_attribution_results(
    results: Dict[str, Any],
    output_json: str,
) -> None:
    """Save attribution results to JSON, creating directories as needed."""
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[saved] {output_json}")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Tokenization Helpers ─────────────────────────────────────────────────────

def tokenize_question(
    question: str,
    tokenizer: AutoTokenizer,
    device: str,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Tokenize a question string for model input."""
    enc = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return {k: v.to(device) for k, v in enc.items()}


def build_chat_input(
    question: str,
    tokenizer: AutoTokenizer,
    device: str,
    system_prompt: Optional[str] = None,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Build chat-template formatted input."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    else:
        enc = tokenizer(question, return_tensors="pt", truncation=True, max_length=max_length)

    return {k: v.to(device) for k, v in enc.items()}


# ── Visualization Helpers ─────────────────────────────────────────────────────

def render_token_heatmap_html(
    tokens: List[str],
    scores: np.ndarray,
    title: str = "Token Attribution",
) -> str:
    """Render a simple HTML heatmap of token-level scores."""
    norm = normalize_scores(scores)
    html_parts = [f"<h3>{title}</h3><p style='font-family:monospace;line-height:2;'>"]
    for tok, s in zip(tokens, norm):
        r = int(255 * s)
        g = int(255 * (1 - s))
        color = f"rgb({r},{g},100)"
        escaped = tok.replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f'<span style="background-color:{color};padding:2px 4px;margin:1px;'
            f'border-radius:3px;" title="{s:.3f}">{escaped}</span> '
        )
    html_parts.append("</p>")
    return "".join(html_parts)


def print_token_scores(
    tokens: List[str],
    scores: np.ndarray,
    top_k: int = 10,
    title: str = "Token Attribution",
) -> None:
    """Print top-k tokens by attribution score."""
    indices = np.argsort(-scores)[:top_k]
    print(f"\n{'='*60}")
    print(f"  {title} (top {top_k})")
    print(f"{'='*60}")
    for rank, idx in enumerate(indices):
        print(f"  {rank+1:3d}. [{idx:3d}] {tokens[idx]:20s}  score={scores[idx]:.6f}")
    print()


# ── Spearman Rank Correlation ─────────────────────────────────────────────────

def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two arrays."""
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else 0.0
