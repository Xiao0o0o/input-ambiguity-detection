#!/usr/bin/env python3
"""
vl_attribution_utils.py

Shared utilities for VL attribution methods on Qwen2.5-VL-Instruct.
Mirrors the interface of `attribution_utils.py` (text-only) but:
  - loads a Qwen2_5_VLForConditionalGeneration + AutoProcessor pair
  - builds (image, text) inputs via the chat template
  - provides merged `inputs_embeds` (post vision fill-in) for gradient methods
  - identifies text-token span vs image-token span so downstream code
    can (a) perturb only text and (b) aggregate word scores on text only

Used by the adapted attribution scripts in this directory; keeps the
original `Experiment/core_code/*.py` scripts untouched.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        print("[device] CUDA")
        return "cuda"
    print("[device] CPU")
    return "cpu"


# ── Model & Processor loading ─────────────────────────────────────────────────

def load_vl_model_and_processor(
    model_path: str,
    device: str,
    cache_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    print(f"[model] Loading VL model {model_path} on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(
        model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Auto-detect model class so other VL models (Gemma3, Mllama, InternVL) work
    from transformers import AutoModelForImageTextToText
    is_gemma = "gemma" in model_path.lower()
    is_mllama = "llama" in model_path.lower() and ("vision" in model_path.lower() or "3.2" in model_path.lower())
    is_internvl = "internvl" in model_path.lower()
    actual_dtype = torch.bfloat16 if (is_gemma or is_mllama or is_internvl) else dtype
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=actual_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device if device != "cuda" else "auto",
    ).eval()
    # Enable gradient checkpointing for memory-heavy models (avoid OOM in VG/diffmask backward)
    is_large_gemma = is_gemma and any(s in model_path.lower() for s in ("27b", "12b"))
    if is_internvl or is_mllama or is_large_gemma:
        try:
            model.gradient_checkpointing_enable()
            print("[model] gradient checkpointing enabled (saves memory in backward)")
        except Exception as e:
            print(f"[model] gradient checkpointing not enabled: {e}")
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[model] Loaded. Parameters: {n:.2f}B")
    return model, processor


# ── Input building ────────────────────────────────────────────────────────────

def build_vl_inputs(
    image_path: str,
    question_text: str,
    processor: AutoProcessor,
    device: str,
    max_length: int = 4096,
) -> Dict[str, Any]:
    """Build processor inputs for (image, question_text).

    Returns a dict with all processor outputs plus:
      text_span_start / text_span_end : slice into input_ids that covers
          the USER-provided text (between <|vision_end|> and the first
          <|im_end|> of the user turn).
      image_span_start / image_span_end : slice covering the <|image_pad|>
          tokens.
    """
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question_text},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_text],
        images=[img],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Locate image span — try multiple token families; pick the pair that
    # actually appears in input_ids (handles Qwen / InternVL3 / Gemma3 etc).
    tok = processor.tokenizer
    ids = inputs["input_ids"][0]
    def _safe_tok(s):
        tid = tok.convert_tokens_to_ids(s)
        if tid is None or tid < 0:
            return None
        if tok.unk_token_id is not None and tid == tok.unk_token_id:
            return None
        return tid

    candidate_pairs = [
        ("<|vision_start|>", "<|vision_end|>"),
        ("<img>", "</img>"),
        ("<start_of_image>", "<end_of_image>"),
    ]
    vs_id = ve_id = None
    for vs_tok, ve_tok in candidate_pairs:
        a = _safe_tok(vs_tok); b = _safe_tok(ve_tok)
        if a is None or b is None: continue
        # Verify both actually appear in this input
        if (ids == a).any() and (ids == b).any():
            vs_id, ve_id = a, b
            break

    im_end_id = _safe_tok("<|im_end|>") or _safe_tok("<end_of_turn>") or tok.eos_token_id

    vs = (ids == vs_id).nonzero(as_tuple=False).flatten() if vs_id is not None else None
    ve = (ids == ve_id).nonzero(as_tuple=False).flatten() if ve_id is not None else None
    image_span_start = int(vs[0].item() + 1) if (vs is not None and len(vs)) else -1
    image_span_end = int(ve[0].item()) if (ve is not None and len(ve)) else -1

    # Text span: after <|vision_end|>, up to the next <|im_end|>
    text_span_start = image_span_end + 1 if image_span_end > 0 else 0
    ie_after = (ids[text_span_start:] == im_end_id).nonzero(as_tuple=False).flatten()
    if len(ie_after):
        text_span_end = int(text_span_start + ie_after[0].item())
    else:
        text_span_end = int(ids.numel())

    inputs["text_span_start"] = text_span_start
    inputs["text_span_end"] = text_span_end
    inputs["image_span_start"] = image_span_start
    inputs["image_span_end"] = image_span_end
    return inputs


@torch.no_grad()
def get_merged_inputs_embeds(
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: Dict[str, Any],
) -> torch.Tensor:
    """First-pass forward that returns the post-vision-merge inputs_embeds.

    The returned tensor has shape [1, seq_len, hidden_size] and can be
    re-fed into the model as `inputs_embeds=...` (without pixel_values)
    to obtain equivalent logits — which is what we need for
    gradient-based attribution on a combined image+text input.
    """
    fwd = {k: v for k, v in inputs.items() if k in (
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
        "pixel_values_videos", "video_grid_thw",
        # Gemma3 extras
        "token_type_ids",
        # Mllama (Llama-3.2-V) extras
        "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask",
    )}
    out = model(**fwd, output_hidden_states=True, use_cache=False)
    return out.hidden_states[0].detach()


# ── Token-word aggregation (TEXT SPAN ONLY) ───────────────────────────────────

_PUNCT = set(".,;:!?\"'`()[]{}<>/\\|")


def aggregate_subword_scores_text_only(
    token_scores: np.ndarray,
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    text_span_start: int,
    text_span_end: int,
) -> List[Dict[str, Any]]:
    """Merge subword token scores into word-level scores, RESTRICTED to
    the user-text span (skipping image tokens and system prefix).

    A word break is triggered by:
      - token starts with a BPE word-start marker (Ġ or ▁)
      - token is a newline marker (Ċ) — also excluded from words
      - token consists only of punctuation — also excluded from words
      - token is a special tokenizer token
    """
    tok = processor.tokenizer
    ids1d = input_ids.squeeze().tolist()
    tokens = tok.convert_ids_to_tokens(ids1d)
    specials = set(tok.all_special_tokens)

    words: List[Dict[str, Any]] = []
    cur_toks: List[str] = []
    cur_idx: List[int] = []

    def flush():
        nonlocal cur_toks, cur_idx
        if cur_toks:
            _flush_word(tok, ids1d, cur_idx, token_scores, words)
        cur_toks, cur_idx = [], []

    for idx in range(text_span_start, text_span_end):
        t = tokens[idx]
        # Hard break on specials or newline markers
        if t in specials or "Ċ" in t:
            flush()
            continue
        clean = t.lstrip("Ġ▁").strip()
        # Pure-punctuation token → break + skip
        if clean and all(c in _PUNCT for c in clean):
            flush()
            continue
        # Word-start markers → flush and start new word
        is_start = t.startswith("Ġ") or t.startswith("▁") or not cur_toks
        if is_start:
            flush()
            cur_toks = [t]
            cur_idx = [idx]
        else:
            cur_toks.append(t)
            cur_idx.append(idx)

    flush()
    return words


def _flush_word(tok, ids1d, cur_idx, token_scores, words):
    word_str = tok.decode([ids1d[i] for i in cur_idx]).strip()
    if not word_str:
        return
    score = float(sum(token_scores[i] for i in cur_idx))
    words.append({
        "word": word_str,
        "score": score,
        "token_indices": list(cur_idx),
    })


# ── Image-token grouping (optional, for later analysis) ───────────────────────

def aggregate_image_score(
    token_scores: np.ndarray,
    image_span_start: int,
    image_span_end: int,
) -> Dict[str, float]:
    """Summarize image-token scores into a single group for bookkeeping."""
    if image_span_end <= image_span_start:
        return {"image_total_score": 0.0, "image_mean_score": 0.0, "n_image_tokens": 0}
    s = token_scores[image_span_start:image_span_end]
    return {
        "image_total_score": float(s.sum()),
        "image_mean_score": float(s.mean()),
        "n_image_tokens": int(s.size),
    }


# ── Score normalization ───────────────────────────────────────────────────────

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = float(scores.min()), float(scores.max())
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


# ── IO ────────────────────────────────────────────────────────────────────────

def resolve_image_paths(data: Dict[str, Any], input_json: str) -> Dict[str, Any]:
    """Make relative `image_path` entries absolute: join with $IMAGE_ROOT if
    set, else with the directory that contains the input JSON."""
    root = os.environ.get("IMAGE_ROOT") or os.path.dirname(os.path.abspath(input_json))
    for entry in data.values():
        p = entry.get("image_path") if isinstance(entry, dict) else None
        if p and not os.path.isabs(p):
            entry["image_path"] = os.path.join(root, p)
    return data


def load_input_data(input_json: str, limit: Optional[int] = None) -> Dict[str, Any]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit and isinstance(data, dict):
        keys = list(data.keys())[:limit]
        data = {k: data[k] for k in keys}
    return resolve_image_paths(data, input_json)


def save_attribution_results(results: Dict[str, Any], output_json: str) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"[saved] {output_json}")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Printing helpers ─────────────────────────────────────────────────────────

def print_token_scores(
    tokens: List[str],
    scores: np.ndarray,
    top_k: int = 10,
    title: str = "Token Attribution",
) -> None:
    indices = np.argsort(-scores)[:top_k]
    print(f"\n{'=' * 60}")
    print(f"  {title} (top {top_k})")
    print(f"{'=' * 60}")
    for rank, idx in enumerate(indices):
        print(f"  {rank + 1:3d}. [{idx:4d}] {tokens[idx]:20s}  score={scores[idx]:.6f}")
