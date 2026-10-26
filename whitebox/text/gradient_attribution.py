#!/usr/bin/env python3
"""
gradient_attribution.py

Task 17a: Input gradient attribution for token-level uncertainty source
identification.

Implements two gradient-based attribution methods:
  1. Vanilla Gradient: L2 norm of the gradient of the loss w.r.t. input
     embeddings at each token position.
  2. Integrated Gradients (Sundararajan et al., 2017): Path-integrated
     gradient from a zero-embedding baseline to the actual input, yielding
     a more faithful attribution that satisfies axioms of completeness
     and sensitivity.

Usage:
    python gradient_attribution.py \
        --model_path meta-llama/Llama-2-7b-chat-hf \
        --input_json /path/to/saup_data.json \
        --method both \
        --n_steps 30 \
        --top_k 10
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Utilities from the shared attribution module (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_utils import (
    aggregate_subword_scores,
    build_chat_input,
    get_device,
    load_input_data,
    load_model_and_tokenizer,
    normalize_scores,
    print_token_scores,
    save_attribution_results,
)


# ---------------------------------------------------------------------------
# Vanilla Gradient
# ---------------------------------------------------------------------------

def compute_vanilla_gradient(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    device: str,
) -> np.ndarray:
    """Compute vanilla input-gradient attribution scores.

    Steps:
        1. Tokenize the question using build_chat_input.
        2. Retrieve the embedding layer and obtain input embeddings.
        3. Cast embeddings to float32, enable gradient tracking.
        4. Forward pass to obtain logits.
        5. Compute cross-entropy loss of the greedy (argmax) prediction
           against the logits themselves.  This measures how "peaked" the
           model distribution is at the greedy token, serving as a proxy
           for model confidence.
        6. Backward pass to obtain dL/d(embeddings).
        7. Return the L2 norm of the gradient vector at each token
           position (shape: [seq_len]).

    Args:
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        question: The raw question string.
        device: Target device string ('cuda', 'xpu', 'cpu').

    Returns:
        numpy array of shape [seq_len] with per-token gradient norms.
    """
    # --- tokenize ---
    encoded = build_chat_input(question, tokenizer, device)
    input_ids = encoded["input_ids"]                     # [1, seq_len]
    attention_mask = encoded.get("attention_mask", None)  # [1, seq_len]

    # --- get embedding layer ---
    embedding_layer = model.get_input_embeddings()

    # --- compute input embeddings in float32 for gradient precision ---
    with torch.no_grad():
        embeds = embedding_layer(input_ids)  # [1, seq_len, hidden]
    embeds = embeds.detach().clone().to(torch.float32).requires_grad_(True)

    # --- forward pass (autocast handles model-dtype model + fp32 embeds) ---
    with torch.amp.autocast(device_type=device.split(":")[0], dtype=model.dtype):
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # --- greedy cross-entropy loss ---
    # Use the last-position logits (next-token prediction).
    last_logits = logits[:, -1, :].float()  # [1, vocab_size], cast to fp32
    greedy_token = last_logits.argmax(dim=-1)  # [1]
    loss = torch.nn.functional.cross_entropy(last_logits, greedy_token)

    # --- backward pass ---
    model.zero_grad()
    loss.backward()

    # --- extract gradient norms ---
    grad = embeds.grad  # [1, seq_len, hidden]
    if grad is None:
        raise RuntimeError(
            "Gradient was not computed on input embeddings. "
            "Ensure the model supports inputs_embeds and the computation "
            "graph is not detached."
        )
    # L2 norm along the hidden dimension -> [seq_len]
    grad_norms = torch.norm(grad.squeeze(0), p=2, dim=-1).detach().cpu().numpy()
    return grad_norms


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def _forward_gradient(
    model: torch.nn.Module,
    embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run a forward + backward pass and return the gradient on *embeds*.

    Args:
        model: A causal language model.
        embeds: Input embeddings tensor with requires_grad=True,
                shape [1, seq_len, hidden], dtype float32.
        attention_mask: Optional attention mask [1, seq_len].

    Returns:
        Gradient tensor of same shape as *embeds*.
    """
    with torch.amp.autocast(device_type=embeds.device.type, dtype=model.dtype):
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask)
    logits = outputs.logits
    last_logits = logits[:, -1, :].float()
    greedy_token = last_logits.argmax(dim=-1)
    loss = torch.nn.functional.cross_entropy(last_logits, greedy_token)

    model.zero_grad()
    if embeds.grad is not None:
        embeds.grad.zero_()
    loss.backward()

    if embeds.grad is None:
        raise RuntimeError("Gradient not computed on interpolated embeddings.")
    return embeds.grad.clone()


def compute_integrated_gradients(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    device: str,
    n_steps: int = 30,
) -> np.ndarray:
    """Compute Integrated Gradients attribution scores.

    Algorithm (Sundararajan et al., 2017):
        1. Obtain actual input embeddings x.
        2. Define a baseline x' (zero embeddings of the same shape).
        3. For k = 0, 1, ..., n_steps, compute the interpolated input
           x' + (k / n_steps) * (x - x').
        4. At each interpolation step, compute the gradient of the loss
           w.r.t. the interpolated input.
        5. Approximate the integral:
           IG = (x - x') * mean_k(gradients_k)
        6. Return L2 norm of IG at each token position.

    Args:
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        question: The raw question string.
        device: Target device string.
        n_steps: Number of interpolation steps (higher = more accurate).

    Returns:
        numpy array of shape [seq_len] with per-token IG norms.
    """
    # --- tokenize ---
    encoded = build_chat_input(question, tokenizer, device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)

    # --- actual embeddings (float32) ---
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        actual_embeds = embedding_layer(input_ids).to(torch.float32)  # [1, S, H]

    # --- baseline: zero embeddings ---
    baseline_embeds = torch.zeros_like(actual_embeds)  # [1, S, H]

    # --- difference vector ---
    delta = actual_embeds - baseline_embeds  # equals actual_embeds since baseline=0

    # --- accumulate gradients along the interpolation path ---
    accumulated_grads = torch.zeros_like(actual_embeds)  # [1, S, H]

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline_embeds + alpha * delta  # [1, S, H]
        interpolated = interpolated.clone().detach().requires_grad_(True)

        grad = _forward_gradient(model, interpolated, attention_mask)
        accumulated_grads += grad

    # --- integrated gradients: (actual - baseline) * mean(grads) ---
    avg_grads = accumulated_grads / (n_steps + 1)
    ig = delta * avg_grads  # [1, S, H]

    # --- L2 norm per token position ---
    ig_norms = torch.norm(ig.squeeze(0), p=2, dim=-1).detach().cpu().numpy()
    return ig_norms


# ---------------------------------------------------------------------------
# Result Packaging Helpers
# ---------------------------------------------------------------------------

def _build_token_info(
    tokenizer: Any,
    input_ids: torch.Tensor,
    raw_scores: np.ndarray,
) -> Dict[str, Any]:
    """Package per-token and per-word scores into a result dict.

    Args:
        tokenizer: The tokenizer used for encoding.
        input_ids: Token IDs tensor [1, seq_len].
        raw_scores: Raw attribution scores [seq_len].

    Returns:
        Dict with 'token_scores' and 'word_scores' lists.
    """
    norm_scores = normalize_scores(raw_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    token_scores = [
        {"token": tok, "score": float(s), "normalized": float(n)}
        for tok, s, n in zip(tokens, raw_scores, norm_scores)
    ]
    word_scores = aggregate_subword_scores(norm_scores, tokenizer, input_ids)
    return {
        "token_scores": token_scores,
        "word_scores": word_scores,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse arguments, load model, run attribution, save results."""
    parser = argparse.ArgumentParser(
        description="Task 17a: Input gradient attribution for token-level "
                    "uncertainty source identification.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to SAUP-format input JSON.",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path for output JSON (auto-generated if omitted).",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=os.environ.get("HF_CACHE_DIR"),
        help="HuggingFace model cache directory.",
    )
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["vanilla", "integrated", "both"],
        help="Attribution method to run.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=30,
        help="Number of interpolation steps for Integrated Gradients.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of questions to process (None = all).",
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Number of top-scoring tokens to display.",
    )

    args = parser.parse_args()

    # --- auto-generate output path if not specified ---
    if args.output_json is None:
        base = os.path.splitext(os.path.basename(args.input_json))[0]
        model_short = args.model_path.rstrip("/").split("/")[-1]
        args.output_json = os.path.join(
            os.path.dirname(args.input_json) or ".",
            f"gradient_attribution_{model_short}_{base}.json",
        )

    # --- device and model ---
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=device,
        cache_dir=args.cache_dir,
        # auto-detect dtype (bf16 for Gemma, fp16 otherwise)
    )
    # Freeze model parameters — VG only needs gradient w.r.t. input embeddings.
    # Without this, backward allocates a full-size grad tensor for every weight,
    # OOM on large models (e.g. ~54GB extra for Gemma-27B in bf16).
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # --- load data ---
    data = load_input_data(args.input_json, limit=args.limit)

    run_vanilla = args.method in ("vanilla", "both")
    run_ig = args.method in ("integrated", "both")

    print(f"\n[config] method={args.method}  n_steps={args.n_steps}  "
          f"top_k={args.top_k}  questions={len(data)}")
    print(f"[config] output -> {args.output_json}\n")

    # --- config record for output ---
    config = {
        "model_path": args.model_path,
        "input_json": args.input_json,
        "method": args.method,
        "n_steps": args.n_steps,
        "limit": args.limit,
        "top_k": args.top_k,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    results: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    # --- main loop ---
    for qid, entry in tqdm(data.items(), desc="Gradient attribution"):
        # Extract question from SAUP format: items[0].question is the
        # original question.
        items = entry.get("items", [])
        if items:
            question = items[0].get("question", "")
        else:
            question = entry.get("question", "")

        if not question:
            errors.append({"qid": qid, "error": "No question text found."})
            continue

        result_entry: Dict[str, Any] = {"question": question}

        try:
            # Tokenize once to get input_ids for word-level aggregation.
            encoded = build_chat_input(question, tokenizer, device)
            input_ids = encoded["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(
                input_ids.squeeze().tolist()
            )

            # --- Vanilla Gradient ---
            if run_vanilla:
                vg_scores = compute_vanilla_gradient(
                    model, tokenizer, question, device,
                )
                result_entry["vanilla_gradient"] = _build_token_info(
                    tokenizer, input_ids, vg_scores,
                )
                norm_vg = normalize_scores(vg_scores)
                print_token_scores(
                    tokens, norm_vg,
                    top_k=args.top_k,
                    title=f"Vanilla Gradient | {qid}",
                )

            # --- Integrated Gradients ---
            if run_ig:
                ig_scores = compute_integrated_gradients(
                    model, tokenizer, question, device,
                    n_steps=args.n_steps,
                )
                result_entry["integrated_gradient"] = _build_token_info(
                    tokenizer, input_ids, ig_scores,
                )
                norm_ig = normalize_scores(ig_scores)
                print_token_scores(
                    tokens, norm_ig,
                    top_k=args.top_k,
                    title=f"Integrated Gradients | {qid}",
                )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            print(f"[error] {qid}: {error_msg}")
            errors.append({"qid": qid, "error": error_msg})
            result_entry["error"] = error_msg

        results[qid] = result_entry

    # --- assemble output ---
    output = {
        "config": config,
        "results": results,
    }
    if errors:
        output["errors"] = errors

    save_attribution_results(output, args.output_json)

    # --- summary ---
    n_success = sum(1 for v in results.values() if "error" not in v)
    n_fail = len(errors)
    print(f"\n[done] {n_success} succeeded, {n_fail} failed.")
    print(f"[done] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
