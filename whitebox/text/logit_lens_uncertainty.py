#!/usr/bin/env python3
"""
logit_lens_uncertainty.py

Task 17c: Logit lens and hidden state trajectory analysis for uncertainty
localization.

Applies the logit lens technique (nostalgebraist 2020) to track how the
model's internal representation of each input token evolves across layers.
Localizes where uncertainty arises in the model's computation by:
  1. Projecting each layer's hidden state through lm_head to vocabulary space.
  2. Computing layer-wise entropy of the projected distributions.
  3. Identifying decision layers (sharp entropy drops) and persistent
     uncertainty positions (entropy stays high across most layers).
  4. Tracking hidden-state norm trajectories across layers.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Import shared utilities from the same directory ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution_utils import (
    get_device,
    load_model_and_tokenizer,
    load_input_data,
    save_attribution_results,
    build_chat_input,
    token_to_word_spans,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Analysis Functions
# ═══════════════════════════════════════════════════════════════════════════════


def compute_logit_lens(
    model: torch.nn.Module,
    hidden_states: Tuple[torch.Tensor, ...],
    input_ids: torch.Tensor,
    tokenizer: Any,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Logit lens: project each layer's hidden state through lm_head to
    vocabulary space and return the top-k predicted tokens.

    Parameters
    ----------
    model : torch.nn.Module
        A causal LM with a `lm_head` attribute.
    hidden_states : tuple of tensors
        (n_layers+1,) tensors, each of shape (batch, seq_len, hidden_dim).
        Index 0 is the embedding layer output; indices 1..n_layers are the
        transformer block outputs.
    input_ids : torch.Tensor
        Shape (batch, seq_len).  Used only for labelling the input tokens.
    tokenizer : transformers.PreTrainedTokenizer
        For decoding token IDs back to strings.
    top_k : int
        Number of top predictions to keep per (layer, position).

    Returns
    -------
    dict with keys:
        "top_tokens"  : list[list[list[str]]]
            Shape conceptually (n_layers, seq_len, top_k).
            Each entry is the decoded token string.
        "top_token_ids" : list[list[list[int]]]
            Same shape, raw token IDs.
        "top_probs"   : list[list[list[float]]]
            Softmax probabilities of the top-k tokens.
        "input_tokens" : list[str]
            The original input tokens for reference.
        "n_layers"    : int
        "seq_len"     : int
    """
    n_layers = len(hidden_states) - 1  # exclude embedding layer
    seq_len = hidden_states[0].shape[1]

    # Decode input tokens for reference
    input_token_ids = input_ids.squeeze(0).tolist()
    input_tokens = [tokenizer.decode([tid]).strip() for tid in input_token_ids]

    top_tokens_all: List[List[List[str]]] = []
    top_ids_all: List[List[List[int]]] = []
    top_probs_all: List[List[List[float]]] = []

    lm_head = model.lm_head

    for layer_idx in range(1, len(hidden_states)):  # skip embedding (index 0)
        h = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
        h_squeezed = h.squeeze(0)  # (seq_len, hidden_dim)

        # Project through lm_head in model dtype, then cast to float32
        with torch.no_grad():
            logits = lm_head(h_squeezed).float()  # (seq_len, vocab_size)

        probs = F.softmax(logits, dim=-1)  # (seq_len, vocab_size)

        top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)  # (seq_len, top_k)

        layer_tokens: List[List[str]] = []
        layer_ids: List[List[int]] = []
        layer_probs: List[List[float]] = []

        for pos in range(seq_len):
            ids_list = top_ids[pos].tolist()
            probs_list = top_probs[pos].tolist()
            tokens_list = [tokenizer.decode([tid]).strip() for tid in ids_list]
            layer_tokens.append(tokens_list)
            layer_ids.append(ids_list)
            layer_probs.append(probs_list)

        top_tokens_all.append(layer_tokens)
        top_ids_all.append(layer_ids)
        top_probs_all.append(layer_probs)

    return {
        "top_tokens": top_tokens_all,
        "top_token_ids": top_ids_all,
        "top_probs": top_probs_all,
        "input_tokens": input_tokens,
        "n_layers": n_layers,
        "seq_len": seq_len,
    }


def compute_layer_entropy(
    model: torch.nn.Module,
    hidden_states: Tuple[torch.Tensor, ...],
) -> np.ndarray:
    """
    Compute entropy of softmax(lm_head(h_l(x_i))) at each (layer, position).

    Parameters
    ----------
    model : torch.nn.Module
        A causal LM with a `lm_head` attribute.
    hidden_states : tuple of tensors
        (n_layers+1,) tensors, each (batch, seq_len, hidden_dim).

    Returns
    -------
    entropy_matrix : np.ndarray, shape (n_layers, seq_len)
        Entropy in nats at each (layer, position).
    """
    n_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    lm_head = model.lm_head

    entropy_matrix = np.zeros((n_layers, seq_len), dtype=np.float64)

    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].squeeze(0)  # (seq_len, hidden_dim)
        with torch.no_grad():
            logits = lm_head(h).float()  # (seq_len, vocab_size) in fp32

        # Log-softmax for stable entropy computation
        log_probs = F.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)
        probs = torch.exp(log_probs)

        # H = -sum(p * log(p));  handle p=0 via p*log_p being 0
        entropy = -(probs * log_probs).sum(dim=-1)  # (seq_len,)
        entropy_matrix[layer_idx - 1] = entropy.cpu().numpy()

    return entropy_matrix


def find_decision_layers(
    entropy_matrix: np.ndarray,
    threshold: float = 0.5,
) -> Dict[int, List[int]]:
    """
    Find layers where entropy drops sharply for each position.

    A "decision layer" for position i is a layer l where:
        (H_{l-1}(x_i) - H_l(x_i)) / H_{l-1}(x_i) > threshold

    i.e. the relative entropy drop exceeds the threshold.

    Parameters
    ----------
    entropy_matrix : np.ndarray, shape (n_layers, seq_len)
    threshold : float
        Minimum relative entropy drop to qualify as a decision layer.

    Returns
    -------
    dict mapping position index -> list of decision layer indices.
    """
    n_layers, seq_len = entropy_matrix.shape
    decision_layers: Dict[int, List[int]] = {}

    for pos in range(seq_len):
        layers_for_pos: List[int] = []
        for layer in range(1, n_layers):
            prev_entropy = entropy_matrix[layer - 1, pos]
            curr_entropy = entropy_matrix[layer, pos]
            if prev_entropy > 1e-8:  # avoid division by zero
                relative_drop = (prev_entropy - curr_entropy) / prev_entropy
                if relative_drop > threshold:
                    layers_for_pos.append(layer)
        if layers_for_pos:
            decision_layers[pos] = layers_for_pos

    return decision_layers


def find_persistent_uncertainty_positions(
    entropy_matrix: np.ndarray,
    percentile: float = 50,
    layer_fraction: float = 0.7,
) -> List[int]:
    """
    Find positions where entropy remains above the per-layer median
    through more than `layer_fraction` of all layers.

    Parameters
    ----------
    entropy_matrix : np.ndarray, shape (n_layers, seq_len)
    percentile : float
        Percentile threshold computed per layer (default 50 = median).
    layer_fraction : float
        Fraction of layers the position must exceed the threshold in.

    Returns
    -------
    List of position indices with persistent high entropy.
    """
    n_layers, seq_len = entropy_matrix.shape

    # Compute per-layer threshold (percentile across positions)
    layer_thresholds = np.percentile(entropy_matrix, percentile, axis=1)  # (n_layers,)

    # Count how many layers each position exceeds the threshold
    above_threshold = np.zeros(seq_len, dtype=np.int64)
    for layer in range(n_layers):
        above_threshold += (entropy_matrix[layer, :] > layer_thresholds[layer]).astype(np.int64)

    # Positions exceeding threshold in more than layer_fraction of layers
    min_layers = int(n_layers * layer_fraction)
    persistent_positions = [
        int(pos) for pos in range(seq_len) if above_threshold[pos] > min_layers
    ]

    return persistent_positions


def compute_norm_trajectory(
    hidden_states: Tuple[torch.Tensor, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the L2 norm of each hidden state across layers and positions,
    and the layer-to-layer delta.

    Parameters
    ----------
    hidden_states : tuple of tensors
        (n_layers+1,) tensors, each (batch, seq_len, hidden_dim).

    Returns
    -------
    norm_matrix : np.ndarray, shape (n_layers, seq_len)
        ||h_l(x_i)|| at each (layer, position).  Layers indexed from
        transformer block outputs (skipping the embedding layer).
    delta_norm_matrix : np.ndarray, shape (n_layers-1, seq_len)
        delta_norm[l, i] = ||h_{l+1}(x_i)|| - ||h_l(x_i)||
    """
    n_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]

    norm_matrix = np.zeros((n_layers, seq_len), dtype=np.float64)

    for layer_idx in range(1, len(hidden_states)):
        h = hidden_states[layer_idx].squeeze(0).float()  # (seq_len, hidden_dim)
        norms = torch.norm(h, dim=-1)  # (seq_len,)
        norm_matrix[layer_idx - 1] = norms.cpu().numpy()

    # Delta norms: difference between consecutive layers
    if n_layers > 1:
        delta_norm_matrix = norm_matrix[1:] - norm_matrix[:-1]  # (n_layers-1, seq_len)
    else:
        delta_norm_matrix = np.zeros((0, seq_len), dtype=np.float64)

    return norm_matrix, delta_norm_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Forward pass with hidden states
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def forward_with_hidden_states(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Run a forward pass and return (logits, hidden_states).

    Parameters
    ----------
    model : torch.nn.Module
    inputs : dict of tensors (input_ids, attention_mask, etc.)

    Returns
    -------
    logits : (batch, seq_len, vocab_size)
    hidden_states : tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
    """
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.logits, outputs.hidden_states


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Summarize logit lens for JSON output
# ═══════════════════════════════════════════════════════════════════════════════


def _summarize_logit_lens_for_json(
    logit_lens_result: Dict[str, Any],
    max_positions: int = 50,
) -> Dict[str, Any]:
    """
    Produce a JSON-friendly summary of the logit lens results.
    To keep output size manageable, we limit the number of positions and
    only keep a subset of layers (first, middle, last few).
    """
    n_layers = logit_lens_result["n_layers"]
    seq_len = logit_lens_result["seq_len"]

    # Select representative layers: first 2, every quarter, last 2
    representative_layers = sorted(set(
        [0, 1]
        + [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
        + [max(0, n_layers - 2), n_layers - 1]
    ))
    representative_layers = [l for l in representative_layers if l < n_layers]

    positions_to_show = min(seq_len, max_positions)

    summary = {
        "n_layers": n_layers,
        "seq_len": seq_len,
        "representative_layers": representative_layers,
        "input_tokens": logit_lens_result["input_tokens"][:positions_to_show],
        "layer_predictions": {},
    }

    for l_idx in representative_layers:
        layer_key = f"layer_{l_idx}"
        summary["layer_predictions"][layer_key] = []
        for pos in range(positions_to_show):
            summary["layer_predictions"][layer_key].append({
                "position": pos,
                "input_token": logit_lens_result["input_tokens"][pos]
                    if pos < len(logit_lens_result["input_tokens"]) else "",
                "top_tokens": logit_lens_result["top_tokens"][l_idx][pos],
                "top_probs": [round(p, 6) for p in logit_lens_result["top_probs"][l_idx][pos]],
            })

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Task 17c: Logit lens and hidden state trajectory analysis "
                    "for uncertainty localization."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to SAUP-format input JSON file.",
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path to write output JSON results.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=os.environ.get("HF_CACHE_DIR"),
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions to process.",
    )
    parser.add_argument(
        "--top_k_vocab", type=int, default=5,
        help="Number of top vocabulary predictions per (layer, position) for logit lens.",
    )
    parser.add_argument(
        "--entropy_drop_threshold", type=float, default=0.5,
        help="Relative entropy drop threshold for decision layer detection.",
    )
    parser.add_argument(
        "--persistent_percentile", type=float, default=50.0,
        help="Percentile for persistent uncertainty detection (default: median).",
    )
    parser.add_argument(
        "--persistent_layer_fraction", type=float, default=0.7,
        help="Fraction of layers a position must exceed threshold to be 'persistent'.",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max tokenization length.",
    )
    args = parser.parse_args()

    # ── Device & model ────────────────────────────────────────────────────────
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, device, cache_dir=args.cache_dir,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_input_data(args.input_json, limit=args.limit)
    print(f"[data] Loaded {len(data)} questions.")

    # ── Process each question ─────────────────────────────────────────────────
    results: Dict[str, Any] = {}
    total = len(data)

    for q_idx, (q_key, q_data) in enumerate(data.items()):
        print(f"\n[{q_idx + 1}/{total}] Processing: {q_key}")

        # Build the question text
        question_text = q_data.get("question", "")
        story = q_data.get("story", "")
        if story:
            full_input = f"{story}\n\nQuestion: {question_text}"
        else:
            full_input = question_text

        # Tokenize with chat template
        inputs = build_chat_input(
            full_input, tokenizer, device, max_length=args.max_length,
        )
        input_ids = inputs["input_ids"]  # (1, seq_len)
        seq_len = input_ids.shape[1]

        print(f"  seq_len = {seq_len}")

        # Forward pass with hidden states
        logits, hidden_states = forward_with_hidden_states(model, inputs)
        n_layers = len(hidden_states) - 1
        print(f"  n_layers = {n_layers}, hidden_dim = {hidden_states[0].shape[-1]}")

        # ── 1. Logit Lens ─────────────────────────────────────────────────────
        logit_lens_result = compute_logit_lens(
            model, hidden_states, input_ids, tokenizer, top_k=args.top_k_vocab,
        )
        logit_lens_summary = _summarize_logit_lens_for_json(logit_lens_result)

        # ── 2. Layer-wise Entropy ─────────────────────────────────────────────
        entropy_matrix = compute_layer_entropy(model, hidden_states)
        # (n_layers, seq_len)

        # ── 3. Decision Layers ────────────────────────────────────────────────
        decision_layers = find_decision_layers(
            entropy_matrix, threshold=args.entropy_drop_threshold,
        )

        # ── 4. Persistent Uncertainty Positions ───────────────────────────────
        persistent_positions = find_persistent_uncertainty_positions(
            entropy_matrix,
            percentile=args.persistent_percentile,
            layer_fraction=args.persistent_layer_fraction,
        )

        # ── 5. Norm Trajectory ────────────────────────────────────────────────
        norm_matrix, delta_norm_matrix = compute_norm_trajectory(hidden_states)

        # ── Decode tokens for labelling ───────────────────────────────────────
        token_ids_list = input_ids.squeeze(0).tolist()
        token_strings = [tokenizer.decode([tid]).strip() for tid in token_ids_list]

        # ── Build per-question result ─────────────────────────────────────────
        # Entropy heatmap: store as list-of-lists (layer x position)
        # To manage output size, round values
        entropy_heatmap = [
            [round(float(entropy_matrix[l, p]), 6) for p in range(seq_len)]
            for l in range(n_layers)
        ]

        # Norm trajectory: similarly rounded
        norm_trajectory = [
            [round(float(norm_matrix[l, p]), 4) for p in range(seq_len)]
            for l in range(n_layers)
        ]
        delta_norm_trajectory = [
            [round(float(delta_norm_matrix[l, p]), 4) for p in range(seq_len)]
            for l in range(delta_norm_matrix.shape[0])
        ]

        # Convert decision_layers keys to strings for JSON
        decision_layers_str = {
            str(pos): layers for pos, layers in decision_layers.items()
        }

        # Annotate persistent uncertainty positions with their tokens
        persistent_info = [
            {
                "position": pos,
                "token": token_strings[pos] if pos < len(token_strings) else "",
                "mean_entropy": round(float(entropy_matrix[:, pos].mean()), 6),
                "max_entropy": round(float(entropy_matrix[:, pos].max()), 6),
                "min_entropy": round(float(entropy_matrix[:, pos].min()), 6),
            }
            for pos in persistent_positions
        ]

        # Summary statistics
        mean_entropy_per_layer = [
            round(float(entropy_matrix[l].mean()), 6) for l in range(n_layers)
        ]
        mean_entropy_per_position = [
            round(float(entropy_matrix[:, p].mean()), 6) for p in range(seq_len)
        ]

        # Top uncertain positions (by mean entropy across layers)
        pos_mean_entropy = entropy_matrix.mean(axis=0)  # (seq_len,)
        top_uncertain_k = min(20, seq_len)
        top_uncertain_indices = np.argsort(-pos_mean_entropy)[:top_uncertain_k]
        top_uncertain_positions = [
            {
                "position": int(idx),
                "token": token_strings[idx] if idx < len(token_strings) else "",
                "mean_entropy": round(float(pos_mean_entropy[idx]), 6),
            }
            for idx in top_uncertain_indices
        ]

        # Top positions by max norm delta (indicating heavy processing)
        if delta_norm_matrix.shape[0] > 0:
            max_delta_per_pos = np.abs(delta_norm_matrix).max(axis=0)  # (seq_len,)
            top_delta_k = min(20, seq_len)
            top_delta_indices = np.argsort(-max_delta_per_pos)[:top_delta_k]
            top_norm_delta_positions = [
                {
                    "position": int(idx),
                    "token": token_strings[idx] if idx < len(token_strings) else "",
                    "max_abs_delta_norm": round(float(max_delta_per_pos[idx]), 4),
                }
                for idx in top_delta_indices
            ]
        else:
            top_norm_delta_positions = []

        results[q_key] = {
            "question": question_text,
            "n_layers": n_layers,
            "seq_len": seq_len,
            "tokens": token_strings,
            "logit_lens_summary": logit_lens_summary,
            "entropy_heatmap": entropy_heatmap,
            "mean_entropy_per_layer": mean_entropy_per_layer,
            "mean_entropy_per_position": mean_entropy_per_position,
            "decision_layers": decision_layers_str,
            "persistent_uncertainty_positions": persistent_info,
            "norm_trajectory": norm_trajectory,
            "delta_norm_trajectory": delta_norm_trajectory,
            "top_uncertain_positions": top_uncertain_positions,
            "top_norm_delta_positions": top_norm_delta_positions,
            "config": {
                "top_k_vocab": args.top_k_vocab,
                "entropy_drop_threshold": args.entropy_drop_threshold,
                "persistent_percentile": args.persistent_percentile,
                "persistent_layer_fraction": args.persistent_layer_fraction,
            },
        }

        # Clean up GPU memory
        del logits, hidden_states, entropy_matrix, norm_matrix, delta_norm_matrix
        del logit_lens_result
        if device != "cpu":
            torch.cuda.empty_cache() if device == "cuda" else None

        print(f"  Done. Decision layers at {len(decision_layers)} positions, "
              f"{len(persistent_positions)} persistent uncertainty positions.")

    # ── Save results ──────────────────────────────────────────────────────────
    save_attribution_results(results, args.output_json)
    print(f"\n[complete] Processed {len(results)} questions. "
          f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
