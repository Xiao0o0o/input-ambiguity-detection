#!/usr/bin/env python3
"""
diffmask_uncertainty.py

Task 17j: Differentiable token masking for combinatorial uncertainty
attribution.

Learns a continuous mask over input tokens via gradient descent to identify
which tokens, when masked out, cause the largest increase in output entropy.
Tokens whose removal most increases uncertainty are flagged as
uncertainty-stabilising (their presence reduces uncertainty).  Tokens whose
removal most decreases uncertainty are uncertainty-causing.

Additionally provides pair-wise interaction detection: for every pair of
critical tokens, the script checks whether masking both simultaneously
produces a super-additive entropy change, revealing combinatorial
interactions between input tokens.

Usage:
    python diffmask_uncertainty.py \
        --model_path meta-llama/Llama-2-7b-chat-hf \
        --input_json /path/to/saup_data.json \
        --output_json /path/to/output.json \
        --n_steps 150 --lr 0.1 --sparsity_weight 0.5 \
        --n_generate 16 --baseline zero \
        --detect_interactions --interaction_top_k 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
# Baseline Embedding Computation
# ---------------------------------------------------------------------------

def compute_baseline_embedding(
    model: nn.Module,
    tokenizer: Any,
    method: str = "zero",
) -> torch.Tensor:
    """Compute a baseline embedding vector.

    Args:
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        method: One of "zero" (zero vector) or "mean" (mean over all vocab
                embeddings).

    Returns:
        Baseline embedding tensor of shape (1, 1, hidden_dim).
    """
    embedding_layer = model.get_input_embeddings()
    hidden_dim = embedding_layer.weight.shape[1]
    device = embedding_layer.weight.device
    dtype = embedding_layer.weight.dtype

    if method == "zero":
        baseline = torch.zeros(1, 1, hidden_dim, device=device, dtype=dtype)
    elif method == "mean":
        with torch.no_grad():
            # Mean embedding over the entire vocabulary.
            baseline = embedding_layer.weight.mean(dim=0, keepdim=True).unsqueeze(0)
            # Shape: (1, 1, hidden_dim)
    else:
        raise ValueError(f"Unknown baseline method: {method!r}. Use 'zero' or 'mean'.")

    return baseline


# ---------------------------------------------------------------------------
# Output Entropy Computation
# ---------------------------------------------------------------------------

def compute_output_entropy(
    model: nn.Module,
    masked_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    n_generate: int = 32,
) -> torch.Tensor:
    """Compute mean output entropy over greedy generation steps.

    Performs n_generate autoregressive greedy decoding steps, recording
    the Shannon entropy of the output distribution at each step.
    Returns the mean entropy as a differentiable scalar.

    Args:
        model: A HuggingFace causal language model (weights frozen,
               but we allow gradient flow through input embeddings).
        masked_embeds: Input embeddings tensor of shape (1, seq_len,
                       hidden_dim), potentially with gradients attached
                       via the mask.
        attention_mask: Attention mask tensor of shape (1, seq_len).
        n_generate: Number of greedy generation steps.

    Returns:
        Mean entropy scalar (differentiable w.r.t. masked_embeds).
    """
    embedding_layer = model.get_input_embeddings()
    entropies = []
    current_embeds = masked_embeds
    current_mask = attention_mask

    for step in range(n_generate):
        outputs = model(
            inputs_embeds=current_embeds,
            attention_mask=current_mask,
        )
        # Logits at the last position: shape (1, vocab_size)
        last_logits = outputs.logits[:, -1, :].float()

        # Compute entropy: H = -sum(p * log(p))
        log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
        probs = torch.nn.functional.softmax(last_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # shape (1,)
        entropies.append(entropy)

        # Greedy decode: pick argmax token, get its embedding, append
        greedy_id = last_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
        next_embed = embedding_layer(greedy_id)  # (1, 1, hidden_dim)
        # Detach the next embedding so gradients flow only through the
        # mask-affected prefix, not through the autoregressive chain.
        next_embed = next_embed.detach().to(current_embeds.dtype)

        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        new_mask = torch.ones(1, 1, device=current_mask.device, dtype=current_mask.dtype)
        current_mask = torch.cat([current_mask, new_mask], dim=1)

    # Mean entropy over generation steps
    mean_entropy = torch.stack(entropies).mean()
    return mean_entropy


# ---------------------------------------------------------------------------
# Differentiable Mask Module
# ---------------------------------------------------------------------------

class DiffMask(nn.Module):
    """Learnable continuous mask over input tokens.

    Each token position has a scalar parameter m_i.  The effective mask
    value is sigmoid(m_i), so it lies in (0, 1).  A mask value near 1
    keeps the original token embedding; a value near 0 replaces it with
    the baseline embedding.

    Attributes:
        m: nn.Parameter of shape (n_tokens,), raw logit-space mask values.
    """

    def __init__(self, n_tokens: int, device: str = "cpu") -> None:
        super().__init__()
        # Initialise at ones so sigmoid(1) ~ 0.73 -- biased toward keeping
        # tokens, encouraging the optimiser to selectively mask only the
        # most impactful ones.
        self.m = nn.Parameter(torch.ones(n_tokens, device=device))

    def forward(
        self,
        actual_embeds: torch.Tensor,
        baseline_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the continuous mask.

        Args:
            actual_embeds: Original input embeddings, shape (1, n_tokens,
                           hidden_dim).
            baseline_embeds: Baseline embeddings, shape broadcastable to
                             actual_embeds (e.g. (1, 1, hidden_dim) or
                             (1, n_tokens, hidden_dim)).

        Returns:
            Masked embeddings, shape (1, n_tokens, hidden_dim).
        """
        # Cast mask to same dtype as embeddings (fp16) to avoid matmul errors
        mask = self.m.sigmoid().unsqueeze(-1).to(actual_embeds.dtype)  # (n_tokens, 1)
        return mask * actual_embeds + (1.0 - mask) * baseline_embeds

    def get_mask_values(self) -> np.ndarray:
        """Return sigmoid(m) as a numpy array of shape (n_tokens,)."""
        with torch.no_grad():
            return self.m.sigmoid().detach().cpu().numpy()

    def get_sparsity_loss(self) -> torch.Tensor:
        """L1 norm of sigmoid(m), encouraging sparse (near-zero) masks."""
        return self.m.sigmoid().sum()


# ---------------------------------------------------------------------------
# Mask Optimisation
# ---------------------------------------------------------------------------

def optimize_mask(
    model: nn.Module,
    tokenizer: Any,
    question: str,
    device: str,
    baseline_method: str = "zero",
    n_steps: int = 150,
    lr: float = 0.1,
    sparsity_weight: float = 0.5,
    n_generate: int = 16,
) -> Tuple[np.ndarray, List[Dict[str, Any]], float, float]:
    """Optimise a differentiable mask to find uncertainty-critical tokens.

    The objective is:
        L = -H(masked_output) + sparsity_weight * L1(mask)

    Minimising L means maximising output entropy (H) under the masked
    input while keeping the mask sparse.  The optimiser pushes mask values
    toward 0 for tokens whose masking increases entropy the most.

    All model parameters are frozen; only the mask parameters receive
    gradients.

    Args:
        model: A HuggingFace causal language model (already on device).
        tokenizer: The corresponding tokenizer.
        question: The raw question string.
        device: Target device string.
        baseline_method: "zero" or "mean" baseline embedding.
        n_steps: Number of optimisation iterations.
        lr: Learning rate for Adam.
        sparsity_weight: Weight for the L1 sparsity penalty.
        n_generate: Number of greedy tokens to generate for entropy
                    computation.

    Returns:
        Tuple of:
          - mask_values: numpy array of shape (seq_len,), final mask values
                         in (0, 1).
          - critical_tokens: list of dicts describing tokens where
                             mask < 0.5, sorted by mask value ascending.
          - unmasked_entropy: float, output entropy with the full (no mask)
                              input.
          - masked_entropy: float, output entropy with the final optimised
                            mask applied.
    """
    # --- tokenize ---
    encoded = build_chat_input(question, tokenizer, device)
    input_ids = encoded["input_ids"]            # (1, seq_len)
    attention_mask = encoded["attention_mask"]   # (1, seq_len)
    seq_len = input_ids.shape[1]

    # --- get actual embeddings ---
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        actual_embeds = embedding_layer(input_ids)
        # (1, seq_len, hidden_dim)

    # --- baseline embedding ---
    baseline_single = compute_baseline_embedding(
        model, tokenizer, method=baseline_method,
    )
    # Expand to (1, seq_len, hidden_dim)
    baseline_embeds = baseline_single.expand_as(actual_embeds)

    # --- compute unmasked entropy (reference) ---
    with torch.no_grad():
        unmasked_entropy_val = compute_output_entropy(
            model, actual_embeds, attention_mask, n_generate=n_generate,
        ).item()

    # --- create mask and optimiser ---
    diff_mask = DiffMask(n_tokens=seq_len, device=device)
    optimizer = torch.optim.Adam(diff_mask.parameters(), lr=lr)

    # --- optimisation loop ---
    print(f"  [diffmask] Optimising mask for {n_steps} steps (seq_len={seq_len})")
    for step in range(n_steps):
        optimizer.zero_grad()

        # Apply mask
        masked_embeds = diff_mask(actual_embeds, baseline_embeds)

        # Compute output entropy with masked input
        H = compute_output_entropy(
            model, masked_embeds, attention_mask, n_generate=n_generate,
        )

        # Sparsity penalty
        sparsity_loss = diff_mask.get_sparsity_loss()

        # Loss: we want to maximise entropy (find tokens whose removal
        # increases entropy), while keeping the mask sparse.
        loss = -H + sparsity_weight * sparsity_loss

        loss.backward()
        optimizer.step()

        if (step + 1) % 25 == 0 or step == 0:
            mv = diff_mask.get_mask_values()
            n_masked = int((mv < 0.5).sum())
            print(
                f"    step {step+1:4d}/{n_steps}  "
                f"loss={loss.item():.4f}  H={H.item():.4f}  "
                f"sparsity={sparsity_loss.item():.2f}  "
                f"n_masked(< 0.5)={n_masked}"
            )

    # --- final mask values ---
    mask_values = diff_mask.get_mask_values()

    # --- compute masked entropy with final mask ---
    with torch.no_grad():
        final_masked_embeds = diff_mask(actual_embeds, baseline_embeds)
        masked_entropy_val = compute_output_entropy(
            model, final_masked_embeds, attention_mask, n_generate=n_generate,
        ).item()

    # --- identify critical tokens (mask < 0.5) ---
    tokens_str = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    critical_tokens = []
    for idx in range(seq_len):
        if mask_values[idx] < 0.5:
            critical_tokens.append({
                "token_idx": int(idx),
                "token_str": tokens_str[idx],
                "mask_value": float(mask_values[idx]),
                "decoded": tokenizer.decode(input_ids[0, idx].item()).strip(),
            })
    # Sort by mask value ascending (most masked first)
    critical_tokens.sort(key=lambda x: x["mask_value"])

    print(
        f"  [diffmask] Done. unmasked_H={unmasked_entropy_val:.4f}  "
        f"masked_H={masked_entropy_val:.4f}  "
        f"critical_tokens={len(critical_tokens)}"
    )

    return mask_values, critical_tokens, unmasked_entropy_val, masked_entropy_val


# ---------------------------------------------------------------------------
# Interaction Detection
# ---------------------------------------------------------------------------

def _compute_entropy_with_positions_masked(
    model: nn.Module,
    actual_embeds: torch.Tensor,
    baseline_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: List[int],
    n_generate: int = 16,
) -> float:
    """Compute output entropy with specific token positions masked.

    Creates a binary mask that replaces the embeddings at the given
    positions with the baseline, then computes the output entropy.

    Args:
        model: Frozen causal LM.
        actual_embeds: Original embeddings, (1, seq_len, hidden_dim).
        baseline_embeds: Baseline embeddings, same shape.
        attention_mask: Attention mask, (1, seq_len).
        positions: List of token indices to mask.
        n_generate: Number of generation steps for entropy.

    Returns:
        Mean output entropy as a float.
    """
    seq_len = actual_embeds.shape[1]
    # Build binary mask: 1 = keep actual, 0 = use baseline
    mask = torch.ones(seq_len, 1, device=actual_embeds.device, dtype=actual_embeds.dtype)
    for pos in positions:
        if 0 <= pos < seq_len:
            mask[pos, 0] = 0.0

    masked_embeds = mask * actual_embeds + (1.0 - mask) * baseline_embeds

    with torch.no_grad():
        entropy_val = compute_output_entropy(
            model, masked_embeds, attention_mask, n_generate=n_generate,
        ).item()
    return entropy_val


def detect_interactions(
    model: nn.Module,
    tokenizer: Any,
    question: str,
    critical_tokens: List[Dict[str, Any]],
    device: str,
    baseline_method: str = "zero",
    n_generate: int = 16,
) -> List[Dict[str, Any]]:
    """Detect super-additive interactions between pairs of critical tokens.

    For each pair (i, j) of critical tokens:
      - Mask only i: compute entropy change delta_i
      - Mask only j: compute entropy change delta_j
      - Mask both: compute entropy change delta_ij
      - If delta_ij > 1.2 * (delta_i + delta_j): super-additive interaction

    Args:
        model: Frozen causal LM.
        tokenizer: The corresponding tokenizer.
        question: The raw question string.
        critical_tokens: List of dicts from optimize_mask with at least
                         'token_idx'.
        device: Device string.
        baseline_method: Baseline embedding method.
        n_generate: Generation steps for entropy.

    Returns:
        List of interaction dicts with keys: token_i, token_j, idx_i, idx_j,
        effect_i, effect_j, effect_both, interaction_type.
    """
    if len(critical_tokens) < 2:
        return []

    # --- tokenize ---
    encoded = build_chat_input(question, tokenizer, device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # --- embeddings ---
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        actual_embeds = embedding_layer(input_ids)
    baseline_single = compute_baseline_embedding(
        model, tokenizer, method=baseline_method,
    )
    baseline_embeds = baseline_single.expand_as(actual_embeds)

    # --- unmasked entropy ---
    unmasked_H = _compute_entropy_with_positions_masked(
        model, actual_embeds, baseline_embeds, attention_mask,
        positions=[], n_generate=n_generate,
    )

    # --- cache single-token masking effects ---
    single_effects: Dict[int, float] = {}
    for ct in critical_tokens:
        idx = ct["token_idx"]
        H_i = _compute_entropy_with_positions_masked(
            model, actual_embeds, baseline_embeds, attention_mask,
            positions=[idx], n_generate=n_generate,
        )
        single_effects[idx] = H_i - unmasked_H  # positive = entropy increased

    # --- pair-wise interactions ---
    interactions = []
    n = len(critical_tokens)
    for a in range(n):
        for b in range(a + 1, n):
            idx_i = critical_tokens[a]["token_idx"]
            idx_j = critical_tokens[b]["token_idx"]
            effect_i = single_effects[idx_i]
            effect_j = single_effects[idx_j]

            H_both = _compute_entropy_with_positions_masked(
                model, actual_embeds, baseline_embeds, attention_mask,
                positions=[idx_i, idx_j], n_generate=n_generate,
            )
            effect_both = H_both - unmasked_H

            # Determine interaction type
            additive_sum = effect_i + effect_j
            if abs(additive_sum) < 1e-8:
                interaction_type = "independent"
            elif effect_both > additive_sum * 1.2:
                interaction_type = "super-additive"
            elif effect_both < additive_sum * 0.8:
                interaction_type = "sub-additive"
            else:
                interaction_type = "approximately-additive"

            interactions.append({
                "idx_i": int(idx_i),
                "idx_j": int(idx_j),
                "token_i": critical_tokens[a].get("token_str", ""),
                "token_j": critical_tokens[b].get("token_str", ""),
                "effect_i": float(effect_i),
                "effect_j": float(effect_j),
                "effect_both": float(effect_both),
                "additive_sum": float(additive_sum),
                "interaction_type": interaction_type,
            })

    return interactions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse arguments, load model, run diffmask, save results."""
    parser = argparse.ArgumentParser(
        description="Task 17j: Differentiable token masking for "
                    "combinatorial uncertainty attribution.",
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
        "--limit", type=int, default=None,
        help="Max number of questions to process (None = all).",
    )
    parser.add_argument(
        "--n_steps", type=int, default=150,
        help="Number of mask optimisation steps.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate for Adam optimiser on mask parameters.",
    )
    parser.add_argument(
        "--sparsity_weight", type=float, default=0.5,
        help="Weight for L1 sparsity penalty on the mask.",
    )
    parser.add_argument(
        "--n_generate", type=int, default=16,
        help="Number of greedy tokens to generate for entropy computation.",
    )
    parser.add_argument(
        "--baseline", type=str, default="zero",
        choices=["zero", "mean"],
        help="Baseline embedding method ('zero' or 'mean').",
    )
    parser.add_argument(
        "--detect_interactions", action="store_true",
        help="Run pair-wise interaction detection on critical tokens.",
    )
    parser.add_argument(
        "--interaction_top_k", type=int, default=5,
        help="Max number of critical tokens to use for interaction "
             "analysis (limits combinatorial explosion).",
    )

    args = parser.parse_args()

    # --- auto-generate output path if not specified ---
    if args.output_json is None:
        base = os.path.splitext(os.path.basename(args.input_json))[0]
        model_short = args.model_path.rstrip("/").split("/")[-1]
        args.output_json = os.path.join(
            os.path.dirname(args.input_json) or ".",
            f"diffmask_uncertainty_{model_short}_{base}.json",
        )

    # --- device and model ---
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=device,
        cache_dir=args.cache_dir,
    )

    # --- freeze all model parameters ---
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("[model] All model parameters frozen (requires_grad=False).")

    # --- load data ---
    data = load_input_data(args.input_json, limit=args.limit)

    print(
        f"\n[config] n_steps={args.n_steps}  lr={args.lr}  "
        f"sparsity_weight={args.sparsity_weight}  "
        f"n_generate={args.n_generate}  baseline={args.baseline}"
    )
    print(
        f"[config] detect_interactions={args.detect_interactions}  "
        f"interaction_top_k={args.interaction_top_k}"
    )
    print(f"[config] questions={len(data)}")
    print(f"[config] output -> {args.output_json}\n")

    # --- config record for output ---
    config = {
        "model_path": args.model_path,
        "input_json": args.input_json,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "sparsity_weight": args.sparsity_weight,
        "n_generate": args.n_generate,
        "baseline": args.baseline,
        "detect_interactions": args.detect_interactions,
        "interaction_top_k": args.interaction_top_k,
        "limit": args.limit,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    results: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    # --- main loop ---
    for qid, entry in tqdm(data.items(), desc="DiffMask uncertainty"):
        # Extract question from SAUP format.
        items = entry.get("items", [])
        if items:
            question = items[0].get("question", "")
        else:
            question = entry.get("question", "")

        if not question:
            errors.append({"qid": qid, "error": "No question text found."})
            continue

        result_entry: Dict[str, Any] = {"question": question}
        t0 = time.time()

        try:
            print(f"\n[qid={qid}] {question[:80]}...")

            # --- optimise mask ---
            mask_values, critical_tokens, unmasked_H, masked_H = optimize_mask(
                model=model,
                tokenizer=tokenizer,
                question=question,
                device=device,
                baseline_method=args.baseline,
                n_steps=args.n_steps,
                lr=args.lr,
                sparsity_weight=args.sparsity_weight,
                n_generate=args.n_generate,
            )

            # --- get token strings for output ---
            encoded = build_chat_input(question, tokenizer, device)
            input_ids = encoded["input_ids"]
            tokens_str = tokenizer.convert_ids_to_tokens(
                input_ids.squeeze().tolist()
            )

            # Per-token mask records
            token_mask_records = []
            for idx in range(len(mask_values)):
                token_mask_records.append({
                    "token_idx": int(idx),
                    "token_str": tokens_str[idx] if idx < len(tokens_str) else "",
                    "mask_value": float(mask_values[idx]),
                })

            # Inverted scores for display: tokens with low mask values are
            # the most important (their masking changes output the most).
            inverted_scores = 1.0 - mask_values
            norm_scores = normalize_scores(inverted_scores)
            print_token_scores(
                tokens_str, norm_scores, top_k=10,
                title=f"DiffMask Importance (1 - mask) | {qid}",
            )

            # Word-level aggregation of inverted mask scores
            word_scores = aggregate_subword_scores(
                norm_scores, tokenizer, input_ids,
            )

            result_entry.update({
                "mask_values": token_mask_records,
                "critical_tokens": critical_tokens,
                "unmasked_entropy": float(unmasked_H),
                "masked_entropy": float(masked_H),
                "entropy_change": float(masked_H - unmasked_H),
                "word_scores": word_scores,
            })

            # --- interaction detection ---
            if args.detect_interactions and len(critical_tokens) >= 2:
                # Limit to top-k critical tokens to avoid combinatorial
                # explosion.
                top_critical = critical_tokens[:args.interaction_top_k]
                print(
                    f"  [interactions] Analysing {len(top_critical)} "
                    f"critical tokens for pair-wise interactions..."
                )

                interactions = detect_interactions(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    critical_tokens=top_critical,
                    device=device,
                    baseline_method=args.baseline,
                    n_generate=args.n_generate,
                )

                n_super = sum(
                    1 for x in interactions
                    if x["interaction_type"] == "super-additive"
                )
                print(
                    f"  [interactions] Found {n_super} super-additive "
                    f"interactions out of {len(interactions)} pairs."
                )
                result_entry["interactions"] = interactions

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            print(f"[error] {qid}: {error_msg}")
            errors.append({"qid": qid, "error": error_msg})
            result_entry["error"] = error_msg

        elapsed = time.time() - t0
        result_entry["elapsed_seconds"] = round(elapsed, 2)
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
