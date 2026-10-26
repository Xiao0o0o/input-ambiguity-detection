#!/usr/bin/env python3
"""
logit_lens_uncertainty.py  (VL variant for Qwen2.5-VL)

Faithful port of the text-only logit-lens uncertainty analysis (Task 17c).
Same five analyses as the original:
  1. Logit lens: project each layer's hidden state through lm_head to the
     vocabulary and record top-k tokens + probs per (layer, position).
  2. Layer-wise entropy matrix [n_layers, seq_len].
  3. Decision-layer detection (sharp entropy drops per position).
  4. Persistent-uncertainty positions (entropy > per-layer threshold in >
     layer_fraction of layers).
  5. Hidden-state norm trajectory + layer-to-layer delta.

Output matches the original schema; additionally, `word_scores` is added
(per-position mean entropy aggregated to words over the text span) so
Pipeline B downstream can consume this method's signal uniformly.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vl_attribution_utils import (
    aggregate_image_score,
    aggregate_subword_scores_text_only,
    build_vl_inputs,
    get_device,
    load_input_data,
    load_vl_model_and_processor,
    normalize_scores,
    save_attribution_results,
)


@torch.no_grad()
def forward_hidden(model, inputs):
    fwd = {k: v for k, v in inputs.items() if k in (
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
        "pixel_values_videos", "video_grid_thw",
    )}
    out = model(**fwd, output_hidden_states=True, use_cache=False)
    return out.logits, out.hidden_states


# ── 1. Logit lens ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_logit_lens(
    model, hidden_states, input_ids, tokenizer, top_k: int = 5,
) -> Dict[str, Any]:
    """Project each layer's hidden state through lm_head; return top-k tokens."""
    n_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    input_token_ids = input_ids.squeeze(0).tolist()
    input_tokens = [tokenizer.decode([tid]).strip() for tid in input_token_ids]

    top_tokens_all: List[List[List[str]]] = []
    top_ids_all: List[List[List[int]]] = []
    top_probs_all: List[List[List[float]]] = []

    lm_head = model.lm_head
    for li in range(1, len(hidden_states)):
        h = hidden_states[li].squeeze(0)
        logits = lm_head(h).float()
        probs = F.softmax(logits, dim=-1)
        tp, tid = torch.topk(probs, k=top_k, dim=-1)
        layer_tokens: List[List[str]] = []
        layer_ids: List[List[int]] = []
        layer_probs: List[List[float]] = []
        for pos in range(seq_len):
            ids_list = tid[pos].tolist()
            probs_list = tp[pos].tolist()
            tokens_list = [tokenizer.decode([t]).strip() for t in ids_list]
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


def summarize_logit_lens(res: Dict[str, Any], max_positions: int = 50) -> Dict[str, Any]:
    n_layers = res["n_layers"]
    seq_len = res["seq_len"]
    rep_layers = sorted(set([
        0, 1, n_layers // 4, n_layers // 2, 3 * n_layers // 4,
        max(0, n_layers - 2), n_layers - 1,
    ]))
    rep_layers = [l for l in rep_layers if l < n_layers]
    positions_to_show = min(seq_len, max_positions)
    summary: Dict[str, Any] = {
        "n_layers": n_layers,
        "seq_len": seq_len,
        "representative_layers": rep_layers,
        "input_tokens": res["input_tokens"][:positions_to_show],
        "layer_predictions": {},
    }
    for l in rep_layers:
        key = f"layer_{l}"
        summary["layer_predictions"][key] = []
        for pos in range(positions_to_show):
            summary["layer_predictions"][key].append({
                "position": pos,
                "input_token": res["input_tokens"][pos] if pos < len(res["input_tokens"]) else "",
                "top_tokens": res["top_tokens"][l][pos],
                "top_probs": [round(p, 6) for p in res["top_probs"][l][pos]],
            })
    return summary


# ── 2. Layer entropy ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_layer_entropy(model, hidden_states) -> np.ndarray:
    n_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    lm_head = model.lm_head
    entropy = np.zeros((n_layers, seq_len), dtype=np.float64)
    for li in range(1, len(hidden_states)):
        h = hidden_states[li].squeeze(0)
        logits = lm_head(h).float()
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        ent = -(p * logp).sum(dim=-1)
        entropy[li - 1] = ent.cpu().numpy()
    return entropy


# ── 3. Decision layers ───────────────────────────────────────────────────────

def find_decision_layers(entropy: np.ndarray, threshold: float = 0.5) -> Dict[int, List[int]]:
    n_layers, seq_len = entropy.shape
    out: Dict[int, List[int]] = {}
    for pos in range(seq_len):
        layers = []
        for l in range(1, n_layers):
            prev = entropy[l - 1, pos]
            cur = entropy[l, pos]
            if prev > 1e-8 and (prev - cur) / prev > threshold:
                layers.append(l)
        if layers:
            out[pos] = layers
    return out


# ── 4. Persistent uncertainty positions ──────────────────────────────────────

def find_persistent_positions(entropy: np.ndarray, percentile: float = 50, layer_fraction: float = 0.7) -> List[int]:
    n_layers, seq_len = entropy.shape
    thresholds = np.percentile(entropy, percentile, axis=1)
    above = np.zeros(seq_len, dtype=np.int64)
    for l in range(n_layers):
        above += (entropy[l] > thresholds[l]).astype(np.int64)
    min_layers = int(n_layers * layer_fraction)
    return [int(p) for p in range(seq_len) if above[p] > min_layers]


# ── 5. Norm trajectory ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_norm_trajectory(hidden_states) -> Tuple[np.ndarray, np.ndarray]:
    n_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    norms = np.zeros((n_layers, seq_len), dtype=np.float64)
    for li in range(1, len(hidden_states)):
        h = hidden_states[li].squeeze(0).float()
        norms[li - 1] = torch.norm(h, dim=-1).cpu().numpy()
    delta = norms[1:] - norms[:-1] if n_layers > 1 else np.zeros((0, seq_len))
    return norms, delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top_k_vocab", type=int, default=5)
    p.add_argument("--entropy_drop_threshold", type=float, default=0.5)
    p.add_argument("--persistent_percentile", type=float, default=50.0)
    p.add_argument("--persistent_layer_fraction", type=float, default=0.7)
    p.add_argument("--save_full_heatmap", action="store_true",
                   help="Save full [n_layers x seq_len] entropy heatmap and norm trajectory to JSON. "
                        "Off by default because output files can be large for long sequences.")
    args = p.parse_args()

    device = get_device()
    model, processor = load_vl_model_and_processor(
        args.model_path, device=device, cache_dir=args.cache_dir, dtype=torch.float16
    )
    tokenizer = processor.tokenizer
    data = load_input_data(args.input_json, limit=args.limit)

    config = {
        "model_path": args.model_path, "input_json": args.input_json, "method": "logit_lens",
        "top_k_vocab": args.top_k_vocab,
        "entropy_drop_threshold": args.entropy_drop_threshold,
        "persistent_percentile": args.persistent_percentile,
        "persistent_layer_fraction": args.persistent_layer_fraction,
        "save_full_heatmap": args.save_full_heatmap,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    results: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    for qid, entry in tqdm(data.items(), desc="VL logit-lens"):
        question = entry.get("question") or entry.get("items", [{}])[0].get("question", "")
        image_path = entry.get("image_path")
        if not question or not image_path:
            errors.append({"qid": qid, "error": "missing question or image_path"})
            continue
        try:
            inputs = build_vl_inputs(image_path, question, processor, device)
            _, hidden_states = forward_hidden(model, inputs)
            input_ids = inputs["input_ids"]

            # 1. Logit lens top-k per (layer, position)
            ll_res = compute_logit_lens(model, hidden_states, input_ids, tokenizer, top_k=args.top_k_vocab)
            ll_summary = summarize_logit_lens(ll_res, max_positions=50)

            # 2. Layer entropy
            entropy = compute_layer_entropy(model, hidden_states)

            # 3. Decision layers
            decision = find_decision_layers(entropy, threshold=args.entropy_drop_threshold)

            # 4. Persistent positions
            persistent = find_persistent_positions(
                entropy, percentile=args.persistent_percentile,
                layer_fraction=args.persistent_layer_fraction,
            )

            # 5. Norm trajectory
            norms, delta_norms = compute_norm_trajectory(hidden_states)

            # Token strings
            token_ids_list = input_ids.squeeze(0).tolist()
            token_strings = [tokenizer.decode([tid]).strip() for tid in token_ids_list]

            # Per-position uncertainty = mean entropy across layers, normalized for word_scores
            per_pos_entropy = entropy.mean(axis=0)
            norm_per_pos = normalize_scores(per_pos_entropy)
            words = aggregate_subword_scores_text_only(
                norm_per_pos, processor, input_ids,
                inputs["text_span_start"], inputs["text_span_end"],
            )
            img_stats = aggregate_image_score(
                norm_per_pos, inputs["image_span_start"], inputs["image_span_end"],
            )

            # Persistent info with token + layer stats
            persistent_info = [
                {
                    "position": pos,
                    "token": token_strings[pos] if pos < len(token_strings) else "",
                    "mean_entropy": round(float(entropy[:, pos].mean()), 6),
                    "max_entropy": round(float(entropy[:, pos].max()), 6),
                    "min_entropy": round(float(entropy[:, pos].min()), 6),
                } for pos in persistent
            ]

            # Top-uncertain positions by mean entropy across layers
            pos_mean_entropy = entropy.mean(axis=0)
            top_k_uncertain = min(20, entropy.shape[1])
            top_un_idx = np.argsort(-pos_mean_entropy)[:top_k_uncertain]
            top_uncertain_positions = [
                {
                    "position": int(i),
                    "token": token_strings[i] if i < len(token_strings) else "",
                    "mean_entropy": round(float(pos_mean_entropy[i]), 6),
                } for i in top_un_idx
            ]

            # Top delta-norm positions
            top_norm_delta_positions: List[Dict[str, Any]] = []
            if delta_norms.shape[0] > 0:
                max_delta_per_pos = np.abs(delta_norms).max(axis=0)
                top_d_idx = np.argsort(-max_delta_per_pos)[:min(20, entropy.shape[1])]
                top_norm_delta_positions = [
                    {
                        "position": int(i),
                        "token": token_strings[i] if i < len(token_strings) else "",
                        "max_abs_delta_norm": round(float(max_delta_per_pos[i]), 4),
                    } for i in top_d_idx
                ]

            mean_entropy_per_layer = [round(float(entropy[l].mean()), 6) for l in range(entropy.shape[0])]
            mean_entropy_per_position = [round(float(x), 6) for x in pos_mean_entropy]

            rec: Dict[str, Any] = {
                "question": entry.get("question_only", question),
                "image_path": image_path,
                "level": entry.get("level"),
                "ambiguity": entry.get("ambiguity"),
                "method": "logit_lens",
                "n_layers": int(entropy.shape[0]),
                "seq_len": int(entropy.shape[1]),
                "tokens": token_strings,
                "logit_lens_summary": ll_summary,
                "mean_entropy_per_layer": mean_entropy_per_layer,
                "mean_entropy_per_position": mean_entropy_per_position,
                "decision_layers": {str(pos): layers for pos, layers in decision.items()},
                "persistent_uncertainty_positions": persistent_info,
                "top_uncertain_positions": top_uncertain_positions,
                "top_norm_delta_positions": top_norm_delta_positions,
                # downstream-consumable fields
                "word_scores": words,
                "image_summary": img_stats,
                "text_span_start": int(inputs["text_span_start"]),
                "text_span_end": int(inputs["text_span_end"]),
                "image_span_start": int(inputs["image_span_start"]),
                "image_span_end": int(inputs["image_span_end"]),
                "full_norm_scores": norm_per_pos.tolist(),
            }
            if args.save_full_heatmap:
                rec["entropy_heatmap"] = [
                    [round(float(entropy[l, p]), 6) for p in range(entropy.shape[1])]
                    for l in range(entropy.shape[0])
                ]
                rec["norm_trajectory"] = [
                    [round(float(norms[l, p]), 4) for p in range(norms.shape[1])]
                    for l in range(norms.shape[0])
                ]
                rec["delta_norm_trajectory"] = [
                    [round(float(delta_norms[l, p]), 4) for p in range(delta_norms.shape[1])]
                    for l in range(delta_norms.shape[0])
                ]

            results[qid] = rec

            del hidden_states
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            errors.append({"qid": qid, "error": f"{type(e).__name__}: {e}"})

    out = {"config": config, "results": results}
    if errors:
        out["errors"] = errors
    save_attribution_results(out, args.output_json)
    print(f"[done] {len(results)} ok, {len(errors)} failed")


if __name__ == "__main__":
    main()
