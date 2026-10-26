#!/usr/bin/env python3
"""
diffmask_uncertainty.py  (VL variant for Qwen2.5-VL)

Faithful port of the text-only DiffMask (Task 17j).
Key change: the mask lives on the MERGED inputs_embeds (image + text).
Masking an image-token position replaces its embedding with the baseline
(zero or mean-vocab) — this gives us image-token importance scores too.
Word-level aggregation is restricted to the text span.
"""
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vl_attribution_utils import (
    aggregate_image_score,
    aggregate_subword_scores_text_only,
    build_vl_inputs,
    get_device,
    get_merged_inputs_embeds,
    load_input_data,
    load_vl_model_and_processor,
    normalize_scores,
    save_attribution_results,
)


def compute_baseline(model, method: str = "zero"):
    embed = model.get_input_embeddings()
    D = embed.weight.shape[1]
    device = embed.weight.device
    dtype = embed.weight.dtype
    if method == "zero":
        return torch.zeros(1, 1, D, device=device, dtype=dtype)
    if method == "mean":
        with torch.no_grad():
            m = embed.weight.mean(dim=0, keepdim=True).unsqueeze(0)
        return m.to(dtype)
    raise ValueError(method)


def compute_output_entropy(model, inputs_embeds, attention_mask, n_generate: int = 16):
    embed = model.get_input_embeddings()
    entropies: List[torch.Tensor] = []
    cur_e = inputs_embeds
    cur_m = attention_mask
    for _ in range(n_generate):
        out = model(inputs_embeds=cur_e, attention_mask=cur_m, use_cache=False)
        last = out.logits[:, -1, :].float()
        logp = torch.nn.functional.log_softmax(last, dim=-1)
        p = torch.nn.functional.softmax(last, dim=-1)
        ent = -(p * logp).sum(dim=-1)
        entropies.append(ent)
        greedy = last.argmax(dim=-1, keepdim=True)
        nxt = embed(greedy).detach().to(cur_e.dtype)
        cur_e = torch.cat([cur_e, nxt], dim=1)
        cur_m = torch.cat([cur_m, torch.ones(1, 1, device=cur_m.device, dtype=cur_m.dtype)], dim=1)
    return torch.stack(entropies).mean()


class DiffMask(nn.Module):
    def __init__(self, n_tokens, device):
        super().__init__()
        self.m = nn.Parameter(torch.ones(n_tokens, device=device))

    def forward(self, actual, baseline):
        mask = self.m.sigmoid().unsqueeze(-1).to(actual.dtype)
        return mask * actual + (1.0 - mask) * baseline

    def values(self):
        with torch.no_grad():
            return self.m.sigmoid().detach().cpu().numpy()

    def sparsity(self):
        return self.m.sigmoid().sum()


def optimize_mask(
    model, processor, image_path: str, question: str, device: str,
    baseline_method: str = "zero", n_steps: int = 30, lr: float = 0.1,
    sparsity_weight: float = 0.5, n_generate: int = 4, verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any], float, float]:
    """Same structure as the original: returns (mask_values, inputs,
    unmasked_entropy, masked_entropy)."""
    inputs = build_vl_inputs(image_path, question, processor, device)
    actual = get_merged_inputs_embeds(model, inputs).clone()  # (1, S, D)
    attention_mask = inputs["attention_mask"]
    seq_len = actual.shape[1]

    baseline_single = compute_baseline(model, method=baseline_method)
    baseline = baseline_single.expand_as(actual)

    with torch.no_grad():
        unmasked_H = compute_output_entropy(model, actual, attention_mask, n_generate=n_generate).item()

    dm = DiffMask(seq_len, device=device)
    opt = torch.optim.Adam(dm.parameters(), lr=lr)

    for step in range(n_steps):
        opt.zero_grad()
        masked = dm(actual, baseline)
        H = compute_output_entropy(model, masked, attention_mask, n_generate=n_generate)
        loss = -H + sparsity_weight * dm.sparsity()
        loss.backward()
        opt.step()
        if verbose and ((step + 1) % 25 == 0 or step == 0):
            mv = dm.values()
            n_masked = int((mv < 0.5).sum())
            print(f"    step {step+1}/{n_steps} loss={loss.item():.4f} "
                  f"H={H.item():.4f} n_masked(<0.5)={n_masked}")

    mv = dm.values()
    with torch.no_grad():
        masked_H = compute_output_entropy(
            model, dm(actual, baseline), attention_mask, n_generate=n_generate,
        ).item()
    return mv, inputs, unmasked_H, masked_H


def _compute_entropy_positions_masked(
    model, actual_embeds, baseline_embeds, attention_mask, positions, n_generate,
):
    """Binary mask helper matching the original's signature."""
    seq_len = actual_embeds.shape[1]
    mask = torch.ones(seq_len, 1, device=actual_embeds.device, dtype=actual_embeds.dtype)
    for pos in positions:
        if 0 <= pos < seq_len:
            mask[pos, 0] = 0.0
    masked_embeds = mask * actual_embeds + (1.0 - mask) * baseline_embeds
    with torch.no_grad():
        H = compute_output_entropy(model, masked_embeds, attention_mask, n_generate=n_generate).item()
    return H


def detect_interactions(
    model, processor, image_path: str, question: str,
    critical_tokens: List[Dict[str, Any]], device: str,
    baseline_method: str = "zero", n_generate: int = 4,
) -> List[Dict[str, Any]]:
    """Detect pair-wise super-additive interactions between critical tokens,
    matching the original's thresholding (> 1.2× additive for super-additive,
    < 0.8× for sub-additive)."""
    if len(critical_tokens) < 2:
        return []
    inputs = build_vl_inputs(image_path, question, processor, device)
    actual = get_merged_inputs_embeds(model, inputs).clone()
    attention_mask = inputs["attention_mask"]
    baseline_single = compute_baseline(model, method=baseline_method)
    baseline = baseline_single.expand_as(actual)

    unmasked_H = _compute_entropy_positions_masked(
        model, actual, baseline, attention_mask, positions=[], n_generate=n_generate,
    )
    single_effects: Dict[int, float] = {}
    for ct in critical_tokens:
        i = int(ct["token_idx"])
        H_i = _compute_entropy_positions_masked(
            model, actual, baseline, attention_mask, [i], n_generate=n_generate,
        )
        single_effects[i] = H_i - unmasked_H

    interactions: List[Dict[str, Any]] = []
    n = len(critical_tokens)
    for a in range(n):
        for b in range(a + 1, n):
            ii, ij = int(critical_tokens[a]["token_idx"]), int(critical_tokens[b]["token_idx"])
            ei, ej = single_effects[ii], single_effects[ij]
            H_both = _compute_entropy_positions_masked(
                model, actual, baseline, attention_mask, [ii, ij], n_generate=n_generate,
            )
            e_both = H_both - unmasked_H
            add = ei + ej
            if abs(add) < 1e-8:
                typ = "independent"
            elif e_both > add * 1.2:
                typ = "super-additive"
            elif e_both < add * 0.8:
                typ = "sub-additive"
            else:
                typ = "approximately-additive"
            interactions.append({
                "idx_i": ii, "idx_j": ij,
                "token_i": critical_tokens[a].get("token_str", ""),
                "token_j": critical_tokens[b].get("token_str", ""),
                "effect_i": float(ei), "effect_j": float(ej),
                "effect_both": float(e_both), "additive_sum": float(add),
                "interaction_type": typ,
            })
    return interactions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--sparsity_weight", type=float, default=0.5)
    p.add_argument("--n_generate", type=int, default=4)
    p.add_argument("--baseline", choices=["zero", "mean"], default="zero")
    p.add_argument("--detect_interactions", action="store_true",
                   help="Run pair-wise interaction detection on critical tokens.")
    p.add_argument("--interaction_top_k", type=int, default=5,
                   help="Cap critical tokens used for interaction analysis.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = get_device()
    model, processor = load_vl_model_and_processor(
        args.model_path, device=device, cache_dir=args.cache_dir, dtype=torch.float16
    )
    for pp in model.parameters():
        pp.requires_grad = False
    model.eval()

    data = load_input_data(args.input_json, limit=args.limit)
    config = {
        "model_path": args.model_path, "input_json": args.input_json, "method": "diffmask",
        "n_steps": args.n_steps, "lr": args.lr, "sparsity_weight": args.sparsity_weight,
        "n_generate": args.n_generate, "baseline": args.baseline,
        "timestamp": datetime.now().isoformat(),
    }

    results: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    for qid, entry in tqdm(data.items(), desc="VL diffmask"):
        question = entry.get("question") or entry.get("items", [{}])[0].get("question", "")
        image_path = entry.get("image_path")
        if not question or not image_path:
            errors.append({"qid": qid, "error": "missing question or image_path"})
            continue
        try:
            mv, inputs, unmasked_H, masked_H = optimize_mask(
                model, processor, image_path, question, device,
                baseline_method=args.baseline, n_steps=args.n_steps, lr=args.lr,
                sparsity_weight=args.sparsity_weight, n_generate=args.n_generate,
                verbose=args.verbose,
            )
            imp = 1.0 - mv
            norm = normalize_scores(imp)
            words = aggregate_subword_scores_text_only(
                norm, processor, inputs["input_ids"],
                inputs["text_span_start"], inputs["text_span_end"],
            )
            img_stats = aggregate_image_score(
                norm, inputs["image_span_start"], inputs["image_span_end"],
            )
            # Critical tokens: mask value < 0.5, ascending (most masked first)
            tok = processor.tokenizer
            ids1d = inputs["input_ids"].squeeze().tolist()
            tokens_str = tok.convert_ids_to_tokens(ids1d)
            critical_tokens = []
            for idx in range(len(mv)):
                if mv[idx] < 0.5:
                    critical_tokens.append({
                        "token_idx": int(idx),
                        "token_str": tokens_str[idx],
                        "mask_value": float(mv[idx]),
                        "decoded": tok.decode([ids1d[idx]]).strip(),
                        "in_text_span": (
                            inputs["text_span_start"] <= idx < inputs["text_span_end"]
                        ),
                    })
            critical_tokens.sort(key=lambda x: x["mask_value"])

            interactions = []
            if args.detect_interactions and len(critical_tokens) >= 2:
                interactions = detect_interactions(
                    model, processor, image_path, question,
                    critical_tokens[:args.interaction_top_k], device,
                    baseline_method=args.baseline, n_generate=args.n_generate,
                )

            results[qid] = {
                "question": entry.get("question_only", question),
                "image_path": image_path,
                "level": entry.get("level"),
                "ambiguity": entry.get("ambiguity"),
                "method": "diffmask",
                "unmasked_entropy": float(unmasked_H),
                "masked_entropy": float(masked_H),
                "critical_tokens": critical_tokens,
                "interactions": interactions,
                "word_scores": words,
                "image_summary": img_stats,
                "text_span_start": int(inputs["text_span_start"]),
                "text_span_end": int(inputs["text_span_end"]),
                "image_span_start": int(inputs["image_span_start"]),
                "image_span_end": int(inputs["image_span_end"]),
                "full_norm_scores": norm.tolist(),
            }
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
