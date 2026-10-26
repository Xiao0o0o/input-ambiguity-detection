#!/usr/bin/env python3
"""
gradient_attribution.py  (VL variant for Qwen2.5-VL)

Ports the text-only gradient attribution (Task 17a) to vision-language inputs.
Attribution is computed on the MERGED inputs_embeds (post vision-token fill-in),
so scores cover both image-patch tokens and text tokens. For downstream
perturbation we only select tokens within the text span, but the image-span
scores are preserved in the JSON for later analysis.

Methods:
  * vanilla   : L2 norm of gradient of loss (cross-entropy at last position)
               w.r.t. merged input embeddings.
  * integrated: Integrated Gradients (Sundararajan et al., 2017) along the
               linear path from zero embeddings to the actual merged
               embeddings, approximated with n_steps + 1 evaluations.

Output JSON (per qid) contains:
  * full_token_scores / full_norm_scores : [seq_len] (image + text)
  * word_scores       : word-level aggregation, TEXT span only
  * image_summary     : {image_total_score, image_mean_score, n_image_tokens}
  * text_span_start / text_span_end   : slice used for perturbation
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
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


# ── Vanilla gradient ─────────────────────────────────────────────────────────

def compute_vanilla_gradient(
    model,
    processor,
    image_path: str,
    question_text: str,
    device: str,
) -> Dict[str, Any]:
    inputs = build_vl_inputs(image_path, question_text, processor, device)

    # Merged embeddings (vision + text) — reusable for the gradient pass
    merged = get_merged_inputs_embeds(model, inputs)
    merged = merged.detach().clone().to(torch.float32).requires_grad_(True)

    attn_mask = inputs["attention_mask"]
    amp_device = device.split(":")[0]

    _amp_dtype = next(model.parameters()).dtype if next(model.parameters()).dtype in (torch.float16, torch.bfloat16) else torch.float16
    with torch.amp.autocast(device_type=amp_device, dtype=_amp_dtype):
        outputs = model(inputs_embeds=merged, attention_mask=attn_mask, use_cache=False)

    last_logits = outputs.logits[:, -1, :].float()
    greedy_token = last_logits.argmax(dim=-1)
    loss = torch.nn.functional.cross_entropy(last_logits, greedy_token)

    model.zero_grad()
    loss.backward()

    grad = merged.grad
    if grad is None:
        raise RuntimeError("No gradient on merged inputs_embeds.")
    grad_norms = torch.norm(grad.squeeze(0), p=2, dim=-1).detach().cpu().numpy()
    return {"inputs": inputs, "scores": grad_norms}


# ── Integrated gradients ─────────────────────────────────────────────────────

def _forward_grad(model, embeds, attn_mask, amp_device):
    _amp_dtype = next(model.parameters()).dtype if next(model.parameters()).dtype in (torch.float16, torch.bfloat16) else torch.float16
    with torch.amp.autocast(device_type=amp_device, dtype=_amp_dtype):
        out = model(inputs_embeds=embeds, attention_mask=attn_mask, use_cache=False)
    last_logits = out.logits[:, -1, :].float()
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
    model,
    processor,
    image_path: str,
    question_text: str,
    device: str,
    n_steps: int = 30,
) -> Dict[str, Any]:
    inputs = build_vl_inputs(image_path, question_text, processor, device)
    merged = get_merged_inputs_embeds(model, inputs).to(torch.float32)

    baseline = torch.zeros_like(merged)
    delta = merged - baseline
    accumulated = torch.zeros_like(merged)

    attn_mask = inputs["attention_mask"]
    amp_device = device.split(":")[0]

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interp = (baseline + alpha * delta).detach().requires_grad_(True)
        accumulated += _forward_grad(model, interp, attn_mask, amp_device)

    avg = accumulated / (n_steps + 1)
    ig = delta * avg
    ig_norms = torch.norm(ig.squeeze(0), p=2, dim=-1).detach().cpu().numpy()
    return {"inputs": inputs, "scores": ig_norms}


# ── Result packaging ─────────────────────────────────────────────────────────

def _package(raw_scores: np.ndarray, processor, inputs) -> Dict[str, Any]:
    """Build the same output shape as the original gradient_attribution.py
    (`token_scores` list of {token,score,normalized} + `word_scores`) plus
    VL-specific additions (`image_summary`, span slices, `full_norm_scores`)."""
    norm = normalize_scores(raw_scores)
    tok = processor.tokenizer
    ids = inputs["input_ids"].squeeze().tolist()
    tokens = tok.convert_ids_to_tokens(ids)
    token_scores = [
        {"token": t, "score": float(s), "normalized": float(n)}
        for t, s, n in zip(tokens, raw_scores, norm)
    ]
    words = aggregate_subword_scores_text_only(
        norm, processor, inputs["input_ids"],
        inputs["text_span_start"], inputs["text_span_end"],
    )
    img_stats = aggregate_image_score(
        norm, inputs["image_span_start"], inputs["image_span_end"],
    )
    return {
        "token_scores": token_scores,
        "word_scores": words,
        "image_summary": img_stats,
        "text_span_start": int(inputs["text_span_start"]),
        "text_span_end": int(inputs["text_span_end"]),
        "image_span_start": int(inputs["image_span_start"]),
        "image_span_end": int(inputs["image_span_end"]),
        "full_norm_scores": norm.tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--method", choices=["vanilla", "integrated", "both"], default="vanilla")
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = get_device()
    model, processor = load_vl_model_and_processor(
        args.model_path, device=device, cache_dir=args.cache_dir, dtype=torch.float16
    )
    # Freeze model params: VG only needs gradient w.r.t. input embeddings, not
    # model weights. Without this, autograd allocates ~param_count bf16 grad
    # buffers (~16GB for 8B, ~54GB for 27B), causing OOM on image-heavy models.
    for _p in model.parameters():
        _p.requires_grad = False
    print(f"[model] froze {sum(1 for _ in model.parameters())} parameter tensors")

    data = load_input_data(args.input_json, limit=args.limit)

    config = {
        "model_path": args.model_path,
        "input_json": args.input_json,
        "method": args.method,
        "n_steps": args.n_steps,
        "device": device,
        "timestamp": datetime.now().isoformat(),
    }

    results: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []

    run_vg = args.method in ("vanilla", "both")
    run_ig = args.method in ("integrated", "both")

    for qid, entry in tqdm(data.items(), desc="VL gradient attribution"):
        question = entry.get("question") or (entry.get("items", [{}])[0].get("question", ""))
        image_path = entry.get("image_path")
        if not question or not image_path:
            errors.append({"qid": qid, "error": "missing question or image_path"})
            continue

        rec: Dict[str, Any] = {
            "question": entry.get("question_only", question),
            "image_path": image_path,
            "level": entry.get("level"),
            "ambiguity": entry.get("ambiguity"),
        }
        try:
            if run_vg:
                r = compute_vanilla_gradient(model, processor, image_path, question, device)
                pkg = _package(r["scores"], processor, r["inputs"])
                # For compatibility with pipeline_b normalizer: flatten under results[qid]
                rec.update(pkg)
                rec["method"] = "vanilla_gradient"
            if run_ig:
                r = compute_integrated_gradients(
                    model, processor, image_path, question, device, n_steps=args.n_steps
                )
                pkg = _package(r["scores"], processor, r["inputs"])
                if run_vg:
                    rec["integrated_gradient"] = pkg
                else:
                    rec.update(pkg)
                    rec["method"] = "integrated_gradient"
            torch.cuda.empty_cache()
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors.append({"qid": qid, "error": msg})
            rec["error"] = msg
        results[qid] = rec

    out = {"config": config, "results": results}
    if errors:
        out["errors"] = errors
    save_attribution_results(out, args.output_json)
    print(f"[done] {sum(1 for v in results.values() if 'error' not in v)} ok, {len(errors)} failed")


if __name__ == "__main__":
    main()
