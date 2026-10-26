#!/usr/bin/env python3
"""Build AmbigQA-with-context input for Pipeline B.

Uses sewon/ambig_qa 'full' validation split. For each example, fetches a short
Wikipedia summary for the first `viewed_doc_titles` entry via the REST API and
prepends it to the question as context. Writes both an input JSON (matching
the AQuA-text schema: {qid: {question, question_only, ambiguity, level, items}})
and a labels JSON ({qid: 'certain'|'uncertain'}).

Output:
  data/ambigqa_with_ctx_500_input.json
  data/ambigqa_with_ctx_500_labels_certain.json

The released data/ambigqa_with_ctx_500_input.json is the exact file used in the
paper. Wikipedia summaries fetched for it are cached in data/build/_wiki_summary_cache.json;
titles that failed at build time are not cached and are re-fetched on a rebuild
(set WIKI_UA to a contact User-Agent), so a rebuild can differ on a few items.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import httpx
from datasets import load_dataset
from tqdm import tqdm

UA = os.environ.get("WIKI_UA", "ambiguity-research/0.1 (research use)")
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
DATA_DIR = Path(__file__).resolve().parent.parent  # data/
CACHE_PATH = Path(__file__).resolve().parent / "_wiki_summary_cache.json"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.load(open(CACHE_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    json.dump(cache, open(CACHE_PATH, "w", encoding="utf-8"))


def fetch_wiki_summary(title: str, client: httpx.Client, cache: dict) -> Optional[str]:
    if title in cache:
        return cache[title]
    safe = title.replace(" ", "_")
    url = WIKI_SUMMARY_URL.format(title=safe)
    try:
        r = client.get(url, headers={"User-Agent": UA}, timeout=10.0)
        if r.status_code == 200:
            extract = r.json().get("extract", "") or ""
            cache[title] = extract
            return extract
        elif r.status_code == 404:
            cache[title] = ""  # cache miss to avoid retry
            return ""
        else:
            return None
    except Exception as e:
        print(f"  [wiki] {title}: {type(e).__name__}: {e}")
        return None


def build_prompt(question: str, ctx: str) -> str:
    ctx_clean = (ctx or "").strip()
    if ctx_clean:
        # Truncate long context to ~3000 chars (4 paragraphs typically)
        if len(ctx_clean) > 3000:
            ctx_clean = ctx_clean[:3000].rsplit(" ", 1)[0] + "..."
        return (
            "You are a helpful QA assistant.\n"
            "Use the following context to answer the question concisely.\n\n"
            f"Context:\n{ctx_clean}\n\n"
            f"Question:\n{question}"
        )
    else:
        return (
            "You are a helpful QA assistant.\n"
            "Answer the question concisely.\n\n"
            f"Question:\n{question}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="Number of items to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_input", default=str(DATA_DIR / "ambigqa_with_ctx_500_input.json"))
    ap.add_argument("--out_labels", default=str(DATA_DIR / "ambigqa_with_ctx_500_labels_certain.json"))
    ap.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    args = ap.parse_args()

    print(f"[load] sewon/ambig_qa full validation")
    ds = load_dataset("sewon/ambig_qa", "full", cache_dir=args.cache_dir, split="validation")
    print(f"  total: {len(ds)}")

    # Bucket by type for stratified sampling
    ambig_idx, unambig_idx = [], []
    for i, ex in enumerate(ds):
        t = ex["annotations"]["type"][0] if ex["annotations"]["type"] else "unknown"
        if t == "multipleQAs":
            ambig_idx.append(i)
        elif t == "singleAnswer":
            unambig_idx.append(i)
    print(f"  ambig (multipleQAs): {len(ambig_idx)}; unambig (singleAnswer): {len(unambig_idx)}")

    rng = random.Random(args.seed)
    half = args.n // 2
    rng.shuffle(ambig_idx); rng.shuffle(unambig_idx)
    pick = sorted(ambig_idx[:half] + unambig_idx[: args.n - half])

    cache = load_cache()
    print(f"[wiki] cache size: {len(cache)}")

    out_data, out_labels = {}, {}
    n_with_ctx = 0
    with httpx.Client() as client:
        for i, idx in enumerate(tqdm(pick, desc="build")):
            ex = ds[int(idx)]
            qid = f"ambigqa_val_{i:04d}"
            q = ex["question"]
            ambig_type = ex["annotations"]["type"][0] if ex["annotations"]["type"] else "unknown"
            ambig_label = "ambiguous" if ambig_type == "multipleQAs" else "unambiguous"
            titles = ex["viewed_doc_titles"] or []
            ctx = ""
            for t in titles[:3]:
                got = fetch_wiki_summary(t, client, cache)
                if got:
                    ctx = got
                    break
            if ctx:
                n_with_ctx += 1
            prompt = build_prompt(q, ctx)
            out_data[qid] = {
                "question": prompt,
                "question_only": q,
                "ambiguity": ambig_label,
                "level": 1 if ambig_label == "ambiguous" else 0,
                "items": [{"question": prompt}],
                "wiki_titles": titles[:3],
            }
            out_labels[qid] = "uncertain" if ambig_label == "ambiguous" else "certain"
            if (i + 1) % 50 == 0:
                save_cache(cache)

    save_cache(cache)
    json.dump(out_data, open(args.out_input, "w", encoding="utf-8"), indent=2)
    json.dump(out_labels, open(args.out_labels, "w", encoding="utf-8"), indent=2)
    print(f"\n[saved] {args.out_input}  ({len(out_data)} items, {n_with_ctx} with wiki context)")
    print(f"[saved] {args.out_labels}")
    print(f"ambig/unambig: {sum(1 for v in out_data.values() if v['ambiguity']=='ambiguous')}/{sum(1 for v in out_data.values() if v['ambiguity']=='unambiguous')}")


if __name__ == "__main__":
    main()
