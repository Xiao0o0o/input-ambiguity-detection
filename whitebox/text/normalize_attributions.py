#!/usr/bin/env python3
"""Generic attribution normalization for one (model, dataset) cell output dir.

Normalizes:
  - vanilla_gradient: lifts results[qid]['vanilla_gradient']['word_scores'] to
                      results[qid]['word_scores']. Idempotent.
  - logit_lens:       re-aggregates token-level entropy to word-level by
                      borrowing VG's word→token_indices mapping (raw LL output
                      lives at <out>/logit_lens_raw.json). Writes
                      <out>/logit_lens.json.
  - diffmask:         already correct (has 'word_scores' inside results[qid]).
                      Verified, no-op.

Pass --output_dir; this script tries each step and skips if files are missing.
"""
import argparse
import json
from pathlib import Path


def normalize_vg(out_dir: Path) -> None:
    p = out_dir / "vanilla_gradient.json"
    if not p.exists():
        print(f"[vg] skip (no {p})"); return
    d = json.load(open(p, encoding="utf-8"))
    if "results" not in d:
        print(f"[vg] unexpected top-level"); return
    res_in = d["results"]
    changed = 0
    res_out = {}
    for qid, v in res_in.items():
        if isinstance(v, dict) and "word_scores" in v:
            res_out[qid] = v  # already normalized
        elif isinstance(v, dict) and "vanilla_gradient" in v:
            ws = v.get("vanilla_gradient", {}).get("word_scores", [])
            res_out[qid] = {"question": v.get("question"), "word_scores": ws}
            changed += 1
        else:
            res_out[qid] = v
    json.dump({"config": d.get("config", {}), "results": res_out}, open(p, "w", encoding="utf-8"))
    print(f"[vg] normalized: {len(res_out)} qids (lifted {changed})")


def normalize_ll(out_dir: Path) -> None:
    raw = out_dir / "logit_lens_raw.json"
    vg_path = out_dir / "vanilla_gradient.json"
    out = out_dir / "logit_lens.json"
    if not raw.exists():
        # Maybe LL was saved already at logit_lens.json directly — try that.
        alt = out_dir / "logit_lens.json"
        if alt.exists():
            d = json.load(open(alt, encoding="utf-8"))
            if "results" in d:
                print(f"[ll] {alt} already normalized"); return
            else:
                raw = alt  # treat alt as raw
        else:
            print(f"[ll] skip (no raw at {raw})"); return
    if not vg_path.exists():
        print(f"[ll] skip (need vanilla_gradient.json for word boundaries)"); return
    vg = json.load(open(vg_path, encoding="utf-8")).get("results", {})
    ll = json.load(open(raw, encoding="utf-8"))
    res_out = {}
    for qid in vg:
        if qid not in ll: continue
        ll_rec = ll[qid]
        scores = ll_rec.get("mean_entropy_per_position", [])
        ws_vg = vg[qid].get("word_scores", [])
        ws_out = []
        for w in ws_vg:
            tidxs = [i for i in w.get("token_indices", []) if isinstance(i, int) and 0 <= i < len(scores)]
            if not tidxs:
                continue
            avg = sum(float(scores[i]) for i in tidxs) / len(tidxs)
            ws_out.append({"word": w["word"], "score": avg, "token_indices": tidxs})
        res_out[qid] = {"question": vg[qid].get("question"), "word_scores": ws_out}
    json.dump({"config": {"method": "logit_lens"}, "results": res_out}, open(out, "w", encoding="utf-8"))
    print(f"[ll] normalized using VG boundaries: {len(res_out)} qids -> {out}")


def verify_diffmask(out_dir: Path) -> None:
    p = out_dir / "diffmask.json"
    if not p.exists():
        print(f"[dm] skip (no {p})"); return
    d = json.load(open(p, encoding="utf-8"))
    if "results" not in d:
        print(f"[dm] unexpected top-level"); return
    first = next(iter(d["results"].values()), {})
    if "word_scores" in first:
        print(f"[dm] already correct: {len(d['results'])} qids")
    else:
        print(f"[dm] missing 'word_scores' — keys: {list(first.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    normalize_vg(out)
    normalize_ll(out)
    verify_diffmask(out)


if __name__ == "__main__":
    main()
