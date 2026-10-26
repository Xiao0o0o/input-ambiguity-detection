#!/usr/bin/env python3
"""Binary ambiguity-detection AUROC from one or more UQ score files.

Reads `pipeline_b_*.json` files (written by compute_uq_from_checkpoint.py or
pipeline_b_perturb_and_uq*.py), whose `per_question[qid]` carries the UQ
scores and an `ambiguity` label ("ambiguous" = positive class).

Usage:
    python compute_auroc.py results/pipeline_b_logit_lens.json \
        [results/pipeline_b_baseline_n10.json ...] \
        [--metrics EigV_entail DegMat_entail] [--intersect] [--output auroc.json]

--intersect     restrict every file to the qids that have a valid score in all
                files (the paper evaluates on items where every method returns
                a valid score).
--tie_decimals  also report AUROC after rounding scores to this many decimals.
                Many questions get the same EigV up to float noise (e.g. when
                all sampled answers are identical); rounding makes those ties
                explicit, so the number no longer depends on NLI float noise.
"""
import argparse
import json
import math
import os

from sklearn.metrics import roc_auc_score

DEFAULT_METRICS = ["EigV_entail", "DegMat_entail", "EigV_disagree", "SemEntropy"]


def _valid(v):
    return isinstance(v, (int, float)) and not math.isnan(v)


def load_scores(path, metrics):
    pq = json.load(open(path, encoding="utf-8")).get("per_question", {})
    out = {}
    for qid, rec in pq.items():
        amb = rec.get("ambiguity")
        if amb not in ("ambiguous", "unambiguous"):
            continue
        out[qid] = {"label": int(amb == "ambiguous"),
                    **{m: rec[m] for m in metrics if _valid(rec.get(m))}}
    return out


def auroc(records, metric, decimals=None):
    ys, xs = [], []
    for r in records.values():
        if metric in r:
            ys.append(r["label"])
            xs.append(r[metric] if decimals is None else round(r[metric], decimals))
    if len(set(ys)) < 2:
        return None, len(ys)
    return float(roc_auc_score(ys, xs)), len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--intersect", action="store_true")
    ap.add_argument("--tie_decimals", type=int, default=6)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    data = {f: load_scores(f, args.metrics) for f in args.files}
    if args.intersect:
        def valid_qids(d):
            present = [m for m in args.metrics if any(m in r for r in d.values())]
            return {q for q, r in d.items() if all(m in r for m in present)}
        common = set.intersection(*(valid_qids(d) for d in data.values()))
        data = {f: {q: r for q, r in d.items() if q in common} for f, d in data.items()}

    summary = {}
    for f, recs in data.items():
        name = os.path.basename(f)
        summary[name] = {}
        print(f"\n== {name}")
        for m in args.metrics:
            a, n = auroc(recs, m)
            if a is None:
                continue
            a_tie, _ = auroc(recs, m, args.tie_decimals)
            summary[name][m] = {"auroc": a, f"auroc_ties_{args.tie_decimals}dp": a_tie, "n": n}
            print(f"  {m:15s} AUROC={a:.4f}   (scores rounded to {args.tie_decimals} dp: {a_tie:.4f})   n={n}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        json.dump(summary, open(args.output, "w", encoding="utf-8"), indent=2)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
