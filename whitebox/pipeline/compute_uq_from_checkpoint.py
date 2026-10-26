#!/usr/bin/env python3
"""Compute pipeline-B style UQ scores from a generation checkpoint.
Bypasses lm_polygraph (which has multiple incompatibilities with newer
transformers). Uses HF deberta-large-mnli directly + numpy estimators.

Reads `_gen_checkpoint_<method>_*.json` (per-qid `all_answers`) and writes
`pipeline_b_<method>_*.json` matching the original schema.

Usage:
    python compute_uq_from_checkpoint.py --checkpoint <path> --output <path>
            [--input_json <path>]   # for level/ambiguity if not in checkpoint
            [--device cuda:0]
            [--nli_model microsoft/deberta-large-mnli]
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# UQ estimators (replicating lm_polygraph behavior)
# ============================================================

def eigv_laplacian(W: np.ndarray) -> float:
    """EigValLaplacian: 1 + sum_{i: lambda_i < 1} (1 - lambda_i)
    over normalized graph Laplacian eigenvalues."""
    K = W.shape[0]
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    deg = np.where(deg > 0, deg, 1.0)
    Dinv = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(K) - Dinv @ W @ Dinv
    eigs = np.sort(np.linalg.eigvalsh(L))
    return 1.0 + sum(max(0.0, 1.0 - e) for e in eigs)


def deg_mat(W: np.ndarray) -> float:
    """DegMat: mean off-diagonal (1 - W). Higher = more disagreement."""
    K = W.shape[0]
    W = (W + W.T) / 2.0
    mask = ~np.eye(K, dtype=bool)
    return float(np.mean(1.0 - W[mask]))


def num_sem_sets_from_classes(classes: np.ndarray) -> int:
    """Number of unique semantic classes (transitive entailment groups)."""
    flat = classes.flatten()
    flat = flat[flat >= 0]
    return len(np.unique(flat)) if len(flat) else 1


def sem_entropy_freq(classes: np.ndarray) -> float:
    flat = classes.flatten()
    flat = flat[flat >= 0]
    if not len(flat): return 0.0
    _, c = np.unique(flat, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def cluster_sem_classes(entail: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Hard clustering: edge if entail_ij > threshold (both directions). Returns
    K×K class matrix with same class id for connected components."""
    K = entail.shape[0]
    # Symmetric entailment: both i→j and j→i exceed threshold
    sim = (entail > threshold) & (entail.T > threshold)
    # Union-find
    parent = list(range(K))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(K):
        for j in range(i+1, K):
            if sim[i,j]: union(i, j)
    classes = np.zeros((K, K), dtype=np.int64)
    for i in range(K):
        for j in range(K):
            classes[i, j] = find(i) if find(i) == find(j) else -1
    return classes


# ============================================================
# NLI batch
# ============================================================

def compute_nli_matrices(answers, tokenizer, model, device, max_len=256):
    """Given K answers, return entail/contra K×K matrices of probabilities.
    label index for deberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment."""
    K = len(answers)
    # Build all K*K pairs (i, j) i!=j; diagonal set to 1.0 entail / 0.0 contra
    pairs = []
    for i in range(K):
        for j in range(K):
            if i == j: continue
            pairs.append((answers[i], answers[j]))

    inputs = tokenizer(
        [p[0] for p in pairs], [p[1] for p in pairs],
        return_tensors="pt", padding=True, truncation=True, max_length=max_len,
    ).to(device)

    with torch.no_grad():
        out = model(**inputs).logits  # (B, 3)
    probs = torch.softmax(out, dim=-1).cpu().numpy()
    contra = probs[:, 0]
    entail = probs[:, 2]

    M_e = np.eye(K, dtype=np.float32)        # diagonal = 1 (self entail)
    M_c = np.zeros((K, K), dtype=np.float32) # diagonal = 0 (no self contra)
    p = 0
    for i in range(K):
        for j in range(K):
            if i == j: continue
            M_e[i, j] = entail[p]
            M_c[i, j] = contra[p]
            p += 1
    return M_e, M_c


# ============================================================
# lm_polygraph-equivalent variant (used for the text-QA numbers)
# ============================================================

def compute_nli_matrices_polygraph(answers, tokenizer, model, device, batch_size=16):
    """Same as lm_polygraph's SemanticMatrixCalculator: NLI on every ordered
    pair of UNIQUE answers (self-pairs included), mapped back to K×K."""
    uniq, inv = np.unique(answers, return_inverse=True)
    pairs = [(a, b) for a in uniq for b in uniq]
    probs = []
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        enc = tokenizer([p[0] for p in chunk], [p[1] for p in chunk],
                        return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probs).reshape(len(uniq), len(uniq), -1)[inv][:, inv]
    return probs[..., 2], probs[..., 0]


def eigv_laplacian_polygraph(W: np.ndarray) -> float:
    """lm_polygraph EigValLaplacian: diagonal kept (NLI self-entailment)."""
    W = (W + W.T) / 2.0
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return float(sum(max(0.0, 1.0 - e) for e in np.linalg.eigvalsh(L)))


def deg_mat_polygraph(W: np.ndarray) -> float:
    """lm_polygraph DegMat: trace(K·I - D) / K²."""
    K = W.shape[0]
    W = (W + W.T) / 2.0
    return float(np.trace(K - np.diag(W.sum(axis=1))) / K ** 2)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to _gen_checkpoint_*.json")
    ap.add_argument("--output", required=True, help="Path for pipeline_b output JSON")
    ap.add_argument("--input_json", default=None, help="SAUP input JSON (for level/ambiguity)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nli_model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--cache_dir", default=os.environ.get("HF_CACHE_DIR"))
    ap.add_argument("--method", default=None, help="method name for config block")
    ap.add_argument("--eigv_variant", choices=["zero_diag", "polygraph"], default="zero_diag",
                    help="zero_diag: Laplacian on off-diagonal NLI entailment (vision-QA "
                         "numbers in the paper). polygraph: lm_polygraph EigValLaplacian/"
                         "DegMat with NLI self-entailment on the diagonal (text-QA numbers).")
    args = ap.parse_args()

    print(f"loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = json.load(open(args.checkpoint, encoding="utf-8"))
    print(f"  qids: {len(ckpt)}", flush=True)

    if args.input_json:
        meta = json.load(open(args.input_json, encoding="utf-8"))
    else:
        meta = {}

    print(f"loading NLI: {args.nli_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.nli_model, cache_dir=args.cache_dir)
    nli = AutoModelForSequenceClassification.from_pretrained(
        args.nli_model, cache_dir=args.cache_dir,
    ).to(args.device).eval()

    per_question = {}
    for qid in tqdm(sorted(ckpt.keys()), desc="UQ"):
        rec = ckpt[qid]
        answers = rec.get("all_answers") or rec.get("answers") or []
        if len(answers) < 2:
            continue
        K = len(answers)
        if args.eigv_variant == "polygraph":
            M_e, M_c = compute_nli_matrices_polygraph(answers, tok, nli, args.device)
            eigv = eigv_laplacian_polygraph(M_e)
            deg = deg_mat_polygraph(M_e)
        else:
            M_e, M_c = compute_nli_matrices(answers, tok, nli, args.device)
            eigv = eigv_laplacian(M_e)
            deg  = deg_mat(M_e)
        classes = cluster_sem_classes(M_e, threshold=0.5)
        # disagree affinity: use NLI contradiction prob (higher → more disagree → higher eigv)
        # Convert to "agreement" graph: 1 - contra (entry-wise), then standard laplacian.
        M_agree = 1.0 - M_c
        np.fill_diagonal(M_agree, 1.0)
        eigv_disagree = eigv_laplacian(M_agree)
        deg_disagree  = deg_mat(M_agree)
        nss = num_sem_sets_from_classes(classes)
        se = sem_entropy_freq(classes)

        meta_q = meta.get(qid, {})
        per_question[qid] = {
            "EigV_entail": float(eigv),
            "DegMat_entail": float(deg),
            "EigV_disagree": float(eigv_disagree),
            "DegMat_disagree": float(deg_disagree),
            "NumSemSets": int(nss),
            "SemEntropy": float(se),
            "n_answers": K,
            "answers": answers,
            "ambiguity": rec.get("ambiguity") or meta_q.get("ambiguity"),
            "level": meta_q.get("level"),
        }

    out = {
        "config": {"method": args.method or "unknown",
                   "n_questions": len(per_question),
                   "nli_model": args.nli_model,
                   "eigv_variant": args.eigv_variant},
        "per_question": per_question,
    }
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # Quick AUROC summary
    from sklearn.metrics import roc_auc_score
    eigv_vals, deg_vals, lvls, ambs = [], [], [], []
    for qid, v in per_question.items():
        if v.get("ambiguity") is None: continue
        eigv_vals.append(v["EigV_entail"])
        deg_vals.append(v["DegMat_entail"])
        ambs.append(1 if v["ambiguity"] == "ambiguous" else 0)
        if v.get("level") is not None: lvls.append(int(v["level"]))
    if len(set(ambs)) >= 2:
        a_e = roc_auc_score(ambs, eigv_vals)
        a_d = roc_auc_score(ambs, deg_vals)
        print(f"\nbinary AUROC: EigV_entail={a_e:.4f}  DegMat_entail={a_d:.4f}  (n={len(ambs)})")
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
