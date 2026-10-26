"""
Compute black-box uncertainty scores for ABG-CoQA results.

Two levels of UQ:
  1. Per-variant UQ:  each variant has 5 responses → compute UQ over those 5
  2. Cross-variant UQ: pool ALL responses from ALL variants → compute UQ over all

Methods (via lm-polygraph):
  - LexicalSimilarity (rouge1, rouge2, rougeL)
  - EigValLaplacian   (NLI entail/contra, Jaccard)
  - DegMat            (NLI entail/contra, Jaccard)
  - Eccentricity      (NLI entail/contra, Jaccard)
  - NumSemSets
  - SemanticEntropy   (frequency)

Usage:
  python blackbox/compute_uq.py --input results/blackbox_qwen3_abg/results.json \
      --output results/blackbox_qwen3_abg/uq.json --device cuda:0
"""

import argparse
import json
import numpy as np
import logging
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Load data — return per-variant and cross-variant response groups
# ---------------------------------------------------------------------------
def load_data(path: str):
    """
    Returns:
        examples: list of dicts with keys:
            example_id, original_question, variants (list of variant dicts)
        per_variant_ids:   list[str]  — "{example_id}|{rule_id}|{idx}"
        per_variant_texts: list[list[str]]  — each inner list has K responses
        cross_variant_ids:   list[str]  — example_id
        cross_variant_texts: list[list[str]]  — all responses pooled per example
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    per_variant_ids = []
    per_variant_texts = []
    per_variant_meta = []  # (example_id, rule_id, rule_name)

    cross_variant_ids = []
    cross_variant_texts = []

    for ex in results:
        eid = ex["example_id"]
        all_responses = []

        for vi, var in enumerate(ex["variants"]):
            rid = var["rule_id"]
            responses = var["responses"]
            if not responses:
                continue

            vid = f"{eid}|{rid}|{vi}"
            per_variant_ids.append(vid)
            per_variant_texts.append(responses)
            per_variant_meta.append((eid, rid, var.get("rule_name", "")))

            all_responses.extend(responses)

        if all_responses:
            cross_variant_ids.append(eid)
            cross_variant_texts.append(all_responses)

    log.info(f"Loaded {len(results)} examples")
    log.info(f"  Per-variant groups: {len(per_variant_ids)}, "
             f"each with {len(per_variant_texts[0])} responses")
    log.info(f"  Cross-variant groups: {len(cross_variant_ids)}, "
             f"each with {len(cross_variant_texts[0])} responses")

    return (results,
            per_variant_ids, per_variant_texts, per_variant_meta,
            cross_variant_ids, cross_variant_texts)


# ---------------------------------------------------------------------------
#  Compute NLI semantic matrices in batches
#  Handles variable-length response lists by grouping by K.
# ---------------------------------------------------------------------------
def _compute_nli_for_group(matrix_calc, sample_texts, mini_bs=16):
    """Compute NLI matrices for a list of samples that all have the same K."""
    N = len(sample_texts)
    all_entail, all_contra, all_classes = [], [], []
    all_entail_logits, all_contra_logits = [], []
    entailment_id = None

    K = len(sample_texts[0])
    for start in tqdm(range(0, N, mini_bs), desc=f"NLI matrix (K={K})", leave=False):
        end = min(start + mini_bs, N)
        batch_samples = sample_texts[start:end]
        deps = {"sample_texts": batch_samples}
        dummy_texts = [""] * len(batch_samples)

        result = matrix_calc(deps, dummy_texts, model=None, max_new_tokens=100)

        all_entail.append(result["semantic_matrix_entail"])
        all_contra.append(result["semantic_matrix_contra"])
        all_classes.append(result["semantic_matrix_classes"])
        all_entail_logits.append(result["semantic_matrix_entail_logits"])
        all_contra_logits.append(result["semantic_matrix_contra_logits"])
        if entailment_id is None:
            entailment_id = result["entailment_id"]

    return {
        "semantic_matrix_entail": np.concatenate(all_entail, axis=0),
        "semantic_matrix_contra": np.concatenate(all_contra, axis=0),
        "semantic_matrix_classes": np.concatenate(all_classes, axis=0),
        "semantic_matrix_entail_logits": np.concatenate(all_entail_logits, axis=0),
        "semantic_matrix_contra_logits": np.concatenate(all_contra_logits, axis=0),
        "entailment_id": entailment_id,
    }



# ---------------------------------------------------------------------------
#  Compute semantic classes (needed for SemanticEntropy)
# ---------------------------------------------------------------------------
def compute_semantic_classes(stats, sample_texts):
    from lm_polygraph.stat_calculators.semantic_classes import SemanticClassesCalculator

    calc = SemanticClassesCalculator()
    deps = {
        "sample_texts": sample_texts,
        "semantic_matrix_classes": stats["semantic_matrix_classes"],
        "semantic_matrix_entail": stats["semantic_matrix_entail"],
        "entailment_id": stats["entailment_id"],
    }
    result = calc(deps, [""] * len(sample_texts), model=None, max_new_tokens=100)
    stats.update(result)
    return stats


# ---------------------------------------------------------------------------
#  Run all estimators
# ---------------------------------------------------------------------------
def build_estimators():
    from lm_polygraph.estimators.lexical_similarity import LexicalSimilarity
    from lm_polygraph.estimators.eig_val_laplacian import EigValLaplacian
    from lm_polygraph.estimators.deg_mat import DegMat
    from lm_polygraph.estimators.eccentricity import Eccentricity
    from lm_polygraph.estimators.num_sem_sets import NumSemSets
    from lm_polygraph.estimators.semantic_entropy import SemanticEntropy

    estimators = OrderedDict()

    estimators["LexicalSimilarity_rougeL"] = LexicalSimilarity(metric="rougeL")
    estimators["LexicalSimilarity_rouge1"] = LexicalSimilarity(metric="rouge1")
    estimators["LexicalSimilarity_rouge2"] = LexicalSimilarity(metric="rouge2")

    estimators["EigValLaplacian_NLI_entail"] = EigValLaplacian(similarity_score="NLI_score", affinity="entail")
    estimators["EigValLaplacian_NLI_contra"] = EigValLaplacian(similarity_score="NLI_score", affinity="contra")
    estimators["EigValLaplacian_Jaccard"] = EigValLaplacian(similarity_score="Jaccard_score")

    estimators["DegMat_NLI_entail"] = DegMat(similarity_score="NLI_score", affinity="entail")
    estimators["DegMat_NLI_contra"] = DegMat(similarity_score="NLI_score", affinity="contra")
    estimators["DegMat_Jaccard"] = DegMat(similarity_score="Jaccard_score")

    estimators["Eccentricity_NLI_entail"] = Eccentricity(similarity_score="NLI_score", affinity="entail")
    estimators["Eccentricity_NLI_contra"] = Eccentricity(similarity_score="NLI_score", affinity="contra")
    estimators["Eccentricity_Jaccard"] = Eccentricity(similarity_score="Jaccard_score")

    estimators["NumSemSets"] = NumSemSets()

    estimators["SemanticEntropy_freq"] = SemanticEntropy(class_probability_estimation="frequency")

    return estimators


def run_estimators(sample_texts, stats):
    estimators = build_estimators()
    stats["sample_texts"] = sample_texts

    results = OrderedDict()
    for name, estimator in estimators.items():
        log.info(f"Running estimator: {name}")
        try:
            scores = estimator(stats)
            results[name] = scores
            log.info(f"  {name}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        except Exception as e:
            log.error(f"  {name} FAILED: {e}")
            results[name] = np.full(len(sample_texts), np.nan)

    return results


# ---------------------------------------------------------------------------
#  Pipeline: NLI → semantic classes → estimators
#  Groups samples by K (response count) so lm-polygraph sees uniform batches.
# ---------------------------------------------------------------------------
def compute_uq_pipeline(sample_texts, device, batch_size, nli_cache=None, skip_nli=False):
    from collections import defaultdict

    # Group by response count K
    k_groups = defaultdict(list)  # K -> [original indices]
    for i, texts in enumerate(sample_texts):
        k_groups[len(texts)].append(i)

    unique_ks = sorted(k_groups.keys())
    N = len(sample_texts)

    log.info(f"Response counts: {', '.join(f'K={k}({len(idxs)})' for k, idxs in sorted(k_groups.items()))}")

    # Load NLI model once (shared across all groups)
    nli_model = None
    matrix_calc = None
    if not skip_nli:
        from lm_polygraph.utils.deberta import Deberta
        from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator
        log.info("Loading DeBERTa NLI model …")
        nli_model = Deberta(
            deberta_path="microsoft/deberta-large-mnli",
            batch_size=batch_size,
            device=device,
        )
        matrix_calc = SemanticMatrixCalculator(nli_model=nli_model)

    estimators = build_estimators()

    # Allocate per-method score arrays
    all_scores = OrderedDict()
    for name in estimators:
        all_scores[name] = np.full(N, np.nan)

    for K in unique_ks:
        idxs = k_groups[K]
        group_texts = [sample_texts[i] for i in idxs]
        log.info(f"--- Processing group K={K}, n={len(idxs)} ---")

        stats = {}
        if not skip_nli:
            cache_path = f"{nli_cache}_K{K}.npz" if nli_cache else None
            if cache_path and Path(cache_path).exists():
                log.info(f"Loading NLI cache: {cache_path}")
                cached = np.load(cache_path, allow_pickle=True)
                stats = {k: cached[k] for k in cached.files}
                stats["entailment_id"] = int(stats["entailment_id"])
            else:
                stats = _compute_nli_for_group(matrix_calc, group_texts)
                if cache_path:
                    log.info(f"Saving NLI cache: {cache_path}")
                    np.savez(cache_path, **stats)
            stats = compute_semantic_classes(stats, group_texts)

        # Run estimators on this group
        stats["sample_texts"] = group_texts
        for name, estimator in estimators.items():
            try:
                group_scores = estimator(stats)
                for j, orig_i in enumerate(idxs):
                    all_scores[name][orig_i] = group_scores[j]
            except Exception as e:
                log.error(f"  {name} FAILED for K={K}: {e}")

    # Log summary
    for name, scores in all_scores.items():
        valid = scores[np.isfinite(scores)]
        if len(valid) > 0:
            log.info(f"  {name}: mean={np.mean(valid):.4f}, std={np.std(valid):.4f}")

    return all_scores


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute UQ for ABG-CoQA results")
    parser.add_argument("--input", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--skip_nli", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        args.output = str(input_path.parent / f"{input_path.stem}_uq.json")

    # 1. Load data
    (examples,
     pv_ids, pv_texts, pv_meta,
     cv_ids, cv_texts) = load_data(args.input)

    # =====================================================================
    # 2. Per-variant UQ  (each variant's 5 responses)
    # =====================================================================
    log.info("=" * 60)
    log.info("Computing PER-VARIANT UQ (5 responses per variant)")
    log.info("=" * 60)

    pv_nli_cache = str(input_path.parent / f"{input_path.stem}_nli_per_variant.npz")
    pv_scores = compute_uq_pipeline(
        pv_texts, args.device, args.batch_size,
        nli_cache=pv_nli_cache, skip_nli=args.skip_nli,
    )

    # =====================================================================
    # 3. Cross-variant UQ  (all responses pooled per example)
    # =====================================================================
    log.info("=" * 60)
    log.info("Computing CROSS-VARIANT UQ (all responses per example)")
    log.info("=" * 60)

    cv_nli_cache = str(input_path.parent / f"{input_path.stem}_nli_cross_variant.npz")
    cv_scores = compute_uq_pipeline(
        cv_texts, args.device, args.batch_size,
        nli_cache=cv_nli_cache, skip_nli=args.skip_nli,
    )

    # =====================================================================
    # 4. Assemble output
    # =====================================================================
    output = OrderedDict()

    # -- per-variant: group by example_id --
    for i, vid in enumerate(pv_ids):
        eid, rid, rname = pv_meta[i]
        if eid not in output:
            output[eid] = {"per_variant": [], "cross_variant": {}}

        entry = OrderedDict()
        entry["variant_id"] = vid
        entry["rule_id"] = rid
        entry["rule_name"] = rname
        for method, scores in pv_scores.items():
            val = scores[i]
            entry[method] = float(val) if np.isfinite(val) else None
        output[eid]["per_variant"].append(entry)

    # -- cross-variant --
    for i, eid in enumerate(cv_ids):
        if eid not in output:
            output[eid] = {"per_variant": [], "cross_variant": {}}
        for method, scores in cv_scores.items():
            val = scores[i]
            output[eid]["cross_variant"][method] = float(val) if np.isfinite(val) else None

    # -- gap: cross_variant_UQ - mean(per_variant_UQ) --
    methods = list(pv_scores.keys())
    for eid, ex_data in output.items():
        ex_data["gap"] = {}
        for method in methods:
            cv_val = ex_data["cross_variant"].get(method)
            pv_vals = [v[method] for v in ex_data["per_variant"] if v.get(method) is not None]
            if cv_val is not None and pv_vals:
                ex_data["gap"][method] = cv_val - float(np.mean(pv_vals))
            else:
                ex_data["gap"][method] = None

    # 5. Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved UQ scores to {args.output}")

    # 6. Print summary
    def print_summary(title, score_dict, n):
        print(f"\n{'=' * 70}")
        print(title)
        print("=" * 70)
        for method, scores in score_dict.items():
            valid = scores[np.isfinite(scores)]
            if len(valid) > 0:
                print(f"  {method:40s}  mean={np.mean(valid):8.4f}  "
                      f"std={np.std(valid):8.4f}  "
                      f"min={np.min(valid):8.4f}  max={np.max(valid):8.4f}")
            else:
                print(f"  {method:40s}  ALL NaN")

    print_summary(f"PER-VARIANT UQ  ({len(pv_ids)} groups, 5 responses each)",
                  pv_scores, len(pv_ids))
    print_summary(f"CROSS-VARIANT UQ  ({len(cv_ids)} examples, "
                  f"{len(cv_texts[0])} responses each)",
                  cv_scores, len(cv_ids))

    # Gap summary
    gap_arrays = OrderedDict()
    for method in methods:
        vals = [output[eid]["gap"][method] for eid in output
                if output[eid]["gap"][method] is not None]
        gap_arrays[method] = np.array(vals)
    print_summary(f"GAP  (cross - mean_per_variant, {len(output)} examples)",
                  gap_arrays, len(output))


if __name__ == "__main__":
    main()
