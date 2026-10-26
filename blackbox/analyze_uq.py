"""
Analyze UQ scores for ABG-CoQA: ambiguous vs unambiguous.

Two axes of analysis:
  A. Per-variant UQ by rule_id:  which perturbation type best separates amb/unamb?
  B. Cross-variant UQ:           does pooling all variants improve detection?

Generates:
  1. AUROC bar chart — per rule_id + cross-variant + ORIGINAL  (all methods)
  2. AUROC ROC curves — per method, one curve per rule_id
  3. Amb vs Unamb mean comparison — grouped bar chart per rule_id
  4. Violin / box plots per rule_id
  5. Effect-size heatmap (Cohen's d)  methods × rule_ids
  6. Statistical summary table

Usage:
  python blackbox/analyze_uq.py \
      --uq results/blackbox_qwen3_abg/uq.json \
      --results results/blackbox_qwen3_abg/results.json \
      --output_dir results/blackbox_qwen3_abg/analysis

The "CROSS" column is the paper's black-box detector: EigV(Agree)
(EigValLaplacian_NLI_entail) over the answers pooled from the original
question and all of its rule variants.
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict, defaultdict
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, roc_curve


# ── Display names ───────────────────────────────────────────────────
RULE_DISPLAY = {
    "ORIGINAL": "Original",
    "A1": "Verb Synonym",
    "A2": "Noun Synonym",
    "A3": "Adjective Synonym",
    "A4": "Adverb Synonym",
    "A5": "Abbreviation Expand",
    "A6": "Abbreviation Contract",
    "B1": "Test-like Prefix",
    "B2": "Conversation-like Prefix",
    "B3": "Instruction Prefix",
    "B4": "WH Expansion",
    "B5": "WH Rephrase",
    "CROSS": "Cross-Variant (all)",
    "GAP": "Gap (cross−mean_pv)",
    "RULE_DISAGREE": "Rule Disagreement",
}

METHOD_SHORT = {
    "LexicalSimilarity_rougeL": "LexSim_rL",
    "LexicalSimilarity_rouge1": "LexSim_r1",
    "LexicalSimilarity_rouge2": "LexSim_r2",
    "EigValLaplacian_NLI_entail": "EigV_entail",
    "EigValLaplacian_NLI_contra": "EigV_contra",
    "EigValLaplacian_Jaccard": "EigV_Jacc",
    "DegMat_NLI_entail": "Deg_entail",
    "DegMat_NLI_contra": "Deg_contra",
    "DegMat_Jaccard": "Deg_Jacc",
    "Eccentricity_NLI_entail": "Ecc_entail",
    "Eccentricity_NLI_contra": "Ecc_contra",
    "Eccentricity_Jaccard": "Ecc_Jacc",
    "NumSemSets": "NumSemSets",
    "SemanticEntropy_freq": "SE_freq",
}

# Canonical display order: ORIGINAL, A-rules, B-rules, then CROSS always last
RULE_ORDER = ["ORIGINAL",
              "A1", "A2", "A3", "A4", "A5", "A6",
              "B1", "B2", "B3", "B4", "B5",
              "CROSS", "GAP", "RULE_DISAGREE"]

COLORS_RULE = {}  # populated at runtime


# ── Load & merge ────────────────────────────────────────────────────
def _build_label_map(results_data):
    label_map = {}
    for ex in results_data["results"]:
        label_map[ex["example_id"]] = ex["ambiguity_label"]  # "ambiguous" / "non_ambiguous"
    return label_map


def _convert_score_groups_to_numpy(score_groups):
    for m in score_groups:
        score_groups[m]["amb"] = np.array(score_groups[m]["amb"])
        score_groups[m]["unamb"] = np.array(score_groups[m]["unamb"])


def _convert_rule_scores_to_numpy(rule_scores):
    for rid in rule_scores:
        _convert_score_groups_to_numpy(rule_scores[rid])


def _finalize_rule_scores(rule_scores, method_names, label_map, n_examples):
    if not method_names:
        raise ValueError("No valid UQ methods found in the input JSON.")

    # Filter rule_ids with enough data (>=10 amb + >=10 unamb)
    valid_rules = []
    for rid in sorted(rule_scores.keys()):
        first_m = method_names[0]
        if first_m not in rule_scores[rid]:
            continue
        n_a = len(rule_scores[rid][first_m]["amb"])
        n_u = len(rule_scores[rid][first_m]["unamb"])
        if n_a >= 10 and n_u >= 10:
            valid_rules.append(rid)

    # Reorder by RULE_ORDER; any unexpected rule_ids go before CROSS
    ordered = [r for r in RULE_ORDER if r in valid_rules]
    ordered_set = set(ordered)
    extras = [r for r in valid_rules if r not in ordered_set]
    if "CROSS" in ordered:
        idx = ordered.index("CROSS")
        ordered = ordered[:idx] + extras + ordered[idx:]
    else:
        ordered += extras

    if not ordered:
        raise ValueError("No rule groups have enough data (need >=10 ambiguous and >=10 non-ambiguous examples).")

    n_amb = sum(1 for v in label_map.values() if v == "ambiguous")
    n_unamb = len(label_map) - n_amb
    print(f"Loaded {n_examples} examples: {n_amb} ambiguous, {n_unamb} non-ambiguous")
    print(f"Valid rule_ids (≥10 amb & unamb): {ordered}")
    print(f"Methods: {len(method_names)}")

    COLORS_RULE.clear()
    cmap = matplotlib.colormaps["tab20"].resampled(len(ordered))
    for i, r in enumerate(ordered):
        COLORS_RULE[r] = cmap(i)

    return rule_scores, method_names, ordered, label_map


def load(uq_path, results_path, rule_disagree_path=None):
    uq = json.load(open(uq_path, encoding="utf-8"))
    results_data = json.load(open(results_path, encoding="utf-8"))
    label_map = _build_label_map(results_data)

    if not uq:
        raise ValueError(f"Empty UQ JSON: {uq_path}")

    first_entry = next(iter(uq.values()))
    standalone_rule_disagree = (
        isinstance(first_entry, dict)
        and "per_variant" not in first_entry
        and "cross_variant" not in first_entry
    )

    # Standalone rule_disagree JSON:
    # {example_id: {n_representatives, method, metric1, metric2, ...}}
    if standalone_rule_disagree:
        if rule_disagree_path:
            print("Warning: --rule_disagree is ignored because --uq is already a standalone rule_disagree JSON.")

        metric_keys = [k for k in first_entry.keys() if k not in ("n_representatives", "method")]
        method_names = []
        for m in metric_keys:
            has_finite = any(
                (val.get(m) is not None) and np.isfinite(val.get(m))
                for val in uq.values()
            )
            if has_finite:
                method_names.append(m)

        rd_scores = defaultdict(lambda: {"amb": [], "unamb": []})
        for eid, val in uq.items():
            lab = label_map.get(eid, "non_ambiguous")
            key = "amb" if lab == "ambiguous" else "unamb"
            for m in method_names:
                v = val.get(m)
                if v is not None and np.isfinite(v):
                    rd_scores[m][key].append(v)

        _convert_score_groups_to_numpy(rd_scores)

        rule_scores = defaultdict(lambda: defaultdict(lambda: {"amb": [], "unamb": []}))
        rule_scores["RULE_DISAGREE"] = dict(rd_scores)
        return _finalize_rule_scores(rule_scores, method_names, label_map, n_examples=len(uq))

    # ---- Per-variant scores grouped by (rule_id, method) ----
    # rule_scores[rule_id][method] = {"amb": [...], "unamb": [...]}
    rule_scores = defaultdict(lambda: defaultdict(lambda: {"amb": [], "unamb": []}))

    # ---- Cross-variant scores ----
    cross_scores = defaultdict(lambda: {"amb": [], "unamb": []})

    method_names = None

    for eid, val in uq.items():
        lab = label_map.get(eid, "non_ambiguous")
        key = "amb" if lab == "ambiguous" else "unamb"

        # per-variant
        for pv in val["per_variant"]:
            rid = pv["rule_id"]
            if method_names is None:
                method_names = [k for k in pv if k not in ("variant_id", "rule_id", "rule_name")]
            for m in method_names:
                v = pv.get(m)
                if v is not None and np.isfinite(v):
                    rule_scores[rid][m][key].append(v)

        # cross-variant
        cv = val.get("cross_variant", {})
        for m in method_names:
            v = cv.get(m)
            if v is not None and np.isfinite(v):
                cross_scores[m][key].append(v)

    # ---- Gap scores (cross - mean_per_variant) ----
    gap_scores = defaultdict(lambda: {"amb": [], "unamb": []})

    for eid, val in uq.items():
        lab = label_map.get(eid, "non_ambiguous")
        key = "amb" if lab == "ambiguous" else "unamb"
        gap = val.get("gap", {})
        for m in method_names:
            v = gap.get(m)
            if v is not None and np.isfinite(v):
                gap_scores[m][key].append(v)

    # Convert lists to numpy
    _convert_rule_scores_to_numpy(rule_scores)
    _convert_score_groups_to_numpy(cross_scores)
    _convert_score_groups_to_numpy(gap_scores)

    # ---- Rule-level disagreement scores ----
    rd_scores = defaultdict(lambda: {"amb": [], "unamb": []})

    if rule_disagree_path:
        rd_data = json.load(open(rule_disagree_path, encoding="utf-8"))
        for eid, val in rd_data.items():
            lab = label_map.get(eid, "non_ambiguous")
            key = "amb" if lab == "ambiguous" else "unamb"
            for m in method_names:
                v = val.get(m)
                if v is not None and np.isfinite(v):
                    rd_scores[m][key].append(v)
        _convert_score_groups_to_numpy(rd_scores)

    # Add pseudo rule_ids only when they have data
    rule_scores["CROSS"] = dict(cross_scores)
    if gap_scores:
        rule_scores["GAP"] = dict(gap_scores)
    if rule_disagree_path and rd_scores:
        rule_scores["RULE_DISAGREE"] = dict(rd_scores)

    return _finalize_rule_scores(rule_scores, method_names, label_map, n_examples=len(uq))


# ── Stats helpers ───────────────────────────────────────────────────
def compute_auroc(amb, unamb):
    if len(amb) < 2 or len(unamb) < 2:
        return np.nan, np.nan
    labels = np.concatenate([np.ones(len(amb)), np.zeros(len(unamb))])
    values = np.concatenate([amb, unamb])
    try:
        auc = roc_auc_score(labels, values)
    except:
        auc = np.nan
    _, p = scipy_stats.mannwhitneyu(amb, unamb, alternative="two-sided")
    return auc, p


def cohens_d(a, b):
    s = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0


# ── Plot 1: AUROC bar chart (one bar per rule_id, one subplot per method) ──
def plot_auroc_bars(rule_scores, method_names, ordered_rules, output_dir):
    n_methods = len(method_names)
    ncols = min(4, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes).reshape(-1)

    for mi, m in enumerate(method_names):
        ax = axes[mi]
        aurocs, pvals = [], []
        for rid in ordered_rules:
            d = rule_scores[rid][m]
            auc, p = compute_auroc(d["amb"], d["unamb"])
            aurocs.append(auc)
            pvals.append(p)

        colors = [COLORS_RULE[r] for r in ordered_rules]
        display_labels = [RULE_DISPLAY.get(r, r) for r in ordered_rules]

        bars = ax.bar(range(len(ordered_rules)), aurocs, color=colors, alpha=0.85, edgecolor="white")
        for i, (auc, p) in enumerate(zip(aurocs, pvals)):
            if np.isnan(auc):
                continue
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(i, auc + 0.005, f"{auc:.3f}{star}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

        ax.set_xticks(range(len(ordered_rules)))
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(y=0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("AUROC")
        ax.set_title(METHOD_SHORT.get(m, m), fontsize=10, fontweight="bold")
        ax.set_ylim(0.35, max(0.75, np.nanmax(aurocs) + 0.08))
        ax.grid(axis="y", alpha=0.3)

    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("AUROC (Amb vs Unamb) per Perturbation Type", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auroc_per_rule.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved auroc_per_rule.png")


# ── Plot 2: ROC curves — one plot per method, one curve per rule_id ──
def plot_roc_curves(rule_scores, method_names, ordered_rules, output_dir):
    n_methods = len(method_names)
    ncols = min(4, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).reshape(-1)

    for mi, m in enumerate(method_names):
        ax = axes[mi]
        for rid in ordered_rules:
            d = rule_scores[rid][m]
            amb, unamb = d["amb"], d["unamb"]
            if len(amb) < 2 or len(unamb) < 2:
                continue
            labels = np.concatenate([np.ones(len(amb)), np.zeros(len(unamb))])
            values = np.concatenate([amb, unamb])
            try:
                fpr, tpr, _ = roc_curve(labels, values)
                auc = roc_auc_score(labels, values)
            except:
                continue
            lbl = RULE_DISPLAY.get(rid, rid)
            ax.plot(fpr, tpr, label=f"{lbl} ({auc:.3f})", color=COLORS_RULE[rid], linewidth=1.2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(METHOD_SHORT.get(m, m), fontsize=10, fontweight="bold")
        ax.legend(fontsize=5, loc="lower right")
        ax.grid(alpha=0.3)

    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("ROC Curves (Amb vs Unamb) per Perturbation Type", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves_per_rule.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved roc_curves_per_rule.png")


# ── Plot 3: Amb vs Unamb mean comparison (grouped bars) ────────────
def plot_mean_comparison(rule_scores, method_names, ordered_rules, output_dir):
    # One figure per method
    for m in method_names:
        display_labels = [RULE_DISPLAY.get(r, r) for r in ordered_rules]
        x = np.arange(len(ordered_rules))
        width = 0.35

        means_a, stds_a, means_u, stds_u = [], [], [], []
        for rid in ordered_rules:
            d = rule_scores[rid][m]
            means_a.append(np.mean(d["amb"]) if len(d["amb"]) else 0)
            stds_a.append(np.std(d["amb"]) if len(d["amb"]) else 0)
            means_u.append(np.mean(d["unamb"]) if len(d["unamb"]) else 0)
            stds_u.append(np.std(d["unamb"]) if len(d["unamb"]) else 0)

        fig, ax = plt.subplots(figsize=(max(10, len(ordered_rules) * 1.2), 6))
        ax.bar(x - width / 2, means_a, width, yerr=stds_a,
               color="#F44336", alpha=0.8, capsize=3, label="Ambiguous", edgecolor="white")
        ax.bar(x + width / 2, means_u, width, yerr=stds_u,
               color="#4CAF50", alpha=0.8, capsize=3, label="Non-ambiguous", edgecolor="white")

        for i in range(len(ordered_rules)):
            if means_a[i] != 0:
                ax.text(x[i] - width / 2, means_a[i] + stds_a[i] + 0.002,
                        f"{means_a[i]:.4f}", ha="center", va="bottom", fontsize=6)
            if means_u[i] != 0:
                ax.text(x[i] + width / 2, means_u[i] + stds_u[i] + 0.002,
                        f"{means_u[i]:.4f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel(f"Mean {METHOD_SHORT.get(m, m)}")
        ax.set_title(f"{METHOD_SHORT.get(m, m)}: Amb vs Unamb per Perturbation", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # dynamic y range
        all_tops = [a + s for a, s in zip(means_a + means_u, stds_a + stds_u)]
        all_bots = [a - s for a, s in zip(means_a + means_u, stds_a + stds_u)]
        ymin = max(0, min(all_bots) - 0.05 * (max(all_tops) - min(all_bots)))
        ymax = max(all_tops) + 0.1 * (max(all_tops) - min(all_bots))
        if ymax > ymin:
            ax.set_ylim(ymin, ymax)

        plt.tight_layout()
        short = METHOD_SHORT.get(m, m).replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"mean_amb_unamb_{short}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved mean_amb_unamb_*.png  ({len(method_names)} files)")


# ── Plot 4: Violin plots ──────────────────────────────────────────
def plot_violins(rule_scores, method_names, ordered_rules, output_dir):
    # Pick a few representative methods
    highlight = ["EigValLaplacian_NLI_entail", "DegMat_NLI_entail",
                 "Eccentricity_NLI_entail", "SemanticEntropy_freq", "NumSemSets"]
    methods = [m for m in highlight if m in method_names]
    if not methods:
        methods = method_names[:4]

    for m in methods:
        n_rules = len(ordered_rules)
        fig, ax = plt.subplots(figsize=(max(14, n_rules * 2), 7))
        positions_a, positions_u = [], []
        data_a, data_u = [], []

        for vi, rid in enumerate(ordered_rules):
            center = vi * 3
            d = rule_scores[rid][m]
            a = d["amb"] if len(d["amb"]) > 0 else np.array([0])
            u = d["unamb"] if len(d["unamb"]) > 0 else np.array([0])
            data_a.append(a)
            data_u.append(u)
            positions_a.append(center - 0.4)
            positions_u.append(center + 0.4)

        vp_a = ax.violinplot(data_a, positions=positions_a, widths=0.7,
                             showmeans=True, showmedians=True)
        for pc in vp_a["bodies"]:
            pc.set_facecolor("#F44336")
            pc.set_alpha(0.6)

        vp_u = ax.violinplot(data_u, positions=positions_u, widths=0.7,
                             showmeans=True, showmedians=True)
        for pc in vp_u["bodies"]:
            pc.set_facecolor("#4CAF50")
            pc.set_alpha(0.6)

        centers = [vi * 3 for vi in range(n_rules)]
        display_labels = [RULE_DISPLAY.get(r, r) for r in ordered_rules]
        ax.set_xticks(centers)
        ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(METHOD_SHORT.get(m, m), fontsize=11)
        ax.set_title(f"{METHOD_SHORT.get(m, m)} Distribution: Amb vs Unamb per Perturbation",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(handles=[
            mpatches.Patch(color="#F44336", alpha=0.6, label="Ambiguous"),
            mpatches.Patch(color="#4CAF50", alpha=0.6, label="Non-ambiguous"),
        ], fontsize=10)

        plt.tight_layout()
        short = METHOD_SHORT.get(m, m).replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"violin_{short}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Saved violin_*.png  ({len(methods)} files)")


# ── Plot 5: Effect-size heatmap (Cohen's d) — methods × rule_ids ──
def plot_effect_heatmap(rule_scores, method_names, ordered_rules, output_dir):
    n_m = len(method_names)
    n_r = len(ordered_rules)
    d_matrix = np.zeros((n_m, n_r))
    p_matrix = np.ones((n_m, n_r))

    for mi, m in enumerate(method_names):
        for ri, rid in enumerate(ordered_rules):
            d = rule_scores[rid][m]
            if len(d["amb"]) >= 2 and len(d["unamb"]) >= 2:
                d_matrix[mi, ri] = cohens_d(d["amb"], d["unamb"])
                _, p_matrix[mi, ri] = scipy_stats.mannwhitneyu(
                    d["amb"], d["unamb"], alternative="two-sided"
                )

    short_methods = [METHOD_SHORT.get(m, m) for m in method_names]
    short_rules = [RULE_DISPLAY.get(r, r) for r in ordered_rules]

    fig, ax = plt.subplots(figsize=(max(10, n_r * 1.5), max(8, n_m * 0.6)))
    vmax = max(abs(d_matrix.min()), abs(d_matrix.max()))
    im = ax.imshow(d_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_r))
    ax.set_yticks(range(n_m))
    ax.set_xticklabels(short_rules, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(short_methods, fontsize=9)

    for i in range(n_m):
        for j in range(n_r):
            p = p_matrix[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            val = d_matrix[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Cohen's d (positive = higher UQ for ambiguous)")
    ax.set_title("Cohen's d: Amb vs Unamb, Methods × Perturbation Types",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "effect_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved effect_heatmap.png")


# ── Plot 6: AUROC heatmap — methods × rule_ids ────────────────────
def plot_auroc_heatmap(rule_scores, method_names, ordered_rules, output_dir):
    n_m = len(method_names)
    n_r = len(ordered_rules)
    auc_matrix = np.full((n_m, n_r), 0.5)
    p_matrix = np.ones((n_m, n_r))

    for mi, m in enumerate(method_names):
        for ri, rid in enumerate(ordered_rules):
            d = rule_scores[rid][m]
            auc, p = compute_auroc(d["amb"], d["unamb"])
            if not np.isnan(auc):
                auc_matrix[mi, ri] = auc
                p_matrix[mi, ri] = p

    short_methods = [METHOD_SHORT.get(m, m) for m in method_names]
    short_rules = [RULE_DISPLAY.get(r, r) for r in ordered_rules]

    fig, ax = plt.subplots(figsize=(max(10, n_r * 1.5), max(8, n_m * 0.6)))
    im = ax.imshow(auc_matrix, cmap="YlOrRd", aspect="auto", vmin=0.4, vmax=0.75)

    ax.set_xticks(range(n_r))
    ax.set_yticks(range(n_m))
    ax.set_xticklabels(short_rules, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(short_methods, fontsize=9)

    for i in range(n_m):
        for j in range(n_r):
            p = p_matrix[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            val = auc_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}{star}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="AUROC")
    ax.set_title("AUROC Heatmap: Amb vs Unamb, Methods × Perturbation Types",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auroc_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved auroc_heatmap.png")


# ── Summary table (printed + saved) ───────────────────────────────
def print_and_save_summary(rule_scores, method_names, ordered_rules, output_dir):
    all_stats = OrderedDict()

    for m in method_names:
        all_stats[m] = OrderedDict()
        for rid in ordered_rules:
            d = rule_scores[rid][m]
            amb, unamb = d["amb"], d["unamb"]
            auc, mw_p = compute_auroc(amb, unamb)
            cd = cohens_d(amb, unamb) if len(amb) >= 2 and len(unamb) >= 2 else 0
            all_stats[m][rid] = {
                "n_amb": len(amb),
                "n_unamb": len(unamb),
                "amb_mean": float(np.mean(amb)) if len(amb) else None,
                "unamb_mean": float(np.mean(unamb)) if len(unamb) else None,
                "mean_diff": float(np.mean(amb) - np.mean(unamb)) if len(amb) and len(unamb) else None,
                "cohens_d": float(cd),
                "auroc": float(auc) if not np.isnan(auc) else None,
                "mannwhitney_p": float(mw_p) if not np.isnan(mw_p) else None,
            }

    # Print compact table
    print(f"\n{'=' * 140}")
    print(f"  Summary: AUROC & Cohen's d  (Amb vs Unamb)")
    print(f"{'=' * 140}")

    header = f"  {'Method':<25}"
    for rid in ordered_rules:
        header += f" {RULE_DISPLAY.get(rid, rid):>14}"
    print(header)
    print(f"  {'-' * (25 + 15 * len(ordered_rules))}")

    for m in method_names:
        line_auc = f"  {METHOD_SHORT.get(m, m):<25}"
        for rid in ordered_rules:
            s = all_stats[m][rid]
            auc = s["auroc"]
            p = s["mannwhitney_p"]
            if auc is not None:
                # p is None when every score ties (Mann-Whitney undefined)
                star = ("" if p is None else
                        "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
                line_auc += f" {auc:>8.3f}{star:>5s}"
            else:
                line_auc += f" {'N/A':>13}"
        print(line_auc)

    # Also print which rule_id has highest AUROC per method
    print(f"\n  Best perturbation per method:")
    for m in method_names:
        best_rid, best_auc = None, 0
        for rid in ordered_rules:
            auc = all_stats[m][rid].get("auroc")
            if auc is not None and auc > best_auc:
                best_auc = auc
                best_rid = rid
        print(f"    {METHOD_SHORT.get(m, m):<25}  {RULE_DISPLAY.get(best_rid, best_rid):<20}  AUROC={best_auc:.4f}")

    # Save
    stats_path = os.path.join(output_dir, "uq_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Saved {stats_path}")

    return all_stats


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyze UQ scores for ABG-CoQA")
    parser.add_argument("--uq", type=str, required=True,
                        help="UQ scores written by compute_uq.py")
    parser.add_argument("--results", type=str, required=True,
                        help="Results file written by run_perturbation.py (for the labels)")
    parser.add_argument("--rule_disagree", type=str, default=None,
                        help="Optional extra per-example scores to overlay: "
                             "JSON {example_id: {metric: score}}")
    parser.add_argument("--output_dir", type=str, default="./uq_llm_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rule_scores, method_names, ordered_rules, label_map = load(
        args.uq, args.results, rule_disagree_path=args.rule_disagree)

    print("\nGenerating plots...")
    plot_auroc_bars(rule_scores, method_names, ordered_rules, args.output_dir)
    plot_roc_curves(rule_scores, method_names, ordered_rules, args.output_dir)
    plot_mean_comparison(rule_scores, method_names, ordered_rules, args.output_dir)
    plot_violins(rule_scores, method_names, ordered_rules, args.output_dir)
    plot_effect_heatmap(rule_scores, method_names, ordered_rules, args.output_dir)
    plot_auroc_heatmap(rule_scores, method_names, ordered_rules, args.output_dir)

    print("\nComputing statistics...")
    print_and_save_summary(rule_scores, method_names, ordered_rules, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
