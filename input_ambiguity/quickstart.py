"""`input-ambiguity quickstart`: AUROC of the paper's quick-start run.

Setting: white-box detector, Logit Lens importance, Qwen2.5-VL-3B-Instruct on
AQuA (720 questions). The per-question EigV(Agree) scores of that run ship
with the package, so this needs no GPU, model download or repository clone.
"""

import argparse
import os
from importlib import resources

from .whitebox.pipeline import compute_auroc

# (file, method, AUROC reported in Table 1 of the paper)
RUNS = [
    ("pipeline_b_baseline_n10.json", "EigV(Agree) baseline (K=10, no perturbation)", 0.644),
    ("pipeline_b_logit_lens.json", "Ours (White-Box, Logit Lens)", 0.708),
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="input-ambiguity quickstart",
        description="AUROC of the stored quick-start run (Qwen2.5-VL-3B x AQuA x Logit Lens)")
    ap.add_argument("--dir", default=None,
                    help="directory with pipeline_b_*.json (default: the copy shipped with the package; "
                         "pass quickstart/outputs to score a re-run)")
    args = ap.parse_args(argv)

    if args.dir:
        _report([os.path.join(args.dir, f) for f, _, _ in RUNS])
        return
    try:
        # installed: quickstart/intermediate/pipeline_b_*.json ship as input_ambiguity.quickstart_data
        data = resources.files(f"{__package__}.quickstart_data")
    except ModuleNotFoundError:
        # plain repository checkout
        data = resources.files(__package__) / ".." / "quickstart" / "intermediate"
    with resources.as_file(data) as d:
        _report([os.path.join(d, f) for f, _, _ in RUNS])


def _report(paths) -> None:
    print("Qwen2.5-VL-3B-Instruct x AQuA (720 questions), score = EigV(Agree)\n")
    print(f"{'method':<48}{'AUROC':>8}{'paper':>8}")
    for path, (_, name, paper) in zip(paths, RUNS):
        recs = compute_auroc.load_scores(path, ["EigV_entail"])
        auc, n = compute_auroc.auroc(recs, "EigV_entail")
        print(f"{name:<48}{auc:>8.4f}{paper:>8.3f}   n={n}")


if __name__ == "__main__":
    main()
