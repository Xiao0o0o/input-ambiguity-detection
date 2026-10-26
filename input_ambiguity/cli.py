"""`input-ambiguity <command> [args...]`: run one of the package's scripts.

Each command runs the corresponding script exactly as `python <script>.py
[args...]` would, so its own `--help` documents the arguments.
"""

import importlib.util
import os
import runpy
import sys

from . import __version__

# command -> (module, description)
COMMANDS = {
    "quickstart": ("quickstart", "AUROC of the paper's quick-start run (CPU, seconds, no download)"),
    # black-box detector
    "rules": ("blackbox.perturbation_rules", "print the 11 rule variants for a few sample questions"),
    "bb-perturb": ("blackbox.run_perturbation", "black-box detector: rule variants -> vLLM sampling"),
    "bb-llm-perturb": ("blackbox.run_llm_perturbation", "ablation: the same rules executed by an LLM rewriter"),
    "bb-paraphrase": ("blackbox.run_paraphrase", "baseline: free LLM paraphrasing"),
    "bb-uq": ("blackbox.compute_uq", "per-variant / cross-variant NLI uncertainty (lm-polygraph)"),
    "bb-analyze": ("blackbox.analyze_uq", "AUROC per rule and pooled, plots"),
    # white-box detector: token importance
    "wb-text-gradient": ("whitebox.text.gradient_attribution", "Vanilla Gradient importance (text LLMs)"),
    "wb-text-logit-lens": ("whitebox.text.logit_lens_uncertainty", "Logit Lens importance (text LLMs)"),
    "wb-text-diffmask": ("whitebox.text.diffmask_uncertainty", "DiffMask importance (text LLMs)"),
    "wb-text-normalize": ("whitebox.text.normalize_attributions", "map text attributions to word_scores"),
    "wb-vision-gradient": ("whitebox.vision.gradient_attribution", "Vanilla Gradient importance (VLMs)"),
    "wb-vision-logit-lens": ("whitebox.vision.logit_lens_uncertainty", "Logit Lens importance (VLMs)"),
    "wb-vision-diffmask": ("whitebox.vision.diffmask_uncertainty", "DiffMask importance (VLMs)"),
    # white-box detector: perturbation, sampling, scoring
    "wb-perturb": ("whitebox.pipeline.pipeline_b_perturb_and_uq", "top-k WordNet perturbation + generation (text)"),
    "wb-perturb-vl": ("whitebox.pipeline.pipeline_b_perturb_and_uq_vl", "top-k WordNet perturbation + generation (vision)"),
    "wb-uq": ("whitebox.pipeline.compute_uq_from_checkpoint", "NLI matrix -> EigV / DegMat / SemEntropy"),
    "auroc": ("whitebox.pipeline.compute_auroc", "AUROC of white-box UQ score files"),
}


def _usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"input-ambiguity {__version__}", "",
             "usage: input-ambiguity <command> [args...]   (input-ambiguity <command> --help for its arguments)",
             "", "commands:"]
    lines += [f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items()]
    return "\n".join(lines)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return
    if argv[0] not in COMMANDS:
        sys.exit(f"unknown command '{argv[0]}'\n\n{_usage()}")

    if argv[0] == "quickstart":
        from . import quickstart
        quickstart.main(argv[1:])
        return

    module = f"{__package__}.{COMMANDS[argv[0]][0]}"
    path = importlib.util.find_spec(module).origin
    # The scripts import their siblings by plain module name.
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [f"input-ambiguity {argv[0]}", *argv[1:]]
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
