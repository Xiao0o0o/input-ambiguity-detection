"""
Black-box detector: rule-based perturbation of ABG-CoQA-style questions.

Pipeline:
  1. Load dataset (CoQA-style conversational QA)
  2. Apply the semantics-blind rule library (perturbation_rules.py) to
     generate question variants
  3. For each variant (+ original), sample N answers from a local LLM via vLLM
  4. Save all results; score them with compute_uq.py

Supported models (auto-detected from --model name):
    Qwen/Qwen3-4B-Instruct-2507        (qwen template)
    meta-llama/Llama-3.1-8B-Instruct   (llama template)
    google/gemma-3-27b-it              (gemma template)

Usage:
    CUDA_VISIBLE_DEVICES=0 python blackbox/run_perturbation.py \
        --input_path data/abg_coqa/coqa_abg_test.json \
        --output_path results/blackbox_llama/results.json \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --temperature 0.5 \
        --num_samples 10

    # Inspect the generated variants without loading a model:
    python blackbox/run_perturbation.py --dry_run \
        --input_path data/abg_coqa/coqa_abg_test.json --output_path variants.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_utils  # noqa: E402  (sets cache / multiprocessing env before vLLM import)

import argparse
import time
from typing import List, Optional
from datetime import datetime

from llm_utils import (
    build_qa_prompt,
    detect_model_type,
    example_record,
    load_examples,
    load_existing_results,
    load_llm,
    save_results,
    stop_tokens,
)
from perturbation_rules import generate_all_variants, ALL_RULES


# ============================================================
# Main Experiment Runner
# ============================================================

def run_experiment(
    input_path: str,
    output_path: str,
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    model_type: str = "qwen",
    temperature: float = 0.5,
    top_p: float = 0.9,
    num_samples: int = 5,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int = 16392,
    rules: Optional[List[str]] = None,
    max_variants_per_rule: int = 3,
    max_examples: Optional[int] = None,
    only_ambiguous: bool = False,
    only_non_ambiguous: bool = False,
    resume: bool = False,
):
    """Run the full perturbation experiment."""
    from vllm import SamplingParams

    examples = load_examples(input_path, max_examples, only_ambiguous, only_non_ambiguous)

    if rules is None:
        rules = list(ALL_RULES.keys())
    print(f"Rules: {rules}")

    existing_results, completed_ids = load_existing_results(output_path, resume)

    # Init vLLM
    stops = stop_tokens(model_type)
    print(f"Loading vLLM model: {model}")
    print(f"  model_type: {model_type}")
    print(f"  tensor_parallel_size: {tensor_parallel_size}")
    print(f"  gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"  stop_tokens: {stops}")
    llm = load_llm(model, model_type, tensor_parallel_size,
                   gpu_memory_utilization, max_model_len)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_samples,
        stop=stops,
    )

    # Pre-generate all prompts for batch processing
    print("Generating variants and building prompts...")
    all_tasks = []  # list of (example_idx, variant_info, prompt)

    for idx, example in enumerate(examples):
        if example["id"] in completed_ids:
            continue

        story = example["story"]
        question = example["target_turn"]["question"]
        history = example.get("history_turns", [])

        # Original question
        prompt = build_qa_prompt(story, history, question, model_type, model)
        all_tasks.append((idx, {
            "rule_id": "ORIGINAL",
            "rule_name": "Original",
            "variant_question": question,
            "replaced_word": "",
            "replacement": "",
        }, prompt))

        # Variants
        variants = generate_all_variants(
            question, rules=rules,
            max_variants_per_rule=max_variants_per_rule
        )
        for variant in variants:
            prompt = build_qa_prompt(story, history, variant["variant_question"],
                                     model_type, model)
            all_tasks.append((idx, variant, prompt))

    print(f"Total prompts to process: {len(all_tasks)}")
    print(f"Total generations (with n={num_samples}): {len(all_tasks) * num_samples}")

    # Batch generate all at once (vLLM handles batching efficiently)
    print("Running vLLM batch inference...")
    start_time = time.time()

    all_prompts = [task[2] for task in all_tasks]
    all_outputs = llm.generate(all_prompts, sampling_params)

    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.1f}s")

    # Organize results by example
    example_results_map = {
        idx: example_record(example)
        for idx, example in enumerate(examples)
        if example["id"] not in completed_ids
    }

    # Fill in responses
    for task_idx, (ex_idx, variant_info, _prompt) in enumerate(all_tasks):
        output = all_outputs[task_idx]
        responses = [o.text.strip() for o in output.outputs]

        example_results_map[ex_idx]["variants"].append({
            "rule_id": variant_info["rule_id"],
            "rule_name": variant_info["rule_name"],
            "variant_question": variant_info["variant_question"],
            "replaced_word": variant_info.get("replaced_word", ""),
            "replacement": variant_info.get("replacement", ""),
            "responses": responses,
        })

    # Combine with existing results
    results = existing_results + [
        example_results_map[idx]
        for idx in sorted(example_results_map.keys())
    ]

    # Save
    total_generations = len(all_tasks) * num_samples
    save_results(output_path, results, {
        "model": model,
        "model_type": model_type,
        "temperature": temperature,
        "top_p": top_p,
        "num_samples": num_samples,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "rules": rules,
        "max_variants_per_rule": max_variants_per_rule,
        "input_path": input_path,
        "total_prompts": len(all_tasks),
        "total_generations": total_generations,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    })

    # Print summary
    print(f"\nDone!")
    print(f"  Total prompts: {len(all_tasks)}")
    print(f"  Total generations: {total_generations}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total_generations / elapsed:.1f} generations/s")
    print(f"  Results saved to {output_path}")

    # Print a few examples
    for r in list(example_results_map.values())[:3]:
        print(f"\n  Q: {r['original_question']}  [{r['ambiguity_label']}]")
        for v in r["variants"][:4]:
            print(f"    [{v['rule_id']:8s}] {v['variant_question']}")
            print(f"             -> {v['responses'][:2]}...")

    return results


# ============================================================
# Dry Run: only generate variants, no model inference
# ============================================================

def dry_run(
    input_path: str,
    output_path: str,
    rules: Optional[List[str]] = None,
    max_variants_per_rule: int = 2,
    max_examples: Optional[int] = None,
):
    """Generate all variants without running any model. For inspection."""
    examples = load_examples(input_path, max_examples)

    if rules is None:
        rules = list(ALL_RULES.keys())

    results = []
    total_variants = 0
    rule_counts = {}

    for example in examples:
        question = example["target_turn"]["question"]
        ambiguity = example["ambiguity"]

        variants = generate_all_variants(
            question, rules=rules,
            max_variants_per_rule=max_variants_per_rule
        )

        for v in variants:
            rid = v["rule_id"]
            rule_counts[rid] = rule_counts.get(rid, 0) + 1
        total_variants += len(variants)

        results.append({
            "example_id": example["id"],
            "original_question": question,
            "ambiguity_label": ambiguity,
            "num_variants": len(variants),
            "variants": [
                {
                    "rule_id": v["rule_id"],
                    "rule_name": v["rule_name"],
                    "variant_question": v["variant_question"],
                    "replaced_word": v.get("replaced_word", ""),
                    "replacement": v.get("replacement", ""),
                }
                for v in variants
            ],
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"DRY RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples: {len(examples)}")
    print(f"Total variants generated: {total_variants}")
    print(f"Avg variants per question: {total_variants/len(examples):.1f}")
    print(f"\nVariants by rule:")
    for rid in sorted(rule_counts.keys()):
        print(f"  {rid}: {rule_counts[rid]:5d} variants")

    no_a = sum(1 for r in results
               if not any(v["rule_id"].startswith("A") for v in r["variants"]))
    print(f"\nQuestions with NO Group A variants: {no_a}/{len(examples)}")

    total_with_original = total_variants + len(examples)
    print(f"\nEstimated total generations (with num_samples=10): {total_with_original * 10}")

    save_results(output_path, results, {
        "mode": "dry_run",
        "rules": rules,
        "max_variants_per_rule": max_variants_per_rule,
        "total_variants": total_variants,
        "rule_counts": rule_counts,
    })
    print(f"\nResults saved to {output_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Black-box (rule-based) perturbation for ambiguity detection"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Only generate variants, no model inference")
    parser.add_argument("--output_path", type=str,
                        default="perturbation_results.json",
                        help="Path to output JSON")
    llm_utils.add_common_args(parser)

    # Perturbation parameters
    parser.add_argument("--rules", nargs="+", default=None,
                        help="Which rules to apply (e.g., A1 A2 B1). Default: all")
    parser.add_argument("--max_variants_per_rule", type=int, default=2,
                        help="Max variants per rule per question")
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type or detect_model_type(args.model)
    print(f"Using model={args.model}, model_type={model_type}")

    if args.dry_run:
        dry_run(
            input_path=args.input_path,
            output_path=args.output_path,
            rules=args.rules,
            max_variants_per_rule=args.max_variants_per_rule,
            max_examples=args.max_examples,
        )
    else:
        run_experiment(
            input_path=args.input_path,
            output_path=args.output_path,
            model=args.model,
            model_type=model_type,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            rules=args.rules,
            max_variants_per_rule=args.max_variants_per_rule,
            max_examples=args.max_examples,
            only_ambiguous=args.only_ambiguous,
            only_non_ambiguous=args.only_non_ambiguous,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
