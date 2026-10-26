"""
Ablation (Table 4): execute the same rule library with an LLM instead of
deterministic code, on ABG-CoQA-style data.

Uses an LLM (via vLLM) to rewrite questions instead of rule-based transformations.
The LLM only sees the question (no story context), generating equivalent rewrites
for each perturbation type (synonym replacement, abbreviation, prefix, WH-word).

Pipeline:
  Phase 1: Load rewriter LLM, batch-generate all question variants
  Phase 2: Load QA LLM (reuses if same model), batch-run QA inference
  Phase 3: Save results (same format as run_perturbation.py)

Supported models (auto-detected from --model name):
    Qwen/Qwen3-4B-Instruct-2507        (qwen template)
    meta-llama/Llama-3.1-8B-Instruct   (llama template)
    google/gemma-3-27b-it              (gemma template)

Usage:
    CUDA_VISIBLE_DEVICES=0 python blackbox/run_llm_perturbation.py \
        --input_path data/abg_coqa/coqa_abg_test.json \
        --output_path results/llm_rules_qwen3_4b/results.json \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --num_samples 10 \
        --max_variants_per_question 10
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_utils  # noqa: E402  (sets cache / multiprocessing env before vLLM import)

import argparse
import time
import re
import gc
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
    wrap_chat,
)


# ============================================================
# Rewrite prompts — one per rule, corresponding to rule-based rules
# ============================================================
RULE_PROMPTS = {
    # Group A: Single Word Replacement
    "A1": (
        "Verb Synonym",
        "Rewrite this question by replacing one main verb with a synonym. "
        "Keep the meaning identical. If there is no suitable verb to replace, "
        "return the question unchanged."
    ),
    "A2": (
        "Noun Synonym",
        "Rewrite this question by replacing one noun (not a proper noun or pronoun) "
        "with a synonym. Keep the meaning identical. If there is no suitable noun, "
        "return the question unchanged."
    ),
    "A3": (
        "Adjective Synonym",
        "Rewrite this question by replacing one adjective with a synonym. "
        "Keep the meaning identical. If there is no adjective to replace, "
        "return the question unchanged."
    ),
    "A4": (
        "Adverb Synonym",
        "Rewrite this question by replacing one adverb with a synonym. "
        "Keep the meaning identical. If there is no adverb to replace, "
        "return the question unchanged."
    ),
    "A5": (
        "Abbreviation Expand",
        "Expand all contractions in this question "
        "(e.g., don't -> do not, what's -> what is, he's -> he is). "
        "If there are no contractions, return the question unchanged."
    ),
    "A6": (
        "Abbreviation Contract",
        "Contract expanded forms into contractions in this question "
        "(e.g., do not -> don't, what is -> what's, he is -> he's). "
        "If there are no forms to contract, return the question unchanged."
    ),
    # Group B: Prefix and WH-word
    "B1": (
        "Test-like Prefix",
        "Add a test/exam-style prefix to this question, such as "
        "'According to the passage,', 'Based on the text,', or 'As stated in the passage,'. "
        "Lowercase the first letter of the original question after the prefix if needed."
    ),
    "B2": (
        "Conversation-like Prefix",
        "Add a casual conversational prefix to this question, such as "
        "'Hey so,', 'OK so,', 'Alright,', or 'So then,'. "
        "Lowercase the first letter of the original question after the prefix if needed."
    ),
    "B3": (
        "Instruction Prefix",
        "Add an instruction-style prefix to this question, such as "
        "'Based on the context above,', 'Given the information provided,', "
        "or 'Considering the passage,'. "
        "Lowercase the first letter of the original question after the prefix if needed."
    ),
    "B4": (
        "WH Expansion",
        "Expand the WH-word at the start of this question to a longer phrase "
        "(e.g., who -> which person, where -> in what place, why -> for what reason, "
        "how -> in what way, when -> at what time). "
        "If there is no WH-word at the start, return the question unchanged."
    ),
    "B5": (
        "WH Rephrase",
        "Replace the WH-word at the start of this question with an alternative "
        "(e.g., who -> what person, how -> by what means, why -> how come, "
        "what -> which, where -> at what place). "
        "If there is no WH-word at the start, return the question unchanged."
    ),
}

GROUP_A_RULES = ["A1", "A2", "A3", "A4", "A5", "A6"]
GROUP_B_RULES = ["B1", "B2", "B3", "B4", "B5"]


# ============================================================
# Prompt builders
# ============================================================

def build_rewrite_prompt(question: str, instruction: str, model_type: str,
                         model_name: Optional[str] = None) -> str:
    """Build a prompt for the rewriter LLM. Only the question is given (no story)."""
    system_msg = (
        "You are a question rewriter. Rewrite questions exactly as instructed. "
        "Output ONLY the rewritten question, nothing else. "
        "No explanation, no quotes, no numbering."
    )
    user_content = f"{instruction}\n\nQuestion: {question}"
    return wrap_chat(system_msg, user_content, model_type, model_name)


# ============================================================
# Post-processing
# ============================================================

def clean_variant(text: str, original: str) -> Optional[str]:
    """Clean LLM rewrite output. Returns None if unusable."""
    text = text.strip()
    if not text:
        return None

    # Remove <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Remove surrounding quotes
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()

    # Remove common LLM prefixes
    for prefix in ["Rewritten question:", "Rewritten:", "Answer:", "Output:",
                    "Result:", "Here is", "Sure,"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Take only the first line if multi-line
    if '\n' in text:
        text = text.split('\n')[0].strip()

    # Skip if empty or unreasonably long
    if not text or len(text) > len(original) * 5:
        return None

    return text


# ============================================================
# Main experiment
# ============================================================

def run_experiment(
    input_path: str,
    output_path: str,
    model: str,
    model_type: str,
    rewriter_model: str,
    rewriter_model_type: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    num_samples: int = 5,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16392,
    rules: Optional[List[str]] = None,
    max_variants_per_rule: int = 3,
    max_variants_per_question: Optional[int] = None,
    max_examples: Optional[int] = None,
    only_ambiguous: bool = False,
    only_non_ambiguous: bool = False,
    resume: bool = False,
    rewriter_temperature: float = 0.7,
):
    """Run the full LLM-based perturbation experiment."""
    from vllm import SamplingParams

    same_model = (rewriter_model == model)

    examples = load_examples(input_path, max_examples, only_ambiguous, only_non_ambiguous)

    if rules is None:
        rules = list(RULE_PROMPTS.keys())
    print(f"Rules: {rules}")

    existing_results, completed_ids = load_existing_results(output_path, resume)

    pending = [(idx, ex) for idx, ex in enumerate(examples)
               if ex["id"] not in completed_ids]
    if not pending:
        print("All examples already completed.")
        return existing_results

    # ===========================================================
    # Phase 1: Generate variants using rewriter LLM
    # ===========================================================
    print(f"\n{'='*60}")
    print(f"Phase 1: Generating variants with {rewriter_model}")
    print(f"{'='*60}")

    rewrite_stop = stop_tokens(rewriter_model_type)
    rewriter_llm = load_llm(rewriter_model, rewriter_model_type, tensor_parallel_size,
                            gpu_memory_utilization, max_model_len)
    rewrite_params = SamplingParams(
        temperature=rewriter_temperature,
        top_p=0.95,
        max_tokens=128,
        n=max_variants_per_rule,
        stop=rewrite_stop,
    )

    # Build all rewrite prompts
    rewrite_tasks = []  # (example_idx, rule_id, rule_name, orig_question, prompt)
    for idx, example in pending:
        question = example["target_turn"]["question"]
        for rule_id in rules:
            if rule_id not in RULE_PROMPTS:
                print(f"Warning: unknown rule '{rule_id}', skipping.")
                continue
            rule_name, instruction = RULE_PROMPTS[rule_id]
            prompt = build_rewrite_prompt(question, instruction, rewriter_model_type,
                                          rewriter_model)
            rewrite_tasks.append((idx, rule_id, rule_name, question, prompt))

    print(f"Rewrite prompts: {len(rewrite_tasks)}")
    print(f"Rewrite generations (n={max_variants_per_rule}): "
          f"{len(rewrite_tasks) * max_variants_per_rule}")

    t0 = time.time()
    rewrite_outputs = rewriter_llm.generate(
        [t[4] for t in rewrite_tasks], rewrite_params
    )
    rewrite_elapsed = time.time() - t0
    print(f"Rewrite phase: {rewrite_elapsed:.1f}s")

    # Parse and deduplicate variants
    variants_by_example = {}  # example_idx -> list of variant dicts
    total_variants = 0
    rule_counts = {}

    for task_idx, (ex_idx, rule_id, rule_name, orig_q, _) in enumerate(rewrite_tasks):
        output = rewrite_outputs[task_idx]
        seen = set()
        for sample in output.outputs:
            vq = clean_variant(sample.text, orig_q)
            if vq is None:
                continue
            if vq.lower().strip("? ") == orig_q.lower().strip("? "):
                continue
            if vq in seen:
                continue
            seen.add(vq)
            variants_by_example.setdefault(ex_idx, []).append({
                "rule_id": rule_id,
                "rule_name": rule_name,
                "variant_question": vq,
                "replaced_word": "",
                "replacement": "",
            })
            total_variants += 1
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

    # Truncate per question if needed
    if max_variants_per_question is not None:
        for ex_idx in variants_by_example:
            if len(variants_by_example[ex_idx]) > max_variants_per_question:
                variants_by_example[ex_idx] = variants_by_example[ex_idx][:max_variants_per_question]
        total_variants = sum(len(v) for v in variants_by_example.values())

    avg_v = total_variants / len(pending) if pending else 0
    print(f"Unique variants: {total_variants} (avg {avg_v:.1f}/question)")
    print("By rule:", {k: rule_counts.get(k, 0) for k in rules})

    # ===========================================================
    # Phase 2: QA inference
    # ===========================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: QA inference with {model}")
    print(f"{'='*60}")

    if same_model:
        qa_llm = rewriter_llm
    else:
        # Free rewriter memory
        del rewriter_llm
        del rewrite_outputs
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        qa_llm = load_llm(model, model_type, tensor_parallel_size,
                          gpu_memory_utilization, max_model_len)
    qa_stop = stop_tokens(model_type)
    qa_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_samples,
        stop=qa_stop,
    )

    # Build QA prompts: original + all variants
    qa_tasks = []  # (example_idx, variant_info, prompt)
    for idx, example in pending:
        story = example["story"]
        question = example["target_turn"]["question"]
        history = example.get("history_turns", [])

        # Original question
        prompt = build_qa_prompt(story, history, question, model_type, model)
        qa_tasks.append((idx, {
            "rule_id": "ORIGINAL", "rule_name": "Original",
            "variant_question": question,
            "replaced_word": "", "replacement": "",
        }, prompt))

        # LLM-generated variants
        for v in variants_by_example.get(idx, []):
            prompt = build_qa_prompt(story, history, v["variant_question"], model_type, model)
            qa_tasks.append((idx, v, prompt))

    print(f"QA prompts: {len(qa_tasks)}")
    print(f"QA generations (n={num_samples}): {len(qa_tasks) * num_samples}")

    t0 = time.time()
    qa_outputs = qa_llm.generate([t[2] for t in qa_tasks], qa_params)
    qa_elapsed = time.time() - t0
    print(f"QA phase: {qa_elapsed:.1f}s")

    # ===========================================================
    # Organize results
    # ===========================================================
    example_results_map = {idx: example_record(example) for idx, example in pending}

    for task_idx, (ex_idx, variant_info, _) in enumerate(qa_tasks):
        output = qa_outputs[task_idx]
        responses = []
        for o in output.outputs:
            r = o.text.strip()
            # Strip thinking tokens from QA responses
            r = re.sub(r'<think>.*?</think>', '', r, flags=re.DOTALL).strip()
            responses.append(r)

        example_results_map[ex_idx]["variants"].append({
            "rule_id": variant_info["rule_id"],
            "rule_name": variant_info["rule_name"],
            "variant_question": variant_info["variant_question"],
            "replaced_word": variant_info.get("replaced_word", ""),
            "replacement": variant_info.get("replacement", ""),
            "responses": responses,
        })

    results = existing_results + [
        example_results_map[idx] for idx in sorted(example_results_map.keys())
    ]

    # ---- Save ----
    total_elapsed = rewrite_elapsed + qa_elapsed
    save_results(output_path, results, {
        "model": model,
        "model_type": model_type,
        "rewriter_model": rewriter_model,
        "rewriter_model_type": rewriter_model_type,
        "rewriter_temperature": rewriter_temperature,
        "temperature": temperature,
        "top_p": top_p,
        "num_samples": num_samples,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "rules": rules,
        "max_variants_per_rule": max_variants_per_rule,
        "input_path": input_path,
        "total_rewrite_prompts": len(rewrite_tasks),
        "total_unique_variants": total_variants,
        "total_qa_prompts": len(qa_tasks),
        "total_qa_generations": len(qa_tasks) * num_samples,
        "rewrite_elapsed_seconds": round(rewrite_elapsed, 1),
        "qa_elapsed_seconds": round(qa_elapsed, 1),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    })

    print(f"\nDone!")
    print(f"  Rewrite: {len(rewrite_tasks)} prompts -> {total_variants} variants "
          f"({rewrite_elapsed:.1f}s)")
    print(f"  QA: {len(qa_tasks)} prompts x {num_samples} samples "
          f"({qa_elapsed:.1f}s)")
    print(f"  Total: {total_elapsed:.1f}s")
    print(f"  Saved to {output_path}")

    # Print examples
    for r in list(example_results_map.values())[:3]:
        print(f"\n  Q: {r['original_question']}  [{r['ambiguity_label']}]")
        for v in r["variants"][:5]:
            print(f"    [{v['rule_id']:8s}] {v['variant_question']}")
            print(f"             -> {v['responses'][:2]}...")

    return results


# ============================================================
# Dry Run: only generate variants, no QA
# ============================================================

def dry_run(
    input_path: str,
    output_path: str,
    rewriter_model: str,
    rewriter_model_type: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16392,
    rules: Optional[List[str]] = None,
    max_variants_per_rule: int = 3,
    max_variants_per_question: Optional[int] = None,
    max_examples: Optional[int] = None,
    rewriter_temperature: float = 0.7,
):
    """Generate all variants without QA inference, for inspection."""
    from vllm import SamplingParams

    examples = load_examples(input_path, max_examples)
    if rules is None:
        rules = list(RULE_PROMPTS.keys())

    rewrite_stop = stop_tokens(rewriter_model_type)

    print(f"Loading rewriter: {rewriter_model}")
    llm = load_llm(rewriter_model, rewriter_model_type, tensor_parallel_size,
                   gpu_memory_utilization, max_model_len)
    params = SamplingParams(
        temperature=rewriter_temperature,
        top_p=0.95,
        max_tokens=128,
        n=max_variants_per_rule,
        stop=rewrite_stop,
    )

    # Build prompts
    tasks = []
    for idx, example in enumerate(examples):
        question = example["target_turn"]["question"]
        for rule_id in rules:
            if rule_id not in RULE_PROMPTS:
                continue
            rule_name, instruction = RULE_PROMPTS[rule_id]
            prompt = build_rewrite_prompt(question, instruction, rewriter_model_type,
                                          rewriter_model)
            tasks.append((idx, rule_id, rule_name, question, prompt))

    print(f"Generating {len(tasks)} x {max_variants_per_rule} rewrite samples...")
    t0 = time.time()
    outputs = llm.generate([t[4] for t in tasks], params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Parse
    variants_by_idx = {}
    rule_counts = {}
    total = 0

    for task_idx, (ex_idx, rule_id, rule_name, orig_q, _) in enumerate(tasks):
        output = outputs[task_idx]
        seen = set()
        for sample in output.outputs:
            vq = clean_variant(sample.text, orig_q)
            if vq and vq.lower().strip("? ") != orig_q.lower().strip("? ") and vq not in seen:
                seen.add(vq)
                variants_by_idx.setdefault(ex_idx, []).append({
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "variant_question": vq,
                    "replaced_word": "",
                    "replacement": "",
                })
                rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1
                total += 1

    # Truncate per question if needed
    if max_variants_per_question is not None:
        for ex_idx in variants_by_idx:
            if len(variants_by_idx[ex_idx]) > max_variants_per_question:
                variants_by_idx[ex_idx] = variants_by_idx[ex_idx][:max_variants_per_question]
        total = sum(len(v) for v in variants_by_idx.values())

    results = []
    for idx, example in enumerate(examples):
        variants = variants_by_idx.get(idx, [])
        results.append({
            "example_id": example["id"],
            "original_question": example["target_turn"]["question"],
            "ambiguity_label": example["ambiguity"],
            "num_variants": len(variants),
            "variants": variants,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"DRY RUN SUMMARY (LLM-based)")
    print(f"{'='*60}")
    print(f"Rewriter model: {rewriter_model}")
    print(f"Total examples: {len(examples)}")
    print(f"Total variants: {total}")
    print(f"Avg per question: {total / len(examples):.1f}")
    print(f"\nVariants by rule:")
    for rid in sorted(rule_counts.keys()):
        print(f"  {rid}: {rule_counts[rid]:5d}")

    no_variant = sum(1 for r in results if r["num_variants"] == 0)
    print(f"\nQuestions with NO variants: {no_variant}/{len(examples)}")
    print(f"Estimated QA generations (n=10): {(total + len(examples)) * 10}")

    save_results(output_path, results, {
        "mode": "dry_run",
        "rewriter_model": rewriter_model,
        "rewriter_model_type": rewriter_model_type,
        "rewriter_temperature": rewriter_temperature,
        "rules": rules,
        "max_variants_per_rule": max_variants_per_rule,
        "total_variants": total,
        "rule_counts": rule_counts,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    })
    print(f"\nSaved to {output_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-executed rule perturbation (ablation) for ambiguity detection"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Only generate variants, no QA inference")
    parser.add_argument("--output_path", type=str,
                        default="results_llm_perturbation.json")
    llm_utils.add_common_args(parser)

    # Rewriter model (defaults to --model if not set)
    parser.add_argument("--rewriter_model", type=str, default=None,
                        help="LLM for rewriting questions (default: same as --model)")
    parser.add_argument("--rewriter_model_type", type=str, default=None,
                        choices=llm_utils.MODEL_TYPES,
                        help="Rewriter model family (auto-detected from name if omitted)")
    parser.add_argument("--rewriter_temperature", type=float, default=0.7,
                        help="Temperature for rewriter (higher = more diverse variants)")

    # Perturbation
    parser.add_argument("--rules", nargs="+", default=None,
                        help="Rules to apply (e.g., A1 A2 B1). Default: all")
    parser.add_argument("--max_variants_per_rule", type=int, default=2,
                        help="Max variants per rule per question")
    parser.add_argument("--max_variants_per_question", type=int, default=None,
                        help="Max total variants per question (truncate after dedup)")
    return parser.parse_args()


def main():
    args = parse_args()

    rewriter = args.rewriter_model or args.model
    model_type = args.model_type or detect_model_type(args.model)
    rewriter_type = args.rewriter_model_type or detect_model_type(rewriter)
    print(f"Using QA model={args.model} ({model_type}), "
          f"rewriter={rewriter} ({rewriter_type})")
    args.model_type = model_type

    if args.dry_run:
        dry_run(
            input_path=args.input_path,
            output_path=args.output_path,
            rewriter_model=rewriter,
            rewriter_model_type=rewriter_type,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            rules=args.rules,
            max_variants_per_rule=args.max_variants_per_rule,
            max_variants_per_question=args.max_variants_per_question,
            max_examples=args.max_examples,
            rewriter_temperature=args.rewriter_temperature,
        )
    else:
        run_experiment(
            input_path=args.input_path,
            output_path=args.output_path,
            model=args.model,
            model_type=args.model_type,
            rewriter_model=rewriter,
            rewriter_model_type=rewriter_type,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            rules=args.rules,
            max_variants_per_rule=args.max_variants_per_rule,
            max_variants_per_question=args.max_variants_per_question,
            max_examples=args.max_examples,
            only_ambiguous=args.only_ambiguous,
            only_non_ambiguous=args.only_non_ambiguous,
            resume=args.resume,
            rewriter_temperature=args.rewriter_temperature,
        )


if __name__ == "__main__":
    main()
