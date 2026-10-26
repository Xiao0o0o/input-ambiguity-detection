"""
LLM paraphrase baseline for ambiguity detection on ABG-CoQA-style data.

Baseline: skip rule-guided perturbation entirely — just ask an LLM to paraphrase
the question N times, then sample K QA responses per paraphrase.

Output schema (identical to run_llm_perturbation.py / run_perturbation.py so
the downstream compute_uq.py / analyze_* scripts work unchanged):
    results[*].variants[*] = {
        rule_id:          "ORIGINAL" | "PARA",
        rule_name:        "Original" | "Paraphrase",
        variant_question: str,
        replaced_word:    "",
        replacement:      "",
        responses:        [str] * num_samples,
    }

Supported models (auto-detected from --model name):
    Qwen/Qwen3-4B-Instruct-2507        (qwen template)
    meta-llama/Llama-3.1-8B-Instruct   (llama template)
    google/gemma-3-27b-it              (gemma template)

Usage:
    CUDA_VISIBLE_DEVICES=0 python blackbox/run_paraphrase.py \
        --input_path data/abg_coqa/coqa_abg_test.json \
        --output_path results/paraphrase_qwen3_4b/results.json \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --num_paraphrases 10 \
        --num_samples 10 \
        --max_examples 1000
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_utils  # noqa: E402  (sets cache / multiprocessing env before vLLM import)

import argparse
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

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


PARAPHRASE_INSTRUCTION = (
    "Rewrite the following question as a paraphrase that keeps the exact "
    "same meaning but uses different wording. "
    "Output ONLY the paraphrased question, nothing else. "
    "No explanation, no quotes, no numbering."
)



def build_paraphrase_prompt(question: str, model_type: str,
                            model_name: Optional[str] = None) -> str:
    system_msg = (
        "You are a question paraphraser. Produce a single paraphrase of the "
        "given question that preserves its meaning exactly. "
        "Output ONLY the paraphrased question."
    )
    user_content = f"{PARAPHRASE_INSTRUCTION}\n\nQuestion: {question}"
    return wrap_chat(system_msg, user_content, model_type, model_name)


# ============================================================
# Paraphrase cleanup
# ============================================================

def clean_paraphrase(text: str, original: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()
    for prefix in ["Paraphrase:", "Paraphrased:", "Rewritten question:",
                   "Rewritten:", "Answer:", "Output:", "Result:",
                   "Here is", "Sure,", "Question:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if "\n" in text:
        text = text.split("\n")[0].strip()
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
    num_paraphrases: int = 10,
    num_samples: int = 10,
    paraphrase_temperature: float = 0.7,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 16392,
    max_examples: Optional[int] = None,
    only_ambiguous: bool = False,
    only_non_ambiguous: bool = False,
    resume: bool = False,
):
    from vllm import SamplingParams

    examples = load_examples(input_path, max_examples, only_ambiguous, only_non_ambiguous)
    existing_results, completed_ids = load_existing_results(output_path, resume)

    pending = [(idx, ex) for idx, ex in enumerate(examples)
               if ex["id"] not in completed_ids]
    if not pending:
        print("All examples already completed.")
        return existing_results

    print(f"\nLoading model: {model} (model_type={model_type})")
    llm = load_llm(model, model_type, tensor_parallel_size,
                   gpu_memory_utilization, max_model_len)
    stops = stop_tokens(model_type)

    # ============================================================
    # Phase 1: paraphrase generation
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Paraphrasing with {model}")
    print(f"  n={num_paraphrases} paraphrases/question, "
          f"temperature={paraphrase_temperature}")
    print(f"{'=' * 60}")

    paraphrase_params = SamplingParams(
        temperature=paraphrase_temperature,
        top_p=0.95,
        max_tokens=128,
        n=num_paraphrases,
        stop=stops,
    )

    para_prompts = [
        build_paraphrase_prompt(ex["target_turn"]["question"], model_type, model)
        for _, ex in pending
    ]
    print(f"Paraphrase prompts: {len(para_prompts)}  "
          f"(generations: {len(para_prompts) * num_paraphrases})")

    t0 = time.time()
    para_outputs = llm.generate(para_prompts, paraphrase_params)
    para_elapsed = time.time() - t0
    print(f"Paraphrase phase: {para_elapsed:.1f}s")

    paraphrases_by_example: Dict[int, List[str]] = {}
    total_paraphrases = 0
    for i, (idx, example) in enumerate(pending):
        orig_q = example["target_turn"]["question"]
        seen = set()
        kept: List[str] = []
        for sample in para_outputs[i].outputs:
            vq = clean_paraphrase(sample.text, orig_q)
            if vq is None:
                continue
            key = vq.lower().strip("? ")
            if key == orig_q.lower().strip("? "):
                continue
            if key in seen:
                continue
            seen.add(key)
            kept.append(vq)
        paraphrases_by_example[idx] = kept
        total_paraphrases += len(kept)

    avg_p = total_paraphrases / len(pending) if pending else 0
    print(f"Unique paraphrases: {total_paraphrases} (avg {avg_p:.1f}/question)")
    n_zero = sum(1 for v in paraphrases_by_example.values() if not v)
    print(f"Questions with NO paraphrases: {n_zero}/{len(pending)}")

    # ============================================================
    # Phase 2: QA inference
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 2: QA inference with {model}")
    print(f"  n={num_samples} samples/prompt, temperature={temperature}")
    print(f"{'=' * 60}")

    qa_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_samples,
        stop=stops,
    )

    qa_tasks = []
    qa_prompts = []
    for idx, example in pending:
        story = example["story"]
        history = example.get("history_turns", [])
        question = example["target_turn"]["question"]

        qa_tasks.append((idx, {
            "rule_id": "ORIGINAL",
            "rule_name": "Original",
            "variant_question": question,
            "replaced_word": "",
            "replacement": "",
        }))
        qa_prompts.append(build_qa_prompt(story, history, question, model_type, model))

        for vq in paraphrases_by_example.get(idx, []):
            qa_tasks.append((idx, {
                "rule_id": "PARA",
                "rule_name": "Paraphrase",
                "variant_question": vq,
                "replaced_word": "",
                "replacement": "",
            }))
            qa_prompts.append(build_qa_prompt(story, history, vq, model_type, model))

    print(f"QA prompts: {len(qa_prompts)}  "
          f"(generations: {len(qa_prompts) * num_samples})")

    t0 = time.time()
    qa_outputs = llm.generate(qa_prompts, qa_params)
    qa_elapsed = time.time() - t0
    print(f"QA phase: {qa_elapsed:.1f}s")

    results_map: Dict[int, Dict] = {idx: example_record(example) for idx, example in pending}

    for i, (ex_idx, variant_info) in enumerate(qa_tasks):
        output = qa_outputs[i]
        responses = []
        for o in output.outputs:
            r = o.text.strip()
            r = re.sub(r"<think>.*?</think>", "", r, flags=re.DOTALL).strip()
            responses.append(r)
        results_map[ex_idx]["variants"].append({
            "rule_id": variant_info["rule_id"],
            "rule_name": variant_info["rule_name"],
            "variant_question": variant_info["variant_question"],
            "replaced_word": "",
            "replacement": "",
            "responses": responses,
        })

    results = existing_results + [
        results_map[idx] for idx in sorted(results_map.keys())
    ]

    total_elapsed = para_elapsed + qa_elapsed
    save_results(output_path, results, {
        "dataset": "abg_coqa",
        "mode": "paraphrase_baseline",
        "model": model,
        "model_type": model_type,
        "num_paraphrases": num_paraphrases,
        "num_samples": num_samples,
        "paraphrase_temperature": paraphrase_temperature,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "input_path": input_path,
        "total_paraphrase_prompts": len(para_prompts),
        "total_unique_paraphrases": total_paraphrases,
        "total_qa_prompts": len(qa_prompts),
        "total_qa_generations": len(qa_prompts) * num_samples,
        "paraphrase_elapsed_seconds": round(para_elapsed, 1),
        "qa_elapsed_seconds": round(qa_elapsed, 1),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    })

    print("\nDone!")
    print(f"  Paraphrase: {len(para_prompts)} prompts -> {total_paraphrases} paraphrases "
          f"({para_elapsed:.1f}s)")
    print(f"  QA: {len(qa_prompts)} prompts x {num_samples} samples "
          f"({qa_elapsed:.1f}s)")
    print(f"  Total: {total_elapsed:.1f}s")
    print(f"  Saved to {output_path}")

    for r in list(results_map.values())[:3]:
        print(f"\n  Q: {r['original_question']}  [{r['ambiguity_label']}]")
        for v in r["variants"][:5]:
            print(f"    [{v['rule_id']:8s}] {v['variant_question']}")
            print(f"             -> {v['responses'][:2]}...")

    return results


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM paraphrase baseline for ambiguity detection"
    )
    parser.add_argument("--output_path", type=str,
                        default="results_paraphrase_qwen3_4b.json")
    llm_utils.add_common_args(parser)
    parser.add_argument("--num_paraphrases", type=int, default=10)
    parser.add_argument("--paraphrase_temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    model_type = args.model_type or detect_model_type(args.model)
    print(f"Using model={args.model}, model_type={model_type}")
    run_experiment(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        model_type=model_type,
        num_paraphrases=args.num_paraphrases,
        num_samples=args.num_samples,
        paraphrase_temperature=args.paraphrase_temperature,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_examples=args.max_examples,
        only_ambiguous=args.only_ambiguous,
        only_non_ambiguous=args.only_non_ambiguous,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
