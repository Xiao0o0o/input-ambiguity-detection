"""
Shared helpers for the black-box scripts: model-family detection, chat
templates, the ABG-CoQA-style QA prompt, vLLM loading and result I/O.

Import this module before anything that may import vLLM / transformers, so
the cache and multiprocessing environment variables below take effect.
"""

import os

# vLLM workers must be spawned (not forked) once CUDA has been initialised.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Same convention as the white-box scripts: HF_CACHE_DIR, if set, is used as
# the HuggingFace hub cache; otherwise the HF default cache is used.
if os.environ.get("HF_CACHE_DIR"):
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_CACHE_DIR"])

import json
import multiprocessing
from typing import Dict, List, Optional

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


SUPPORTED_MODELS = {
    "Qwen/Qwen3-4B-Instruct-2507": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "google/gemma-3-27b-it": "gemma",
}

MODEL_TYPES = ["qwen", "llama", "gemma", "phi", "deepseek", "mistral", "generic"]


def detect_model_type(model_name: str) -> str:
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    m = model_name.lower()
    if "qwen" in m:
        return "qwen"
    if "llama" in m:
        return "llama"
    if "gemma" in m:
        return "gemma"
    if "phi" in m:
        return "phi"
    if "mistral" in m:
        return "mistral"
    if "deepseek" in m:
        return "deepseek"
    return "generic"


# ============================================================
# Stop tokens per model family
# ============================================================
STOP_TOKENS_MAP = {
    "qwen": ["</s>", "<|im_end|>", "<|endoftext|>"],
    "llama": ["<|eot_id|>", "<|end_of_text|>"],
    "gemma": ["<end_of_turn>", "<eos>"],
    "phi": ["<|im_end|>"],
    "deepseek": ["</s>", "<|end▁of▁sentence|>"],
    "mistral": ["</s>"],
}


def stop_tokens(model_type: str) -> List[str]:
    return STOP_TOKENS_MAP.get(model_type, ["</s>"])


# ============================================================
# Chat templates
#
# Gemma-3: a hand-coded "<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n"
# string makes Gemma-3-27B-IT generate 100% empty outputs. Routing through
# tokenizer.apply_chat_template renders the model's official template, and
# stripping the leading literal BOS keeps vLLM's auto-prepended BOS from
# doubling up.
# ============================================================
_TOKENIZER_CACHE = {}


def _get_tokenizer(model_name):
    from transformers import AutoTokenizer
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return _TOKENIZER_CACHE[model_name]


def _chat_format_gemma(system_msg: str, user_content: str, model_name: str) -> str:
    """Format a single-turn chat for Gemma via apply_chat_template + strip BOS."""
    tok = _get_tokenizer(model_name)
    # Gemma-3 doesn't accept a separate system role; fold into user turn.
    merged = (f"{system_msg}\n\n{user_content}"
              if system_msg else user_content)
    msgs = [{"role": "user", "content": merged}]
    out = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    # Strip leading literal BOS so vLLM's auto-prepended BOS isn't duplicated.
    bos_str = getattr(tok, "bos_token", None)
    if bos_str and out.startswith(bos_str):
        out = out[len(bos_str):]
        if out.startswith("\n"):
            out = out[1:]
    return out


def wrap_chat(system_msg: str, user_content: str, model_type: str,
              model_name: Optional[str] = None) -> str:
    """Render a single-turn (system, user) chat as a raw prompt string.
    `model_name` is only needed for Gemma, whose template comes from its tokenizer."""
    if model_type in ("qwen", "phi"):
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if model_type == "llama":
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if model_type == "gemma":
        if model_name is None:
            raise ValueError("model_name is required for the Gemma chat template")
        return _chat_format_gemma(system_msg, user_content, model_name)
    if model_type == "mistral":
        return f"<s>[INST] {system_msg}\n\n{user_content} [/INST]"
    return f"System: {system_msg}\n\nUser: {user_content}\n\nAssistant:"


# ============================================================
# QA prompt (CoQA-style: passage + conversation history + question)
# ============================================================
QA_SYSTEM_MSG = (
    "You are a helpful assistant answering questions about a passage. "
    "Answer concisely based on the given passage and conversation history. "
    "Give only the answer, no explanation."
)


def build_qa_prompt(story: str, history_turns: List[Dict], question: str,
                    model_type: str, model_name: Optional[str] = None) -> str:
    history_str = ""
    for turn in history_turns:
        history_str += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
    history_section = ("Previous conversation:\n" + history_str) if history_str else ""
    user_content = (
        f"Passage:\n{story}\n\n"
        f"{history_section}"
        f"Current question: {question}\n\n"
        f"Answer:"
    )
    return wrap_chat(QA_SYSTEM_MSG, user_content, model_type, model_name)


# ============================================================
# vLLM
# ============================================================
def load_llm(model: str, model_type: str, tensor_parallel_size: int = 1,
             gpu_memory_utilization: float = 0.9, max_model_len: int = 16392):
    from vllm import LLM
    # Gemma-3 is multimodal; vLLM otherwise allocates encoder cache for
    # 64 phantom image items and the model emits <image_soft_token> (id=262144)
    # which decodes to empty string → 100% empty outputs. Disabling images fixes it.
    extra = {"limit_mm_per_prompt": {"image": 0}} if model_type == "gemma" else {}
    return LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        enforce_eager=True,
        **extra,
    )


# ============================================================
# Data / result I/O
# ============================================================
def load_examples(input_path: str, max_examples: Optional[int] = None,
                  only_ambiguous: bool = False,
                  only_non_ambiguous: bool = False) -> List[Dict]:
    """Load an ABG-CoQA-format file ({"data": [...]}) and apply the filters.
    max_examples <= 0 or None keeps every example."""
    print(f"Loading data from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        examples = json.load(f)["data"]

    if only_ambiguous:
        examples = [e for e in examples if e["ambiguity"] == "ambiguous"]
        print(f"Filtered to {len(examples)} ambiguous examples")
    elif only_non_ambiguous:
        examples = [e for e in examples if e["ambiguity"] == "non_ambiguous"]
        print(f"Filtered to {len(examples)} non-ambiguous examples")

    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    print(f"Processing {len(examples)} examples")
    return examples


def load_existing_results(output_path: str, resume: bool):
    """Return (existing_results, completed_ids) for --resume."""
    if not (resume and os.path.exists(output_path)):
        return [], set()
    print(f"Resuming from {output_path}...")
    with open(output_path, encoding="utf-8") as f:
        existing_results = json.load(f).get("results", [])
    completed_ids = {r["example_id"] for r in existing_results}
    print(f"Found {len(completed_ids)} completed examples")
    return existing_results, completed_ids


def example_record(example: Dict) -> Dict:
    """Per-example header of the results file (variants are appended later)."""
    target = example["target_turn"]
    return {
        "example_id": example["id"],
        "original_question": target["question"],
        "gold_answer": target["answer"],
        "ambiguity_label": example["ambiguity"],
        "clarification": example.get("clarification_turn", {}),
        "source": example.get("source", ""),
        "variants": [],
    }


def save_results(output_path: str, results: List[Dict], config: Dict):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results}, f,
                  indent=2, ensure_ascii=False)


def add_common_args(parser):
    """Arguments shared by run_perturbation / run_llm_perturbation / run_paraphrase."""
    parser.add_argument("--input_path", type=str, required=True,
                        help="ABG-CoQA-format input JSON (see blackbox/README.md)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="HuggingFace model name. Tested: " + ", ".join(SUPPORTED_MODELS))
    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES,
                        help="Model family (auto-detected from --model if omitted)")

    # vLLM
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=16392,
                        help="Max model context length")

    # QA generation
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sampled answers per question variant")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens per response")

    # Filtering
    parser.add_argument("--max_examples", type=int, default=1000,
                        help="Max examples to process (<= 0 for the full set)")
    parser.add_argument("--only_ambiguous", action="store_true",
                        help="Only process ambiguous examples")
    parser.add_argument("--only_non_ambiguous", action="store_true",
                        help="Only process non-ambiguous examples")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing output file")
