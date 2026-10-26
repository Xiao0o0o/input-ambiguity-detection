#!/usr/bin/env python3
"""
Pipeline B (VL variant for Qwen2.5-VL + AQuA).

Same algorithm as the text-only pipeline_b_perturb_and_uq.py:
  1. Load attribution results → extract top-k important TEXT words
  2. Perturb those words in the TEXT portion of the prompt via WordNet synonyms
  3. Generate answers with the VL model (image + perturbed text) for original + variants
  4. Pool answers → compute EigValLaplacian_entail, DegMat_entail via DeBERTa NLI
  5. AUROC vs. ambiguity label

Generation backend:
  --backend vllm : fast, batched Qwen2.5-VL inference via vLLM
  --backend hf   : reference HF generate (slower, safer fallback)

Baseline mode (`--baseline`): no attribution / no perturbation; 10-sample
multi-sampling on original (image, question) pairs, same UQ metrics.
"""
import argparse
import json
import math
import os
import re
import sys
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# WordNet perturbation (identical to original pipeline_b_perturb_and_uq.py)
# ═══════════════════════════════════════════════════════════════════════════════

WORD_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from",
    "he", "her", "him", "his", "how", "if", "in", "into", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "with",
}


def _treebank_to_wordnet_pos(tag):
    if tag.startswith("J"): return "a"
    if tag.startswith("V"): return "v"
    if tag.startswith("N"): return "n"
    if tag.startswith("R"): return "r"
    return None


def _wordnet_synonyms(wn, word, wn_pos, max_syns=5):
    out, seen = [], set()
    base = word.lower()
    synsets = wn.synsets(base, pos=wn_pos) if wn_pos else []
    if not synsets:
        synsets = wn.synsets(base)
    for syn in synsets[:5]:
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ").strip().lower()
            if " " in cand or cand == base or cand in seen:
                continue
            if not WORD_TOKEN_RE.fullmatch(cand):
                continue
            if cand in STOPWORDS:
                continue
            seen.add(cand)
            out.append(cand)
            if len(out) >= max_syns:
                return out
    return out


def _apply_pos_surface_form(lemma, tb_pos):
    w = lemma
    if tb_pos in {"VBD", "VBN"}:
        if w.endswith("e"): return w + "d"
        if len(w) > 1 and w.endswith("y") and w[-2] not in "aeiou":
            return w[:-1] + "ied"
        return w + "ed"
    if tb_pos == "VBG":
        if w.endswith("ie"): return w[:-2] + "ying"
        if w.endswith("e") and len(w) > 2: return w[:-1] + "ing"
        return w + "ing"
    if tb_pos == "VBZ":
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        if w.endswith(("s", "x", "z", "ch", "sh", "o")): return w + "es"
        return w + "s"
    if tb_pos == "NNS":
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        if w.endswith(("s", "x", "z", "ch", "sh")): return w + "es"
        return w + "s"
    return w


def _match_token_case(src, tgt):
    if src.isupper(): return tgt.upper()
    if src and src[0].isupper(): return tgt.capitalize()
    return tgt


def _normalize_spaces(t):
    return re.sub(r"  +", " ", t).strip()


def get_wordnet_perturbations(question, target_words, n_variants=4, max_synonyms_per_token=5):
    """Perturb `question` by substituting WordNet synonyms of `target_words`.
    Identical algorithm to the text-only pipeline_b."""
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.tag import pos_tag
    for res in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"taggers/{res}" if "tagger" in res else f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

    matches = list(WORD_TOKEN_RE.finditer(question))
    tokens = [m.group(0) for m in matches]
    tagged = pos_tag(tokens)
    word_positions: Dict[str, List] = {}
    for idx, m in enumerate(matches):
        word_positions.setdefault(m.group(0).lower(), []).append((m, tagged[idx][1]))

    perturbations = []
    seen = set()
    for tw in target_words:
        tl = tw.lower().strip()
        if tl not in word_positions:
            continue
        for match_obj, tb_pos in word_positions[tl]:
            wn_pos = _treebank_to_wordnet_pos(tb_pos)
            for syn in _wordnet_synonyms(wn, tl, wn_pos, max_synonyms_per_token):
                surfaced = _apply_pos_surface_form(syn, tb_pos) if wn_pos else syn
                new_token = _match_token_case(match_obj.group(0), surfaced)
                new_q = _normalize_spaces(
                    question[:match_obj.start()] + new_token + question[match_obj.end():]
                )
                if new_q == question or new_q in seen:
                    continue
                seen.add(new_q)
                perturbations.append({
                    "perturbed_question": new_q,
                    "changed_word": match_obj.group(0),
                    "replacement": new_token,
                    "tb_pos": tb_pos,
                    "strategy": "wordnet",
                })
                if len(perturbations) >= n_variants:
                    return perturbations
    return perturbations


# ═══════════════════════════════════════════════════════════════════════════════
# Top-k word extraction from attribution JSON
# ═══════════════════════════════════════════════════════════════════════════════

SKIP_WORDS = {
    "the", "and", "a", "an", "of", "in", "to", "is", "are", "was",
    "were", "that", "this", "it", "for", "on", "with", "as", "at",
    "by", "from", "he", "she", "they", "we", "you", "his", "her",
    "its", "their", "my", "your", "our", "who", "what", "where",
    "when", "how", "which", "do", "does", "did", "not", "no", "yes",
    "or", "but", "if", "so", "be", "been", "being", "has", "had",
    "have", "will", "would", "could", "should", "may", "might", "can",
    "shall", "there", "here", "then", "than", "also", "just", "very",
    "too", "more", "most", "some", "any", "all", "each", "every",
    "about", "up", "out", "into", "over", "after", "before",
    "between", "through", "during", "without", "against",
    "assistant", "helpful", "visual", "look", "image", "answer",
    "question", "concisely", "system", "user",
    "", " ", "\n",
}


def extract_top_k_words(attr_res: Dict[str, Any], k: int = 3, question_text: str = "") -> List[str]:
    """Extract top-k content words from an attribution result, filtered to
    those appearing in question_text. Identical to original."""
    word_scores = attr_res.get("word_scores", [])
    valid = []
    for e in word_scores:
        s = e.get("score", 0)
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        valid.append(e)
    if not valid:
        valid = word_scores

    srt = sorted(
        valid,
        key=lambda x: x.get("score", 0) if isinstance(x.get("score", 0), (int, float)) and not math.isnan(x.get("score", 0)) else 0,
        reverse=True,
    )
    q_lower = question_text.lower() if question_text else ""
    top, seen = [], set()
    for e in srt:
        w = e.get("word", "").strip()
        if not w or "\n" in w or "<|" in w or "|>" in w:
            continue
        clean = re.sub(r"^[^A-Za-z]+", "", w)
        clean = re.sub(r"[^A-Za-z]+$", "", clean)
        if not clean or len(clean) <= 1:
            continue
        if clean.lower() in SKIP_WORDS:
            continue
        if not WORD_TOKEN_RE.search(clean):
            continue
        if clean.lower() in seen:
            continue
        if q_lower and clean.lower() not in q_lower:
            continue
        seen.add(clean.lower())
        top.append(clean)
        if len(top) >= k:
            break
    return top


# ═══════════════════════════════════════════════════════════════════════════════
# Generation backends
# ═══════════════════════════════════════════════════════════════════════════════

class VLGenerator:
    """Abstract interface — implementations: HFVLGenerator, VLLMVLGenerator."""
    def generate(self, image_path: str, prompt_text: str, *,
                 n_greedy: int = 1, n_sampled: int = 0,
                 temperature: float = 0.5, max_new_tokens: int = 128) -> List[str]:
        raise NotImplementedError


class HFVLGenerator(VLGenerator):
    def __init__(self, model_path: str, cache_dir: str, device: str = "cuda", dtype="float16"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        # Models needing bfloat16 (float16 causes overflow → CUDA asserts)
        is_gemma3 = "gemma" in model_path.lower() and "3" in model_path
        is_mllama = "llama" in model_path.lower() and ("vision" in model_path.lower() or "3.2" in model_path.lower())
        is_internvl = "internvl" in model_path.lower()
        if is_gemma3 or is_mllama or is_internvl:
            _dtype = torch.bfloat16
        else:
            _dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, cache_dir=cache_dir,
            torch_dtype=_dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.is_gemma3 = type(self.model).__name__.startswith("Gemma3")
        self.is_mllama = type(self.model).__name__.startswith("Mllama")
        self.is_internvl = type(self.model).__name__.startswith("InternVL")

    def _build_inputs(self, image_path: str, prompt_text: str):
        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user",
                     "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
        if self.is_gemma3 or self.is_internvl:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
        else:
            chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[chat_text], images=[img],
                return_tensors="pt", truncation=True, max_length=4096,
            )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate(self, image_path, prompt_text, *, n_greedy=1, n_sampled=0,
                 temperature=0.5, max_new_tokens=128):
        inputs = self._build_inputs(image_path, prompt_text)
        out: List[str] = []
        # Greedy
        for _ in range(n_greedy):
            with self.torch.no_grad():
                ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            new_ids = ids[0][inputs["input_ids"].shape[1]:]
            text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            if text:
                out.append(text)
        # Sampled
        for _ in range(n_sampled):
            with self.torch.no_grad():
                ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=temperature, top_p=0.9,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            new_ids = ids[0][inputs["input_ids"].shape[1]:]
            text = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            if text:
                out.append(text)
        return out


class VLLMVLGenerator(VLGenerator):
    """vLLM-backed generation for Qwen2.5-VL.

    Uses `LLM.chat` with PIL image content blocks. Batches per-question calls
    natively (caller drives with a loop; vLLM engine batches under the hood).
    """
    def __init__(self, model_path: str, cache_dir: str, max_model_len: int = 4096,
                 dtype: str = "float16", gpu_mem_util: float = 0.85,
                 limit_mm_per_prompt: int = 1):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        os.environ.setdefault("HF_HOME", cache_dir)
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=True,
            download_dir=cache_dir,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            limit_mm_per_prompt={"image": limit_mm_per_prompt},
        )

    def _build_messages(self, image_path: str, prompt_text: str):
        img = Image.open(image_path).convert("RGB")
        return [{"role": "user",
                 "content": [{"type": "image_pil", "image_pil": img},
                             {"type": "text", "text": prompt_text}]}]

    def generate(self, image_path, prompt_text, *, n_greedy=1, n_sampled=0,
                 temperature=0.5, max_new_tokens=128):
        out: List[str] = []
        messages = self._build_messages(image_path, prompt_text)
        if n_greedy > 0:
            sp = self.SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=n_greedy)
            try:
                res = self.llm.chat(messages, sp, use_tqdm=False)
            except TypeError:
                res = self.llm.chat([messages], sp, use_tqdm=False)
            for out_req in res:
                for c in out_req.outputs:
                    if c.text.strip():
                        out.append(c.text.strip())
        if n_sampled > 0:
            sp = self.SamplingParams(temperature=temperature, top_p=0.9,
                                     max_tokens=max_new_tokens, n=n_sampled)
            try:
                res = self.llm.chat(messages, sp, use_tqdm=False)
            except TypeError:
                res = self.llm.chat([messages], sp, use_tqdm=False)
            for out_req in res:
                for c in out_req.outputs:
                    if c.text.strip():
                        out.append(c.text.strip())
        return out

    def generate_batch(self, items: List[Tuple[str, str]], *,
                       n_greedy: int = 0, n_sampled: int = 0,
                       temperature: float = 0.5, max_new_tokens: int = 128
                       ) -> List[List[str]]:
        """Batched call: items = [(image_path, prompt_text), ...]. Returns a
        list of lists of strings — one entry per input."""
        from vllm.utils import identity  # no-op
        all_messages = [self._build_messages(ip, pt) for ip, pt in items]
        results: List[List[str]] = [[] for _ in items]
        if n_greedy > 0:
            sp = self.SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=n_greedy)
            res = self.llm.chat(all_messages, sp, use_tqdm=False)
            for i, out_req in enumerate(res):
                for c in out_req.outputs:
                    if c.text.strip():
                        results[i].append(c.text.strip())
        if n_sampled > 0:
            sp = self.SamplingParams(temperature=temperature, top_p=0.9,
                                     max_tokens=max_new_tokens, n=n_sampled)
            res = self.llm.chat(all_messages, sp, use_tqdm=False)
            for i, out_req in enumerate(res):
                for c in out_req.outputs:
                    if c.text.strip():
                        results[i].append(c.text.strip())
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# UQ via lm_polygraph (unchanged from original pipeline_b)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_uq_metrics(answers: List[str], deberta, sem_calc, eigv_est, degmat_est) -> Dict[str, float]:
    if len(answers) < 2:
        return {"EigV_entail": float("nan"), "DegMat_entail": float("nan")}
    deps = {"sample_texts": [answers]}
    stats = sem_calc(deps, texts=[""], model=None, max_new_tokens=1)
    stats["sample_texts"] = [answers]
    eigv_val = eigv_est(stats)[0]
    degmat_val = degmat_est(stats)[0]
    return {"EigV_entail": float(eigv_val), "DegMat_entail": float(degmat_val)}


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    valid = ~np.isnan(scores)
    if valid.sum() < 5:
        return float("nan")
    return roc_auc_score(labels[valid], scores[valid])


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt manipulation: isolate the question text from the VL prompt so
# perturbation only modifies the user-question part.
# ═══════════════════════════════════════════════════════════════════════════════

def split_vl_prompt(full_prompt: str, question_only: str) -> Tuple[str, str, str]:
    """Return (prefix, question_only, suffix) such that
    prefix + question_only + suffix == full_prompt. The perturbation only
    modifies the question_only portion."""
    idx = full_prompt.rfind(question_only)
    if idx < 0:
        # Fall back: prefix=empty, question=full, suffix=empty
        return "", full_prompt, ""
    return full_prompt[:idx], question_only, full_prompt[idx + len(question_only):]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attribution_json", type=str, default=None)
    p.add_argument("--input_json", type=str, required=True,
                   help="AQuA 720 input JSON with image_path per qid")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--cache_dir", type=str, default=os.environ.get("HF_CACHE_DIR"))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--backend", choices=["hf", "vllm"], default="vllm",
                   help="Generation backend")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--n_variants", type=int, default=4)
    p.add_argument("--n_sampled", type=int, default=2,
                   help="Sampled generations for ORIGINAL (perturbed variants are greedy-only).")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--nli_model", type=str, default="microsoft/deberta-large-mnli")
    p.add_argument("--nli_batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--method_name", type=str, default=None)
    p.add_argument("--resume_from_gen", type=str, default=None)
    p.add_argument("--baseline", action="store_true",
                   help="Baseline mode: 1 greedy + (N-1) sampled on originals, no perturbation.")
    p.add_argument("--baseline_n_samples", type=int, default=10)
    p.add_argument("--gpu_mem_util", type=float, default=0.80)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gen_only", action="store_true",
                   help="Stop after saving the generation checkpoint (compute UQ "
                        "separately with compute_uq_from_checkpoint.py).")
    p.add_argument("--perturb_only", action="store_true",
                   help="Stop after saving the perturbation checkpoint.")
    p.add_argument("--perturbations_json", type=str, default=None,
                   help="Saved _perturbations_<method>.json; skips attribution → perturbation.")
    args = p.parse_args()

    if not args.baseline and args.attribution_json is None and args.perturbations_json is None:
        p.error("--attribution_json (or --perturbations_json) is required unless --baseline is set.")

    if args.method_name is None:
        if args.baseline:
            args.method_name = "baseline"
        elif args.attribution_json:
            bn = os.path.basename(args.attribution_json)
            args.method_name = bn.split("_Qwen")[0]
        else:
            bn = os.path.basename(args.perturbations_json)
            args.method_name = bn.replace("_perturbations_", "").rsplit(".json", 1)[0]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Pipeline B VL] method={args.method_name} backend={args.backend} "
          f"top_k={args.top_k} n_variants={args.n_variants} "
          f"temperature={args.temperature} baseline={args.baseline}")

    # ── Load input ────────────────────────────────────────────────────────────
    with open(args.input_json, encoding="utf-8") as f:
        input_data = json.load(f)
    if args.limit:
        keys = list(input_data.keys())[:args.limit]
        input_data = {k: input_data[k] for k in keys}
    # Relative image paths resolve against $IMAGE_ROOT, else the input JSON's dir
    img_root = os.environ.get("IMAGE_ROOT") or os.path.dirname(os.path.abspath(args.input_json))
    for entry in input_data.values():
        if entry.get("image_path") and not os.path.isabs(entry["image_path"]):
            entry["image_path"] = os.path.join(img_root, entry["image_path"])
    print(f"[data] {len(input_data)} items from {args.input_json}")

    # ── Build generation tasks ────────────────────────────────────────────────
    # Each qid → dict{ image_path, original_prompt, perturbed_prompts, top_words, ambiguity, level }
    gen_tasks: Dict[str, Dict[str, Any]] = {}

    if args.baseline:
        for qid, entry in input_data.items():
            gen_tasks[qid] = {
                "image_path": entry["image_path"],
                "original_prompt": entry.get("question", ""),
                "perturbed_prompts": [],
                "top_words": [],
                "ambiguity": entry.get("ambiguity", ""),
                "level": entry.get("level"),
            }
    elif args.perturbations_json:
        # Resume from a saved perturbation checkpoint (skips attribution → perturbation)
        print(f"[pert] loading {args.perturbations_json}")
        with open(args.perturbations_json, encoding="utf-8") as f:
            gen_tasks = json.load(f)
        gen_tasks = {q: t for q, t in gen_tasks.items() if q in input_data}
        for qid, task in gen_tasks.items():
            task["image_path"] = input_data[qid]["image_path"]
        print(f"[pert] {len(gen_tasks)} qids loaded")
    else:
        print(f"[attr] loading {args.attribution_json}")
        with open(args.attribution_json, encoding="utf-8") as f:
            attr_data = json.load(f)
        attr_results = attr_data.get("results", attr_data)

        n_no_pert = 0
        for qid, attr_res in tqdm(attr_results.items(), desc="building perturbations"):
            if "error" in attr_res: continue
            if qid not in input_data: continue
            entry = input_data[qid]
            full_prompt = entry.get("question", "")
            question_only = entry.get("question_only", "")
            if not full_prompt or not question_only:
                continue
            # Pick top-k words from the attribution, restricted to the question span
            top_words = extract_top_k_words(attr_res, k=args.top_k, question_text=question_only)
            if not top_words:
                n_no_pert += 1
                gen_tasks[qid] = {
                    "image_path": entry["image_path"],
                    "original_prompt": full_prompt,
                    "perturbed_prompts": [],
                    "top_words": [],
                    "ambiguity": entry.get("ambiguity", ""),
                    "level": entry.get("level"),
                }
                continue
            # Perturb ONLY the question_only span, then splice back
            prefix, q_only, suffix = split_vl_prompt(full_prompt, question_only)
            pert_dicts = get_wordnet_perturbations(q_only, top_words, n_variants=args.n_variants)
            perturbed_prompts = [prefix + pd["perturbed_question"] + suffix for pd in pert_dicts]
            gen_tasks[qid] = {
                "image_path": entry["image_path"],
                "original_prompt": full_prompt,
                "perturbed_prompts": perturbed_prompts,
                "top_words": top_words,
                "perturbation_info": pert_dicts,
                "ambiguity": entry.get("ambiguity", ""),
                "level": entry.get("level"),
            }
        print(f"[pert] {len(gen_tasks)} qids total; {n_no_pert} without valid perturbations")

    # Save perturbation checkpoint
    if not args.baseline and not args.perturbations_json:
        pert_ckpt = os.path.join(args.output_dir, f"_perturbations_{args.method_name}.json")
        with open(pert_ckpt, "w", encoding="utf-8") as f:
            json.dump(gen_tasks, f, indent=2, ensure_ascii=False)
        print(f"[ckpt] {pert_ckpt}")
    if args.perturb_only:
        print("[perturb_only] done.")
        return

    # ── Generation step (or resume) ───────────────────────────────────────────
    generation_data: Dict[str, Dict[str, Any]] = {}

    if args.resume_from_gen:
        print(f"[resume] loading {args.resume_from_gen}")
        with open(args.resume_from_gen, encoding="utf-8") as f:
            generation_data = json.load(f)
    else:
        # Load generation backend
        print(f"[gen] loading {args.backend} backend...")
        if args.backend == "vllm":
            gen = VLLMVLGenerator(
                args.model_path, cache_dir=args.cache_dir,
                max_model_len=args.max_model_len,
                gpu_mem_util=args.gpu_mem_util,
            )
        else:
            gen = HFVLGenerator(args.model_path, cache_dir=args.cache_dir, device=args.device)

        for qid, task in tqdm(gen_tasks.items(), desc="generation"):
            image_path = task["image_path"]
            orig_prompt = task["original_prompt"]
            pert_prompts = task["perturbed_prompts"]
            # Original: 1 greedy + n_sampled stochastic
            if args.baseline:
                orig_answers = gen.generate(
                    image_path, orig_prompt,
                    n_greedy=1, n_sampled=args.baseline_n_samples - 1,
                    temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                )
                all_answers = orig_answers
                n_perturb = 0
            else:
                orig_answers = gen.generate(
                    image_path, orig_prompt,
                    n_greedy=1, n_sampled=args.n_sampled,
                    temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                )
                # Perturbed: 1 greedy each (vllm batch if supported)
                pert_answers = []
                if pert_prompts and args.backend == "vllm" and hasattr(gen, "generate_batch"):
                    batch_items = [(image_path, p) for p in pert_prompts]
                    rs = gen.generate_batch(batch_items, n_greedy=1, n_sampled=0,
                                            max_new_tokens=args.max_new_tokens)
                    for ans_list in rs:
                        pert_answers.extend(ans_list)
                else:
                    for pp in pert_prompts:
                        pert_answers.extend(gen.generate(
                            image_path, pp, n_greedy=1, n_sampled=0,
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                        ))
                all_answers = orig_answers + pert_answers
                n_perturb = len(pert_prompts)

            generation_data[qid] = {
                "all_answers": all_answers,
                "n_answers": len(all_answers),
                "ambiguity": task["ambiguity"],
                "level": task.get("level"),
                "top_words": task["top_words"],
                "n_perturbations": n_perturb,
            }

        # Free generator
        del gen
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Save generation checkpoint
        if args.baseline:
            gen_ckpt = os.path.join(args.output_dir,
                                    f"_gen_checkpoint_baseline_n{args.baseline_n_samples}.json")
        else:
            gen_ckpt = os.path.join(args.output_dir,
                                    f"_gen_checkpoint_{args.method_name}.json")
        with open(gen_ckpt, "w", encoding="utf-8") as f:
            json.dump(generation_data, f, indent=2, ensure_ascii=False)
        print(f"[ckpt] {gen_ckpt}")

    if args.gen_only:
        print("[gen_only] done; run compute_uq_from_checkpoint.py for UQ.")
        return

    # ── UQ computation ────────────────────────────────────────────────────────
    print("[uq] loading DeBERTa NLI model...")
    try:
        from lm_polygraph.utils.deberta import Deberta
        from lm_polygraph.stat_calculators.semantic_matrix import SemanticMatrixCalculator
        from lm_polygraph.estimators import EigValLaplacian, DegMat
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        print(f"   [warn] lm_polygraph unavailable ({type(e).__name__}: {e}).")
        print(f"   [info] Generation checkpoint saved; run compute_uq_from_checkpoint.py to compute UQ.")
        return

    deberta = Deberta(
        deberta_path=args.nli_model, batch_size=args.nli_batch_size,
        device=args.device, hf_cache=args.cache_dir,
    )
    sem_calc = SemanticMatrixCalculator(nli_model=deberta)
    eigv_est = EigValLaplacian()
    degmat_est = DegMat()

    uq_results: Dict[str, Any] = {}
    for qid, gdata in tqdm(generation_data.items(), desc="computing UQ"):
        uq = compute_uq_metrics(gdata["all_answers"], deberta, sem_calc, eigv_est, degmat_est)
        uq_results[qid] = {
            **uq,
            "n_answers": len(gdata["all_answers"]),
            "answers": gdata["all_answers"],
            "ambiguity": gdata["ambiguity"],
            "level": gdata.get("level"),
            "top_words": gdata.get("top_words", []),
            "n_perturbations": gdata.get("n_perturbations", 0),
        }

    del deberta, sem_calc
    gc.collect()

    # ── AUROC ─────────────────────────────────────────────────────────────────
    labels, eigv_s, deg_s = [], [], []
    for qid, uq in uq_results.items():
        labels.append(1 if uq["ambiguity"] == "ambiguous" else 0)
        eigv_s.append(uq["EigV_entail"])
        deg_s.append(uq["DegMat_entail"])
    labels = np.array(labels)
    eigv_s = np.array(eigv_s)
    deg_s = np.array(deg_s)

    auroc_eigv = compute_auroc(labels, eigv_s)
    auroc_deg = compute_auroc(labels, deg_s)
    np.random.seed(42)
    auroc_rand = compute_auroc(labels, np.random.rand(len(labels)))

    print("\n" + "=" * 60)
    print(f"  AUROC: {args.method_name}")
    print("=" * 60)
    print(f"  EigV_entail   : {auroc_eigv:.4f}")
    print(f"  DegMat_entail : {auroc_deg:.4f}")
    print(f"  random        : {auroc_rand:.4f}")
    print(f"  N={len(labels)}  amb={int(labels.sum())}  unamb={len(labels)-int(labels.sum())}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "method": args.method_name,
            "model": args.model_path,
            "backend": args.backend,
            "top_k": args.top_k,
            "n_variants": args.n_variants,
            "n_sampled": args.n_sampled,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "perturbation_strategy": "wordnet",
            "baseline": args.baseline,
            "baseline_n_samples": args.baseline_n_samples if args.baseline else None,
            "n_questions": len(uq_results),
            "timestamp": datetime.now().isoformat(),
        },
        "auroc": {
            "EigV_entail": auroc_eigv,
            "DegMat_entail": auroc_deg,
            "random_baseline": auroc_rand,
        },
        "per_question": uq_results,
    }

    if args.baseline:
        outfile = os.path.join(args.output_dir,
                               f"pipeline_b_baseline_n{args.baseline_n_samples}.json")
    else:
        outfile = os.path.join(args.output_dir,
                               f"pipeline_b_{args.method_name}_top{args.top_k}_question_only.json")
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False,
                  default=lambda o: float(o) if isinstance(o, np.floating) else
                                   int(o) if isinstance(o, np.integer) else
                                   o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"[saved] {outfile}")


if __name__ == "__main__":
    main()
