#!/usr/bin/env python3
"""
Prepare AQuA validation-set input JSON for VL attribution methods.

AQuA fields: image (PIL), filename, level (0-3), question, answer.
  level 0 = unambiguous; level 1/2/3 = ambiguous (referential / multiple / clarification).

Binary label for AUROC: level 0 -> unambiguous, level 1/2/3 -> ambiguous.
The 4-level info is preserved under `level` for later fine-grained analysis.

Images are saved once under data/images/aqua/<filename> (paths stored
relative to data/) so attribution scripts can load from disk without
re-reading the HF dataset.

Usage:
    python data/build/build_aqua_input.py
"""
import json
import os
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent  # data/
CACHE_DIR = os.environ.get("HF_CACHE_DIR")
IMG_DIR = DATA_DIR / "images" / "aqua"
OUT_INPUT = DATA_DIR / "aqua_720_input.json"
OUT_LABELS = DATA_DIR / "aqua_720_labels.json"
OUT_LABELS_NUM = DATA_DIR / "aqua_720_labels_numeric.json"

USER_INSTRUCTION = (
    "You are a helpful visual QA assistant.\n"
    "Look at the image and answer the question concisely."
)


def build_text_prompt(question: str) -> str:
    return USER_INSTRUCTION + "\n\nQuestion:\n" + question


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading AQuA validation split ...")
    ds = load_dataset("jihyoung/AQuA", split="validation", cache_dir=CACHE_DIR)
    print(f"       n = {len(ds)}")

    inputs = {}
    labels = {}
    labels_num = {}

    print(f"[2/3] Saving images to {IMG_DIR}")
    for i, row in enumerate(ds):
        qid = f"aqua_val_{i:04d}"
        img_path = IMG_DIR / row["filename"]
        if not img_path.exists():
            row["image"].save(img_path)

        level = int(row["level"])
        ambiguity = "unambiguous" if level == 0 else "ambiguous"
        question = row["question"]
        gold_answer = row["answer"]

        text_prompt = build_text_prompt(question)

        inputs[qid] = {
            "question": text_prompt,
            "question_only": question,
            "image_path": str(img_path.relative_to(DATA_DIR)),
            "filename": row["filename"],
            "level": level,
            "ambiguity": ambiguity,
            "gold_answer": gold_answer,
            "items": [{"question": text_prompt}],
        }
        labels[qid] = ambiguity
        labels_num[qid] = 1 if ambiguity == "ambiguous" else 0

    print(f"[3/3] Writing JSONs")
    with open(OUT_INPUT, "w", encoding="utf-8") as f:
        json.dump(inputs, f, indent=2, ensure_ascii=False)
    with open(OUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    with open(OUT_LABELS_NUM, "w", encoding="utf-8") as f:
        json.dump(labels_num, f, indent=2)

    n_amb = sum(1 for v in inputs.values() if v["ambiguity"] == "ambiguous")
    n_un = len(inputs) - n_amb
    level_counts = {}
    for v in inputs.values():
        level_counts[v["level"]] = level_counts.get(v["level"], 0) + 1
    print(f"\nTotal: {len(inputs)}  amb={n_amb}  unamb={n_un}")
    print(f"Levels: {dict(sorted(level_counts.items()))}")
    print(f"Input JSON : {OUT_INPUT}")
    print(f"Labels     : {OUT_LABELS}")
    print(f"Labels(num): {OUT_LABELS_NUM}")


if __name__ == "__main__":
    main()
