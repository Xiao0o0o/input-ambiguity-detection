#!/usr/bin/env python3
"""Build SAUP-format input + labels for VQ-FocusAmbiguity test split (n=5334).
Ambig labels derived from mask count (>1 mask = ambiguous; verified 97% on train, 89% on val).

Expects the VQ-FocusAmbiguity release unpacked under $VQFA_ROOT (default:
data/vq_focus_ambiguity) with focus_ambiguity_public/{test_public.json,images/}
and masks_gt_v3/test/<masks_id>/*.png. The 720-item subset's images are
copied to data/images/vqfa/ and stored as paths relative to data/.

Usage:
    VQFA_ROOT=/path/to/vq_focus_ambiguity python data/build/build_vqfa_input.py
"""
import json
import os
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent  # data/
ROOT = Path(os.environ.get("VQFA_ROOT", DATA_DIR / "vq_focus_ambiguity"))
TEST_JSON = ROOT / "focus_ambiguity_public/test_public.json"
IMG_DIR = ROOT / "focus_ambiguity_public/images"
MASKS_DIR = ROOT / "masks_gt_v3/test"

OUT_INPUT = DATA_DIR / "vqfa_input.json"
OUT_LABELS = DATA_DIR / "vqfa_labels_certain.json"
OUT_INPUT_720 = DATA_DIR / "vqfa_720_input.json"
OUT_LABELS_720 = DATA_DIR / "vqfa_720_labels_certain.json"
OUT_IMG_DIR = DATA_DIR / "images" / "vqfa"
SUBSAMPLE_N = 720

USER_INSTRUCTION = (
    "You are a helpful visual QA assistant.\n"
    "Look at the image and answer the question concisely."
)


def main():
    test = json.load(open(TEST_JSON, encoding="utf-8"))
    print(f"loading {len(test)} test entries")

    out = {}
    labels = {}
    n_amb, n_unamb, n_skip = 0, 0, 0

    for ex in test:
        masks_id = ex["masks_id"]
        mdir = MASKS_DIR / masks_id
        if not mdir.is_dir():
            n_skip += 1; continue
        n_masks = len([f for f in os.listdir(mdir) if f.endswith(".png")])
        if n_masks == 0:
            n_skip += 1; continue

        is_ambig = (n_masks > 1)
        img_path = IMG_DIR / ex["file_name"]
        if not img_path.is_file():
            n_skip += 1; continue

        qid = f"vqfa_test_{ex['id']:05d}"
        q = ex["question"]
        full_prompt = f"{USER_INSTRUCTION}\n\nQuestion:\n{q}"

        out[qid] = {
            "question": full_prompt,
            "question_only": q,
            "image_path": str(img_path),
            "filename": ex["file_name"],
            "masks_id": masks_id,
            "n_masks": n_masks,
            "level": 1 if is_ambig else 0,  # binary; no fine-grained levels
            "ambiguity": "ambiguous" if is_ambig else "unambiguous",
            "items": [{"question": full_prompt}],
        }
        labels[qid] = "uncertain" if is_ambig else "certain"
        if is_ambig: n_amb += 1
        else: n_unamb += 1

    print(f"built {len(out)} entries; ambig={n_amb}, unamb={n_unamb}, skipped={n_skip}")
    json.dump(out, open(OUT_INPUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(labels, open(OUT_LABELS, "w", encoding="utf-8"), indent=2)
    print(f"[saved] {OUT_INPUT}")
    print(f"[saved] {OUT_LABELS}")

    # Subsample to 720 for parity with AQuA (stratified by ambig label)
    import random
    random.seed(42)
    ambig_qids = sorted([q for q, v in out.items() if v["ambiguity"] == "ambiguous"])
    unamb_qids = sorted([q for q, v in out.items() if v["ambiguity"] == "unambiguous"])
    n_amb_target = int(SUBSAMPLE_N * len(ambig_qids) / (len(ambig_qids) + len(unamb_qids)))
    n_unamb_target = SUBSAMPLE_N - n_amb_target
    sampled_amb = random.sample(ambig_qids, n_amb_target)
    sampled_unamb = random.sample(unamb_qids, n_unamb_target)
    sampled = set(sampled_amb + sampled_unamb)
    out720 = {q: v for q, v in out.items() if q in sampled}
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for v in out720.values():
        dst = OUT_IMG_DIR / v["filename"]
        if not dst.exists():
            shutil.copy(v["image_path"], dst)
        v["image_path"] = str(dst.relative_to(DATA_DIR))
    labels720 = {q: v for q, v in labels.items() if q in sampled}
    json.dump(out720, open(OUT_INPUT_720, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(labels720, open(OUT_LABELS_720, "w", encoding="utf-8"), indent=2)
    print(f"[saved] {OUT_INPUT_720} (n={len(out720)}, ambig={n_amb_target}, unamb={n_unamb_target})")
    print(f"[saved] {OUT_LABELS_720}")


if __name__ == "__main__":
    main()
