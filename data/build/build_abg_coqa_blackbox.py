#!/usr/bin/env python3
"""
Build the black-box input for ABG-CoQA: the same 200 questions as
data/abg_coqa_200_input.json, in the original ABG-CoQA record format
({"data": [{id, story, history_turns, target_turn, ambiguity, ...}]}) that
blackbox/run_*.py read.

The ids of data/abg_coqa_200_input.json are `abg_coqa_<i>`, where <i> is the
row index in the official ABG-CoQA test split. The official file is
downloaded from https://github.com/MeiqiGuo/AKBC2021-Abg-CoQA (MIT) unless
--abg_test points to a local copy.

Usage:
    python data/build/build_abg_coqa_blackbox.py
    python data/build/build_abg_coqa_blackbox.py --abg_test /path/to/coqa_abg_test.json
"""
import argparse
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent  # data/
ABG_TEST_URL = ("https://raw.githubusercontent.com/MeiqiGuo/AKBC2021-Abg-CoQA/"
                "main/abg-coqa/coqa_abg_test.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abg_test", default=None,
                    help="local coqa_abg_test.json (downloaded if omitted)")
    ap.add_argument("--subset", default=str(DATA_DIR / "abg_coqa_200_input.json"))
    ap.add_argument("--out", default=str(DATA_DIR / "abg_coqa_200_blackbox.json"))
    args = ap.parse_args()

    if args.abg_test:
        with open(args.abg_test, encoding="utf-8") as f:
            test = json.load(f)
    else:
        print(f"downloading {ABG_TEST_URL}")
        with urllib.request.urlopen(ABG_TEST_URL) as r:
            test = json.loads(r.read().decode("utf-8"))
    rows = test["data"]

    with open(args.subset, encoding="utf-8") as f:
        subset = json.load(f)

    out = []
    for qid, item in subset.items():
        row = rows[int(qid.rsplit("_", 1)[1])]
        # Guard against a different version of the test split.
        assert row["target_turn"]["question"].strip() == item["question_only"].strip(), qid
        # Use the shared qid so black-box and white-box outputs line up.
        out.append(dict(row, id=qid, abg_id=row["id"]))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"version": test.get("version"), "data": out}, f,
                  indent=2, ensure_ascii=False)
    n_amb = sum(r["ambiguity"] == "ambiguous" for r in out)
    print(f"wrote {args.out}: {len(out)} items ({n_amb} ambiguous / {len(out) - n_amb} non-ambiguous)")


if __name__ == "__main__":
    main()
