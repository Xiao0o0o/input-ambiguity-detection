"""Black-box shared helpers (blackbox/llm_utils.py)."""
import json

import llm_utils as U


def test_detect_model_type():
    assert U.detect_model_type("Qwen/Qwen3-4B-Instruct-2507") == "qwen"
    assert U.detect_model_type("meta-llama/Llama-3.1-8B-Instruct") == "llama"
    assert U.detect_model_type("google/gemma-3-27b-it") == "gemma"
    assert U.detect_model_type("some/unknown-model") == "generic"


def test_qa_prompt_contains_passage_history_and_question():
    hist = [{"question": "who came?", "answer": "Sam"}]
    p = U.build_qa_prompt("A passage.", hist, "did he?", "qwen")
    assert p.startswith("<|im_start|>system\n" + U.QA_SYSTEM_MSG)
    assert "Passage:\nA passage.\n\nPrevious conversation:\nQ: who came?\nA: Sam\n\n" in p
    assert p.endswith("Current question: did he?\n\nAnswer:<|im_end|>\n<|im_start|>assistant\n")
    assert "Previous conversation" not in U.build_qa_prompt("A passage.", [], "did he?", "llama")


def test_load_examples_filters(tmp_path):
    data = {"data": [{"id": str(i), "ambiguity": "ambiguous" if i % 2 else "non_ambiguous"}
                     for i in range(5)]}
    path = tmp_path / "in.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert len(U.load_examples(str(path), -1)) == 5
    assert len(U.load_examples(str(path), 0)) == 5
    assert len(U.load_examples(str(path), 3)) == 3
    assert {e["id"] for e in U.load_examples(str(path), None, only_ambiguous=True)} == {"1", "3"}
    assert len(U.load_examples(str(path), None, only_non_ambiguous=True)) == 3


def test_blackbox_abg_coqa_subset_matches_whitebox_ids():
    root = U.os.path.dirname(U.os.path.dirname(U.os.path.abspath(U.__file__)))
    with open(f"{root}/data/abg_coqa_200_blackbox.json", encoding="utf-8") as f:
        bb = json.load(f)["data"]
    with open(f"{root}/data/abg_coqa_200_input.json", encoding="utf-8") as f:
        wb = json.load(f)
    assert [e["id"] for e in bb] == list(wb)
    for e in bb:
        assert e["target_turn"]["question"].strip() == wb[e["id"]]["question_only"].strip()
        assert (e["ambiguity"] == "ambiguous") == (wb[e["id"]]["ambiguity"] == "ambiguous")
