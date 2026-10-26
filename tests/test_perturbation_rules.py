"""Black-box rule library (blackbox/perturbation_rules.py)."""
import pytest

pr = pytest.importorskip("perturbation_rules")


def variants(question, rule, k=2):
    return [v["variant_question"] for v in pr.generate_all_variants(question, rules=[rule],
                                                                     max_variants_per_rule=k)]


def test_registry_has_all_eleven_rules():
    assert list(pr.ALL_RULES) == pr.GROUP_A_RULES + pr.GROUP_B_RULES
    assert len(pr.ALL_RULES) == 11


def test_contraction_expand_and_contract():
    assert variants("Who's the author?", "A5") == ["Who is the author?"]
    assert variants("Who is the author?", "A6") == ["Who's the author?"]
    assert variants("Who wrote it?", "A5") == []


@pytest.mark.parametrize("rule,prefix", [
    ("B1", "According to the passage,"),
    ("B2", "Hey so,"),
    ("B3", "Based on the context above,"),
])
def test_prefix_rules_keep_question_stem(rule, prefix):
    out = variants("Who won?", rule)
    assert out[0] == f"{prefix} who won?"
    assert len(out) == 2
    assert all(v.endswith("who won?") for v in out)


def test_wh_expansion_and_rephrase():
    assert variants("Who won?", "B4") == ["Which person won?"]
    assert variants("How did it happen?", "B5") == ["By what means did it happen?",
                                                    "In what manner did it happen?"]
    assert variants("How many people came?", "B4") == ["What number of people came?"]


def test_wh_rules_guard_determiner_use():
    # "What color" is a determiner: no expansion, only the what -> which swap.
    assert variants("What color was it?", "B4") == []
    assert variants("What color was it?", "B5") == ["Which color was it?"]


def test_no_wh_word_no_template_variant():
    assert variants("In what year?", "B4") == []
    assert variants("In what year?", "B5") == []


@pytest.mark.skipif(not pr.USE_WORDNET, reason="WordNet data not installed")
def test_synonym_rules_change_one_word():
    q = "What is the largest city?"
    out = variants(q, "A3")
    assert out, "expected an adjective synonym"
    for v in out:
        a, b = q.rstrip("?").split(), v.rstrip(" ?").split()
        assert len(a) == len(b)
        assert sum(x != y for x, y in zip(a, b)) == 1


@pytest.mark.skipif(not (pr.USE_WORDNET and pr.USE_SPACY), reason="WordNet / spaCy not installed")
def test_verb_synonym_keeps_tense():
    out = variants("Who helped?", "A1")
    assert out and all(v.split()[1].endswith("ed") for v in out)
