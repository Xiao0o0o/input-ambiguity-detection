"""White-box scoring (EigV variants) and the quick-start AUROC."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
QS = ROOT / "quickstart" / "intermediate"


def block_agreement(sizes):
    """Perfect-agreement NLI matrix with one block per semantic cluster."""
    n = sum(sizes)
    W = np.zeros((n, n))
    i = 0
    for s in sizes:
        W[i:i + s, i:i + s] = 1.0
        i += s
    return W


@pytest.fixture(scope="module")
def uq():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import compute_uq_from_checkpoint
    return compute_uq_from_checkpoint


@pytest.mark.parametrize("sizes", [[6], [3, 3], [2, 2, 2], [4, 1, 1]])
def test_polygraph_eigv_counts_clusters(uq, sizes):
    # With self-entailment on the diagonal, EigV(Agree) equals the number of clusters.
    assert uq.eigv_laplacian_polygraph(block_agreement(sizes)) == pytest.approx(len(sizes))


def test_eigv_increases_with_disagreement(uq):
    for f in (uq.eigv_laplacian, uq.eigv_laplacian_polygraph):
        scores = [f(block_agreement(s)) for s in ([6], [3, 3], [2, 2, 2])]
        assert scores == sorted(scores) and scores[0] < scores[-1]


def test_semantic_clusters(uq):
    classes = uq.cluster_sem_classes(block_agreement([2, 3]))
    assert uq.num_sem_sets_from_classes(classes) == 2
    assert uq.sem_entropy_freq(uq.cluster_sem_classes(block_agreement([5]))) == pytest.approx(0.0, abs=1e-9)


def test_quickstart_auroc_matches_paper():
    import compute_auroc
    for fname, expected in [("pipeline_b_logit_lens.json", 0.708),   # Ours (White-Box), Table 1
                            ("pipeline_b_baseline_n10.json", 0.644)]:  # EigV(Agree) baseline
        recs = compute_auroc.load_scores(str(QS / fname), ["EigV_entail"])
        # the paper reports 3 decimals (0.64346 is printed as 0.644)
        assert compute_auroc.auroc(recs, "EigV_entail")[0] == pytest.approx(expected, abs=1e-3)


def test_quickstart_auroc_matches_stored_summary():
    import compute_auroc
    stored = json.loads((QS / "auroc.json").read_text(encoding="utf-8"))
    for fname, metrics in stored.items():
        recs = compute_auroc.load_scores(str(QS / fname), list(metrics))
        for m, v in metrics.items():
            assert compute_auroc.auroc(recs, m)[0] == pytest.approx(v["auroc"])


def test_packaged_quickstart_command(capsys):
    qs = pytest.importorskip("input_ambiguity.quickstart")  # needs `pip install -e .`
    qs.main([])
    out = capsys.readouterr().out
    assert "0.7083" in out and "0.6435" in out
