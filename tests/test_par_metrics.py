"""Masked attribute metrics and the voting curve, hand-computed."""

import numpy as np
import pytest

from reid.eval import par_metrics as pm

PROBS = np.array([[0.9, 0.2, 0.9], [0.6, 0.1, 0.4], [0.2, 0.8, 0.1]])
TARGETS = np.array([[1, 0, -1], [0, 0, 1], [1, 1, -1]], np.int8)


def test_label_stats_and_mA():
    st = pm.label_stats(PROBS, TARGETS)
    assert st["n_pos"].tolist() == [2, 1, 1] and st["n_neg"].tolist() == [1, 2, 0]
    assert st["pos_recall"][0] == pytest.approx(0.5) and st["neg_recall"][0] == 0.0
    assert st["ma_term"][:2].tolist() == pytest.approx([0.25, 1.0])
    assert np.isnan(st["ma_term"][2])  # no negative: not evaluated
    assert pm.mean_accuracy(PROBS, TARGETS) == pytest.approx(0.625)
    assert pm.mean_accuracy(PROBS, TARGETS, cols=[1]) == pytest.approx(1.0)
    assert np.isnan(pm.mean_accuracy(PROBS, TARGETS, cols=[2]))


def test_unknown_targets_are_ignored():
    p2 = PROBS.copy()
    p2[0, 2] = 0.0  # prediction on an unknown target: no effect
    assert pm.mean_accuracy(p2, TARGETS) == pytest.approx(0.625)
    st = pm.label_stats(PROBS, TARGETS, threshold=np.array([0.5, 0.15, 0.5]))
    assert st["neg_recall"][1] == pytest.approx(0.5)  # per-label thresholds


def test_instance_metrics():
    m = pm.instance_metrics(PROBS, TARGETS)
    assert m["acc"] == pytest.approx(0.5)
    assert m["prec"] == pytest.approx(2 / 3) and m["rec"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m["f1_per_image"] == pytest.approx((1 + 0 + 2 / 3) / 3)


def test_instance_metrics_skips_empty_images():
    t = np.array([[0, 0], [1, 0]], np.int8)
    p = np.array([[0.1, 0.1], [0.9, 0.1]])  # image 0: no positive anywhere: skipped
    m = pm.instance_metrics(p, t)
    assert m["acc"] == m["prec"] == m["rec"] == m["f1"] == 1.0
    assert np.isnan(pm.instance_metrics(p[:1], t[:1])["acc"])


def test_voting_curve_improves_with_k():
    rng = np.random.default_rng(0)
    n_people, n_lab = 60, 4
    truth = rng.integers(0, 2, (n_people, n_lab)).astype(np.int8)
    counts = np.where(np.arange(n_people) < 40, 8, 2)  # 40 people with 8 images, 20 with 2
    persons = np.repeat([f"p{i}" for i in range(n_people)], counts)
    tgt = np.repeat(truth, counts, axis=0)
    probs = np.clip(0.5 + (tgt - 0.5) * 0.15 + rng.normal(0, 0.3, tgt.shape), 0, 1)
    curve = pm.voting_curve(probs, tgt, persons, ks=(1, 2, 8), draws=4, min_persons=30)
    by_k = {r["k"]: r for r in curve}
    assert set(by_k) == {1, 2, 8}  # the largest feasible k (8) is already listed
    assert by_k[1]["n_persons"] == 60 and by_k[2]["n_persons"] == 60 and by_k[8]["n_persons"] == 40
    assert by_k[8]["ma"] > by_k[2]["ma"] > by_k[1]["ma"] - 0.02
    assert by_k[8]["ma"] > by_k[8]["ma_k1"]
    assert pm.voting_curve(probs, tgt, persons, ks=(16,), min_persons=100) == []
