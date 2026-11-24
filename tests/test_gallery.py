import numpy as np
import pytest

from reid.identity import gallery as G
from tests.synth import centers, jitter


def ang(*deg):
    a = np.deg2rad(np.array(deg, dtype=np.float64))
    return np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)


def test_diversity_gate_min_reference():
    members = ang(0, 30)  # D(C) = 1 - cos 30 deg = 0.134
    assert not G.passes_diversity_gate(members, ang(2)[0])  # near-duplicate of a member
    assert G.passes_diversity_gate(members, ang(70)[0])  # adds diversity
    assert G.passes_diversity_gate(members[:1], ang(1)[0])  # below 2 members: always
    # 'mean' reference (mean nn distance) is stricter than the paper's 'min'
    m = ang(0, 10, 90)  # nn distances 0.015, 0.015, 0.826 -> min 0.015, mean 0.285
    x = ang(45)[0]  # distance to the nearest member: 1 - cos 35 deg = 0.181
    assert G.passes_diversity_gate(m, x, "min")
    assert not G.passes_diversity_gate(m, x, "mean")


def test_drop_rule_redundancy_with_single_class():
    e = ang(0, 1, 90)  # entropy term is 0 with N=1 -> drops the most redundant sample
    i = G.drop_index(e, np.array([[1.0, 0.0]], np.float32), gamma=0.6, temp=10)
    assert i in (0, 1)


def test_drop_rule_entropy_dominates():
    e = ang(0, 5, 45)  # the 45 deg sample is equally close to both classes: max entropy
    cents = np.array([[1, 0], [0, 1]], np.float32)
    assert G.drop_index(e, cents, gamma=0.6, temp=10) == 2
    # with gamma = 0 only redundancy counts: one of the near-duplicates goes instead
    assert G.drop_index(e, cents, gamma=0.0, temp=10) in (0, 1)


def test_normalized_entropy():
    p = G.softmax(np.array([[0.5, 0.5, 0.5, 0.5]]), 10)
    assert G.normalized_entropy(p, 4)[0] == pytest.approx(1.0, abs=1e-6)
    assert G.normalized_entropy(p, 1)[0] == 0.0


def test_budget_respected_on_insert_and_merge():
    rng = np.random.default_rng(0)
    c = centers(3, seed=1)
    bank = G.GalleryBank(budget=6, gamma=0.6, temp=10, diversity_ref="min")
    sid = 0
    for gid in (1, 2):
        embs = jitter(c[gid - 1], 400, 0.6, rng)
        for e in embs:
            bank.insert(gid, e, 0.8, sid)
            sid += 1
        assert len(bank.galleries[gid]) == 6  # full, never above the budget
    # merge: union of two full galleries is re-pruned to the budget (v1 B7 fixed)
    moved, dropped = bank.merge(1, 2)
    assert 2 not in bank and len(bank.galleries[1]) == 6
    assert len(dropped) == 6 and len(moved) <= 6
    assert len(set(bank.galleries[1].ids)) == 6


def test_insert_many_reports_membership_changes():
    rng = np.random.default_rng(1)
    c = centers(2, seed=2)
    bank = G.GalleryBank(budget=5)
    e = jitter(c[0], 20, 0.6, rng)
    added, removed = bank.insert_many(1, e, np.linspace(0.5, 0.9, 20), list(range(20)))
    assert set(added) == set(bank.galleries[1].ids) and not removed
    added2, removed2 = bank.insert_many(
        1, jitter(c[0], 20, 0.6, rng), np.full(20, 0.7), list(range(20, 40))
    )
    assert set(bank.galleries[1].ids) == (set(added) | set(added2)) - set(removed2) | set()
    assert len(bank.galleries[1]) <= 5


def test_softmax_ratio_accept_reject():
    gids = [10, 20, 30]
    cents = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    kw = dict(temp=10, tau=2.0, min_cos=0.78)
    # clearly closest to gid 10
    x = G.l2n(np.array([[0.95, 0.2, 0.1]]))[0]
    d = G.softmax_ratio_test(x, gids, cents, **kw)
    assert d.accepted and d.gid == 10 and d.ratio >= 2
    # equidistant between 10 and 20: ratio ~ 1 -> reject (gid still reported)
    x = G.l2n(np.array([[1, 1, 0.05]]))[0]
    d = G.softmax_ratio_test(x, gids, cents, **dict(kw, min_cos=0.5))
    assert not d.accepted and d.ratio < 1.5
    # decisive margin but below the cosine floor
    x = G.l2n(np.array([[0.6, 0.0, 0.0]]) + np.array([[0, 0.5, 0.5]]) * 0.9)[0]
    d = G.softmax_ratio_test(x, gids, cents, temp=10, tau=1.0, min_cos=0.78)
    assert d.cos < 0.78 and not d.accepted
    # blocking the winner makes the runner-up the candidate; one candidate left -> ratio inf
    x = G.l2n(np.array([[0.95, 0.2, 0.1]]))[0]
    d = G.softmax_ratio_test(x, gids, cents, blocked={10}, **dict(kw, min_cos=0.1))
    assert d.gid == 20
    d = G.softmax_ratio_test(x, [10, 20], cents[:2], blocked={20}, **kw)
    assert d.accepted and d.ratio == float("inf")
    assert G.softmax_ratio_test(x, [10], cents[:1], blocked={10}, **kw).gid is None


def test_temperature_matches_paper_equivalence():
    # ratio = exp(temp * (s1 - s2)): temp 10 needs a score margin of ln(2)/10 = 0.0693 for tau = 2
    cents = np.eye(2, dtype=np.float32)
    for margin, ok in ((0.05, False), (0.09, True)):
        x = np.array([0.8, 0.8 - margin], np.float32)  # scores are the raw coordinates
        d = G.softmax_ratio_test(x, [1, 2], cents, temp=10, tau=2.0, min_cos=0.0)
        assert d.accepted is ok
        assert d.ratio == pytest.approx(np.exp(10 * margin), rel=1e-4)
