"""Zero-shot external evaluation: metrics, label filter, identity vote, run (CPU, synthetic)."""

import json

import numpy as np
import pytest

from reid.attributes import schema as s
from reid.attributes.datasets import group_by_person
from reid.attributes.external import bootstrap_ma, evaluate_external, in_domain, vote
from reid.attributes.external_run import find_in_domain_metrics, run_eval_external
from reid.eval.par_metrics import label_auc
from tests.attr_market_synth import make_market
from tests.attr_synth import make_head


def labelled_head(in_dim: int = 8, thr: float = 0.5):
    head = make_head(in_dim, labels=list(s.LABELS), body_model="fake", thr=thr)
    return head


def fake_split(n_ids=40, per_id=6, seed=0):
    """Targets for 3 labels (balanced, rare, untrained) and an oracle probability matrix."""
    rng = np.random.default_rng(seed)
    persons = [f"{k:04d}" for k in range(n_ids) for _ in range(per_id)]
    ids = np.repeat(np.arange(n_ids), per_id)
    t = np.full((len(ids), s.N_LABELS), -1, np.int8)
    t[:, s.INDEX["female"]] = ids % 2
    t[:, s.INDEX["hat"]] = (ids < 3).astype(np.int8)  # 3 positive identities only
    t[:, s.INDEX["backpack"]] = (ids % 3 == 0).astype(np.int8)
    p = np.full(t.shape, 0.5, np.float32)
    for lab in ("female", "hat", "backpack"):
        j = s.INDEX[lab]
        p[:, j] = np.where(t[:, j] == 1, 0.9, 0.1) + rng.normal(0, 0.01, len(ids))
    return persons, t, p


def test_label_auc():
    p = np.array([[0.1], [0.4], [0.35], [0.8]])
    t = np.array([[0], [0], [1], [1]], np.int8)
    assert label_auc(p, t)[0] == pytest.approx(0.75)
    assert np.isnan(label_auc(p, np.ones_like(t))[0])
    t2 = t.copy()
    t2[0] = -1  # unknown rows are ignored
    assert label_auc(p, t2)[0] == pytest.approx(0.5)  # 0.8 > 0.4 but 0.35 < 0.4


def test_evaluate_oracle_probabilities_and_label_filter():
    persons, t, p = fake_split()
    head = labelled_head()
    head.trained_labels[s.INDEX["backpack"]] = False
    res = evaluate_external(head, p, t, persons, min_pos=5, repeats=3, k=4)
    assert res["used_labels"] == ["female"]  # hat: 3 positive ids < 5; backpack: untrained
    assert set(res["excluded"]) == {"hat", "backpack"} and "3 positive" in res["excluded"]["hat"]
    assert res["n_images"] == 240 and res["n_identities"] == 40
    assert res["image"]["ma"] == pytest.approx(1.0) and res["image"]["f1"] == pytest.approx(1.0)
    assert res["identity"]["summary"]["ma"]["mean"] == pytest.approx(1.0)
    assert res["identity"]["summary"]["ma"]["std"] == pytest.approx(0.0)
    assert res["labels"]["hat"]["n_pos_ids"] == 3 and res["labels"]["hat"]["n_pos_images"] == 18
    assert res["labels"]["female"]["auc"] == pytest.approx(1.0)
    assert res["image"]["ma_all_mapped"] == pytest.approx(1.0)  # hat counted here
    assert [r["k"] for r in res["voting"]][:3] == [1, 2, 4]
    c = res["consistency"]["female"]
    assert c["pos_rate"] == 1.0 and c["neg_rate"] == 0.0 and c["pos_mixed"] == 0.0


def test_identity_vote_uses_all_crops_when_fewer_than_k():
    persons = ["a", "a", "b"]
    probs = np.array([[0.0], [1.0], [0.3]], np.float32)
    out = vote(probs, group_by_person(persons), 16, np.random.default_rng(0))
    assert out.ravel().tolist() == pytest.approx([0.5, 0.3])


def test_identity_vote_reduces_noise():
    rng = np.random.default_rng(1)
    persons, t, _ = fake_split(n_ids=60, per_id=16)
    j = s.INDEX["female"]
    p = np.full(t.shape, 0.5, np.float32)
    p[:, j] = np.clip(0.5 + (t[:, j] - 0.5) * 0.3 + rng.normal(0, 0.35, len(t)), 0, 1)
    res = evaluate_external(labelled_head(), p, t, persons, min_pos=5, repeats=5, k=16)
    assert res["identity"]["summary"]["ma"]["mean"] > res["image"]["ma"] + 0.05


def test_in_domain_reference_filters_by_positives():
    metrics = {
        "datasets": {
            "pa100k": {
                "labels": {"female": {"n_pos": 99, "ma": 0.9}, "hat": {"n_pos": 5, "ma": 0.7}}
            },
            "peta": {"labels": {"female": {"n_pos": 99, "ma": 0.8}}},
            "other": {"labels": {"jeans": {"n_pos": 99, "ma": 0.8}}},
        }
    }
    ref = in_domain(metrics, ["female", "hat"])
    assert set(ref) == {"pa100k", "peta"}
    assert ref["pa100k"]["labels"] == {"female": 0.9} and ref["peta"]["mean"] == 0.8


def test_find_in_domain_metrics(tmp_path):
    (tmp_path / "head").mkdir()
    (tmp_path / "report").mkdir()
    (tmp_path / "head" / "head.npz").write_bytes(b"")
    assert find_in_domain_metrics(tmp_path / "head") is None
    (tmp_path / "report" / "metrics_x.json").write_text("{}")
    assert find_in_domain_metrics(tmp_path / "head" / "head.npz").name == "metrics_x.json"


def test_run_eval_external_end_to_end_and_resume(tmp_path):
    pytest.importorskip("torch")
    from reid.attributes.head import save_head
    from tests.test_attr_embed import FakeBody

    make_market(tmp_path / "data", n_ids=12, per_id=4)
    save_head(tmp_path / "head", labelled_head())
    out = tmp_path / "out"
    body = FakeBody()
    kw = {"dataset": "market1501", "root": tmp_path / "data", "head_path": tmp_path / "head"}
    path = run_eval_external(**kw, out=out, repeats=2, min_pos=3, embedder=body, batch=8, workers=0)
    assert path.name == "RESULTS_market1501.md" and body.items == 48  # no flipped copies
    m = json.loads((out / "metrics_market1501.json").read_text())
    assert m["result"]["n_identities"] == 12 and m["meta"]["body_model"] == "fake"
    assert m["in_domain"] == {} and "female" in m["result"]["mapped_labels"]
    assert "Headline" in path.read_text() and "Voting curve" in path.read_text()
    shards = sorted((out / "emb" / "market1501" / "test").glob("shard_*.npy"))
    assert shards
    # second run: embeddings complete, so no model is built (embedder=None would build one)
    run_eval_external(**kw, out=out, repeats=2, min_pos=3)
    # a head trained on another backbone is refused
    other = labelled_head()
    other.body_model = "other"
    save_head(tmp_path / "head2", other)
    with pytest.raises(ValueError, match="trained on other"):
        run_eval_external(
            "market1501", tmp_path / "data", tmp_path / "head2", tmp_path / "out2",
            embedder=FakeBody(), workers=0,
        )  # fmt: skip


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    persons, t, _ = fake_split(n_ids=80, per_id=5)
    j = s.INDEX["female"]
    p = np.full(t.shape, 0.5, np.float32)
    p[:, j] = np.clip(0.5 + (t[:, j] - 0.5) * 0.6 + rng.normal(0, 0.4, len(t)), 0, 1)
    groups = group_by_person(persons)
    used = np.array([j])
    ci = bootstrap_ma(p, t, groups, used, n_boot=200)
    ma = float(
        np.nanmean(((p[:, j] > 0.5)[t[:, j] == 1].mean(), (p[:, j] <= 0.5)[t[:, j] == 0].mean()))
    )
    assert ci["lo"] < ma < ci["hi"] and ci["hi"] - ci["lo"] < 0.2
    oracle = fake_split(n_ids=80, per_id=5)[2]
    exact = bootstrap_ma(oracle, t, groups, used, n_boot=50)
    assert exact["lo"] == exact["hi"] == 1.0
    res = evaluate_external(labelled_head(), p, t, persons, min_pos=5, repeats=2, k=4)
    assert res["image"]["ma_ci95_identities"]["lo"] < res["image"]["ma"]
    assert res["identity"]["ma_ci95_identities"]["hi"] > res["identity"]["summary"]["ma"]["mean"]
