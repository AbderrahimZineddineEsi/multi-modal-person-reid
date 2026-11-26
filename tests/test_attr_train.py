"""Head training on synthetic embeddings with planted rules and masked labels."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reid.attributes import store, train  # noqa: E402
from reid.attributes.head import load_head, predict  # noqa: E402
from reid.attributes.schema import INDEX, N_LABELS  # noqa: E402
from reid.eval.par_metrics import mean_accuracy  # noqa: E402

DIM = 32
RULES = {"hat": 0, "scarf": 1, "jacket": 2, "tshirt": 3}  # label -> embedding dim it follows


def make_split(emb_dir, ds, split, n, seed, flips=True, wrong_test=False):
    """Write a finished split whose labels follow ``x[:, dim] > 0`` (+ masks, + a rare label)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, DIM)).astype(np.float32)
    t = np.full((n, N_LABELS), -1, np.int8)
    for name, dim in RULES.items():
        if ds == "pa100k" and name in ("scarf", "jacket"):
            continue  # one dataset does not annotate these (masked)
        t[:, INDEX[name]] = (x[:, dim] > 0).astype(np.int8)
    t[:3, INDEX["stroller"]] = 1  # fewer than 10 positives: never trained
    t[3:, INDEX["stroller"]] = 0
    if wrong_test:
        t[:, INDEX["hat"]] = 1 - t[:, INDEX["hat"]]
    sdir = store.split_dir(emb_dir, ds, split)
    allx = np.concatenate([x, x + 0.01]) if flips else x
    index = {"images": [f"{i}.png" for i in range(n)], "model": "fake", "neck_feat": "after",
             "shard_size": 50, "with_flip": flips, "dim": DIM, "complete": True}  # fmt: skip
    store.prepare_split(sdir, index, t, [])
    for k, (a, b) in enumerate(store.shard_bounds(len(allx), 50)):
        store.write_shard(sdir, k, allx[a:b])
    return x, t


def make_all(emb_dir, with_test=True, wrong_test=False):
    for i, ds in enumerate(("pa100k", "peta")):
        make_split(emb_dir, ds, "train", 400, 10 + i)
        make_split(emb_dir, ds, "val", 150, 20 + i)
        if with_test:
            make_split(emb_dir, ds, "test", 150, 30 + i, flips=False, wrong_test=wrong_test)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


CFG = train.TrainConfig(hidden=32, epochs=25, patience=6, batch=128, lr=5e-3)


def test_training_learns_planted_rules(tmp_path):
    make_all(tmp_path)
    path = train.run_train(tmp_path, tmp_path / "head", CFG)
    head = load_head(path)
    assert head.hparams["in_dim"] == DIM and head.body_model == "fake"
    cols = [INDEX[n] for n in RULES]
    assert head.trained_labels[cols].all()
    assert not head.trained_labels[INDEX["stroller"]]  # < 10 positives
    assert head.trained_labels.sum() == len(RULES)
    x, t = make_split(tmp_path / "other", "peta", "test", 300, 99, flips=False)
    ma = mean_accuracy(predict(head, x), t, 0.5, cols)
    assert ma > 0.93
    assert ((head.thresholds > 0) & (head.thresholds < 1)).all()
    assert len(head.train_log) >= 1 and "val_ma_mean" in head.train_log[0]


def test_masked_labels_do_not_influence_loss():
    torch.manual_seed(0)
    logits = torch.randn(16, N_LABELS, requires_grad=True)
    t = torch.full((16, N_LABELS), -1, dtype=torch.int64)
    t[:, 3] = torch.randint(0, 2, (16,))
    t[:8, 7] = torch.randint(0, 2, (8,))
    pw, nw = torch.full((N_LABELS,), 1.5), torch.full((N_LABELS,), 1.1)
    on = torch.ones(N_LABELS, dtype=torch.bool)
    loss = train.masked_bce(logits, t, pw, nw, on)
    loss.backward()
    known = t != -1
    assert (logits.grad[~known] == 0).all() and (logits.grad[known] != 0).any()
    other = logits.detach().clone()
    other[~known] += 5.0  # predictions on unknown targets change nothing
    assert torch.allclose(train.masked_bce(other, t, pw, nw, on), loss)
    off = on.clone()
    off[7] = False  # an untrained label is masked as well
    assert not torch.allclose(train.masked_bce(logits, t, pw, nw, off), loss)
    g = logits.detach().clone().requires_grad_()
    train.masked_bce(g, t, pw, nw, off).backward()
    assert (g.grad[:, 7] == 0).all()


def test_label_weights():
    t = np.array([[1, -1], [0, -1], [0, -1], [0, -1]], np.int8)
    pw, nw = train.label_weights(t)
    assert pw[0] == pytest.approx(np.exp(0.75)) and nw[0] == pytest.approx(np.exp(0.25))
    assert pw[1] == pytest.approx(np.e) and nw[1] == pytest.approx(1.0)  # nothing known


def test_selection_uses_validation_only(tmp_path, monkeypatch):
    make_all(tmp_path / "a")
    make_all(tmp_path / "b", wrong_test=True)  # test labels flipped: must not matter
    calls = []
    real = store.load_split

    def spy(emb, ds, split, include_flips=True):
        calls.append(split)
        return real(emb, ds, split, include_flips)

    monkeypatch.setattr(train.store, "load_split", spy)
    h1 = train.train_head(train.load_train_data(tmp_path / "a"), CFG)
    h2 = train.train_head(train.load_train_data(tmp_path / "b"), CFG)
    assert "test" not in calls
    np.testing.assert_allclose(h1.W1, h2.W1, atol=1e-5)
    assert h1.hparams["best_val_ma"] == pytest.approx(h2.hparams["best_val_ma"])


def test_early_stopping_and_peta_weight(tmp_path):
    make_all(tmp_path, with_test=False)
    cfg = train.TrainConfig(hidden=16, epochs=40, patience=2, batch=128, lr=5e-3, peta_weight=2.0)
    head = train.train_head(train.load_train_data(tmp_path), cfg)
    assert len(head.train_log) < 40  # stopped early on validation
    assert head.hparams["peta_weight"] == 2.0


def test_tune_thresholds_maximises_balanced_accuracy():
    p = np.array([[0.9], [0.8], [0.35], [0.3], [0.2], [0.1]])
    y = np.array([[1], [1], [1], [0], [0], [0]], np.int8)
    thr = train.tune_thresholds(p, y, np.array([True]))
    assert ((p[:, 0] > thr[0]) == (y[:, 0] == 1)).all()  # separates 0.35 from 0.3
    assert 0.3 <= thr[0] <= 0.35
    assert train.tune_thresholds(p, y, np.array([False]))[0] == 0.5  # untrained label
