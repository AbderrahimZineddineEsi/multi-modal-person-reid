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


def _spy_loss(monkeypatch):
    """Record the arguments of every ``masked_bce`` call made by ``train_head``."""
    calls = []
    real = train.masked_bce

    def spy(logits, t, pw, nw, trained, sample_w=None):
        calls.append((t.clone(), pw.clone(), nw.clone(), trained.clone()))
        return real(logits, t, pw, nw, trained, sample_w)

    monkeypatch.setattr(train, "masked_bce", spy)
    return calls


def test_defaults_reproduce_step1_training(tmp_path, monkeypatch):
    cfg = train.TrainConfig()
    assert (cfg.weighting, cfg.use_flips, cfg.train_view) == ("ratio", True, True)
    make_all(tmp_path, with_test=False)
    data = train.load_train_data(tmp_path)
    assert data.flip.sum() == len(data.x) // 2 and len(data.flip) == len(data.x)
    calls = _spy_loss(monkeypatch)
    one = train.TrainConfig(hidden=16, epochs=1, batch=4096)
    train.train_head(data, one)
    t, pw, nw, trained = calls[0]
    assert len(t) == len(data.x)  # flipped rows included
    exp_pw, exp_nw = train.label_weights(data.t)
    np.testing.assert_allclose(pw.numpy(), exp_pw)
    np.testing.assert_allclose(nw.numpy(), exp_nw)
    np.testing.assert_array_equal(trained.numpy(), data.train_pos >= one.min_pos)


def test_use_flips_false_drops_flipped_rows(tmp_path, monkeypatch):
    make_all(tmp_path, with_test=False)
    data = train.load_train_data(tmp_path)
    calls = _spy_loss(monkeypatch)
    head = train.train_head(
        data, train.TrainConfig(hidden=16, epochs=1, batch=4096, use_flips=False)
    )
    assert len(calls[0][0]) == int((~data.flip).sum()) == len(data.x) // 2
    assert head.hparams["use_flips"] is False


def test_train_view_false_masks_view_labels(tmp_path, monkeypatch):
    make_all(tmp_path, with_test=False)
    data = train.load_train_data(tmp_path)
    view = [INDEX[n] for n in ("view_front", "view_side", "view_back")]
    assert (train.VIEW_COLS == view).all()
    data.t[:, view[0]] = (data.x[:, 5] > 0).astype(np.int8)  # a view label with plenty of positives
    data.train_pos[view[0]] = 100
    for tv in (True, False):
        calls = _spy_loss(monkeypatch)
        head = train.train_head(data, train.TrainConfig(hidden=16, epochs=1, train_view=tv))
        assert bool(calls[0][3][view[0]]) is tv and bool(head.trained_labels[view[0]]) is tv
    # masked in the loss: the view logits get exactly zero gradient
    logits = torch.randn(8, N_LABELS, requires_grad=True)
    t = torch.randint(0, 2, (8, N_LABELS))
    on = torch.ones(N_LABELS, dtype=torch.bool)
    on[view] = False
    train.masked_bce(logits, t, torch.ones(N_LABELS), torch.ones(N_LABELS), on).backward()
    assert (logits.grad[:, view] == 0).all() and (logits.grad[:, 0] != 0).any()


def test_validation_score_excludes_view_labels(tmp_path):
    make_all(tmp_path, with_test=False)
    data = train.load_train_data(tmp_path)
    view = [INDEX[n] for n in ("view_front", "view_side", "view_back")]
    trained = np.zeros(N_LABELS, bool)
    trained[[INDEX["hat"], *view]] = True
    assert list(train.val_columns(trained)) == [INDEX["hat"]]
    assert data.val  # sanity


def test_weighting_none_is_unweighted_bce(tmp_path, monkeypatch):
    torch.manual_seed(1)
    logits = torch.randn(32, N_LABELS)
    t = torch.randint(-1, 2, (32, N_LABELS))
    ones = torch.ones(N_LABELS)
    on = torch.ones(N_LABELS, dtype=torch.bool)
    known = t != -1
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[known], (t[known] == 1).float()
    )
    assert torch.allclose(train.masked_bce(logits, t, ones, ones, on), ref, atol=1e-6)
    make_all(tmp_path, with_test=False)
    data = train.load_train_data(tmp_path)
    calls = _spy_loss(monkeypatch)
    train.train_head(data, train.TrainConfig(hidden=16, epochs=1, weighting="none"))
    assert (calls[0][1] == 1).all() and (calls[0][2] == 1).all()
    with pytest.raises(ValueError):
        train.train_head(data, train.TrainConfig(weighting="bogus"))


def test_on_epoch_callback_gets_validation_score_and_can_stop(tmp_path):
    make_all(tmp_path, with_test=False)
    data = train.load_train_data(tmp_path)
    seen = []
    head = train.train_head(
        data, train.TrainConfig(hidden=16, epochs=5, patience=9, batch=128),
        on_epoch=lambda e, s: seen.append((e, s)),
    )  # fmt: skip
    assert [e for e, _ in seen] == list(range(5)) == [r["epoch"] for r in head.train_log]
    assert [s for _, s in seen] == [r["val_ma_mean"] for r in head.train_log]

    class Stop(Exception):
        pass

    def boom(epoch, score):
        seen.append(epoch)
        if epoch == 2:
            raise Stop

    seen.clear()
    with pytest.raises(Stop):
        train.train_head(data, train.TrainConfig(hidden=16, epochs=9, batch=128), on_epoch=boom)
    assert seen == [0, 1, 2]
