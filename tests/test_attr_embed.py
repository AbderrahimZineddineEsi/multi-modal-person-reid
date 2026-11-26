"""Embedding with a fake body model: shards, resume after a crash, flips only for train/val."""

import cv2
import numpy as np
import pytest

from reid.attributes import embed, store
from reid.attributes.datasets import ParSplit
from reid.attributes.schema import N_LABELS

torch = pytest.importorskip("torch")

H, W = 16, 8


class FakeBody:
    """Deterministic stand-in for BodyReID: left/right half means (flip-sensitive), L2-normed."""

    name, neck_feat, dim, h, w = "fake", "after", 8, H, W

    def __init__(self, crash_after: int | None = None) -> None:
        self.calls = 0
        self.items = 0
        self.crash_after = crash_after

    def embed(self, crops):
        self.calls += 1
        if self.crash_after is not None and self.calls > self.crash_after:
            raise RuntimeError("simulated crash")
        self.items += len(crops)
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            c = np.asarray(c, np.float32)
            half = c.shape[1] // 2
            out[i, :3] = c[:, :half].reshape(-1, 3).mean(0)
            out[i, 3:6] = c[:, half:].reshape(-1, 3).mean(0)
            out[i, 6:] = 1.0
            out[i] /= np.linalg.norm(out[i])
        return out


def make_images(root, n, name="img"):
    rng = np.random.default_rng(0)
    rel = []
    for i in range(n):
        img = rng.integers(0, 255, (20 + i, 11, 3)).astype(np.uint8)  # sizes differ, not 16x8
        cv2.imwrite(str(root / f"{name}{i}.png"), img)
        rel.append(f"{name}{i}.png")
    return rel


def split(root, ds, sp, n, bad=False):
    rel = make_images(root, n, f"{ds}{sp}")
    if bad:
        (root / rel[1]).write_bytes(b"not an image")
    t = np.zeros((n, N_LABELS), np.int8)
    t[:, 0] = np.arange(n) % 2
    persons = [f"s/{i // 2}" for i in range(n)] if ds == "peta" else []
    return ParSplit(ds, sp, root, rel, t, persons)


def test_embeddings_flips_and_shards(tmp_path):
    img, out = tmp_path / "img", tmp_path / "emb"
    img.mkdir()
    sets = {"peta": {s: split(img, "peta", s, 5) for s in ("train", "val", "test")}}
    fake = FakeBody()
    embed.embed_datasets(fake, sets, out, batch=2, workers=0, shard_size=4)
    assert fake.items == 10 + 10 + 5  # flips for train and val only
    tr = store.load_split(out, "peta", "train")
    te = store.load_split(out, "peta", "test")
    assert tr.x.shape == (10, 8) and tr.x.dtype == np.float16 and te.x.shape == (5, 8)
    assert tr.flip.tolist() == [False] * 5 + [True] * 5 and not te.flip.any()
    assert store.n_items(5, True) == 10 and len(list((out / "peta/train").glob("shard_*.npy"))) == 3
    assert tr.persons.tolist() == sets["peta"]["train"].persons
    assert (tr.row_targets == np.tile(tr.targets, (2, 1))).all()
    # a flipped row is the embedding of the mirrored resized image
    img0 = cv2.imread(str(img / sets["peta"]["train"].images[0]))
    crop = cv2.resize(cv2.flip(img0, 1), (W, H))
    np.testing.assert_allclose(tr.x[5], FakeBody().embed([crop])[0], atol=1e-3)
    assert not np.allclose(tr.x[0], tr.x[5], atol=1e-3)  # mirror differs
    idx = store.read_index(out / "peta/train")
    assert idx["complete"] and idx["model"] == "fake" and idx["fp16"] and idx["with_flip"]
    assert idx["neck_feat"] == "after" and "reid_version" in idx and idx["shard_size"] == 4
    assert len(store.load_split(out, "peta", "train", include_flips=False).x) == 5


def test_resume_after_crash(tmp_path):
    img, out = tmp_path / "img", tmp_path / "emb"
    img.mkdir()
    sp = split(img, "pa100k", "train", 6)
    with pytest.raises(RuntimeError):
        embed.embed_datasets(FakeBody(crash_after=2), {"pa100k": {"train": sp}}, out, batch=2,
                             workers=0, shard_size=4)  # fmt: skip
    sdir = out / "pa100k/train"
    assert len(list(sdir.glob("shard_*.npy"))) == 1  # shard 0 (4 items = 2 batches) survived
    assert not store.read_index(sdir)["complete"]
    with pytest.raises(FileNotFoundError):
        store.load_split(out, "pa100k", "train")
    fake = FakeBody()
    embed.embed_datasets(fake, {"pa100k": {"train": sp}}, out, batch=2, workers=0, shard_size=4)
    assert fake.items == 12 - 4  # only the missing shards were computed
    full = tmp_path / "full"
    embed.embed_datasets(
        FakeBody(), {"pa100k": {"train": sp}}, full, batch=2, workers=0, shard_size=4
    )
    np.testing.assert_array_equal(
        store.load_split(out, "pa100k", "train").x, store.load_split(full, "pa100k", "train").x
    )
    again = FakeBody()  # nothing left to do
    embed.embed_datasets(again, {"pa100k": {"train": sp}}, out, batch=2, workers=0, shard_size=4)
    assert again.items == 0


def test_changed_inputs_restart(tmp_path):
    img, out = tmp_path / "img", tmp_path / "emb"
    img.mkdir()
    sp = split(img, "pa100k", "val", 4)
    embed.embed_datasets(FakeBody(), {"pa100k": {"val": sp}}, out, batch=2, workers=0, shard_size=4)
    sp2 = split(img, "pa100k", "val", 4)
    sp2.images = sp2.images[::-1]  # another image list: the old shards must not be reused
    fake = FakeBody()
    embed.embed_datasets(fake, {"pa100k": {"val": sp2}}, out, batch=2, workers=0, shard_size=4)
    assert fake.items == 8


def test_unreadable_image_gives_zero_vector_and_workers(tmp_path):
    img, out = tmp_path / "img", tmp_path / "emb"
    img.mkdir()
    sp = split(img, "peta", "test", 4, bad=True)
    embed.embed_datasets(FakeBody(), {"peta": {"test": sp}}, out, batch=2, workers=2, shard_size=8)
    x = store.load_split(out, "peta", "test").x
    assert (
        np.linalg.norm(x[1].astype(np.float32)) == 0 and np.linalg.norm(x[0].astype(np.float32)) > 0
    )
    assert store.read_index(out / "peta/test")["failed"] == 1


def test_preresize_in_workers_is_exact():
    """BodyReID.embed resizes to its input size again: pre-resizing must not change its input."""
    from reid.models.body_reid import preprocess_crops

    rng = np.random.default_rng(1)
    raw = rng.integers(0, 255, (37, 19, 3)).astype(np.uint8)
    pre = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
    a = preprocess_crops([raw], H, W, (0.5,) * 3, (0.5,) * 3)
    b = preprocess_crops([pre], H, W, (0.5,) * 3, (0.5,) * 3)
    assert torch.equal(a, b)


def test_eta_log_line():
    t = [0.0]
    eta = embed.EtaLog(100, interval_s=0.0, clock=lambda: t[0])
    t[0] = 10.0
    eta.update(50, "peta/train")
    assert eta.last_line.startswith("ETA | peta/train | 50/100") and "5.0 img/s" in eta.last_line
    assert "~10 s left" in eta.last_line


def test_parts_then_final_pass_equal_single_run(tmp_path):
    img = tmp_path / "img"
    img.mkdir()
    sets = {"peta": {s: split(img, "peta", s, 7) for s in ("train", "test")}}
    one = tmp_path / "one"
    embed.embed_datasets(FakeBody(), sets, one, batch=2, workers=0, shard_size=3)
    par = tmp_path / "par"
    a, b = FakeBody(), FakeBody()
    embed.embed_datasets(a, sets, par, batch=2, workers=0, shard_size=3, part=(0, 2))
    assert not store.read_index(par / "peta/train")["complete"]  # half the shards only
    embed.embed_datasets(b, sets, par, batch=2, workers=0, shard_size=3, part=(1, 2))
    assert a.items + b.items == 14 + 7  # every item embedded exactly once over the two parts
    final = FakeBody()
    embed.embed_datasets(final, sets, par, batch=2, workers=0, shard_size=3)
    assert final.items == 0  # the final pass only verifies and marks complete
    for s in ("train", "test"):
        assert store.read_index(par / f"peta/{s}")["complete"]
        np.testing.assert_array_equal(
            store.load_split(par, "peta", s).x, store.load_split(one, "peta", s).x
        )
    assert embed.pending_items(sets["peta"]["train"], par, 3, (1, 2)) == 0
