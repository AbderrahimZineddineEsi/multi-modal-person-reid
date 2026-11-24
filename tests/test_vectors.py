import numpy as np
import pytest

from reid.storage.vectors import VectorCollection, VectorStore, faiss

BACKENDS = ["numpy"] + (["faiss"] if faiss is not None else [])


def _rand(n, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


@pytest.fixture(params=BACKENDS)
def col(request):
    return VectorCollection(backend=request.param)


def test_add_normalizes_and_query(col):
    v = _rand(50)
    col.add(np.arange(50), v * 7.0, labels=np.arange(50) % 5)
    assert len(col) == 50 and col.dim == 16
    np.testing.assert_allclose(np.linalg.norm(col.vectors(), axis=1), 1.0, rtol=1e-5)
    res = col.query(v[10] * 3, top_k=3)
    assert res[0][0] == 10 and res[0][1] == 0 and res[0][2] == pytest.approx(1.0, abs=1e-5)
    assert [r[2] for r in res] == sorted((r[2] for r in res), reverse=True)
    assert len(col.query(v[0], top_k=1000)) == 50
    assert all(r[1] != 0 for r in col.query(v[10], 5, exclude_labels={0}))
    assert col.query(np.ones(16), 3) and VectorCollection().query(np.ones(16)) == []


def test_best_per_label(col):
    v = _rand(30, seed=1)
    col.add(np.arange(30), v, labels=np.arange(30) % 3)
    col.add([100], v[:1], labels=[-1])  # unlabelled never reported
    res = col.best_per_label(v[4])
    assert [r[0] for r in res][0] == 4 % 3 and res[0][1] == 4
    assert sorted(r[0] for r in res) == [0, 1, 2]
    assert all(res[i][2] >= res[i + 1][2] for i in range(2))
    assert len(col.best_per_label(v[4], top_k=2)) == 2
    assert 4 % 3 not in [r[0] for r in col.best_per_label(v[4], exclude_labels={1})]


def test_replace_remove_relabel(col):
    v = _rand(10, seed=2)
    col.add(np.arange(10), v, labels=[0] * 10)
    col.add([3], v[7:8], labels=[9])  # replaces key 3
    assert len(col) == 10 and col.label_of(3) == 9
    assert col.query(v[7], 1)[0][2] == pytest.approx(1.0, abs=1e-5)
    assert col.remove([3, 4, 12345]) == 2 and len(col) == 8 and 3 not in col
    assert 4 not in {r[0] for r in col.query(v[4], 10)}
    col.relabel({0: 5, 1: 5})
    col.relabel(keys=[2], label=6)
    assert [col.label_of(k) for k in (0, 1, 2, 5)] == [5, 5, 6, 0]
    col.merge_labels(5, 6)
    assert sorted(col.keys_for_label(6).tolist()) == [0, 1, 2]
    assert col.remove_label(6) == 3 and len(col) == 5
    with pytest.raises(ValueError):
        col.add([1000], _rand(1, d=8))
    with pytest.raises(ValueError):
        col.add([1, 1], _rand(2))


def test_relabel_after_search_keeps_index_valid(col):
    v = _rand(20, seed=3)
    col.add(np.arange(20), v, labels=np.arange(20))
    col.query(v[0], 3)
    col.relabel({0: 99})
    assert col.query(v[0], 1)[0][:2] == (0, 99)


def test_persist_roundtrip(tmp_path):
    store = VectorStore(tmp_path / "vectors")
    v = _rand(25, seed=4)
    store.get("body_tracklet").add(np.arange(25) + 1000, v, labels=np.arange(25) % 4)
    store.get("face_tracklet", dim=16).add([1], v[:1], labels=[2])
    store.save()
    assert (tmp_path / "vectors" / "body_tracklet.npy").exists()
    assert (tmp_path / "vectors" / "body_tracklet.ids.npy").exists()
    assert not list((tmp_path / "vectors").glob("*.tmp"))

    store2 = VectorStore(tmp_path / "vectors")
    assert store2.names() == ["body_tracklet", "face_tracklet"]
    c = store2.get("body_tracklet")
    assert len(c) == 25 and not c.dirty
    assert c.query(v[7], 1)[0][:2] == (1007, 7 % 4)
    np.testing.assert_allclose(c.vectors(), store.get("body_tracklet").vectors())
    c.remove([1000])
    store2.save()
    assert len(VectorStore(tmp_path / "vectors").get("body_tracklet")) == 24
    store2.drop("face_tracklet")
    assert VectorStore(tmp_path / "vectors").names() == ["body_tracklet"]


def test_empty_collection_persist(tmp_path):
    store = VectorStore(tmp_path)
    store.get("e", dim=8).add([1], _rand(1, d=8))
    store.get("e").remove([1])
    store.save()
    c = VectorStore(tmp_path).get("e")
    assert len(c) == 0 and c.query(np.ones(8)) == []
    c.add([5], _rand(1, d=8))
    assert len(c) == 1
