"""Delete an identity or a run, wipe a data_dir: rows, vectors and files stay consistent."""

import numpy as np
import pytest

from reid.app import environments as envs
from reid.identity.associate import IdentityEngine
from reid.storage import db, files, maintenance
from reid.storage.vectors import VectorStore
from tests.synth import centers, make_env, make_tracklet


@pytest.fixture
def populated(tmp_path):
    """Two people over two runs, with crop files, and prototypes built."""
    rng = np.random.default_rng(0)
    c = centers(2, seed=5, shared=0.0)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    for i in range(4):
        t = make_tracklet(env, "c1", 10 * i, 10 * i + 5, c[i % 2], rng)
        eng.on_tracklet(t)
        for p in (files.crop_path(tmp_path, t.id), files.face_path(tmp_path, t.id)):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    eng.end_run(env.run_id)
    run2 = db.create_run(env.conn, env.cfg.to_json())
    env.run_id = run2
    t = make_tracklet(env, "c1", 100, 105, c[0], rng)
    eng.on_tracklet(t)
    eng.end_run(run2)
    env.vectors.save()
    return env


def _check_consistent(env):
    """Every vector label / key refers to something that still exists in the database."""
    vec = VectorStore(env.cfg.data_dir / "vectors")
    gids = {i.gid for i in db.list_identities(env.conn)}
    tids = {t.id for t in db.get_tracklets(env.conn)}
    sids = {r["id"] for r in db.get_samples(env.conn)}
    assert {int(k) for k in vec.get("body_samples").keys} <= sids
    for name in ("body_tracklet", "face_tracklet"):
        assert {int(k) for k in vec.get(name).keys} <= tids
    for name in ("body_samples", "body_tracklet", "face_tracklet", "body_proto"):
        assert {int(x) for x in vec.get(name).labels if x != -1} <= gids


def test_delete_identity_removes_tracklets_and_vectors(populated):
    env = populated
    gid = db.list_identities(env.conn)[0].gid
    before = len(db.get_tracklets(env.conn))
    n = len(db.get_tracklets(env.conn, gid=gid))
    out = maintenance.delete_identity(env.conn, env.vectors, env.cfg, gid)
    assert out["tracklets_deleted"] == n and n > 0
    assert db.get_identity(env.conn, gid) is None
    assert len(db.get_tracklets(env.conn)) == before - n
    assert gid not in set(env.vectors.get("body_proto").labels)
    _check_consistent(env)


def test_delete_identity_keep_tracklets(populated):
    env = populated
    gid = db.list_identities(env.conn)[0].gid
    ids = [t.id for t in db.get_tracklets(env.conn, gid=gid)]
    maintenance.delete_identity(env.conn, env.vectors, env.cfg, gid, keep_tracklets=True)
    assert all(db.get_tracklet(env.conn, i).gid is None for i in ids)
    assert gid not in set(env.vectors.get("body_samples").labels)
    _check_consistent(env)
    with pytest.raises(KeyError):
        maintenance.delete_identity(env.conn, env.vectors, env.cfg, 999)


def test_delete_run_drops_empty_identities(populated):
    env = populated
    d = env.cfg.data_dir
    first = db.list_runs(env.conn)[0]["id"]
    maintenance.delete_run(env.conn, env.vectors, env.cfg, env.run_id)
    assert [r["id"] for r in db.list_runs(env.conn)] == [first]
    assert all(t.run_id == first for t in db.get_tracklets(env.conn))
    assert all(i.n_tracklets > 0 for i in db.list_identities(env.conn))
    _check_consistent(env)
    out = maintenance.delete_run(env.conn, env.vectors, env.cfg, first)
    assert out["identities_dropped"] == 2 and db.list_identities(env.conn) == []
    assert len(env.vectors.get("body_samples")) == 0 and len(env.vectors.get("body_proto")) == 0
    assert not list((d / "crops").rglob("*.jpg")) and not list((d / "faces").rglob("*.jpg"))
    _check_consistent(env)


def test_wipe_keeps_env_marker(tmp_path):
    ws = tmp_path / "ws"
    path = envs.create_environment(ws, "demo", "first")
    (path / "reid.db").write_bytes(b"x")
    (path / "crops" / "a.jpg").write_bytes(b"x")
    maintenance.wipe_data_dir(path)
    assert not (path / "reid.db").exists() and not list((path / "crops").iterdir())
    assert maintenance.read_marker(path)["description"] == "first"
    assert [e.name for e in envs.list_environments(ws)] == ["demo"]


def test_environment_helpers(tmp_path):
    ws = tmp_path / "ws"
    (ws / "videos").mkdir(parents=True)  # not an environment
    envs.create_environment(ws, "a_1")
    with pytest.raises(ValueError):
        envs.create_environment(ws, "a_1")
    for bad in ("", "x y", "../up", "a/b", "é"):
        with pytest.raises(ValueError):
            envs.validate_name(bad)
    assert [e.name for e in envs.list_environments(ws)] == ["a_1"]
    assert envs.describe(ws / "a_1").stats == "0 cameras · 0 people · 0 runs"
    with pytest.raises(ValueError):
        envs.delete_environment(ws, "videos")
    envs.delete_environment(ws, "a_1")
    assert envs.list_environments(ws) == []
