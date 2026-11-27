"""Schema v3, annotating tracklets, head resolution, backfill, maintenance, export/import, CLI."""

import json
import logging

import numpy as np
import pytest

from reid import cli, find
from reid.attributes import annotate as an
from reid.attributes.head import predict, save_head
from reid.config import load_config
from reid.storage import attributes_db as adb
from reid.storage import db, files, maintenance, portable
from reid.storage.vectors import VectorStore
from tests.attr_synth import add_camera, add_tracklet, make_head
from tests.synth import make_env


def add_samples(conn, vectors, tid, embs, quals):
    sids = db.add_samples(conn, tid, [(i, q) for i, q in enumerate(quals)])
    vectors.get("body_samples").add(sids, embs)
    return sids


@pytest.fixture
def world(tmp_path):
    """Env with 3 tracklets: two with samples (different qualities) and one without."""
    env = make_env(tmp_path / "env", cams=("c1",))
    rng = np.random.default_rng(0)
    emb = {k: rng.normal(size=(n, 8)).astype(np.float32) for k, n in (("a", 4), ("b", 2))}
    ta = add_tracklet(env.conn, env.run_id, "c1", 0, 5, gid=1)
    tb = add_tracklet(env.conn, env.run_id, "c1", 6, 9, gid=2)
    tn = add_tracklet(env.conn, env.run_id, "c1", 10, 12)
    add_samples(env.conn, env.vectors, ta, emb["a"], [0.9, 0.8, 0.2, 0.1])
    add_samples(env.conn, env.vectors, tb, emb["b"], [0.5, 0.5])
    return env, emb, (ta, tb, tn)


def test_migration_v2_and_v1_to_v3(tmp_path):
    path = tmp_path / "old.db"
    c = db.connect(path)
    for version in (2, 1):
        c.executescript("DROP TABLE tracklet_attributes; DROP TABLE attribute_heads;")
        c.execute("UPDATE schema_version SET version=?", (version,))
        c.close()
        c = db.connect(path)
        assert db.schema_version(c) == 3 == db.SCHEMA_VERSION
        assert c.execute("SELECT COUNT(*) FROM tracklet_attributes").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM attribute_heads").fetchone()[0] == 0
    c.execute("UPDATE schema_version SET version=4")
    c.close()
    with pytest.raises(RuntimeError, match="migration"):
        db.connect(path)


def test_head_rows_are_deduplicated_by_content(conn):
    h = make_head()
    a, b = adb.upsert_head(conn, h), adb.upsert_head(conn, make_head())
    assert a.id == b.id and a.sha256 == adb.head_sha256(h)
    assert adb.upsert_head(conn, make_head(seed=1)).id != a.id
    rec = adb.get_head(conn, a.id)
    assert rec.labels == h.labels and rec.index["backpack"] == 1
    assert rec.thresholds.tolist() == h.thresholds.tolist() and rec.trained["best_val_ma"] == 0.75
    assert adb.head_by_sha(conn, a.sha256).id == a.id and adb.head_by_sha(conn, "x") is None


def test_annotate_weighted_mean_idempotent_force(world):
    env, emb, (ta, tb, tn) = world
    head = make_head()
    assert an.annotate(env.conn, env.vectors, head) == 2  # the tracklet without samples is skipped
    rec = adb.head_by_sha(env.conn, adb.head_sha256(head))
    ids, probs, n = adb.read_tracklet_attributes(env.conn, rec.id, len(head.labels))
    assert ids.tolist() == [ta, tb] and n.tolist() == [4, 2] and probs.dtype == np.float32
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    w = np.array([0.9, 0.8, 0.2, 0.1])
    ref = (predict(head, unit(emb["a"])) * (w / w.sum())[:, None]).sum(0)
    np.testing.assert_allclose(probs[0], ref, atol=1e-5)
    assert an.annotate(env.conn, env.vectors, head) == 0  # idempotent
    env.conn.execute("UPDATE tracklet_attributes SET n_samples=99")
    assert an.annotate(env.conn, env.vectors, head, [ta]) == 0
    assert an.annotate(env.conn, env.vectors, head, [ta], force=True) == 1
    assert adb.read_tracklet_attributes(env.conn, rec.id, 7)[2].tolist() == [4, 99]
    assert an.annotate(env.conn, env.vectors, head, force=True) == 2
    assert adb.count_annotated(env.conn) == adb.count_annotated(env.conn, rec.id) == 2


def test_annotate_wrong_dimension_is_skipped(world):
    env = world[0]
    assert an.annotate(env.conn, env.vectors, make_head(in_dim=5)) == 0


def test_second_head_keeps_both(world):
    env = world[0]
    an.annotate(env.conn, env.vectors, make_head())
    an.annotate(env.conn, env.vectors, make_head(seed=3))
    assert env.conn.execute("SELECT COUNT(*) FROM tracklet_attributes").fetchone()[0] == 4
    assert adb.count_annotated(env.conn) == 2
    assert adb.latest_annotated_head(env.conn).sha256 == adb.head_sha256(make_head(seed=3))


def _cfg(tmp_path, **over):
    base = {"paths.weights_dir": str(tmp_path / "w"), "body_reid.dim": 8}
    return load_config(None, base | over)


def test_resolve_head(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    assert an.resolve_head(cfg) is None  # nothing installed: silent
    path = an.default_head_path(cfg)
    assert path == tmp_path / "w" / "attributes" / "head.npz"
    save_head(path, make_head())
    assert an.resolve_head(cfg).in_dim == 8
    assert an.resolve_head(_cfg(tmp_path, **{"attributes.enabled": False})) is None
    own = tmp_path / "mine" / "head.npz"
    save_head(own, make_head(seed=2))
    got = an.resolve_head(_cfg(tmp_path, **{"attributes.head": str(own)}))
    assert adb.head_sha256(got) == adb.head_sha256(make_head(seed=2))
    lite = _cfg(tmp_path, **{"body_reid.model": "osnet_x1_0_msmt17", "body_reid.dim": 512})
    with caplog.at_level(logging.WARNING, logger="reid.attributes.annotate"):
        assert an.resolve_head(lite) is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and "osnet" in caplog.text
    assert an.load_configured_head(lite) is not None  # the app still wants to explain
    assert an.resolve_head(_cfg(tmp_path, **{"body_reid.neck_feat": "before"}), warn=False) is None


def _make_env_dir(root, preset="balanced", body_dim=8):
    files.ensure_dirs(root)
    (root / "env.json").write_text(json.dumps({"name": root.name}))
    over = {"body_reid.dim": body_dim}
    if preset == "lite":
        over["body_reid.model"] = "osnet_x1_0_msmt17"
    conn = db.connect(files.db_path(root))
    run_id = db.create_run(conn, load_config(None, over).to_json())
    add_camera(conn, "c1")
    tid = add_tracklet(conn, run_id, "c1", 0, 5, gid=1)
    vec = VectorStore(files.vectors_dir(root))
    add_samples(conn, vec, tid, np.random.default_rng(1).normal(size=(3, body_dim)), [0.5] * 3)
    vec.save()
    return root, conn


def test_backfill_env_and_workspace(tmp_path):
    ws = tmp_path / "ws"
    ok, c1 = _make_env_dir(ws / "ok")
    lite, c2 = _make_env_dir(ws / "lite", "lite", 8)
    c1.close(), c2.close()
    head = make_head()
    res = {r.env: r for r in an.backfill_workspace(ws, head)}
    assert res["ok"].state == "annotated" and res["ok"].annotated == res["ok"].total == 1
    assert res["lite"].state == "skipped" and "osnet" in res["lite"].reason
    again = an.backfill_env(ws / "ok", head)
    assert again.annotated == 0 and again.total == 1
    assert an.backfill_env(ws / "ok", head, force=True).annotated == 1
    conn = db.connect(files.db_path(ws / "lite"))
    assert adb.count_annotated(conn) == 0
    st = an.env_status(conn, ws / "lite", head)
    assert st.state == "incompatible" and "osnet" in st.reason
    assert an.env_status(conn, ws / "lite", None).state == "no_head"
    conn.close()
    conn = db.connect(files.db_path(ws / "ok"))
    assert an.env_status(conn, ws / "ok", None).state == "ready"
    conn.execute("DELETE FROM tracklet_attributes")
    assert an.env_status(conn, ws / "ok", head).state == "not_annotated"
    conn.close()


def test_maintenance_deletes_attribute_rows(world):
    env, _, (ta, tb, _) = world
    cfg = env.cfg
    an.annotate(env.conn, env.vectors, make_head())
    db.refresh_identity_stats(env.conn)  # creates the identity rows of gids 1 and 2
    maintenance.delete_identity(env.conn, env.vectors, cfg, 1)
    left = [r[0] for r in env.conn.execute("SELECT tracklet_id FROM tracklet_attributes")]
    assert left == [tb]
    maintenance.delete_run(env.conn, env.vectors, cfg, env.run_id)
    assert env.conn.execute("SELECT COUNT(*) FROM tracklet_attributes").fetchone()[0] == 0
    assert env.conn.execute("SELECT COUNT(*) FROM attribute_heads").fetchone()[0] == 1


def test_export_import_keeps_attributes(tmp_path):
    root, conn = _make_env_dir(tmp_path / "e1")
    an.annotate(conn, VectorStore(files.vectors_dir(root)), make_head())
    before = find.load_table(conn)
    conn.close()
    portable.export_env(root, tmp_path / "e.zip")
    r = portable.import_env(tmp_path / "e.zip", tmp_path / "ws", name="copy")
    conn = db.connect(files.db_path(r.path))
    after = find.load_table(conn)
    assert after.head.sha256 == before.head.sha256 and after.head.labels == before.head.labels
    np.testing.assert_array_equal(after.probs, before.probs)
    assert find.find(conn, find.FindQuery(require={"female": True})).total_tracklets == 1
    conn.close()


def test_cli_use_head_backfill_find(tmp_path, capsys):
    w = tmp_path / "w"
    save_head(tmp_path / "src" / "head.npz", make_head())
    base = ["--set", f"paths.weights_dir={w}", "--set", "body_reid.dim=8"]
    assert cli.main(["attributes", *base, "use-head", str(tmp_path / "src")]) == 0
    info = json.loads(capsys.readouterr().out.split("\nnext:")[0])
    assert info["labels"] == 7 and info["body_model"] == "clipreid_msmt17"
    assert (w / "attributes" / "head.npz").is_file()
    ws = tmp_path / "ws"
    ok, conn = _make_env_dir(ws / "ok")
    conn.close()
    assert cli.main(["attributes", *base, "backfill", "--workspace", str(ws)]) == 0
    assert "ok" in capsys.readouterr().out
    assert cli.main(["find", "--data-dir", str(ok), "--attr", "female", "--any",
                     "upper_red,upper_pink", "--attr", "backpack=no", *base]) == 0  # fmt: skip
    out = json.loads(capsys.readouterr().out)
    assert out["total_tracklets"] == 1 and out["group"] == "person"
    assert (
        out["results"][0]["best"]["tracklet_id"] and "female" in out["results"][0]["best"]["probs"]
    )
    assert cli.main(["find", "--data-dir", str(ok), "--group", "appearance", "--strict",
                     "--attr", "female", *base]) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out)["group"] == "appearance"
    assert cli.main(["find", "--data-dir", str(ok), "--attr", "flying", *base]) == 2
    empty = tmp_path / "empty"
    files.ensure_dirs(empty)
    assert cli.main(["attributes", *base, "backfill", "--data-dir", str(ok), "--force"]) == 0
    assert "+1 annotated" in capsys.readouterr().out
    assert cli.main(["find", "--data-dir", str(empty), "--attr", "female", *base]) == 1
