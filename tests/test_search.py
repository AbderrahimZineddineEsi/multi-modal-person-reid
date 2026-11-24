import numpy as np
import pytest

from reid import search as S
from reid.identity.associate import IdentityEngine
from reid.storage import db
from reid.types import SearchStatus
from tests.synth import D_FACE, centers, jitter, make_env, make_tracklet, unit


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    c = centers(3, seed=21, shared=0.0)
    f = centers(3, d=D_FACE, seed=22, shared=0.0)
    env = make_env(tmp_path, **{"search.body_min_cos": 0.5})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    tl = []
    for p in range(3):
        for k, cam in enumerate(("c1", "c2")):
            tl.append(
                make_tracklet(
                    env,
                    cam,
                    100 * p + 40 * k,
                    100 * p + 40 * k + 10,
                    c[p],
                    rng,
                    face_center=f[p] if p < 2 else None,
                )
            )
            eng.on_tracklet(tl[-1])
    eng.end_run(env.run_id)
    gids = [db.get_tracklet(env.conn, tl[2 * p].id).gid for p in range(3)]
    return env, c, f, gids


def test_body_match_ambiguous_unknown(world):
    env, c, _, gids = world
    q = jitter(c[0], 1, 0.2, np.random.default_rng(1))[0]
    hit = S.search_body(q, env.conn, env.vectors, env.cfg)
    assert (
        hit.status == SearchStatus.MATCH and hit.gid == gids[0] and hit.ratio >= env.cfg.search.tau
    )
    assert [s.camera_id for s in hit.journey] == ["c1", "c2"]  # enriched with the journey
    assert hit.candidates[0][0] == gids[0] and hit.second_gid in gids[1:]
    # halfway between persons 0 and 1: top-2 scores nearly equal -> AMBIGUOUS, best guess kept
    hit = S.search_body(unit(c[0] + c[1]), env.conn, env.vectors, env.cfg)
    assert hit.status == SearchStatus.AMBIGUOUS and hit.gid in gids[:2] and hit.ratio < 1.5
    assert hit.journey  # journey of the best guess
    # a stranger: below the minimum cosine
    rng = np.random.default_rng(7)
    hit = S.search_body(unit(rng.normal(size=c.shape[1])), env.conn, env.vectors, env.cfg)
    assert hit.status == SearchStatus.UNKNOWN and hit.gid is None and not hit.journey


def test_body_search_falls_back_without_prototypes(world):
    env, c, _, gids = world
    env.vectors.drop("body_proto")
    hit = S.search_body(c[1], env.conn, env.vectors, env.cfg)
    assert hit.status == SearchStatus.MATCH and hit.gid == gids[1]


def test_face_statuses(world):
    env, _, f, gids = world
    hit = S.search_face(f[0], env.conn, env.vectors, env.cfg)
    assert hit.status == SearchStatus.MATCH and hit.gid == gids[0] and hit.modality == "face"
    # between the two enrolled faces: both clear the threshold, lead < margin
    hit = S.search_face(unit(f[0] + f[1]), env.conn, env.vectors, env.cfg)
    assert hit.status == SearchStatus.AMBIGUOUS
    stranger = unit(np.random.default_rng(3).normal(size=D_FACE))
    assert S.search_face(stranger, env.conn, env.vectors, env.cfg).status == SearchStatus.UNKNOWN


class FakeBody:
    def __init__(self, emb):
        self.emb = emb

    def embed(self, crops):
        return np.stack([self.emb] * len(crops))


class FakeFace:
    """Mimics FaceModel: ``analyze`` returns (box, kps, score, emb) tuples."""

    def __init__(self, emb):
        self.emb = emb

    def analyze(self, img):
        if self.emb is None:
            return []
        small = (np.array([0, 0, 5, 5.0]), np.zeros((5, 2)), 0.5, -self.emb)  # smaller face
        big = (np.array([0, 0, 20, 20.0]), np.zeros((5, 2)), 0.9, self.emb)
        return [small, big]

    def embed_aligned(self, chip):
        return self.emb


class FakeModels:
    """Stands in for reid.models.registry.Models: no weights needed."""

    def __init__(self, body, face):
        self.body, self.face = FakeBody(body), FakeFace(face)


def test_image_queries_use_models_and_no_face(world):
    env, c, f, gids = world
    img = np.zeros((64, 32, 3), np.uint8)
    m = FakeModels(c[2], None)
    assert (
        S.search_face(img, env.conn, env.vectors, env.cfg, models=m).status == SearchStatus.NO_FACE
    )
    hit = S.search_face(img, env.conn, env.vectors, env.cfg, models=m, is_crop=True)
    assert hit.status == SearchStatus.NO_FACE
    hit = S.search_body(img, env.conn, env.vectors, env.cfg, models=m)
    assert hit.status == SearchStatus.MATCH and hit.gid == gids[2]
    # the largest detected face is the one searched; a face-less model bundle gives NO_FACE
    m = FakeModels(c[2], f[1])
    hit = S.search_face(img, env.conn, env.vectors, env.cfg, models=m)
    assert hit.gid == gids[1] and hit.status == SearchStatus.MATCH
    m.face = None
    assert (
        S.search_face(img, env.conn, env.vectors, env.cfg, models=m).status == SearchStatus.NO_FACE
    )


def test_both_face_priority_and_body_fallback(world):
    env, c, f, gids = world
    img = np.zeros((64, 32, 3), np.uint8)
    # face says person 0, body says person 2: the face wins (modality tells which decided)
    m = FakeModels(c[2], f[0])
    hit = S.search_both(img, env.conn, env.vectors, env.cfg, models=m)
    assert hit.gid == gids[0] and hit.modality == "face" and hit.status == SearchStatus.MATCH
    # no face in the query: body fallback
    hit = S.search_both(img, env.conn, env.vectors, env.cfg, models=FakeModels(c[2], None))
    assert hit.gid == gids[2] and hit.modality == "body"
    # precomputed embeddings, unknown face -> body fallback
    stranger = unit(np.random.default_rng(3).normal(size=D_FACE))
    hit = S.search_both(
        None, env.conn, env.vectors, env.cfg, body_embedding=c[1], face_embedding=stranger
    )
    assert hit.gid == gids[1] and hit.modality == "body"


def test_cli_search(world, tmp_path, monkeypatch, capsys):
    import json

    import cv2

    from reid.cli import main

    env, c, f, gids = world
    env.vectors.save()
    env.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    path = tmp_path / "q.png"
    cv2.imwrite(str(path), np.zeros((64, 32, 3), np.uint8))
    monkeypatch.setattr(S, "_models", lambda models, cfg: FakeModels(c[0], f[0]))
    base = ["search", "--image", str(path), "--data-dir", env.cfg.paths.data_dir]
    base += ["--set", "search.body_min_cos=0.5"]
    for mode, modality in (("body", "body"), ("face", "face"), ("both", "face")):
        rc = main([*base, "--mode", mode])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0 and out["status"] == "MATCH" and out["gid"] == gids[0]
        assert out["modality"] == modality and out["journey"]
    assert main(["search", "--image", str(tmp_path / "missing.png"), "--data-dir", "x"]) == 1
