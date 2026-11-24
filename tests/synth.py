"""Synthetic embeddings and tracklets for identity tests (clustered Gaussians on the sphere)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reid.config import Config, load_config
from reid.storage import db
from reid.storage.vectors import VectorStore
from reid.types import CameraInfo, FaceObs, Sample, Tracklet

D_BODY, D_FACE = 64, 32


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def centers(n: int, d: int = D_BODY, seed: int = 0, shared: float = 0.5) -> np.ndarray:
    """``n`` unit vectors that share a common component (pairwise cosine around 0.2-0.3)."""
    rng = np.random.default_rng(seed)
    common = unit(rng.normal(size=d))
    return unit(shared * common + rng.normal(size=(n, d)) / np.sqrt(d))


def jitter(c: np.ndarray, n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples around centre ``c`` (cos to the centre is about 1/sqrt(1+noise^2))."""
    d = c.shape[-1]
    return unit(c + noise * rng.normal(size=(n, d)) / np.sqrt(d))


def cfg_for(tmp_path: Path, **over) -> Config:
    base = {"paths.data_dir": str(tmp_path), "tracklets.min_samples": 2}
    base.update(over)
    return load_config(None, base)


@dataclass
class Env:
    cfg: Config
    conn: object
    vectors: VectorStore
    run_id: int


def make_env(tmp_path: Path, cams=("c1", "c2", "c3"), face_cams=("c1", "c2"), **over) -> Env:
    cfg = cfg_for(tmp_path, **over)
    conn = db.connect(tmp_path / "reid.db")
    for c in cams:
        db.upsert_camera(conn, CameraInfo(c, c, f"{c}.mp4", 10.0, 640, 480, 0.0, c in face_cams))
    run_id = db.create_run(conn, cfg.to_json())
    return Env(cfg, conn, VectorStore(tmp_path / "vectors"), run_id)


def make_tracklet(
    env: Env,
    cam: str,
    t0: float,
    t1: float,
    body_center: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10,
    noise: float = 0.3,
    face_center: np.ndarray | None = None,
    n_faces: int = 3,
    face_noise: float = 0.2,
    insert: bool = True,
) -> Tracklet:
    """A tracklet with synthetic samples (and faces); inserted into the DB (``t.id`` set)."""
    f0, f1 = int(t0 * 10), int(t1 * 10)
    t = Tracklet(cam, int(rng.integers(1_000_000)), f0, f1, t0, t1)
    embs = jitter(body_center, n_samples, noise, rng)
    q = rng.uniform(0.5, 0.95, n_samples)
    t.samples = [Sample(f0 + i, float(q[i]), embs[i].astype(np.float32)) for i in range(n_samples)]
    t.n_samples = n_samples
    if face_center is not None:
        fe = jitter(face_center, n_faces, face_noise, rng)
        t.faces = [
            FaceObs(f0 + i, (0, 0, 1, 1), 0.9, 0.8, fe[i].astype(np.float32))
            for i in range(n_faces)
        ]
        t.face_quality = 0.8
    if insert:
        db.insert_tracklets(env.conn, env.run_id, [t])
    return t
