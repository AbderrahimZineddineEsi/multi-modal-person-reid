"""Synthetic attribute heads and annotated tracklets (no models, no real embeddings)."""

from __future__ import annotations

import numpy as np

from reid.attributes.head import HeadParams
from reid.storage import attributes_db as adb
from reid.storage import db
from reid.types import CameraInfo, Tracklet

LABELS = ["female", "backpack", "upper_red", "upper_pink", "upper_blue", "child", "adult"]


def make_head(
    in_dim: int = 8, labels: list[str] | None = None, body_model: str = "clipreid_msmt17",
    neck_feat: str = "after", seed: int = 0, thr: float = 0.5,
) -> HeadParams:  # fmt: skip
    """A small deterministic head (identity BN, random weights) with a single threshold."""
    labels = labels or LABELS
    rng = np.random.default_rng(seed)
    hidden = 6
    return HeadParams(
        W1=rng.normal(size=(hidden, in_dim)).astype(np.float32),
        b1=np.zeros(hidden, np.float32),
        bn_mean=np.zeros(hidden, np.float32),
        bn_var=np.ones(hidden, np.float32),
        bn_weight=np.ones(hidden, np.float32),
        bn_bias=np.zeros(hidden, np.float32),
        W2=rng.normal(size=(len(labels), hidden)).astype(np.float32),
        b2=np.zeros(len(labels), np.float32),
        labels=list(labels),
        groups=["g"] * len(labels),
        thresholds=np.full(len(labels), thr, np.float32),
        trained_labels=np.ones(len(labels), bool),
        body_model=body_model,
        neck_feat=neck_feat,
        hparams={"best_val_ma": 0.75},
    )


def add_tracklet(
    conn, run_id: int, cam: str, t0: float, t1: float, gid: int | None = None
) -> int:  # fmt: skip
    """A bare tracklet row (no points, no samples); returns its id."""
    t = Tracklet(cam, 1, int(t0 * 10), int(t1 * 10), t0, t1, gid=gid, n_samples=3)
    (tid,) = db.insert_tracklets(conn, run_id, [t])
    return tid


def add_camera(conn, cam: str) -> None:
    db.upsert_camera(conn, CameraInfo(cam, cam, f"{cam}.mp4", 10.0, 640, 480))


def annotated_env(conn, rows: list[tuple[str, float, float, int | None, dict[str, float]]]):
    """Create tracklets + attributes. ``rows``: (camera, t0, t1, gid, {label: prob}); unlisted
    labels get 0.02. Returns ``(head_record, tracklet_ids)``."""
    head = make_head()
    rec = adb.upsert_head(conn, head)
    run_id = db.create_run(conn, "{}")
    ids, out = [], []
    for cam, t0, t1, gid, probs in rows:
        add_camera(conn, cam)
        tid = add_tracklet(conn, run_id, cam, t0, t1, gid)
        p = np.full(len(LABELS), 0.02, np.float32)
        for k, v in probs.items():
            p[LABELS.index(k)] = v
        out.append((tid, p, 4))
        ids.append(tid)
    adb.upsert_tracklet_attributes(conn, rec.id, out)
    return rec, ids
