"""Glue between the SQLite database, the vector collections and the in-memory galleries.

Not a paper component: it keeps the three stores consistent (every gid change relabels the
vectors, every gallery change flips ``samples.in_gallery``) so that the identity state can be
reloaded exactly from ``data_dir`` (fixes v1 B5/B13/B14: nothing lives only in RAM).

Vector collections (keys, labels): ``body_samples`` (sample id, gid), ``body_tracklet``
(tracklet id, gid), ``face_tracklet`` (tracklet id, gid), ``body_proto`` (prototype key, gid).
Label -1 means "tracklet has no identity".
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable

import numpy as np

from reid.config import CameraSet, Config
from reid.identity.constraints import Span
from reid.identity.gallery import GalleryBank, l2n
from reid.storage import db
from reid.storage.vectors import NO_LABEL, VectorCollection, VectorStore

log = logging.getLogger(__name__)

PROTO_STRIDE = 1000  # prototype key = gid * PROTO_STRIDE + index


class IdentityStore:
    """Identity state: galleries in memory, everything else in db/vectors."""

    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        vectors: VectorStore,
        cameras: CameraSet | None = None,
    ) -> None:
        self.cfg, self.conn, self.vectors, self.cameras = cfg, conn, vectors, cameras
        self.bank = self._new_bank()
        self.load()

    # ---- collections
    @property
    def body_samples(self) -> VectorCollection:
        return self.vectors.get("body_samples")

    @property
    def body_tracklet(self) -> VectorCollection:
        return self.vectors.get("body_tracklet")

    @property
    def face_tracklet(self) -> VectorCollection:
        return self.vectors.get("face_tracklet")

    @property
    def body_proto(self) -> VectorCollection:
        return self.vectors.get("body_proto")

    def _new_bank(self) -> GalleryBank:
        a = self.cfg.association
        return GalleryBank(a.gallery_budget, a.gallery_gamma, a.softmax_temp, a.diversity_ref)

    def load(self) -> None:
        """(Re)build the galleries from ``samples.in_gallery`` and ``body_samples``."""
        self.bank = self._new_bank()
        rows = [r for r in db.get_samples(self.conn, in_gallery=True) if r["gid"] is not None]
        by_gid: dict[int, list[sqlite3.Row]] = {}
        for r in rows:
            by_gid.setdefault(r["gid"], []).append(r)
        samples = self.body_samples
        for gid, rs in by_gid.items():
            rs = [r for r in rs if r["id"] in samples]  # vectors may lag the DB after a crash
            if not rs:
                continue
            ids = [r["id"] for r in rs]
            self.bank.restore(
                gid,
                self.body_samples.vectors(ids),
                np.array([r["quality"] for r in rs], dtype=np.float32),
                ids,
            )

    def save(self) -> None:
        """Persist the vector collections."""
        self.vectors.save()

    # ---- queries
    def spans(self) -> list[Span]:
        """A span for every tracklet in the database."""
        return [
            Span(t.id, t.camera_id, t.start_ts, t.end_ts, t.gid, t.run_id)  # type: ignore[arg-type]
            for t in db.get_tracklets(self.conn)
        ]

    def gid_faces(self) -> dict[int, np.ndarray]:
        """Mean face descriptor of every identity that has face tracklets."""
        col = self.face_tracklet
        if len(col) == 0:
            return {}
        labs, vecs = col.labels, col.vectors()
        out: dict[int, np.ndarray] = {}
        for g in np.unique(labs[labs != NO_LABEL]):
            out[int(g)] = l2n(vecs[labs == g].mean(0))
        return out

    # ---- mutation
    def mint(self) -> int:
        """Create a new identity row and return its gid."""
        return db.create_identity(self.conn)

    def relabel_tracklet_vectors(self, tracklet_id: int, gid: int | None) -> None:
        """Set the vector labels of one tracklet (samples, descriptors) to ``gid``."""
        lab = NO_LABEL if gid is None else gid
        keys = [r["id"] for r in db.get_samples(self.conn, tracklet_id=tracklet_id)]
        if keys:
            self.body_samples.relabel(keys=keys, label=lab)
        for col in (self.body_tracklet, self.face_tracklet):
            if tracklet_id in col:
                col.relabel({tracklet_id: lab})

    def assign(
        self,
        items: Iterable[tuple[int, int | None, float | None, str]],
        reason: str,
    ) -> int:
        """Write ``(tracklet_id, gid, score, source)`` to the DB (with history) and vectors."""
        items = list(items)
        n = db.set_gids(self.conn, items, reason=reason)
        for tid, gid, _s, _src in items:
            self.relabel_tracklet_vectors(tid, gid)
        return n

    def add_gallery_samples(
        self, gid: int, embs: np.ndarray, weights: np.ndarray, ids: list[int]
    ) -> None:
        """Gate-insert samples into ``gid``'s gallery and persist the membership changes."""
        added, removed = self.bank.insert_many(gid, embs, weights, ids)
        db.set_in_gallery(self.conn, added, True)
        db.set_in_gallery(self.conn, removed, False)

    def rebuild_gallery(self, gid: int) -> None:
        """Recompute a gallery from all samples of the identity's tracklets (gated)."""
        rows = db.get_samples(self.conn, gid=gid)
        db.set_in_gallery(self.conn, [r["id"] for r in rows if r["in_gallery"]], False)
        self.bank.remove(gid)
        if rows:
            ids = [r["id"] for r in rows]
            self.add_gallery_samples(
                gid,
                self.body_samples.vectors(ids),
                np.array([r["quality"] for r in rows], dtype=np.float32),
                ids,
            )

    def merge_gids(self, keep: int, absorb: int, reason: str) -> None:
        """Fold identity ``absorb`` into ``keep`` (history kept; galleries merged and pruned)."""
        if keep == absorb:
            return
        moved = [
            (t.id, keep, t.assign_score, "consolidate")
            for t in db.get_tracklets(self.conn, gid=absorb)
            if t.id is not None
        ]
        db.set_gids(self.conn, moved, reason=reason)
        for col in (self.body_samples, self.body_tracklet, self.face_tracklet):
            col.merge_labels(absorb, keep)
        _, dropped = self.bank.merge(keep, absorb)
        db.set_in_gallery(self.conn, dropped, False)
        db.remap_cannot_links(self.conn, absorb, keep)
        self.body_proto.remove_label(absorb)
        db.delete_identity(self.conn, absorb)
