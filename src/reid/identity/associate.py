"""Online tracklet -> identity assignment (the :class:`IdentityEngine`).

Implements, at the **tracklet level**, Casao et al. 2023 (Sensors 23(5):2662): self-adaptive
gallery (``gallery.py``, Sec. 3.1-3.2), softmax-ratio open-set decision (Sec. 3.3) and Unknown
Data Manager (``udm.py``, Sec. 3.3), plus the project's own face evidence and constraints
(``constraints.py``). Flow per finished tracklet (PLAN section 3.3):

1. descriptor = quality-weighted mean of the tracklet's sample embeddings (and of its best faces);
2. **face first, with accumulated evidence**: best gid over ``face_tracklet`` must reach
   ``face_match``, lead the next *different-looking* gid by ``face_margin`` and be supported by at
   least ``face_min_obs`` of the tracklet's own faces. Cannot-linked gids are excluded;
3. **body test** (Casao): softmax over quality-weighted centroids with temperature ``1/v``,
   ``p1/p2 >= tau`` and ``cos >= min_cos``, on the tracklet descriptor, and (optionally) a vote
   of the individual samples; cannot-linked gids are excluded;
4. **verify-before-update**: samples enter the gallery only after a confident assignment, through
   Casao's diversity gate and budget drop (never raw frames). A face-assigned tracklet whose body
   confidently contradicts the face gid is *not* inserted and records a must-link instead;
5. unmatched tracklets are pooled in the Unknown Data Manager (``end_camera`` / every
   ``udm_every`` tracklets); during bootstrap (< ``bootstrap_q`` identities) a well-sampled
   unmatched tracklet mints directly (Casao's initialisation stage, joined above cos 0.9);
   ``end_run`` flushes the pool (nobody vanishes, v1 B8), runs consolidation and rebuilds the
   prototypes.

Deviations from the paper: decisions per tracklet not per frame (v1 W2), temperature in config,
``min_cos`` floor from v1, sample vote, face evidence and constraints (not in the paper); see the
sub-module docstrings for the rest. State is persisted (DB, vectors) and reloaded on init, so runs
accumulate across sessions and a reload yields identical decisions.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from reid.config import CameraSet, Config
from reid.identity.consolidate import ConsolidationResult, Merge, consolidate
from reid.identity.constraints import (
    Span,
    blocked_gids,
    face_ambiguous,
    violates,
)
from reid.identity.gallery import BodyDecision, l2n, softmax_ratio_test, weighted_mean
from reid.identity.prototypes import rebuild_prototypes
from reid.identity.store import IdentityStore
from reid.identity.udm import PoolItem, UdmResolution, UnknownDataManager
from reid.storage import db
from reid.storage.vectors import VectorStore
from reid.types import Tracklet

log = logging.getLogger(__name__)


@dataclass
class Decision:
    """Audit record of one assignment decision."""

    tracklet_id: int
    gid: int | None
    source: str  # 'face' | 'body' | 'udm' | 'flush' | 'bootstrap' | ''
    score: float | None = None
    note: str = ""


@dataclass
class RunIdentityReport:
    """Summary returned by :meth:`IdentityEngine.end_run`."""

    run_id: int
    n_tracklets: int = 0
    n_identified: int = 0
    n_unidentified: int = 0
    n_identities: int = 0  # identities in data_dir after the run
    n_new_identities: int = 0  # minted during this engine's life
    by_source: dict[str, int] = field(default_factory=dict)
    merges: list[Merge] = field(default_factory=list)
    must_links: list[tuple[int, int]] = field(default_factory=list)
    unidentified_tracklet_ids: list[int] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)


class IdentityEngine:
    """Assigns tracklets to global identities, persisting everything under ``data_dir``.

    ``cameras`` is optional (needed only for topology cannot-links).
    """

    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        vectors: VectorStore,
        cameras: CameraSet | None = None,
    ) -> None:
        self.cfg, self.conn, self.cameras = cfg, conn, cameras
        self.store = IdentityStore(cfg, conn, vectors, cameras)
        self.udm = UnknownDataManager(cfg.association)
        self.decisions: list[Decision] = []
        self.must_links: set[tuple[int, int]] = set()
        self._minted: set[int] = set()
        self._touched: set[int] = set()
        self._by_cam: dict[str, list[Span]] = {}
        self._span: dict[int, Span] = {}
        for s in self.store.spans():
            self._register(s)

    # ------------------------------------------------------------------ helpers
    def _register(self, s: Span) -> None:
        self._span[s.id] = s
        self._by_cam.setdefault(s.camera_id, []).append(s)

    def _others(self, span: Span) -> list[Span]:
        a = self.cfg.association
        out = list(self._by_cam.get(span.camera_id, []))
        if a.use_topology and self.cameras is not None:
            for cam, lst in self._by_cam.items():
                if cam != span.camera_id and self.cameras.min_transit(span.camera_id, cam):
                    out.extend(lst)
        return out

    def _blocked(self, span: Span, face: np.ndarray | None, n_face: int) -> set[int]:
        a = self.cfg.association
        use_face = face is not None and n_face >= a.face_min_obs
        return blocked_gids(
            span,
            self._others(span),
            a,
            self.cameras,
            face if use_face else None,
            self.store.gid_faces() if use_face else None,
        )

    def _violates(self, x: Span, y: Span) -> bool:
        return violates(x, y, self.cfg.association, self.cameras)

    def _set(
        self, tid: int, gid: int | None, score: float | None, source: str, reason: str
    ) -> None:
        self.store.assign([(tid, gid, score, source)], reason)
        self._span[tid].gid = gid
        if gid is not None:
            self._touched.add(gid)

    def _log(
        self, tid: int, gid: int | None, source: str, score: float | None, note: str = ""
    ) -> None:
        self.decisions.append(Decision(tid, gid, source, score, note))

    # ------------------------------------------------------------------ face step
    def _face_step(
        self, desc: np.ndarray | None, faces: np.ndarray, blocked: set[int]
    ) -> tuple[int | None, float]:
        a = self.cfg.association
        if desc is None or len(faces) < a.face_min_obs:
            return None, 0.0
        hits = self.store.face_tracklet.best_per_label(desc, exclude_labels=blocked)
        if not hits or hits[0][2] < a.face_match:
            return None, 0.0
        best, score = hits[0][0], hits[0][2]
        cents = self.store.gid_faces()
        sims = {g: s for g, _k, s in hits}
        pair = {
            (min(g, best), max(g, best)): float(cents[g] @ cents[best])
            for g in sims
            if g != best and g in cents and best in cents
        }
        if face_ambiguous(best, sims, pair, a):
            return None, 0.0
        support = int(np.sum(faces @ cents[best] >= a.face_match)) if best in cents else 0
        return (best, score) if support >= a.face_min_obs else (None, 0.0)

    # ------------------------------------------------------------------ body step
    def _body_step(self, desc: np.ndarray, embs: np.ndarray, blocked: set[int]) -> BodyDecision:
        a = self.cfg.association
        gids, cents = self.store.bank.centroids()
        if not gids:
            return BodyDecision(None)
        dec = softmax_ratio_test(
            desc, gids, cents, blocked=blocked, temp=a.softmax_temp, tau=a.tau, min_cos=a.min_cos
        )
        if dec.accepted and a.sample_vote > 0 and len(embs) > 1:
            keep = [i for i, g in enumerate(gids) if g not in blocked]
            top = np.argmax(embs @ cents[keep].T, axis=1)
            frac = float(np.mean([gids[keep[i]] == dec.gid for i in top]))
            if frac < a.sample_vote:
                dec.accepted = False
        return dec

    # ------------------------------------------------------------------ public API
    def on_tracklet(self, t: Tracklet) -> int | None:
        """Decide the identity of a finished tracklet; returns the gid or None (pooled)."""
        if t.id is None:
            raise ValueError("tracklet must be inserted in the DB first (t.id is None)")
        a, st = self.cfg.association, self.store
        if db.get_samples(self.conn, tracklet_id=t.id):  # already processed (idempotent)
            return db.get_tracklet(self.conn, t.id).gid  # type: ignore[union-attr]
        span = Span(t.id, t.camera_id, t.start_ts, t.end_ts, None, t.run_id)
        self._register(span)
        smp = [s for s in t.samples if s.embedding is not None]
        if len(smp) < max(1, self.cfg.tracklets.min_samples):
            self._log(t.id, None, "", None, "too few samples")
            return None
        embs = l2n(np.stack([s.embedding for s in smp]))
        qual = np.array([s.quality for s in smp], dtype=np.float32)
        desc = (
            l2n(t.body_embedding.reshape(1, -1))[0]
            if t.body_embedding is not None
            else weighted_mean(embs, qual)
        )
        fobs = sorted((f for f in t.faces if f.embedding is not None), key=lambda f: -f.quality)[
            : self.cfg.face.max_faces
        ]
        fembs = l2n(np.stack([f.embedding for f in fobs])) if fobs else np.zeros((0, 0), np.float32)
        fdesc = None
        if fobs:
            fdesc = (
                l2n(t.face_embedding.reshape(1, -1))[0]
                if t.face_embedding is not None
                else weighted_mean(fembs, np.array([f.quality for f in fobs], np.float32))
            )
        sids = db.add_samples(self.conn, t.id, [(s.frame, s.quality) for s in smp])
        st.body_samples.add(sids, embs)
        st.body_tracklet.add([t.id], desc[None])
        if fdesc is not None and len(fobs) >= a.face_min_obs:  # accumulated evidence only
            st.face_tracklet.add([t.id], fdesc[None])

        blocked = self._blocked(span, fdesc, len(fobs))
        fgid, fscore = self._face_step(fdesc, fembs, blocked)
        body = self._body_step(desc, embs, blocked)

        gid: int | None
        if fgid is not None:
            gid, source, score, insert = fgid, "face", fscore, True
            if body.accepted and body.gid != fgid:
                insert = False  # verify-before-update: body disagrees, keep the gallery clean
                self.must_links.add((min(fgid, body.gid), max(fgid, body.gid)))  # type: ignore[type-var]
        elif body.accepted:
            gid, source, score, insert = body.gid, "body", body.cos, True
        elif len(st.bank.gids()) < a.bootstrap_q and len(smp) >= a.bootstrap_min_samples:
            if body.gid is not None and body.cos >= a.bootstrap_join_cos:
                gid, source, score = body.gid, "bootstrap", body.cos
            else:
                gid, source, score = st.mint(), "bootstrap", None
                self._minted.add(gid)
            insert = True
        else:
            self.udm.add(PoolItem(span, desc, embs, qual, sids, fdesc, len(fobs)))
            self._log(t.id, None, "", body.cos if body.gid is not None else None, "pooled")
            if len(self.udm) % a.udm_every == 0:
                self._run_udm(flush=False)
            return db.get_tracklet(self.conn, t.id).gid  # type: ignore[union-attr]

        assert gid is not None
        self._set(t.id, gid, score, "face" if source == "face" else "body", f"online:{source}")
        t.gid, t.assign_score, t.assign_source = gid, score, "face" if source == "face" else "body"
        if insert:
            st.add_gallery_samples(gid, embs, qual, sids)
        self._log(t.id, gid, source, score, "" if insert else "not inserted: body contradicts")
        return gid

    def _run_udm(self, flush: bool) -> list[UdmResolution]:
        st = self.store

        def blocked_for(p: PoolItem) -> set[int]:
            return self._blocked(p.span, p.face, p.n_face_obs)

        res = self.udm.process(st.bank, self._violates, blocked_for, flush=flush)
        for r in res:
            gid = r.gid
            blocked: set[int] = set()
            for p in r.items:
                blocked |= blocked_for(p)  # re-check: earlier resolutions may have changed spans
            if gid is not None and gid in blocked:
                gid = None
            if gid is None:
                gid = st.mint()
                self._minted.add(gid)
            embs = np.concatenate([p.embs for p in r.items])
            w = np.concatenate([p.weights for p in r.items])
            sids = [i for p in r.items for i in p.sample_ids]
            for p in r.items:
                self._set(
                    p.span.id,
                    gid,
                    1.0 - r.distance if np.isfinite(r.distance) else None,
                    "body",
                    f"udm:{r.reason}",
                )
                self._log(p.span.id, gid, r.reason, None, f"cluster of {len(r.items)}")
            st.add_gallery_samples(gid, embs, w, sids)
        return res

    def end_camera(self, camera_id: str) -> None:
        """Run the Unknown Data Manager on the pool and persist the vectors."""
        log.info("end_camera %s: pool=%d", camera_id, len(self.udm))
        self._run_udm(flush=False)
        self.store.save()

    def end_run(self, run_id: int) -> RunIdentityReport:
        """Flush the pool, consolidate identities, rebuild prototypes, persist, report."""
        st = self.store
        self._run_udm(flush=True)
        cons = ConsolidationResult()
        if self.cfg.consolidation.enabled:
            restrict = None
            if not self.cfg.consolidation.include_existing:
                restrict = {t.gid for t in db.get_tracklets(self.conn, run_id=run_id) if t.gid}
            cons = consolidate(st, restrict)
            for m in cons.merges:
                self._touched.add(m.kept)
                self._touched.discard(m.absorbed)
                self._minted.discard(m.absorbed)
            self.must_links |= {(a, b) for a, b in cons.must_links}
        live = {
            r["gid"]
            for r in self.conn.execute("SELECT DISTINCT gid FROM tracklets WHERE gid IS NOT NULL")
        }
        have = {int(x) for x in st.body_proto.labels if x != -1}
        rebuild_prototypes(
            self.conn, st.vectors, self.cfg, sorted((self._touched | (live - have)) & live)
        )
        db.refresh_identity_stats(self.conn)
        st.save()
        self._touched.clear()
        tl = db.get_tracklets(self.conn, run_id=run_id)
        by_source: dict[str, int] = {}
        for t in tl:
            if t.gid is not None:
                by_source[t.assign_source] = by_source.get(t.assign_source, 0) + 1
        un = [t.id for t in tl if t.gid is None]
        return RunIdentityReport(
            run_id=run_id,
            n_tracklets=len(tl),
            n_identified=len(tl) - len(un),
            n_unidentified=len(un),
            n_identities=len(db.list_identities(self.conn)),
            n_new_identities=len(self._minted),
            by_source=by_source,
            merges=cons.merges,
            must_links=sorted(self.must_links),
            unidentified_tracklet_ids=[i for i in un if i is not None],
            decisions=list(self.decisions),
        )
