"""Cannot-link and must-link constraints between tracklets and identities.

Not in Casao et al. 2023 (single camera, frame level). Multi-camera extension, motivated by
v1 review B1/B3/B4 and docs/dev/pipeline_decisions.md row 8 and 7:

* **Cannot-link (hard)**
  - same camera and the two tracklets overlap in time by more than ``overlap_tolerance_s``
    (one person cannot be in two places of the same view);
  - face conflict: both sides have a face descriptor and their cosine is below
    ``face_conflict``;
  - topology (optional): two cameras with a configured ``min_transit_s`` > 0 and a time gap
    (``max(start) - min(end)``, negative if they overlap) shorter than that transit time.
* **Must-link (soft, resolved in consolidation)**: two identities whose face centroids are
  within ``face_match`` of each other with a margin: no third identity that is as close to one
  of them (within ``face_margin``) while being a different face (below ``face_match`` to the
  other). v1 merged on a single frame with no margin (B4); here a margin and accumulated
  evidence are required and the link never merges anything by itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from reid.config import AssociationCfg, CameraSet


@dataclass(slots=True)
class Span:
    """Where and when a tracklet exists."""

    id: int
    camera_id: str
    start_ts: float
    end_ts: float
    gid: int | None = None


def time_gap(a: Span, b: Span) -> float:
    """Seconds between two spans (negative when they overlap)."""
    return max(a.start_ts, b.start_ts) - min(a.end_ts, b.end_ts)


def violates(a: Span, b: Span, cfg: AssociationCfg, cameras: CameraSet | None = None) -> bool:
    """True if the two tracklets can never be the same person (time/topology rules)."""
    gap = time_gap(a, b)
    if a.camera_id == b.camera_id:
        return -gap > cfg.overlap_tolerance_s
    if cfg.use_topology and cameras is not None:
        mt = cameras.min_transit(a.camera_id, b.camera_id)
        if mt is not None and mt > 0 and gap < mt:
            return True
    return False


def face_conflict(fa: np.ndarray | None, fb: np.ndarray | None, thr: float) -> bool:
    """True if both face descriptors exist and are less similar than ``thr``."""
    return fa is not None and fb is not None and float(fa @ fb) < thr


def blocked_gids(
    span: Span,
    others: Iterable[Span],
    cfg: AssociationCfg,
    cameras: CameraSet | None = None,
    face: np.ndarray | None = None,
    gid_faces: dict[int, np.ndarray] | None = None,
) -> set[int]:
    """Gids this tracklet may not join: a time/topology violation with one of the gid's
    tracklets, or a face conflict with the gid's face centroid."""
    out: set[int] = set()
    for o in others:
        if (
            o.gid is not None
            and o.id != span.id
            and o.gid not in out
            and violates(span, o, cfg, cameras)
        ):
            out.add(o.gid)
    if face is not None and gid_faces:
        out |= {g for g, f in gid_faces.items() if face_conflict(face, f, cfg.face_conflict)}
    return out


def violation_matrix(
    spans: Sequence[Span], cfg: AssociationCfg, cameras: CameraSet | None = None
) -> np.ndarray:
    """Boolean (n, n) matrix of time/topology cannot-links between spans (vectorized)."""
    n = len(spans)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)
    cams = sorted({s.camera_id for s in spans})
    ci = {c: i for i, c in enumerate(cams)}
    cam = np.array([ci[s.camera_id] for s in spans])
    st = np.array([s.start_ts for s in spans])
    en = np.array([s.end_ts for s in spans])
    gap = np.maximum(st[:, None], st[None, :]) - np.minimum(en[:, None], en[None, :])
    same = cam[:, None] == cam[None, :]
    out = same & (-gap > cfg.overlap_tolerance_s)
    if cfg.use_topology and cameras is not None:
        mt = np.zeros((len(cams), len(cams)))
        for a in cams:
            for b in cams:
                v = cameras.min_transit(a, b) if a != b else None
                mt[ci[a], ci[b]] = v if v else 0.0
        need = mt[cam[:, None], cam[None, :]]
        out |= (~same) & (need > 0) & (gap < need)
    np.fill_diagonal(out, False)
    return out


def gid_cannot_links(
    spans: Sequence[Span],
    gids: Sequence[int],
    cfg: AssociationCfg,
    cameras: CameraSet | None = None,
    gid_faces: dict[int, np.ndarray] | None = None,
    use_faces: bool = True,
    persisted: Iterable[tuple[int, int]] = (),
) -> np.ndarray:
    """Boolean (G, G) cannot-link matrix between identities (aligned with ``gids``).

    Two identities cannot merge if any pair of their tracklets violates a time/topology rule,
    if their face centroids conflict, or if a cannot-link was persisted (e.g. manual split).
    """
    idx = {g: i for i, g in enumerate(gids)}
    out = np.zeros((len(gids), len(gids)), dtype=bool)
    sp = [s for s in spans if s.gid in idx]
    viol = violation_matrix(sp, cfg, cameras)
    gi = np.array([idx[s.gid] for s in sp], dtype=int)  # type: ignore[index]
    for i, j in zip(*np.nonzero(viol), strict=True):
        out[gi[i], gi[j]] = True
    if use_faces and gid_faces:
        fg = [g for g in gids if g in gid_faces]
        for a in range(len(fg)):
            for b in range(a + 1, len(fg)):
                if face_conflict(gid_faces[fg[a]], gid_faces[fg[b]], cfg.face_conflict):
                    out[idx[fg[a]], idx[fg[b]]] = out[idx[fg[b]], idx[fg[a]]] = True
    for a, b in persisted:
        if a in idx and b in idx:
            out[idx[a], idx[b]] = out[idx[b], idx[a]] = True
    return out | out.T


def face_ambiguous(
    best_gid: int,
    sims: dict[int, float],
    centroid_sim: dict[tuple[int, int], float] | None,
    cfg: AssociationCfg,
) -> bool:
    """True if another *different-looking* gid scores within ``face_margin`` of ``best_gid``.

    ``sims`` maps gid -> similarity to the query. A competitor that is itself face-similar to
    ``best_gid`` (>= ``face_match``) is the same person fragmented over two gids and does not
    make the match ambiguous.
    """
    s1 = sims[best_gid]
    for g, s in sims.items():
        if g == best_gid or s < s1 - cfg.face_margin:
            continue
        cs = (centroid_sim or {}).get((min(g, best_gid), max(g, best_gid)))
        if cs is None or cs < cfg.face_match:
            return True
    return False


def must_links(gid_faces: dict[int, np.ndarray], cfg: AssociationCfg) -> set[tuple[int, int]]:
    """Face must-links ``(a, b)``, ``a < b``, between identities (see module docstring)."""
    gids = sorted(gid_faces)
    if len(gids) < 2:
        return set()
    f = np.stack([gid_faces[g] for g in gids])
    s = f @ f.T
    np.fill_diagonal(s, -1.0)
    n = len(gids)

    def clear(a: int, b: int) -> bool:
        """No third gid competes with b for a while being a different face from b."""
        for c in range(n):
            if c in (a, b):
                continue
            if s[a, c] >= s[a, b] - cfg.face_margin and s[b, c] < cfg.face_match:
                return False
        return True

    out: set[tuple[int, int]] = set()
    for a in range(n):
        for b in range(a + 1, n):
            if s[a, b] >= cfg.face_match and clear(a, b) and clear(b, a):
                out.add((gids[a], gids[b]))
    return out
