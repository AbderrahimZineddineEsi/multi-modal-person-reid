"""TrackletBuilder: turns per-frame tracker output into finalized :class:`Tracklet` objects.

Implements steps 2-6 of docs/dev/PLAN.md section 3.1: every tracked box becomes a
``TrackPoint``; a subset of boxes is sampled (gates on size, edge truncation and keypoint
visibility), embedded in batches, and the best ``max_samples_per_tracklet`` are kept; on face
cameras the best face in the head region is kept; tracklets are finalized when their track has
been lost for ``track_lost_s`` and at the end of the video.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from reid.config import Config
from reid.models.face import face_quality
from reid.tracking.tracker import TrackRow
from reid.types import CameraInfo, FaceObs, Sample, Tracklet, TrackPoint

if TYPE_CHECKING:
    from reid.models.registry import Models

log = logging.getLogger(__name__)


def visibility_ratio(kps: np.ndarray | None, kp_conf: float) -> float:
    """Fraction of the 17 keypoints with confidence above ``kp_conf`` (None -> 0)."""
    if kps is None:
        return 0.0
    return float((np.asarray(kps)[:, 2] > kp_conf).sum()) / 17.0


def is_truncated(box: tuple[float, float, float, float], w: int, h: int, margin: int) -> bool:
    """True if the box touches the frame border (within ``margin`` px)."""
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def weighted_mean(vecs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Quality-weighted mean of rows, L2-normalized (float32)."""
    w = np.asarray(weights, np.float64)
    m = (np.asarray(vecs, np.float64) * (w[:, None] if w.sum() > 0 else 1.0)).sum(0)
    return (m / max(np.linalg.norm(m), 1e-12)).astype(np.float32)


def box_iou(a, b) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def split_indices(embs: np.ndarray, min_sim: float, min_samples: int) -> list[int]:
    """Change points of a chronological embedding sequence (sorted indices into ``embs``).

    For a range of n >= 2*min_samples rows, the split ``k`` minimising the cosine similarity of
    ``mean(e[:k])`` and ``mean(e[k:])`` (k in [min_samples, n - min_samples]) is taken if that
    similarity is below ``min_sim``; both sides are then searched recursively. Index ``k`` means
    "rows ``< k`` belong to the earlier piece".
    """
    out: list[int] = []

    def rec(a: int, b: int) -> None:
        n = b - a
        if n < 2 * min_samples:
            return
        e = np.asarray(embs[a:b], np.float64)
        cs = np.cumsum(e, axis=0)
        ks = np.arange(min_samples, n - min_samples + 1)
        left = cs[ks - 1]
        right = cs[-1] - left
        sims = (left * right).sum(1) / np.maximum(
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
        )
        i = int(np.argmin(sims))
        if sims[i] < min_sim:
            k = a + int(ks[i])
            rec(a, k)
            out.append(k)
            rec(k, b)

    rec(0, len(embs))
    return out


def choose_split_frame(
    frames: list[int], overlaps: list[float], lo_frame: int, hi_frame: int
) -> int:
    """Frame at which to cut, among track points with ``lo_frame < frame <= hi_frame``.

    The cut goes where the track overlapped another one the most (where people crossed); among
    equal maxima (including "no overlap at all") the middle candidate is used, which minimises
    the expected error when nothing else is known. Falls back to ``hi_frame``.
    """
    cand = [(f, o) for f, o in zip(frames, overlaps, strict=True) if lo_frame < f <= hi_frame]
    if not cand:
        return hi_frame
    top = max(o for _, o in cand)
    best = [f for f, o in cand if o >= top - 1e-9]
    return best[len(best) // 2]


@dataclass
class _Track:
    local_id: int
    first_frame: int
    last_frame: int
    points: list[TrackPoint] = field(default_factory=list)
    overlaps: list[float] = field(default_factory=list)  # parallel to points: max IoU, other tracks
    history: list[Sample] = field(default_factory=list)  # every embedded sample (not just top-K)
    jpegs: list[bytes] = field(default_factory=list)  # padded crop of each history sample
    faces: list[FaceObs] = field(default_factory=list)
    last_sample_proc: int = -(10**9)


def thin_history(tr: _Track) -> None:
    """Halve the density of the older half of ``tr.history`` (keeps the best-quality sample).

    Recent samples stay dense, so a switch near the end is still found precisely; an old switch
    is still found because both sides keep plenty of (sparser) samples.
    """
    order = sorted(range(len(tr.history)), key=lambda i: tr.history[i].frame)
    best = max(order, key=lambda i: tr.history[i].quality)
    half = len(order) // 2
    keep = [i for j, i in enumerate(order) if j >= half or j % 2 == 0 or i == best]
    tr.history = [tr.history[i] for i in keep]
    tr.jpegs = [tr.jpegs[i] for i in keep]


@dataclass
class _Pending:
    track: _Track
    frame: int
    quality: float
    crop: np.ndarray
    jpg: bytes


class TrackletBuilder:
    """Accumulates tracker output for one camera. Call :meth:`process` per processed frame and
    :meth:`flush` at the end; both return the tracklets finalized by that call."""

    def __init__(self, cfg: Config, cam: CameraInfo, models: Models | None, fps: float) -> None:
        self.cfg, self.cam, self.models, self.fps = cfg, cam, models, fps
        self.q, self.t, self.f = cfg.quality, cfg.tracklets, cfg.face
        self.tracks: dict[int, _Track] = {}
        self.pending: list[_Pending] = []
        self.proc_idx = 0
        self.rejects: Counter[str] = Counter()  # why due samples were rejected
        self.ms = {"pose": 0.0, "reid": 0.0, "face": 0.0}
        self.n_pose_calls = self.n_embedded = self.n_faces_found = self.n_splits = 0
        self._warned_nopose = False

    # ------------------------------------------------------------------ per frame
    def process(
        self,
        frame_idx: int,
        frame: np.ndarray,
        tracks: list[TrackRow],
        kps: np.ndarray | None = None,
        det_embs: np.ndarray | None = None,
    ) -> list[Tracklet]:
        """Consume one frame. ``kps`` are detector keypoints indexed by ``det_index`` (or None).

        ``det_embs`` are body embeddings of the detections (row = det_index, computed on the
        unpadded box crop) when the tracker already needed them; candidates backed by a detection
        use them directly instead of being embedded again.
        """
        self.proc_idx += 1
        h, w = frame.shape[:2]
        ts = self.cam.start_ts + frame_idx / self.fps
        cand: list[tuple[_Track, tuple, float, int]] = []
        for tid, box, conf, det_i in tracks:
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = _Track(tid, frame_idx, frame_idx)
            tr.last_frame = frame_idx
            tr.points.append(TrackPoint(frame_idx, ts, *box, conf))
            tr.overlaps.append(
                max((box_iou(box, o[1]) for o in tracks if o[0] != tid), default=0.0)
            )
            if self._passes_cheap_gates(tr, box, conf, w, h):
                cand.append((tr, box, conf, det_i))
        if cand:
            self._sample(frame, frame_idx, cand, kps, det_embs)
        if len(self.pending) >= self.cfg.runtime.batch_size:
            self._embed_pending()
        return self._finalize_lost(frame_idx)

    def flush(self) -> list[Tracklet]:
        """Finalize every open track (end of video)."""
        self._embed_pending()
        return [t for tr in list(self.tracks.values()) for t in self._finalize(tr)]

    # ------------------------------------------------------------------ gates / sampling
    def _passes_cheap_gates(self, tr: _Track, box, conf: float, w: int, h: int) -> bool:
        if self.proc_idx - tr.last_sample_proc < self.q.sample_every:
            return False  # not a rejection: simply not due yet
        if conf < self.q.min_det_conf:
            self.rejects["det_conf"] += 1
        elif (box[3] - box[1]) < self.q.min_box_h:
            self.rejects["box_h"] += 1
        elif self.q.reject_truncated and is_truncated(box, w, h, self.q.edge_margin_px):
            self.rejects["truncated"] += 1
        else:
            return True
        return False

    def _keypoints_for(self, frame, cand, kps) -> list[np.ndarray | None]:
        """Keypoints of each candidate: from the detector (own det row) or the pose model."""
        if kps is not None:
            return [kps[d] if 0 <= d < len(kps) else None for _, _, _, d in cand]
        if self.models is not None and self.models.pose is not None:
            t0 = time.perf_counter()
            with self.models.stage("pose"):
                got = self.models.pose.keypoints(frame, np.array([b for _, b, _, _ in cand]))
            self.ms["pose"] += (time.perf_counter() - t0) * 1e3
            self.n_pose_calls += 1
            return list(got)
        if not self._warned_nopose:
            log.warning("no keypoints available: visibility gate disabled")
            self._warned_nopose = True
        return [np.tile([0.0, 0.0, 1.0], (17, 1)) for _ in cand]

    def _sample(self, frame, frame_idx, cand, kps, det_embs=None) -> None:
        h, w = frame.shape[:2]
        for (tr, box, conf, det_i), kp in zip(
            cand, self._keypoints_for(frame, cand, kps), strict=True
        ):
            ratio = visibility_ratio(kp, self.q.kp_conf)
            q = ratio * conf
            if ratio < self.q.min_visible_ratio:
                self.rejects["visibility"] += 1
                continue
            if q < self.q.min_quality:
                self.rejects["quality"] += 1
                continue
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tr.last_sample_proc = self.proc_idx
            ok, buf = cv2.imencode(
                ".jpg", self._padded_crop(frame, box), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            jpg = buf.tobytes() if ok else b""
            if det_embs is not None and 0 <= det_i < len(det_embs):
                self._add_sample(tr, Sample(frame_idx, q, np.asarray(det_embs[det_i])), jpg)
            else:
                self.pending.append(_Pending(tr, frame_idx, q, crop.copy(), jpg))
            if self.cam.is_face_cam and self.f.enabled and self.models and self.models.face:
                self._detect_face(tr, frame, frame_idx, box)

    def _padded_crop(self, frame: np.ndarray, box) -> np.ndarray:
        h, w = frame.shape[:2]
        p = self.t.crop_pad
        bw, bh = box[2] - box[0], box[3] - box[1]
        x1, y1 = max(0, int(box[0] - p * bw)), max(0, int(box[1] - p * bh))
        x2, y2 = min(w, int(np.ceil(box[2] + p * bw))), min(h, int(np.ceil(box[3] + p * bh)))
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ embedding
    def _embed_pending(self) -> None:
        if not self.pending:
            return
        if self.models is None:  # extraction without embeddings (tests, dry runs)
            self.pending = []
            return
        pend, self.pending = self.pending, []
        t0 = time.perf_counter()
        with self.models.stage("reid"):
            emb = self.models.body.embed([p.crop for p in pend])
        self.ms["reid"] += (time.perf_counter() - t0) * 1e3
        self.n_embedded += len(pend)
        for p, e in zip(pend, emb, strict=True):
            self._add_sample(p.track, Sample(p.frame, p.quality, e), p.jpg)

    def _add_sample(self, tr: _Track, sample: Sample, jpg: bytes) -> None:
        tr.history.append(sample)
        tr.jpegs.append(jpg)
        if len(tr.history) > self.t.max_history:
            thin_history(tr)

    # ------------------------------------------------------------------ faces
    def _detect_face(self, tr: _Track, frame: np.ndarray, frame_idx: int, box) -> None:
        """Best face whose centre lies in the head region of ``box`` (fixes v1 B18)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        head_y2 = y1 + self.f.head_region * (y2 - y1)
        mx, my = 0.1 * (x2 - x1), 0.1 * (head_y2 - y1)
        cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        cx2, cy2 = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(head_y2 + my)))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return
        region = frame[cy1:cy2, cx1:cx2]
        t0 = time.perf_counter()
        found = self.models.face.analyze(region)  # type: ignore[union-attr]
        self.ms["face"] += (time.perf_counter() - t0) * 1e3
        best = None
        for fb, kps5, score, emb in found:
            fb = fb + np.array([cx1, cy1, cx1, cy1], np.float32)
            fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            if score <= self.f.det_conf or not (x1 <= fcx <= x2 and y1 <= fcy <= head_y2):
                continue
            fq = face_quality(score, fb, kps5, self.f.min_face_px, self.f.full_size_px)
            if fq > 0 and (best is None or fq > best.quality):
                best = FaceObs(
                    frame_idx,
                    tuple(float(v) for v in fb),
                    score,
                    fq,
                    emb,
                    self._face_chip(frame, fb),
                )
        if best is not None:
            self.n_faces_found += 1
            tr.faces.append(best)
            if len(tr.faces) > self.f.max_faces:
                tr.faces.remove(min(tr.faces, key=lambda x: x.quality))

    def _face_chip(self, frame: np.ndarray, fb: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        m = self.f.crop_margin
        bw, bh = fb[2] - fb[0], fb[3] - fb[1]
        x1, y1 = max(0, int(fb[0] - m * bw)), max(0, int(fb[1] - m * bh))
        x2, y2 = min(w, int(np.ceil(fb[2] + m * bw))), min(h, int(np.ceil(fb[3] + m * bh)))
        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ finalization
    def _finalize_lost(self, frame_idx: int) -> list[Tracklet]:
        lost = [
            tr
            for tr in self.tracks.values()
            if (frame_idx - tr.last_frame) / self.fps >= self.t.track_lost_s
        ]
        if not lost:
            return []
        self._embed_pending()  # samples of closing tracks must be in before they are built
        return [t for tr in lost for t in self._finalize(tr)]

    def _finalize(self, tr: _Track) -> list[Tracklet]:
        """Build the tracklet(s) of a closed track: one, or several if its appearance switched."""
        self.tracks.pop(tr.local_id, None)
        order = sorted(range(len(tr.history)), key=lambda i: tr.history[i].frame)
        hist = [tr.history[i] for i in order]
        jpgs = [tr.jpegs[i] for i in order]
        cuts: list[int] = []  # frames: a piece starts at each of these
        if self.t.split_on_appearance and len(hist) >= 2 * self.t.split_min_samples:
            frames = [p.frame for p in tr.points]
            sims = split_indices(
                np.stack([s.embedding for s in hist]),
                self.t.split_min_sim,
                self.t.split_min_samples,
            )
            for k in sims:
                cut = choose_split_frame(frames, tr.overlaps, hist[k - 1].frame, hist[k].frame)
                cuts.append(cut)
                self.n_splits += 1
                log.info(
                    "%s track %d: appearance change, split at frame %d (%s)",
                    self.cam.id,
                    tr.local_id,
                    cut,
                    self._split_sim(hist, k),
                )
        bounds = [-(10**12), *cuts, 10**12]
        out = []
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            t = self._build_piece(tr, hist, jpgs, lo, hi)
            if t is not None:
                out.append(t)
        return out

    @staticmethod
    def _split_sim(hist: list[Sample], k: int) -> str:
        a = np.mean([s.embedding for s in hist[:k]], axis=0)
        b = np.mean([s.embedding for s in hist[k:]], axis=0)
        sim = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        return f"similarity {sim:.2f}, {k}/{len(hist) - k} samples"

    def _build_piece(
        self, tr: _Track, hist: list[Sample], jpgs: list[bytes], lo: int, hi: int
    ) -> Tracklet | None:
        """Tracklet of the part of ``tr`` with ``lo <= frame < hi``."""
        points = [p for p in tr.points if lo <= p.frame < hi]
        if len(points) < self.t.min_points:
            return None
        idx = [i for i, s in enumerate(hist) if lo <= s.frame < hi]
        top = sorted(idx, key=lambda i: -hist[i].quality)
        samples = [hist[i] for i in top[: self.t.max_samples_per_tracklet]]
        kept = {s.frame for s in samples}
        for p in points:
            p.sampled = p.frame in kept
        best_crop = None
        if top and jpgs[top[0]]:
            best_crop = cv2.imdecode(np.frombuffer(jpgs[top[0]], np.uint8), cv2.IMREAD_COLOR)
        faces = [f for f in tr.faces if lo <= f.frame < hi]
        tl = Tracklet(
            camera_id=self.cam.id,
            local_track_id=tr.local_id,
            start_frame=points[0].frame,
            end_frame=points[-1].frame,
            start_ts=points[0].ts,
            end_ts=points[-1].ts,
            n_points=len(points),
            n_samples=len(samples),
            points=points,
            samples=samples,
            faces=sorted(faces, key=lambda f: -f.quality),
            best_crop=best_crop,
        )
        if tl.samples:
            q = np.array([s.quality for s in tl.samples])
            tl.mean_quality = float(q.mean())
            tl.body_embedding = weighted_mean(tl.embeddings, q)
        if tl.faces:
            tl.face_quality = tl.faces[0].quality
            tl.face_embedding = weighted_mean(
                np.stack([f.embedding for f in tl.faces]), np.array([f.quality for f in tl.faces])
            )
        return tl
