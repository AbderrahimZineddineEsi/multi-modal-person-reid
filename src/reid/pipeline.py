"""Per-camera extraction pipeline: ``run(cfg, cameras, engine=None, progress=None)``.

Cameras are processed sequentially into one run and one shared identity space. For each camera:
read frames, detect people, track, build tracklets (quality-gated samples, faces), persist
tracklets + every track point, then hand each finalized tracklet to the identity engine.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from reid.config import CameraSet, CameraSpec, Config
from reid.models.gpu_memory import GpuMemoryMonitor, format_peak
from reid.models.registry import Models, build_models
from reid.storage import db, files
from reid.tracking.tracker import Tracker, appearance_active
from reid.tracking.tracklets import TrackletBuilder
from reid.types import CameraInfo, Tracklet
from reid.viz.video_io import fit_frame, processed_size

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]  # (camera_id, frames_done, frames_total)


class IdentityEngineLike(Protocol):
    """The slice of ``reid.identity.IdentityEngine`` the pipeline calls (PLAN section 3.3)."""

    def on_tracklet(self, t: Tracklet) -> int | None: ...
    def end_camera(self, camera_id: str) -> None: ...
    def end_run(self, run_id: int) -> Any: ...


@dataclass
class CameraStats:
    """Counts for one camera."""

    camera_id: str
    fps: float = 0.0
    frames_read: int = 0
    frames_processed: int = 0
    tracklets: int = 0
    track_points: int = 0
    samples: int = 0
    tracklets_with_face: int = 0
    wall_s: float = 0.0
    rejects: dict[str, int] = field(default_factory=dict)  # sample-gate rejections by reason
    splits: int = 0  # tracklets cut at an appearance change (ID switches inside the tracker)


@dataclass
class RunResult:
    """Outcome of :func:`run`: counts, per-stage timings and VRAM."""

    run_id: int
    frames_processed: int = 0
    tracklets: int = 0
    track_points: int = 0
    samples: int = 0
    tracklets_with_face: int = 0
    tracklets_unidentified: int = 0
    wall_s: float = 0.0
    model_load_s: float = 0.0
    fps: float = 0.0  # processed frames per wall second (excluding model loading)
    ms_per_frame: dict[str, float] = field(default_factory=dict)  # detect/track/pose/reid/face/read
    vram_peak_mb: dict[str, float] = field(default_factory=dict)  # PyTorch peak per stage
    # whole-process GPU memory (CUDA context + PyTorch cache + ONNX Runtime), see gpu_memory.py
    gpu_memory: dict[str, Any] = field(default_factory=dict)
    cameras: list[CameraStats] = field(default_factory=list)
    identity_report: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        rep = self.identity_report
        if rep is not None and is_dataclass(rep):
            rep = {k: v for k, v in asdict(rep).items() if k != "decisions"}
        d["identity_report"] = rep
        return d


def read_frames(
    cap: cv2.VideoCapture,
    start_f: int,
    end_f: int | None,
    stride: int,
    max_frames: int | None,
    max_side: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR frame)`` for every ``stride``-th frame in ``[start_f, end_f)``.

    Skipped frames are only ``grab``-ed (no decode to image). ``max_frames`` caps frames read.
    Frames are downscaled to ``max_side`` (longer side, 0 = never) before they are yielded.
    """
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    idx, read = start_f, 0
    while (end_f is None or idx < end_f) and (max_frames is None or read < max_frames):
        if not cap.grab():
            break
        read += 1
        if (idx - start_f) % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx, fit_frame(frame, max_side)
        idx += 1


def _store_crops(conn: sqlite3.Connection, data_dir, cfg: Config, t: Tracklet) -> None:
    """Write the best body/face crop of an inserted tracklet and record their paths."""
    assert t.id is not None
    if cfg.tracklets.save_crops and t.best_crop is not None and t.best_crop.size:
        p = files.crop_path(data_dir, t.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(p), t.best_crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            t.best_crop_path = files.rel_to_data(data_dir, p)
    if cfg.face.save_faces and t.faces and t.faces[0].image is not None:
        p = files.face_path(data_dir, t.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(p), t.faces[0].image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            t.best_face_path = files.rel_to_data(data_dir, p)
    if t.best_crop_path or t.best_face_path:
        conn.execute(
            "UPDATE tracklets SET best_crop_path=?, best_face_path=? WHERE id=?",
            (t.best_crop_path, t.best_face_path, t.id),
        )


def detection_embeddings(models: Models, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Body embeddings of every detection box (unpadded crops, clamped to the frame)."""
    h, w = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        a, b = min(max(int(x1), 0), w - 2), min(max(int(y1), 0), h - 2)
        c, d = max(min(int(np.ceil(x2)), w), a + 2), max(min(int(np.ceil(y2)), h), b + 2)
        crops.append(frame[b:d, a:c])
    return models.body.embed(crops)


def _camera_info(spec: CameraSpec, fps: float, w: int, h: int) -> CameraInfo:
    return CameraInfo(
        id=spec.id,
        name=spec.name,
        video_path=spec.video_path,
        fps=fps,
        width=w,
        height=h,
        start_ts=spec.start_ts,
        is_face_cam=spec.is_face_cam,
        homography=spec.homography,
        background_path=spec.background,
    )


def _process_camera(
    cfg: Config,
    spec: CameraSpec,
    models: Models,
    conn: sqlite3.Connection,
    run_id: int,
    engine: IdentityEngineLike | None,
    progress: ProgressFn | None,
    ms: dict[str, float],
) -> CameraStats:
    cap = cv2.VideoCapture(spec.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {spec.video_path!r} (camera {spec.id})")
    rt = cfg.runtime
    try:
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = spec.fps or file_fps
        if fps <= 0:
            log.warning("%s: no fps in file or config, assuming 30", spec.id)
            fps = 30.0
        w, h = processed_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            rt.max_side,
        )
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        start_f = int(round(rt.start_s * fps))
        end_f = int(round(rt.end_s * fps)) if rt.end_s is not None else n_total
        if end_f is not None and n_total:
            end_f = min(end_f, n_total)
        todo = (end_f - start_f) if end_f is not None else 0
        if rt.max_frames is not None:
            todo = min(todo, rt.max_frames) if todo else rt.max_frames
        db.upsert_camera(conn, cam := _camera_info(spec, fps, w, h))
        log.info("camera %s: %dx%d @ %.3f fps, frames %d..%s", spec.id, w, h, fps, start_f, end_f)

        tracker = Tracker(cfg.tracking)
        use_app = appearance_active(cfg.tracking)
        builder = TrackletBuilder(cfg, cam, models, fps)
        stats = CameraStats(spec.id, fps=fps)
        t_cam = time.perf_counter()

        def commit(done: list[Tracklet]) -> None:
            for t in done:
                db.insert_tracklets(conn, run_id, [t])
                _store_crops(conn, cfg.data_dir, cfg, t)
                stats.tracklets += 1
                stats.track_points += t.n_points
                stats.samples += t.n_samples
                stats.tracklets_with_face += bool(t.faces)
                if engine is not None:
                    engine.on_tracklet(t)
                t.best_crop = None  # free memory; points/samples already persisted/handed over
                t.points = []
                for f in t.faces:
                    f.image = None

        for frame_idx, frame in _timed(
            read_frames(cap, start_f, end_f, rt.frame_stride, rt.max_frames, rt.max_side),
            ms,
            "read",
        ):
            t0 = time.perf_counter()
            with models.stage("detect"):
                boxes, confs, kps = models.detector.detect(frame)
            t1 = time.perf_counter()
            embs = None
            if use_app:  # the tracker's appearance cue; also reused for tracklet samples
                with models.stage("reid"):
                    embs = detection_embeddings(models, frame, boxes)
            t2 = time.perf_counter()
            ms["reid"] += (t2 - t1) * 1e3
            tracks = tracker.update(boxes, confs, frame, embs)
            t3 = time.perf_counter()
            ms["detect"] += (t1 - t0) * 1e3
            ms["track"] += (t3 - t2) * 1e3
            commit(builder.process(frame_idx, frame, tracks, kps, embs))
            stats.frames_processed += 1
            if progress is not None:
                progress(spec.id, frame_idx - start_f + 1, todo)
        stats.frames_read = stats.frames_processed * rt.frame_stride
        commit(builder.flush())
        for k, v in builder.ms.items():
            ms[k] += v
        stats.wall_s = time.perf_counter() - t_cam
        stats.rejects = dict(builder.rejects)
        stats.splits = builder.n_splits
        log.info("camera %s: rejected sample candidates %s", spec.id, stats.rejects)
        return stats
    finally:
        cap.release()


def _timed(it: Iterator, ms: dict[str, float], key: str) -> Iterator:
    """Accumulate the time spent waiting on ``it`` (frame decoding) into ``ms[key]``."""
    while True:
        t0 = time.perf_counter()
        try:
            item = next(it)
        except StopIteration:
            return
        ms[key] += (time.perf_counter() - t0) * 1e3
        yield item


def run(
    cfg: Config,
    cameras: CameraSet,
    engine: IdentityEngineLike | None = None,
    progress: ProgressFn | None = None,
    *,
    models: Models | None = None,
    conn: sqlite3.Connection | None = None,
) -> RunResult:
    """Process all cameras of ``cameras`` into ``cfg.data_dir`` and return a :class:`RunResult`.

    With ``engine=None`` only extraction runs (tracklets, points, samples, faces; gids stay
    NULL). ``models`` and ``conn`` may be supplied to reuse loaded models / an open database
    connection (the identity engine is typically built on the same connection).
    """
    np.random.seed(cfg.runtime.seed)
    try:
        import torch

        torch.manual_seed(cfg.runtime.seed)
    except ImportError:  # extraction tests without torch
        pass
    data_dir = files.ensure_dirs(cfg.data_dir)
    own_conn = conn is None
    conn = conn or db.connect(files.db_path(data_dir))
    run_id = db.create_run(conn, cfg.to_json())
    models = models or build_models(cfg)
    res = RunResult(run_id=run_id)
    monitor = GpuMemoryMonitor(_cuda_index(cfg.runtime.detect_device)).__enter__()
    t_load = time.perf_counter()
    models.warm()
    res.model_load_s = time.perf_counter() - t_load
    ms = dict.fromkeys(("read", "detect", "track", "pose", "reid", "face"), 0.0)
    t_all = time.perf_counter()
    try:
        for spec in cameras.cameras:
            st = _process_camera(cfg, spec, models, conn, run_id, engine, progress, ms)
            res.cameras.append(st)
            if engine is not None:
                engine.end_camera(spec.id)
        if engine is not None:
            res.identity_report = engine.end_run(run_id)
        db.set_run_status(conn, run_id, "done")
        res.tracklets_unidentified = conn.execute(
            "SELECT COUNT(*) FROM tracklets WHERE run_id=? AND gid IS NULL", (run_id,)
        ).fetchone()[0]
    except BaseException:
        db.set_run_status(conn, run_id, "failed")
        raise
    finally:
        monitor.__exit__(None, None, None)
        if own_conn:
            conn.close()
    res.gpu_memory = monitor.peak.to_dict()
    res.wall_s = time.perf_counter() - t_all
    res.frames_processed = sum(c.frames_processed for c in res.cameras)
    res.tracklets = sum(c.tracklets for c in res.cameras)
    res.track_points = sum(c.track_points for c in res.cameras)
    res.samples = sum(c.samples for c in res.cameras)
    res.tracklets_with_face = sum(c.tracklets_with_face for c in res.cameras)
    n = max(res.frames_processed, 1)
    res.ms_per_frame = {k: v / n for k, v in ms.items()}
    res.fps = res.frames_processed / res.wall_s if res.wall_s > 0 else 0.0
    res.vram_peak_mb = models.vram_report()
    log.info("run %d done: %s", run_id, res.ms_per_frame)
    log.info("run %d: %s", run_id, format_peak(monitor.peak))
    return res


def _cuda_index(device: str) -> int:
    """NVML index of a ``cuda:N`` device string (0 for ``cuda``/``auto``/anything else)."""
    _, _, idx = str(device).partition("cuda:")
    return int(idx) if idx.isdigit() else 0
