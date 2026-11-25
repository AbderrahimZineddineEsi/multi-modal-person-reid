"""Remaining-time estimate of one ``reid benchmark aicity24 run`` process.

Work is counted in source frames x cameras over all the scenes of the job. Before a scene is
fetched its size is estimated as ``DEFAULT_FRAMES`` x its camera count (all AI City videos are
about 800 s long); afterwards the real frame count is used. Speed is measured on the processing
steps only (no fetch). The estimate adds the non-GPU steps measured so far (fetch, 2D
evaluation, export, per scene) for the scenes that still have to run them, plus one 720p encode
for the tail (the last scene's encode is not overlapped with GPU work). It is an extrapolation
from what has been measured, not a promise.

Every ``interval_s`` seconds, at each camera end and at each step change a line is logged::

    ETA | scene 075 camera 5/16 | 38% of this job | 31.2 fps (processed) | ~1 h 52 min left
    | ends ~14:40 UTC      (one line)

and the same numbers go to ``<workspace>/progress_<job>.json``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_FRAMES = 23_994  # every full AI City 2024 camera video
CAMERAS_BY_SCENE = {"089": 15}  # the others have 16 (dataset README)
NON_GPU_STEPS = ("fetch", "eval_2d", "export")


def fmt_remaining(seconds: float) -> str:
    """``1 h 52 min``, ``37 min`` or ``40 s``."""
    s = max(0, round(seconds))
    if s < 60:
        return f"{s} s"
    m = round(s / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


class ProgressTracker:
    """Collects progress of a job and logs / writes its ETA (thread-safe)."""

    def __init__(
        self,
        job: str,
        workspace: str | Path,
        scenes: Sequence[str],
        frame_stride: int = 2,
        *,
        interval_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.job, self.workspace = job, Path(workspace)
        self.stride = max(1, frame_stride)
        self.interval_s = interval_s
        self._clock, self._wall = clock, wall
        self._lock = threading.RLock()
        self.cameras = {s: CAMERAS_BY_SCENE.get(s, 16) for s in scenes}
        self.units = {s: float(DEFAULT_FRAMES * self.cameras[s]) for s in scenes}
        self._processed: set[str] = set()  # scenes whose processing is over (done or resumed)
        self._cam_done: dict[tuple[str, str], float] = {}
        self._seen: dict[str, list[str]] = {}
        self._steps: dict[str, dict[str, float]] = {}
        self._ended: set[tuple[str, str]] = set()
        self._running_since: float | None = None
        self._process_s = 0.0
        self.scene: str | None = None
        self._last_emit = -1e18
        self.last_line = ""

    # ------------------------------------------------------------------ inputs

    def set_scene(self, scene: str, cameras: int, frames: int) -> None:
        """Real size of a scene (known after its fetch)."""
        with self._lock:
            self.cameras[scene] = cameras
            self.units[scene] = float(frames * cameras)

    def skip_scene(self, scene: str) -> None:
        """A scene that needs no work (already complete): remove it from the job."""
        with self._lock:
            self.units.pop(scene, None)
            self.cameras.pop(scene, None)

    def start_scene(self, scene: str) -> None:
        with self._lock:
            self.scene = scene
            self._seen.setdefault(scene, [])

    def scene_processed(self, scene: str) -> None:
        """Processing of ``scene`` is over (or was done in an earlier session)."""
        with self._lock:
            self._processed.add(scene)

    def process_start(self) -> None:
        with self._lock:
            self._running_since = self._clock()

    def process_end(self) -> None:
        with self._lock:
            if self._running_since is not None:
                self._process_s += self._clock() - self._running_since
                self._running_since = None

    def step_done(self, scene: str, step: str, seconds: float) -> None:
        """Seconds a non-GPU step took (``fetch``, ``eval_2d``, ``export``, ``videos_720p``)."""
        with self._lock:
            self._steps.setdefault(scene, {})[step] = seconds

    def note(self, step: str) -> None:
        """Log an ETA line at a step change."""
        with self._lock:
            self._emit(step)

    def progress(self, cam: str, done: int, total: int) -> None:
        """Pipeline callback ``(camera, source frames done, source frames total)``."""
        with self._lock:
            sc = self.scene
            if sc is None:
                return
            seen = self._seen.setdefault(sc, [])
            new_cam = cam not in seen
            if new_cam:
                seen.append(cam)
            self._cam_done[(sc, cam)] = float(done)
            end = total > 0 and done + self.stride >= total and (sc, cam) not in self._ended
            if end:
                self._ended.add((sc, cam))
            if new_cam or end or self._clock() - self._last_emit >= self.interval_s:
                self._emit("processing")

    # ------------------------------------------------------------------ estimate

    def _done_units(self) -> tuple[float, float]:
        """(units done over the whole job, units processed in this session)."""
        session = sum(self._cam_done.values())
        done = session
        for s in self._processed:
            if s in self.units:
                done += self.units[s] - sum(v for (sc, _), v in self._cam_done.items() if sc == s)
        return done, session

    def _non_gpu_allowance(self) -> float:
        """Seconds of fetch / 2D eval / export still ahead, from the scenes measured so far."""
        full = [sum(st.get(k, 0.0) for k in NON_GPU_STEPS) for st in self._steps.values()
                if all(k in st for k in NON_GPU_STEPS)]  # fmt: skip
        fetches = [st["fetch"] for st in self._steps.values() if "fetch" in st]
        per_scene = (
            sum(full) / len(full) if full else (sum(fetches) / len(fetches) if fetches else 0)
        )
        total = 0.0
        for s in self.units:
            if s in self._steps and all(k in self._steps[s] for k in NON_GPU_STEPS):
                continue  # nothing left for this scene
            total += max(
                per_scene - sum(self._steps.get(s, {}).get(k, 0) for k in NON_GPU_STEPS), 0
            )
        enc = [st["videos_720p"] for st in self._steps.values() if "videos_720p" in st]
        return total + (sum(enc) / len(enc) if enc else 0.0)

    def snapshot(self, step: str = "processing") -> dict[str, Any]:
        with self._lock:
            total = sum(self.units.values())
            done, session = self._done_units()
            elapsed = self._process_s + (
                self._clock() - self._running_since if self._running_since is not None else 0.0
            )
            speed = session / elapsed if elapsed > 1 and session > 0 else 0.0  # units / s
            sc = self.scene
            cams = self.cameras.get(sc or "", 0)
            snap: dict[str, Any] = {
                "job": self.job, "scene": sc, "step": step,
                "camera_index": len(self._seen.get(sc or "", [])), "cameras": cams,
                "percent": round(100 * min(done, total) / total, 1) if total else 100.0,
                "fps_processed": round(speed / self.stride, 2),
                "remaining_s": None, "ends_at": None, "updated": self._wall(),
            }  # fmt: skip
            if speed > 0:
                rem = max(total - done, 0) / speed + self._non_gpu_allowance()
                snap["remaining_s"] = round(rem)
                snap["ends_at"] = self._wall() + rem
            return snap

    def format_line(self, snap: Mapping[str, Any]) -> str:
        where = f"scene {snap['scene']}"
        if snap["step"] == "processing":
            where += f" camera {snap['camera_index']}/{snap['cameras']}"
        else:
            where += f" {snap['step']}"
        parts = ["ETA", where, f"{snap['percent']:.0f}% of this job"]
        if snap["remaining_s"] is None:
            parts.append("measuring speed")
        else:
            ends = time.strftime("%H:%M", time.gmtime(snap["ends_at"]))
            parts += [
                f"{snap['fps_processed']:.1f} fps (processed)",
                f"~{fmt_remaining(snap['remaining_s'])} left",
                f"ends ~{ends} UTC",
            ]
        return " | ".join(parts)

    def _emit(self, step: str) -> None:
        snap = self.snapshot(step)
        self.last_line = self.format_line(snap)
        self._last_emit = self._clock()
        log.info(self.last_line)
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.workspace, prefix=".progress-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({**snap, "line": self.last_line}, f, indent=1)
            os.replace(tmp, self.workspace / f"progress_{self.job}.json")
        except OSError:  # the estimate must never stop a run
            log.debug("could not write the progress file", exc_info=True)
