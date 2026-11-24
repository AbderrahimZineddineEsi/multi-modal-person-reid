"""Video reading and H.264 writing (ffmpeg stdin pipe with an OpenCV fallback)."""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str | None:
    """Path of an ffmpeg that can encode H.264, else None."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if "libx264" in out else None


class VideoSink:
    """Write BGR frames to an H.264 yuv420p mp4 (``-movflags +faststart``).

    Uses an ffmpeg process fed through stdin; falls back to ``cv2.VideoWriter`` (mp4v, which
    browsers do not play) when ffmpeg or libx264 is missing. Dimensions are rounded down to
    even numbers, as yuv420p requires.
    """

    def __init__(
        self, path: str | Path, width: int, height: int, fps: float, crf: int = 23
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = int(width) // 2 * 2, int(height) // 2 * 2
        self.fps = float(fps)
        self.n = 0
        self._proc: subprocess.Popen | None = None
        self._cv: cv2.VideoWriter | None = None
        exe = ffmpeg_exe()
        if exe:
            cmd = [
                exe, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
                "-r", f"{self.fps:.6f}", "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
            ]  # fmt: skip
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            log.warning("ffmpeg/libx264 not found: falling back to cv2.VideoWriter (mp4v)")
            self._cv = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h)
            )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            frame = frame[: self.h, : self.w]
            if frame.shape[0] != self.h or frame.shape[1] != self.w:
                frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.write(frame.tobytes())
        elif self._cv is not None:
            self._cv.write(frame)
        self.n += 1

    def close(self) -> None:
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            rc = self._proc.wait()
            self._proc = None
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {rc} writing {self.path}")
        if self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self) -> VideoSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FrameSource:
    """Random-access-ish reader: sequential reads are cheap, a backward jump seeks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0  # index of the frame the next ``cap.read()`` returns
        self._last: tuple[int, np.ndarray] | None = None

    def get(self, idx: int) -> np.ndarray | None:
        """Frame ``idx`` (None past the end or before the start)."""
        if idx < 0 or (self.n_frames > 0 and idx >= self.n_frames):
            return None
        if self._last is not None and self._last[0] == idx:
            return self._last[1]
        if idx < self._next or idx > self._next + 120:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next = idx
        while self._next < idx:
            if not self.cap.grab():
                return None
            self._next += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last = (idx, frame)
        self._next = idx + 1
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def resolve_video(video_path: str, data_dir: str | Path | None = None) -> Path:
    """Locate a camera's video (as stored, else relative to ``data_dir``)."""
    p = Path(video_path).expanduser()
    if p.exists():
        return p
    if data_dir is not None and (Path(data_dir) / p).exists():
        return Path(data_dir) / p
    raise FileNotFoundError(f"video not found: {video_path}")
