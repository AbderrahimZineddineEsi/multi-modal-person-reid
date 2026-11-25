"""Video helpers for the dataset fetchers: probe a file or URL, trim and downscale with ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    """Basic properties of a video file."""

    width: int
    height: int
    fps: float
    frames: int


def probe_video(path: str | Path) -> VideoInfo:
    """Width, height, fps and frame count of a video (local path or URL) via OpenCV."""
    if str(path).startswith(("http://", "https://")):  # OpenCV's ffmpeg has no CA bundle
        return _probe_url(str(path))
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path!r}")
        return VideoInfo(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def _probe_url(url: str) -> VideoInfo:
    """:func:`probe_video` for a URL, through ``ffprobe`` (``frames`` may be 0 if unknown)."""
    out = subprocess.run(  # noqa: S603
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", url],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    st = json.loads(out)["streams"][0]
    num, _, den = st.get("r_frame_rate", "30/1").partition("/")
    nb = st.get("nb_frames", "0")
    return VideoInfo(
        int(st["width"]), int(st["height"]), float(num) / float(den or 1),
        int(nb) if str(nb).isdigit() else 0,
    )  # fmt: skip


def trim_video(
    src: str | Path,
    dst: str | Path,
    seconds: float | None,
    height: int | None,
    *,
    crf: int = 23,
    nice: int | None = None,
    threads: int | None = None,
) -> VideoInfo:
    """Re-encode the first ``seconds`` of ``src`` (path or URL) to ``dst``: H.264, no audio,
    scaled to ``height`` pixels (width follows, even), every frame kept (frame indices stay
    identical to the source, which the converted GT relies on). ``nice`` runs ffmpeg at a lower CPU
    priority and ``threads`` caps its encoder threads (for encoding next to GPU work). Needs
    ``ffmpeg`` on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim/downscale videos (apt install ffmpeg)")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if nice is not None and shutil.which("nice"):
        cmd = ["nice", "-n", str(nice), *cmd]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.3f}"]
    if height:
        cmd += ["-vf", f"scale=-2:{int(height)}"]
    cmd += ["-an", "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast"]
    if threads:
        cmd += ["-threads", str(int(threads))]
    cmd += ["-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)  # noqa: S603
    return probe_video(dst)
