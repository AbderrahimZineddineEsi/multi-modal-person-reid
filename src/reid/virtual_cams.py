"""Virtual cameras: derive N "cameras" from one video (crops with overlap, mirror, delay, scale).

Used to smoke-test the multi-camera plumbing with a single recording. This proves that the
pipeline runs end to end and that identities can be linked across overlapping views; it does
**not** demonstrate cross-view generalization (all views share lighting, resolution and
optics, and the overlap region is the very same pixels).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VirtualCam:
    """Geometry of one virtual camera inside the source frame."""

    id: str
    x: int  # left edge of the crop in source pixels
    w: int  # crop width (even)
    h: int  # crop height (even; full source height)
    mirror: bool = False
    delay_s: float = 0.0  # becomes the camera's start_ts (timeline offset)
    out_w: int | None = None  # output width after downscale (None = no scaling)

    @property
    def out_h(self) -> int:
        if self.out_w is None:
            return self.h
        return int(round(self.h * self.out_w / self.w / 2)) * 2


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def plan_cameras(
    width: int,
    height: int,
    n: int,
    overlap: float = 0.15,
    mirror: list[int] | None = None,
    delays: list[float] | None = None,
    out_width: int | None = None,
    prefix: str = "cam_virt",
) -> list[VirtualCam]:
    """Split a ``width`` x ``height`` frame into ``n`` horizontal crops.

    Adjacent crops overlap by ``overlap`` x crop width; the crops together span the full width.
    ``mirror`` lists camera indices to flip horizontally, ``delays`` gives per-camera timeline
    offsets in seconds, ``out_width`` downscales each crop to that width.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    cw = _even(width / (n - (n - 1) * overlap))
    step = (width - cw) / (n - 1) if n > 1 else 0.0
    cams = []
    for i in range(n):
        x = min(max(0, int(round(i * step / 2)) * 2), width - cw)
        cams.append(
            VirtualCam(
                id=f"{prefix}_{i}",
                x=x,
                w=cw,
                h=_even(height),
                mirror=i in (mirror or []),
                delay_s=float((delays or [])[i]) if delays and i < len(delays) else 0.0,
                out_w=out_width,
            )
        )
    return cams


def _ffmpeg_cmd(src: str, dst: Path, c: VirtualCam, start_s: float, dur_s: float | None) -> list:
    vf = [f"crop={c.w}:{c.h}:{c.x}:0"]
    if c.mirror:
        vf.append("hflip")
    if c.out_w:
        vf.append(f"scale={_even(c.out_w)}:{c.out_h}")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s}", "-i", src]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s}"]
    return cmd + [
        "-vf",
        ",".join(vf),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]


def _cv2_render(src: str, dst: Path, c: VirtualCam, start_s: float, dur_s: float | None) -> None:
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    size = (c.out_w or c.w, c.out_h)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(dur_s * fps) if dur_s is not None else 10**9
    while n > 0:
        ok, fr = cap.read()
        if not ok:
            break
        fr = fr[: c.h, c.x : c.x + c.w]
        if c.mirror:
            fr = fr[:, ::-1]
        if c.out_w:
            fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
        out.write(fr)
        n -= 1
    out.release()
    cap.release()


def make_virtual_cameras(
    video: str | Path,
    out_dir: str | Path,
    n: int = 2,
    overlap: float = 0.15,
    mirror: list[int] | None = None,
    delays: list[float] | None = None,
    out_width: int | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    face_cams: list[int] | None = None,
    prefix: str = "cam_virt",
    yaml_path: str | Path | None = None,
) -> Path:
    """Write ``n`` virtual camera videos plus a camera-set YAML; returns the YAML path."""
    video = str(video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    cams = plan_cameras(w, h, n, overlap, mirror, delays, out_width, prefix)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = None if end_s is None else end_s - start_s
    entries = []
    for i, c in enumerate(cams):
        dst = out_dir / f"{c.id}.mp4"
        if shutil.which("ffmpeg"):
            subprocess.run(_ffmpeg_cmd(video, dst, c, start_s, dur), check=True)
        else:
            log.warning("ffmpeg not found; using OpenCV writer (mp4v)")
            _cv2_render(video, dst, c, start_s, dur)
        entry = {
            "id": c.id,
            "name": c.id,
            "video_path": str(dst),
            "start_ts": c.delay_s,
            "fps": float(fps) if fps else None,
            "is_face_cam": i in (face_cams or []),
        }
        entries.append({k: v for k, v in entry.items() if v is not None})
    yp = Path(yaml_path) if yaml_path else out_dir / "cameras.yaml"
    yp.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Virtual cameras derived from one video: proves the plumbing, not cross-view\n"
        "# generalization (all views share lighting/optics; overlaps are identical pixels).\n"
    )
    yp.write_text(header + yaml.safe_dump({"cameras": entries}, sort_keys=False))
    return yp
