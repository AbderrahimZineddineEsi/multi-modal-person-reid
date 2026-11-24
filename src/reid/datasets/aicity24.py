"""Fetch a small multi-camera subset of the AI City 2024 MTMC dataset from Hugging Face.

Source: ``nvidia/PhysicalAI-SmartSpaces`` (dataset repo, CC-BY-4.0, synthetic, no gating). Layout
verified against the repo file listing on 2026-11-24::

    MTMC_Tracking_2024/{train,val,test}/scene_XXX/camera_YYYY/video.mp4      (1080p, 30 fps)
    MTMC_Tracking_2024/{train,val,test}/scene_XXX/ground_truth.txt           (all cameras)

``ground_truth.txt`` lines are ``<camera> <obj_id> <frame> <x> <y> <w> <h> <xworld> <yworld>``
with ``obj_id`` consistent across the cameras of a scene. Scenes 071-080 are a retail space plus
a storage room (dataset README); each has 16 usable cameras (``scene_071/camera_0649`` is corrupt).

What :func:`fetch_aicity24` does: list the repo, resolve the scene, download only the GT plus the
chosen cameras' videos (or, with ``stream=True``, let ffmpeg read just the first minutes over
HTTP), trim and downscale each video, convert the GT to the uniform format of
:mod:`reid.datasets.gt` (camera ids ``cam_<number>``, boxes scaled to the output resolution, cut
at the same frame), and write a camera YAML in which every camera is synchronized
(``start_ts: 0``) and no camera is a face camera (the people are synthetic).
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from reid.datasets.cameras import write_camera_yaml
from reid.datasets.gt import GT_COLUMNS, write_gt

log = logging.getLogger(__name__)

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
ROOT = "MTMC_Tracking_2024"
LICENSE = "CC-BY-4.0, NVIDIA, https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces"
KNOWN_CORRUPT = {("071", 649)}  # (scene, camera number), per the dataset README
DEFAULT_SCENE = "075"
_VIDEO_RE = re.compile(rf"^{ROOT}/(train|val|test)/scene_(\d+)/camera_(\d+)/video\.mp4$")
_GT_RE = re.compile(rf"^{ROOT}/(train|val|test)/scene_(\d+)/ground_truth\.txt$")


@dataclass
class SceneLayout:
    """Repo paths of one scene: ``cameras`` maps camera number -> video path in the repo."""

    split: str
    scene: str
    gt: str
    cameras: dict[int, str] = field(default_factory=dict)


@dataclass
class VideoInfo:
    """Basic properties of a video file."""

    width: int
    height: int
    fps: float
    frames: int


@dataclass
class FetchResult:
    """Everything :func:`fetch_aicity24` produced."""

    out_dir: Path
    gt_path: Path
    camera_yaml: Path
    cameras: list[str]
    videos: dict[str, Path]
    n_people: int
    n_boxes: int
    frames: int
    size: tuple[int, int]
    fps: float
    disk_bytes: int


def parse_scene_id(scene: str | int) -> str:
    """``75``, ``"075"`` and ``"scene_075"`` -> ``"075"``."""
    m = re.search(r"(\d+)$", str(scene).strip())
    if not m:
        raise ValueError(f"cannot parse scene id {scene!r}")
    return f"{int(m.group(1)):03d}"


def list_repo_files(repo_id: str = REPO_ID) -> list[str]:
    """File listing of the dataset repo (network; needs ``huggingface_hub``)."""
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo_id, repo_type="dataset")


def resolve_scene(files: Sequence[str], scene: str | int) -> SceneLayout:
    """Find ``scene`` in a repo file listing: its split, ground truth and camera videos.

    Raises ``ValueError`` listing the available scenes if it is absent or incomplete.
    """
    sid = parse_scene_id(scene)
    layout: SceneLayout | None = None
    for f in files:
        if m := _GT_RE.match(f):
            if m.group(2) == sid:
                layout = SceneLayout(m.group(1), sid, f)
    if layout is None:
        scenes = sorted({m.group(2) for f in files if (m := _GT_RE.match(f))})
        raise ValueError(f"scene {sid} not in {ROOT} (available: {', '.join(scenes) or 'none'})")
    for f in files:
        m = _VIDEO_RE.match(f)
        if m and m.group(2) == sid and m.group(1) == layout.split:
            num = int(m.group(3))
            if (sid, num) not in KNOWN_CORRUPT:
                layout.cameras[num] = f
    if not layout.cameras:
        raise ValueError(f"scene {sid} has a ground truth but no camera videos in the listing")
    return layout


def load_raw_gt(path: str | Path) -> pd.DataFrame:
    """Parse an official ``ground_truth.txt`` into columns ``cam,gid,f,x,y,w,h``."""
    return pd.read_csv(
        path,
        sep=" ",
        header=None,
        usecols=range(7),
        names=["cam", "gid", "f", "x", "y", "w", "h"],
        dtype={"cam": int, "gid": int, "f": int, "x": float, "y": float, "w": float, "h": float},
    )


def select_cameras(
    raw: pd.DataFrame,
    available: Sequence[int],
    n: int,
    max_frame: int | None = None,
    *,
    min_h: float = 60.0,
    max_people: int = 14,
) -> tuple[list[int], dict[str, int]]:
    """Choose ``n`` cameras whose views overlap and that see a moderate number of people.

    Considers boxes taller than ``min_h`` source pixels in frames ``< max_frame``. Among the
    camera combinations that see at most ``max_people`` distinct people (if none does, the ones
    with the fewest), maximizes the number of people visible in two or more of the chosen
    cameras (what cross-camera association needs), then prefers fewer people. Every chosen
    camera must see at least two people. Returns ``(cameras, stats)``.
    """
    d = raw[raw["h"] >= min_h]
    if max_frame is not None:
        d = d[d["f"] < max_frame]
    avail = {int(c) for c in available}
    seen = {int(c): set(g["gid"].unique()) for c, g in d.groupby("cam") if int(c) in avail}
    cams = sorted(c for c, s in seen.items() if len(s) >= 2)
    n = min(n, len(cams))
    if n == 0:
        raise ValueError("no camera sees at least two people in the requested window")
    best: tuple | None = None
    combos = itertools.combinations(cams, n)
    for combo in itertools.islice(combos, 200_000):
        everyone = set().union(*(seen[c] for c in combo))
        multi = sum(sum(g in seen[c] for c in combo) >= 2 for g in everyone)
        key = (len(everyone) <= max_people, multi if len(everyone) <= max_people else 0)
        key = (*key, -len(everyone), tuple(-c for c in combo))
        if best is None or key > best[0]:
            best = (key, combo, len(everyone), multi)
    assert best is not None
    _, combo, total, multi = best
    return list(combo), {"people": total, "people_in_2plus_cameras": multi}


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
) -> VideoInfo:
    """Re-encode the first ``seconds`` of ``src`` (path or URL) to ``dst``: H.264, no audio,
    scaled to ``height`` pixels (width follows, even), every frame kept (frame indices stay
    identical to the source, which the converted GT relies on). Needs ``ffmpeg`` on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim/downscale videos (apt install ffmpeg)")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.3f}"]
    if height:
        cmd += ["-vf", f"scale=-2:{int(height)}"]
    cmd += ["-an", "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast"]
    cmd += ["-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)  # noqa: S603
    return probe_video(dst)


def convert_gt(
    raw: pd.DataFrame,
    cameras: dict[int, str],
    *,
    max_frame: int | None,
    scale: dict[int, tuple[float, float]],
    size: dict[int, tuple[int, int]],
    min_height: float = 0.0,
) -> pd.DataFrame:
    """Official GT -> uniform GT (:data:`reid.datasets.gt.GT_COLUMNS`).

    Keeps the cameras in ``cameras`` (number -> pipeline id) and frames ``< max_frame``, scales
    boxes per camera by ``scale[num] = (sx, sy)``, clips them to ``size[num] = (w, h)`` and drops
    boxes that end up shorter than ``min_height`` pixels or empty.
    """
    d = raw[raw["cam"].isin(cameras)]
    if max_frame is not None:
        d = d[d["f"] < max_frame]
    parts = []
    for num, cid in cameras.items():
        c = d[d["cam"] == num]
        sx, sy = scale[num]
        w, h = size[num]
        x1, y1 = (c["x"] * sx).clip(0, w), (c["y"] * sy).clip(0, h)
        x2, y2 = ((c["x"] + c["w"]) * sx).clip(0, w), ((c["y"] + c["h"]) * sy).clip(0, h)
        part = pd.DataFrame(
            {
                "camera": cid,
                "frame": c["f"],
                "gid": c["gid"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        parts.append(part[((y2 - y1) > max(min_height, 0.0)) & ((x2 - x1) > 0)])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=GT_COLUMNS)
    return out[GT_COLUMNS]


def _default_download(repo_path: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, repo_path, repo_type="dataset", local_dir=str(local_dir)))


def fetch_aicity24(
    scene: str | int = DEFAULT_SCENE,
    *,
    cameras: int = 4,
    camera_ids: Sequence[int] | None = None,
    minutes: float | None = 2.0,
    out: str | Path,
    height: int | None = 720,
    yaml_path: str | Path | None = None,
    stream: bool = False,
    keep_raw: bool = False,
    files: Sequence[str] | None = None,
    download: Callable[[str, Path], Path] | None = None,
) -> FetchResult:
    """Download, trim and convert a scene subset into ``out`` (see the module docstring).

    ``camera_ids`` (camera numbers such as ``700``) overrides the automatic choice of
    ``cameras`` overlapping cameras. ``minutes=None`` keeps the whole video. ``stream=True``
    avoids downloading the full videos: ffmpeg reads only the needed part over HTTP (the GT
    file, 50-90 MB, is always downloaded). ``files`` / ``download`` are injection points for
    tests. Outputs: ``cam_<n>.mp4``, ``gt.csv``, ``meta.json`` in ``out`` and the camera YAML
    (default ``configs/cameras/aicity24_<scene>.yaml``).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layout = resolve_scene(files if files is not None else list_repo_files(), scene)
    download = download or _default_download
    raw_dir = out / "raw"
    log.info(
        "scene %s (%s): %d cameras in the repo", layout.scene, layout.split, len(layout.cameras)
    )
    raw_gt = load_raw_gt(download(layout.gt, raw_dir))

    secs = None if minutes is None else float(minutes) * 60.0
    window = None if secs is None else int(round(secs * 30))  # camera choice assumes 30 fps
    if camera_ids:
        chosen = [int(c) for c in camera_ids]
        unknown = [c for c in chosen if c not in layout.cameras]
        if unknown:
            raise ValueError(
                f"cameras {unknown} not in scene {layout.scene}: {sorted(layout.cameras)}"
            )
        stats: dict[str, int] = {}
    else:
        chosen, stats = select_cameras(raw_gt, list(layout.cameras), cameras, window)
    log.info("cameras %s %s", chosen, stats)

    ids = {c: f"cam_{c:04d}" for c in chosen}
    videos: dict[str, Path] = {}
    scale: dict[int, tuple[float, float]] = {}
    size: dict[int, tuple[int, int]] = {}
    infos: dict[int, VideoInfo] = {}
    for c in chosen:
        if stream:
            from huggingface_hub import hf_hub_url

            src: str | Path = hf_hub_url(REPO_ID, layout.cameras[c], repo_type="dataset")
        else:
            src = download(layout.cameras[c], raw_dir)
        src_info = probe_video(src)
        dst = out / f"{ids[c]}.mp4"
        info = trim_video(src, dst, secs, height)
        scale[c] = (info.width / src_info.width, info.height / src_info.height)
        size[c], infos[c], videos[ids[c]] = (info.width, info.height), info, dst
        if not keep_raw and not stream:
            Path(src).unlink(missing_ok=True)
    if not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)

    frames = min(i.frames for i in infos.values())
    fps = infos[chosen[0]].fps
    gt = convert_gt(raw_gt, ids, max_frame=frames, scale=scale, size=size)
    gt_path = write_gt(gt, out / "gt.csv")
    yp = write_camera_yaml(
        yaml_path or f"configs/cameras/aicity24_{layout.scene}.yaml",
        [{"id": ids[c], "name": ids[c], "video_path": str(videos[ids[c]])} for c in chosen],
        header=(
            f"AI City 2024 MTMC, scene {layout.scene}, first {minutes} min, {height}p, "
            f"cameras synchronized, synthetic people (no face cameras). {LICENSE}"
        ),
    )
    meta = {
        "dataset": "aicity24",
        "license": LICENSE,
        "scene": layout.scene,
        "split": layout.split,
        "cameras": ids,
        "selection": stats,
        "minutes": minutes,
        "height": height,
        "frames": frames,
        "fps": fps,
        "n_people": int(gt["gid"].nunique()),
        "n_boxes": len(gt),
        "gt": gt_path.name,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    disk = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return FetchResult(
        out, gt_path, yp, list(ids.values()), videos, meta["n_people"], len(gt), frames,
        next(iter(size.values())), fps, disk,
    )  # fmt: skip


def result_dict(res: FetchResult) -> dict:
    """JSON-friendly view of a :class:`FetchResult`."""
    d = asdict(res)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()} | {
        "videos": {k: str(v) for k, v in res.videos.items()}
    }
