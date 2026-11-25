"""Fetch a small multi-camera subset of the AI City 2024 MTMC dataset from Hugging Face.

Source: ``nvidia/PhysicalAI-SmartSpaces`` (dataset repo, CC-BY-4.0, synthetic, no gating). Layout
verified against the repo file listing on 2026-11-24::

    MTMC_Tracking_2024/{train,val,test}/scene_XXX/camera_YYYY/video.mp4      (1080p, 30 fps)
    MTMC_Tracking_2024/{train,val,test}/scene_XXX/ground_truth.txt           (all cameras)

``ground_truth.txt`` lines are ``<camera> <obj_id> <frame> <x> <y> <w> <h> <xworld> <yworld>``
with ``obj_id`` consistent across the cameras of a scene. Scenes 071-080 are a retail space plus
a storage room (dataset README); each has 16 usable cameras (``scene_071/camera_0649`` is corrupt).
``scene_XXX/calibration_2025_format.json`` holds one homography per camera
(:mod:`reid.datasets.aicity24_calib`).

What :func:`fetch_aicity24` does: list the repo, resolve the scene, download only the GT plus the
chosen cameras' videos (or, with ``stream=True``, let ffmpeg read just the first minutes over
HTTP), trim and downscale each video, convert the GT to the uniform format of
:mod:`reid.datasets.gt` (camera ids ``cam_<number>``, boxes scaled to the output resolution, cut
at the same frame), and write a camera YAML in which every camera is synchronized
(``start_ts: 0``) and no camera is a face camera (the people are synthetic). With
``minutes=None`` and ``height=None`` the original videos are kept as they are (no re-encode), and
``cameras=0`` takes every camera of the scene: this is the full-scene benchmark input, which also
gets each camera's image-to-floor homography (source pixels), ``gt_official.txt`` (the official
GT rows of the chosen cameras) and ``calibration.json``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reid.datasets.aicity24_calib import write_scene_extras
from reid.datasets.aicity24_gt import convert_gt, load_raw_gt, select_cameras
from reid.datasets.cameras import write_camera_yaml
from reid.datasets.gt import write_gt
from reid.datasets.videos import VideoInfo, probe_video, trim_video

__all__ = [  # re-exported: the helpers moved to videos.py / aicity24_gt.py
    "VideoInfo", "convert_gt", "fetch_aicity24", "load_raw_gt", "probe_video", "resolve_scene",
    "select_cameras", "trim_video",
]  # fmt: skip

log = logging.getLogger(__name__)

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
ROOT = "MTMC_Tracking_2024"
LICENSE = "CC-BY-4.0, NVIDIA, https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces"
KNOWN_CORRUPT = {("071", 649)}  # (scene, camera number), per the dataset README
DEFAULT_SCENE = "075"
_VIDEO_RE = re.compile(rf"^{ROOT}/(train|val|test)/scene_(\d+)/camera_(\d+)/video\.mp4$")
_GT_RE = re.compile(rf"^{ROOT}/(train|val|test)/scene_(\d+)/ground_truth\.txt$")
_CALIB_RE = re.compile(rf"^{ROOT}/(train|val|test)/scene_(\d+)/calibration_2025_format\.json$")


@dataclass
class SceneLayout:
    """Repo paths of one scene: ``cameras`` maps camera number -> video path in the repo."""

    split: str
    scene: str
    gt: str
    cameras: dict[int, str] = field(default_factory=dict)
    calibration: str | None = None


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
        elif (m := _CALIB_RE.match(f)) and m.group(2) == sid and m.group(1) == layout.split:
            layout.calibration = f
    if not layout.cameras:
        raise ValueError(f"scene {sid} has a ground truth but no camera videos in the listing")
    return layout


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
    raw_dir: str | Path | None = None,
) -> FetchResult:
    """Download, trim and convert a scene subset into ``out`` (see the module docstring).

    ``camera_ids`` (camera numbers such as ``700``) overrides the automatic choice of
    ``cameras`` overlapping cameras; ``cameras=0`` takes every camera of the scene.
    ``minutes=None`` keeps the whole video; with ``height=None`` too, the original files are
    moved to ``out`` without re-encoding. ``raw_dir`` (default ``out/raw``) receives the
    downloads and is cleaned up afterwards. ``stream=True``
    avoids downloading the full videos: ffmpeg reads only the needed part over HTTP (the GT
    file, 50-90 MB, is always downloaded). ``files`` / ``download`` are injection points for
    tests. Outputs: ``cam_<n>.mp4``, ``gt.csv``, ``meta.json`` in ``out`` and the camera YAML
    (default ``configs/cameras/aicity24_<scene>.yaml``).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    layout = resolve_scene(files if files is not None else list_repo_files(), scene)
    download = download or _default_download
    own_raw = raw_dir is None
    raw = Path(raw_dir) if raw_dir is not None else out / "raw"
    log.info(
        "scene %s (%s): %d cameras in the repo", layout.scene, layout.split, len(layout.cameras)
    )
    raw_gt_path = download(layout.gt, raw)
    raw_gt = load_raw_gt(raw_gt_path)

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
    elif cameras <= 0:
        chosen, stats = sorted(layout.cameras), {}
    else:
        chosen, stats = select_cameras(raw_gt, list(layout.cameras), cameras, window)
    log.info("cameras %s %s", chosen, stats)

    ids = {c: f"cam_{c:04d}" for c in chosen}
    videos: dict[str, Path] = {}
    scale: dict[int, tuple[float, float]] = {}
    size: dict[int, tuple[int, int]] = {}
    infos: dict[int, VideoInfo] = {}
    src_sizes: dict[str, list[int]] = {}
    keep_original = secs is None and not height  # no trimming, no scaling: do not re-encode
    for c in chosen:
        if stream and not keep_original:
            from huggingface_hub import hf_hub_url

            src: str | Path = hf_hub_url(REPO_ID, layout.cameras[c], repo_type="dataset")
        else:
            src = download(layout.cameras[c], raw)
        src_info = probe_video(src)
        dst = out / f"{ids[c]}.mp4"
        if keep_original:
            shutil.move(src, dst)
            info = src_info
        else:
            info = trim_video(src, dst, secs, height)
        scale[c] = (info.width / src_info.width, info.height / src_info.height)
        size[c], infos[c], videos[ids[c]] = (info.width, info.height), info, dst
        src_sizes[ids[c]] = [src_info.width, src_info.height]
        if not keep_raw and not stream:
            Path(src).unlink(missing_ok=True)
    frames = min(i.frames for i in infos.values())
    fps = infos[chosen[0]].fps
    calib_src = download(layout.calibration, raw) if layout.calibration else None
    homs, n_official = write_scene_extras(
        calib_src, raw_gt_path, out, chosen, frames if secs is not None else None
    )
    if not keep_raw:
        shutil.rmtree(
            raw if own_raw else raw / ROOT / layout.split / f"scene_{layout.scene}",
            ignore_errors=True,
        )
    gt = convert_gt(raw_gt, ids, max_frame=frames, scale=scale, size=size)
    gt_path = write_gt(gt, out / "gt.csv")
    yp = write_camera_yaml(
        yaml_path or f"configs/cameras/aicity24_{layout.scene}.yaml",
        [
            {"id": ids[c], "name": ids[c], "video_path": str(videos[ids[c]])}
            | ({"homography": homs[c]} if c in homs else {})
            for c in chosen
        ],
        header=(
            f"AI City 2024 MTMC, scene {layout.scene}, "
            f"{'full length' if secs is None else f'first {minutes} min'}, "
            f"{f'{height}p' if height else 'original resolution'}, cameras synchronized, "
            f"synthetic people (no face cameras). {LICENSE}"
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
        "camera_numbers": {ids[c]: c for c in chosen},
        "src_sizes": src_sizes,
        "homography_cameras": sorted(homs),
        "gt_official_rows": n_official,
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
