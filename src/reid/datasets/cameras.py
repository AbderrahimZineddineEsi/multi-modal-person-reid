"""Write a camera-set YAML (the input of ``reid process``) from a list of videos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def write_camera_yaml(
    path: str | Path,
    cameras: Sequence[str | Path | Mapping[str, Any]],
    *,
    face_cams: Sequence[str] = (),
    header: str = "",
) -> Path:
    """Write a camera YAML with every camera synchronized (``start_ts: 0``).

    ``cameras`` items are either a video path (the id becomes ``cam_<n>``, the name the file
    stem) or a mapping with at least ``video_path`` and optionally ``id``, ``name``,
    ``start_ts``, ``is_face_cam``, ``fps``. ``face_cams`` lists ids flagged as face cameras;
    all others get ``is_face_cam: false``. Paths are written as given.
    """
    entries: list[dict[str, Any]] = []
    for i, item in enumerate(cameras):
        spec = dict(item) if isinstance(item, Mapping) else {"video_path": str(item)}
        if "video_path" not in spec:
            raise ValueError(f"camera {i}: missing video_path")
        spec["video_path"] = str(spec["video_path"])
        spec.setdefault("id", f"cam_{i + 1:02d}")
        spec.setdefault("name", Path(spec["video_path"]).stem)
        spec.setdefault("start_ts", 0.0)
        spec["is_face_cam"] = bool(spec.get("is_face_cam", False) or spec["id"] in face_cams)
        entries.append(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"cameras": entries}, sort_keys=False)
    path.write_text((f"# {header}\n" if header else "") + body, encoding="utf-8")
    return path
