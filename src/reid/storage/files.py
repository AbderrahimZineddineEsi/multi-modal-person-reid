"""Path helpers for everything stored under ``data_dir``, plus a guarded reset."""

from __future__ import annotations

import shutil
from pathlib import Path

MARKER = ".reid_data"
SUBDIRS = ("vectors", "crops", "faces", "renders", "eval")


def db_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "reid.db"


def vectors_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "vectors"


def crop_path(data_dir: str | Path, tracklet_id: int) -> Path:
    """Best body crop of a tracklet (``crops/<id//1000:05d>/t<id>.jpg``, sharded)."""
    return Path(data_dir) / "crops" / f"{tracklet_id // 1000:05d}" / f"t{tracklet_id:08d}.jpg"


def face_path(data_dir: str | Path, tracklet_id: int) -> Path:
    """Best face crop of a tracklet."""
    return Path(data_dir) / "faces" / f"{tracklet_id // 1000:05d}" / f"t{tracklet_id:08d}.jpg"


def render_path(data_dir: str | Path, name: str) -> Path:
    return Path(data_dir) / "renders" / name


def eval_path(data_dir: str | Path, run_name: str) -> Path:
    return Path(data_dir) / "eval" / f"{run_name}.json"


def rel_to_data(data_dir: str | Path, path: str | Path) -> str:
    """Path stored in the DB: relative to ``data_dir`` so the directory can move."""
    return Path(path).relative_to(data_dir).as_posix()


def ensure_dirs(data_dir: str | Path) -> Path:
    """Create ``data_dir`` and its standard subdirectories; returns ``data_dir``."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).touch()
    for sub in SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    return root


def reset_data_dir(data_dir: str | Path, *, confirm: bool) -> None:
    """Delete everything inside ``data_dir`` (the directory itself is kept) and recreate it.

    ``confirm`` must be literally ``True``. Refuses to touch a non-empty directory that does
    not carry the ``.reid_data`` marker written by :func:`ensure_dirs`, and refuses ``/``,
    the home directory and the working directory.
    """
    if confirm is not True:
        raise ValueError("reset_data_dir requires confirm=True")
    root = Path(data_dir).expanduser().resolve()
    if root in (Path("/"), Path.home().resolve(), Path.cwd().resolve()):
        raise ValueError(f"refusing to reset {root}")
    if root.exists():
        if any(root.iterdir()) and not (root / MARKER).exists():
            raise ValueError(f"{root} is not a reid data directory (no {MARKER} marker)")
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    ensure_dirs(root)


def data_stats(data_dir: str | Path) -> dict[str, int | bool]:
    """File-level statistics of a data directory (no DB access)."""
    root = Path(data_dir)
    db = db_path(root)

    def count(sub: str, pattern: str) -> int:
        d = root / sub
        return sum(1 for _ in d.rglob(pattern)) if d.exists() else 0

    return {
        "exists": root.exists(),
        "db_bytes": db.stat().st_size if db.exists() else 0,
        "vector_collections": count("vectors", "*.ids.npy"),
        "crops": count("crops", "*.jpg"),
        "faces": count("faces", "*.jpg"),
        "renders": count("renders", "*.*"),
    }
