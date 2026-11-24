"""Move environments between machines (Colab -> PC): list, export, import, relink; model rule.

An environment is one ``data_dir``. ``export_env`` zips it with a *consistent* copy of the
SQLite database (online backup API, so a WAL file or a running writer cannot corrupt it) and a
``reid_export.json`` manifest. With videos, each camera's source file is bundled as
``videos/<camera_id><ext>`` and the manifest records it so ``import_env`` rewrites
``cameras.video_path`` to that relative path (resolved against the environment folder).
No models and no UI are imported here.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from reid.config import Config
from reid.storage import db, files
from reid.storage.maintenance import ENV_MARKER, read_marker

MANIFEST = "reid_export.json"
MANIFEST_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_STORED_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mkv", ".mov", ".webm", ".gif"}
_SKIP_NAMES = {"reid.db-wal", "reid.db-shm", "reid.db-journal"}


# --------------------------------------------------------------------------- helpers


def _basename(video_path: str) -> str:
    """File name of a stored path, whichever separator style (Colab, Windows) it uses."""
    return re.split(r"[\\/]", video_path.rstrip("\\/"))[-1]


def _read_cameras(data_dir: Path) -> list[tuple[str, str]]:
    """``(camera_id, video_path)`` pairs, read-only (no schema creation)."""
    dbp = files.db_path(data_dir)
    if not dbp.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        try:
            return [(r[0], r[1]) for r in conn.execute("SELECT id, video_path FROM cameras")]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def find_video(video_path: str, data_dir: Path, camera_id: str | None = None) -> Path | None:
    """Locate a camera video: stored path, relative to ``data_dir``, then ``uploads/``/``videos/``.

    A relative stored path (what the app writes for videos kept inside the environment) is tried
    against ``data_dir`` first. The ``uploads/`` and ``videos/`` fallbacks (by file name, then
    ``<camera_id>.*`` in ``videos/``) are only used when they match exactly one file, so a wrong
    video is never picked silently.
    """
    p = Path(video_path).expanduser()
    if not p.is_absolute() and (data_dir / p).is_file():
        return data_dir / p
    if p.exists():
        return p
    for sub in ("uploads", "videos"):
        vdir = data_dir / sub
        if not vdir.is_dir():
            continue
        cand = vdir / _basename(video_path)
        if cand.is_file():
            return cand
        if camera_id and sub == "videos":
            hits = [h for h in vdir.glob(f"{camera_id}.*") if h.is_file()]
            if len(hits) == 1:
                return hits[0]
    return None


def relativize_videos(data_dir: str | Path) -> int:
    """Store ``cameras.video_path`` relative to the environment for videos that live inside it.

    Lets the environment move (Colab <-> PC, sync) without the stored path going stale.
    Returns how many cameras were rewritten.
    """
    root = Path(data_dir).resolve()
    conn = db.connect(files.db_path(root))
    n = 0
    try:
        for cam in db.list_cameras(conn):
            p = Path(cam.video_path)
            if not p.is_absolute():
                continue
            try:
                rel = p.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            conn.execute("UPDATE cameras SET video_path=? WHERE id=?", (rel, cam.id))
            n += 1
        conn.commit()
    finally:
        conn.close()
    return n


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- list


@dataclass(slots=True)
class EnvSummary:
    name: str
    path: Path
    cameras: int
    people: int
    runs: int
    size_bytes: int
    videos_present: int
    preset: str | None = None


def is_env(path: Path) -> bool:
    return path.is_dir() and ((path / "reid.db").exists() or (path / ENV_MARKER).exists())


def _count(data_dir: Path, sql: str) -> int:
    dbp = files.db_path(data_dir)
    if not dbp.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        try:
            return int(conn.execute(sql).fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def list_envs(workspace: Path) -> list[EnvSummary]:
    """Summaries of the environments directly under ``workspace`` (no gradio import)."""
    out = []
    root = Path(workspace)
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        if not is_env(path):
            continue
        cams = _read_cameras(path)
        present = sum(find_video(vp, path, cid) is not None for cid, vp in cams)
        out.append(
            EnvSummary(
                path.name,
                path,
                len(cams),
                _count(path, "SELECT COUNT(*) FROM identities WHERE n_tracklets > 0"),
                _count(path, "SELECT COUNT(*) FROM runs"),
                _dir_size(path),
                present,
                _env_preset(path),
            )
        )
    return out


# --------------------------------------------------------------------------- export


@dataclass(slots=True)
class ExportResult:
    out: Path
    size_bytes: int
    videos: dict[str, str] = field(default_factory=dict)
    missing_videos: list[str] = field(default_factory=list)


def _backup_db(src: Path, dst: Path) -> None:
    """Consistent copy of the database (works while it is open elsewhere, WAL included)."""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
        d.execute("PRAGMA journal_mode=DELETE")
    finally:
        d.close()
        s.close()


def export_env(
    data_dir: str | Path,
    out: str | Path,
    *,
    with_videos: bool = False,
    with_renders: bool = False,
) -> ExportResult:
    root = Path(data_dir).resolve()
    dbp = files.db_path(root)
    if not dbp.exists():
        raise ValueError(f"{root} is not an environment (no reid.db)")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip_dirs = {"videos"} | (set() if with_renders else {"renders"})
    tmp_db = out.with_name(out.name + ".db.tmp")
    tmp_zip = out.with_name(out.name + ".part")
    res = ExportResult(out, 0)
    try:
        _backup_db(dbp, tmp_db)
        cams = _read_cameras(root)
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            zf.write(tmp_db, "reid.db")
            for f in sorted(root.rglob("*")):
                rel = f.relative_to(root)
                if (
                    not f.is_file()
                    or f.name in _SKIP_NAMES
                    or f == dbp
                    or rel.parts[0] in skip_dirs
                ):
                    continue
                ctype = zipfile.ZIP_STORED if f.suffix.lower() in _STORED_SUFFIXES else None
                zf.write(f, rel.as_posix(), compress_type=ctype)
            if with_videos:
                for cid, vp in cams:
                    src = find_video(vp, root, cid)
                    if src is None:
                        res.missing_videos.append(cid)
                        continue
                    arc = f"videos/{cid}{src.suffix}"
                    zf.write(src, arc, compress_type=zipfile.ZIP_STORED)
                    res.videos[cid] = arc
            zf.writestr(
                MANIFEST,
                json.dumps(
                    {
                        "manifest_version": MANIFEST_VERSION,
                        "name": root.name,
                        "created_at": time.time(),
                        "schema_version": db.SCHEMA_VERSION,
                        "with_renders": with_renders,
                        "videos": res.videos,
                    },
                    indent=2,
                ),
            )
        tmp_zip.replace(out)
    finally:
        tmp_db.unlink(missing_ok=True)
        tmp_zip.unlink(missing_ok=True)
    res.size_bytes = out.stat().st_size
    return res


# --------------------------------------------------------------------------- import


@dataclass(slots=True)
class ImportResult:
    name: str
    path: Path
    cameras: int
    schema_version: int
    videos_relinked: int
    videos_missing: list[str] = field(default_factory=list)


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for m in zf.infolist():
        p = PurePosixPath(m.filename)
        if p.is_absolute() or ".." in p.parts or "\\" in m.filename:
            raise ValueError(f"unsafe path in archive: {m.filename!r}")
        members.append(m)
    return members


def import_env(
    archive: str | Path,
    workspace: str | Path,
    *,
    name: str | None = None,
    force: bool = False,
) -> ImportResult:
    archive, workspace = Path(archive), Path(workspace)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"{archive} is not a zip file")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if "reid.db" not in names:
            raise ValueError(f"{archive} is not a reid environment export (no reid.db)")
        manifest: dict[str, Any] = json.loads(zf.read(MANIFEST)) if MANIFEST in names else {}
        marker: dict[str, Any] = json.loads(zf.read(ENV_MARKER)) if ENV_MARKER in names else {}
        name = (name or manifest.get("name") or marker.get("name") or archive.stem).strip()
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid environment name {name!r} (use letters, digits, - or _)")
        dest = workspace / name
        if dest.exists():
            if not force:
                raise FileExistsError(f"{dest} already exists (use --force to replace it)")
            if not is_env(dest):
                raise ValueError(f"{dest} exists and is not an environment; refusing to replace")
        workspace.mkdir(parents=True, exist_ok=True)
        stage = workspace / f".import_{name}.tmp"
        shutil.rmtree(stage, ignore_errors=True)
        try:
            members = _safe_members(zf)
            for m in members:
                if m.filename != MANIFEST:
                    zf.extract(m, stage)
            res = _finalize(stage, name, manifest)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    if dest.exists():
        shutil.rmtree(dest)
    stage.replace(dest)
    res.path = dest
    return res


def _finalize(stage: Path, name: str, manifest: dict[str, Any]) -> ImportResult:
    files.ensure_dirs(stage)
    (stage / ENV_MARKER).write_text(
        json.dumps({**read_marker(stage), "name": name, "imported_at": time.time()})
    )
    for stray in _SKIP_NAMES:
        (stage / stray).unlink(missing_ok=True)
    conn = db.connect(files.db_path(stage))  # raises on a schema mismatch
    try:
        version = db.schema_version(conn)
        videos: dict[str, str] = manifest.get("videos", {})
        relinked = 0
        for cid, rel in videos.items():
            if (stage / rel).is_file():
                conn.execute("UPDATE cameras SET video_path=? WHERE id=?", (rel, cid))
                relinked += 1
        cams = [(c.id, c.video_path) for c in db.list_cameras(conn)]
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    for stray in _SKIP_NAMES:
        (stage / stray).unlink(missing_ok=True)
    missing = [cid for cid, vp in cams if find_video(vp, stage, cid) is None]
    return ImportResult(name, stage, len(cams), version, relinked, missing)


# --------------------------------------------------------------------------- relink


@dataclass(slots=True)
class RelinkResult:
    relinked: dict[str, str] = field(default_factory=dict)
    already_ok: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)


def relink_env(data_dir: str | Path, videos_dir: str | Path) -> RelinkResult:
    """Point cameras at local copies found under ``videos_dir`` (recursive).

    A camera matches a file named like the stored path's basename, else ``<camera_id>.*``.
    Cameras whose video already resolves are left alone; several matches are reported, not
    guessed. Paths are stored absolute.
    """
    root, vdir = Path(data_dir).resolve(), Path(videos_dir).expanduser().resolve()
    if not vdir.is_dir():
        raise ValueError(f"{vdir} is not a directory")
    candidates = [f for f in vdir.rglob("*") if f.is_file()]
    res = RelinkResult()
    conn = db.connect(files.db_path(root))
    try:
        for cam in db.list_cameras(conn):
            if find_video(cam.video_path, root, cam.id) is not None:
                res.already_ok.append(cam.id)
                continue
            base = _basename(cam.video_path)
            hits = [f for f in candidates if f.name == base]
            if not hits:
                hits = [f for f in candidates if f.stem == cam.id]
            if len(hits) == 1:
                conn.execute("UPDATE cameras SET video_path=? WHERE id=?", (str(hits[0]), cam.id))
                res.relinked[cam.id] = str(hits[0])
            elif hits:
                res.ambiguous[cam.id] = sorted(str(h) for h in hits)
            else:
                res.missing.append(cam.id)
    finally:
        conn.close()
    return res


# --------------------------------------------------------------------------- model consistency

_BODY_KEYS = ("model", "dim", "neck_feat", "input_h", "input_w", "weights_url")
_FACE_KEYS = ("model", "det_size")


def _latest_run_config(data_dir: Path) -> Config | None:
    dbp = files.db_path(data_dir)
    if not dbp.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT config_json FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        finally:
            conn.close()
        return Config.model_validate(json.loads(row[0])) if row else None
    except (sqlite3.Error, ValueError):
        return None


def _env_preset(data_dir: Path) -> str | None:
    cfg = _latest_run_config(data_dir)
    if cfg is None:
        return None
    from reid.eval.hypothesis import infer_preset

    return infer_preset(cfg)


def env_model_overrides(data_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Body/face embedder settings of the environment's latest run (``{}`` if it has no run).

    Search queries must be embedded by the model that produced the stored vectors, whatever
    preset the app or CLI was started with.
    """
    cfg = _latest_run_config(Path(data_dir))
    if cfg is None:
        return {}
    return {
        "body_reid": {k: getattr(cfg.body_reid, k) for k in _BODY_KEYS},
        "face": {k: getattr(cfg.face, k) for k in _FACE_KEYS},
    }


def apply_env_models(cfg: Config, data_dir: str | Path | None = None) -> tuple[Config, str | None]:
    """``cfg`` with the environment's embedders, plus a short notice when they differ.

    Returns ``(cfg, None)`` unchanged when the environment has no run or already matches.
    """
    ov = env_model_overrides(data_dir or cfg.data_dir)
    changes = [
        f"{sec}.{k}: {getattr(getattr(cfg, sec), k)} -> {v}"
        for sec, vals in ov.items()
        for k, v in vals.items()
        if getattr(getattr(cfg, sec), k) != v
    ]
    if not changes:
        return cfg, None
    from reid.config import deep_merge

    new = Config.model_validate(deep_merge(cfg.model_dump(), ov))
    return new, "search uses the models this environment was processed with (" + "; ".join(
        changes
    ) + ")"
