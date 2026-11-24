"""Environments: isolated data_dirs inside one workspace folder (no UI, no models).

An environment is ``<workspace>/<name>/`` holding ``reid.db`` or an ``env.json`` marker. Other
folders in the workspace (datasets, videos) are ignored.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from reid.storage import files
from reid.storage.maintenance import ENV_MARKER, read_marker

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
DEFAULT_ENV = "default"


@dataclass(frozen=True, slots=True)
class EnvInfo:
    name: str
    path: Path
    cameras: int = 0
    people: int = 0
    runs: int = 0
    description: str = ""

    @property
    def stats(self) -> str:
        parts = (
            f"{self.cameras} camera{'' if self.cameras == 1 else 's'}",
            f"{self.people} {'person' if self.people == 1 else 'people'}",
            f"{self.runs} run{'' if self.runs == 1 else 's'}",
        )
        return " · ".join(parts)

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.stats})"


def validate_name(name: str) -> str:
    """The stripped name, or ``ValueError`` (letters, digits, ``-`` and ``_`` only)."""
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError("Use 1 to 40 letters, digits, - or _ for the environment name.")
    return name


def is_env(path: Path) -> bool:
    return path.is_dir() and ((path / "reid.db").exists() or (path / ENV_MARKER).exists())


def _count(db_file: Path, sql: str) -> int:
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            return int(conn.execute(sql).fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def describe(path: Path) -> EnvInfo:
    db_file = files.db_path(path)
    has_db = db_file.exists()
    return EnvInfo(
        name=path.name,
        path=path,
        cameras=_count(db_file, "SELECT COUNT(*) FROM cameras") if has_db else 0,
        people=_count(db_file, "SELECT COUNT(*) FROM identities WHERE n_tracklets > 0")
        if has_db
        else 0,
        runs=_count(db_file, "SELECT COUNT(*) FROM runs") if has_db else 0,
        description=str(read_marker(path).get("description", "")),
    )


def list_environments(workspace: Path) -> list[EnvInfo]:
    if not workspace.is_dir():
        return []
    return [describe(p) for p in sorted(workspace.iterdir()) if is_env(p)]


def create_environment(workspace: Path, name: str, description: str = "") -> Path:
    """Create ``<workspace>/<name>`` with ``env.json`` and the standard subfolders."""
    name = validate_name(name)
    path = workspace / name
    if path.exists():
        raise ValueError(f"'{name}' already exists in the workspace.")
    files.ensure_dirs(path)
    (path / ENV_MARKER).write_text(
        json.dumps({"name": name, "created_at": time.time(), "description": description.strip()})
    )
    return path


def ensure_environment(workspace: Path, name: str) -> Path:
    """Path of environment ``name``, created when missing (used for the initial selection).

    An existing folder (e.g. one given with ``--data-dir``) is adopted as it is.
    """
    path = workspace / name
    if not path.exists():
        return create_environment(workspace, name)
    files.ensure_dirs(path)
    if not is_env(path):
        (path / ENV_MARKER).write_text(json.dumps({"name": path.name, "created_at": time.time()}))
    return path


def delete_environment(workspace: Path, name: str) -> None:
    """Remove an environment folder; only direct, recognised children of ``workspace``."""
    path = (workspace / name).resolve()
    if path.parent != workspace.resolve() or not is_env(path):
        raise ValueError(f"'{name}' is not an environment of this workspace.")
    shutil.rmtree(path)
