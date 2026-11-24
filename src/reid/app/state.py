"""Shared application state: config, per-call database access, lazy models, cached renders."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from reid.config import Config, load_config
from reid.models.registry import Models
from reid.storage import db, files
from reid.storage.vectors import VectorStore

# Config sections that decide which weights are loaded; a change means rebuilding the models.
_MODEL_SECTIONS = ("runtime", "detection", "quality", "body_reid", "face", "paths")


class AppContext:
    """Everything the tabs share. Holds no models until a handler asks for them.

    SQLite connections are opened per call (``with ctx.db() as conn``), so Gradio's worker
    threads never share one. Renders are cached under ``data_dir/renders/app`` keyed by their
    parameters and by a fingerprint of the identity assignment, so editing identities
    invalidates them.
    """

    def __init__(self, cfg: Config, admin: bool = False) -> None:
        self.cfg = cfg
        self.admin = admin
        self.data_dir = cfg.data_dir.resolve()
        self.ui: dict[str, Any] = {}  # components that tabs register for each other
        self._models: Models | None = None
        self._models_key = ""
        self._lock = threading.Lock()
        files.ensure_dirs(self.data_dir)

    # ------------------------------------------------------------------ database

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        conn = db.connect(files.db_path(self.data_dir))
        try:
            yield conn
        finally:
            conn.close()

    def vectors(self) -> VectorStore:
        return VectorStore(files.vectors_dir(self.data_dir))

    def latest_run_id(self, conn: sqlite3.Connection) -> int | None:
        runs = db.list_runs(conn)
        return int(runs[-1]["id"]) if runs else None

    # ------------------------------------------------------------------ config / models

    def derive_cfg(self, preset: str | None, overrides: dict[str, Any] | None = None) -> Config:
        """Config for a processing run: ``preset`` + ``overrides`` on this app's data_dir."""
        merged = {"paths.data_dir": str(self.data_dir), **(overrides or {})}
        return load_config(None, merged, preset=preset or None)

    def models(self, cfg: Config | None = None) -> Models:
        """Models for ``cfg`` (default: the app config); built once, reused while unchanged."""
        from reid.models.registry import build_models

        cfg = cfg or self.cfg
        dump = cfg.model_dump()
        key = json.dumps({k: dump[k] for k in _MODEL_SECTIONS}, sort_keys=True)
        with self._lock:
            if self._models is None or key != self._models_key:
                self._models = build_models(cfg)
                self._models_key = key
            return self._models

    def models_loaded(self) -> bool:
        return self._models is not None

    # ------------------------------------------------------------------ render cache

    def fingerprint(self, conn: sqlite3.Connection) -> str:
        """Changes whenever tracklets or their identity assignment change."""
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0), TOTAL(COALESCE(gid, 0) * id) FROM tracklets"
        ).fetchone()
        return ":".join(str(x) for x in tuple(row))

    def cached(
        self,
        conn: sqlite3.Connection,
        kind: str,
        params: dict[str, Any],
        suffix: str,
        make: Callable[[Path], object],
    ) -> Path:
        """Path of ``make``'s output for these parameters, calling ``make(path)`` on a miss."""
        key = json.dumps([kind, params, self.fingerprint(conn)], sort_keys=True, default=str)
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        out = files.render_path(self.data_dir, f"app/{kind}_{digest}{suffix}")
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f"tmp_{out.name}")
            make(tmp)
            tmp.replace(out)
        return out

    # ------------------------------------------------------------------ run statistics

    def _stats_path(self, run_id: int) -> Path:
        return self.data_dir / "cache" / f"run_{run_id}.json"

    def save_run_stats(self, run_id: int, stats: dict[str, Any]) -> None:
        """Persist per-stage timings and VRAM of a run (the database does not store them)."""
        p = self._stats_path(run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, default=str, indent=2))

    def load_run_stats(self, run_id: int | None) -> dict[str, Any] | None:
        if run_id is None or not self._stats_path(run_id).exists():
            return None
        return json.loads(self._stats_path(run_id).read_text())
