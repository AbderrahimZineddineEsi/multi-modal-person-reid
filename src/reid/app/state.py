"""Shared application state: config, per-call database access, lazy models, cached renders."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from reid.app import environments as envs
from reid.config import Config, load_config
from reid.models.registry import Models
from reid.storage import db, files
from reid.storage.portable import apply_env_models
from reid.storage.vectors import VectorStore

log = logging.getLogger(__name__)

# Config sections that decide which weights are loaded; a change means rebuilding the models.
_MODEL_SECTIONS = ("runtime", "detection", "quality", "body_reid", "face")


def _slug(text: str) -> str:
    """File-name safe version of ``text`` (letters, digits, ``-``, ``_``)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40]


class AppContext:
    """Everything the tabs share. Holds no models until a handler asks for them.

    The app works on one *environment* at a time: ``<workspace>/<env>`` is the data_dir every
    tab reads and writes. The selected environment lives here, not in per-session state, so all
    browser sessions share it. That is deliberate for a single-user demo: switching in one tab
    re-points the whole app, and the other tabs pick it up on their next refresh.

    SQLite connections are opened per call (``with ctx.db() as conn``), so Gradio's worker
    threads never share one. Renders are cached under ``data_dir/renders/app`` keyed by their
    parameters and by a fingerprint of the identity assignment, so editing identities
    invalidates them.
    """

    def __init__(
        self,
        cfg: Config,
        read_only: bool = False,
        workspace: Path | None = None,
        env: str | None = None,
    ) -> None:
        self.read_only = read_only
        self.ui: dict[str, Any] = {}  # components that tabs register for each other
        self.find_cache: dict[str, Any] = {}  # Find tab: attribute table of the selected env
        self._models: Models | None = None
        self._models_key = ""
        self._lock = threading.Lock()
        data_dir = cfg.data_dir.resolve()
        if workspace is None:  # a plain data_dir: its parent is the workspace
            workspace, env = data_dir.parent, data_dir.name
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.env = ""
        self.data_dir = data_dir
        self.select_env(env or envs.DEFAULT_ENV, create=True)

    # ------------------------------------------------------------------ environments

    def select_env(self, name: str, create: bool = False) -> None:
        """Point the app at ``<workspace>/<name>`` (created when ``create``)."""
        path = envs.ensure_environment(self.workspace, name) if create else self.workspace / name
        if not create and not envs.is_env(path):
            raise ValueError(f"Environment '{name}' does not exist.")
        self.env = name
        self.data_dir = path.resolve()
        # --- model consistency hook: search must embed with the environment's own models
        self._base_cfg = getattr(self, "_base_cfg", None) or self.cfg
        self.cfg = self._base_cfg.model_copy(deep=True)
        self.cfg.paths.data_dir = str(self.data_dir)
        self.cfg, self.model_note = apply_env_models(self.cfg, self.data_dir)
        if self.model_note:
            log.info("environment '%s': %s", name, self.model_note)
        # --- end hook
        files.ensure_dirs(self.data_dir)

    def environments(self) -> list[envs.EnvInfo]:
        return envs.list_environments(self.workspace)

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
        cfg = load_config(None, merged, preset=preset or None)
        # an environment's vectors all come from one body/face model: new runs keep it
        cfg, note = apply_env_models(cfg, self.data_dir)
        if note:
            log.info("processing into '%s' keeps its embedding models: %s", self.env, note)
        return cfg

    def models(self, cfg: Config | None = None) -> Models:
        """Models for ``cfg`` (default: the app config); built once, reused while unchanged."""
        from reid.models.registry import build_models

        cfg = cfg or self.cfg
        dump = cfg.model_dump()
        parts = {k: dump[k] for k in _MODEL_SECTIONS} | {"weights": dump["paths"]["weights_dir"]}
        key = json.dumps(parts, sort_keys=True)  # data_dir is not part of it: no reload per env
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
        tag: str = "",
    ) -> Path:
        """Path of ``make``'s output for these parameters, calling ``make(path)`` on a miss.

        The file is ``<kind>_<tag>_<hash>``: ``tag`` is for people (``gid5``, ``cam_01_gid3``),
        the hash keys the cache.
        """
        key = json.dumps([kind, params, self.fingerprint(conn)], sort_keys=True, default=str)
        digest = hashlib.sha1(key.encode()).hexdigest()[:8]
        name = "_".join(x for x in (kind, _slug(tag), digest) if x)
        out = files.render_path(self.data_dir, f"app/{name}{suffix}")
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
