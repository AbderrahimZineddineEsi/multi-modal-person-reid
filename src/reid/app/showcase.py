"""Showcase mode: the app blurs every person except a hand-picked allow-list (default deny).

For screen recordings that are published. The policy is a small JSON file::

    {"max_female": 0.6, "envs": {"<environment>": {"allow_gids": [3, 4] | "all"}}}

A person image (body or face crop of one tracklet) is sharp only when the environment is listed,
the tracklet's gid is on its allow-list (or the list is ``"all"``) and the tracklet's ``female``
attribute probability is below ``max_female`` (no attribute row: the gid alone decides). Anything
else is blurred hard. :class:`ShowView` is the per-environment answer built from the database;
the rendered videos use the same allow-list through ``render.blur_*`` (see :func:`apply_to_cfg`).
Videos are decided per identity and are stricter: a gid is shown only when none of its tracklets
fails the female guard.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from reid.app.media import thumb_uri
from reid.config import Config
from reid.storage import attributes_db as adb
from reid.storage import files
from reid.viz.privacy import strong_blur

log = logging.getLogger(__name__)

FEMALE = "female"  # label of the attribute head used by the guard


@dataclass(frozen=True)
class Showcase:
    """The policy file: ``envs`` maps an environment to its allowed gids (None = all)."""

    max_female: float = 0.6
    envs: dict[str, frozenset[int] | None] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Showcase:
        """Read and validate a policy file; raises ``ValueError`` on anything unexpected."""
        try:
            raw = json.loads(Path(path).expanduser().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"showcase file {path}: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("envs", {}), dict):
            raise ValueError(f"showcase file {path}: expected an object with an 'envs' object")
        max_female = float(raw.get("max_female", 0.6))
        envs: dict[str, frozenset[int] | None] = {}
        for name, entry in raw.get("envs", {}).items():
            allow = entry.get("allow_gids") if isinstance(entry, dict) else None
            if allow == "all":
                envs[str(name)] = None
            elif isinstance(allow, list) and all(isinstance(g, int) for g in allow):
                envs[str(name)] = frozenset(allow)
            else:
                raise ValueError(f"showcase file {path}: env '{name}' needs allow_gids")
        return cls(max_female, envs)

    def key(self) -> str:
        """Stable text form of the whole policy, for render cache keys."""
        envs = {k: ("all" if v is None else sorted(v)) for k, v in sorted(self.envs.items())}
        return json.dumps({"max_female": self.max_female, "envs": envs}, sort_keys=True)

    def permits(self, env: str, gid: int | None, female: float | None = None) -> bool:
        """May a tracklet of ``gid`` with this ``female`` probability be shown sharp in ``env``?"""
        if env not in self.envs:
            return False
        allow = self.envs[env]
        if allow is not None and (gid is None or gid not in allow):
            return False
        return female is None or female < self.max_female  # NaN fails: blurred


@dataclass(frozen=True)
class ShowView:
    """What is sharp in one environment right now (built by :func:`build_view`)."""

    env: str
    data_dir: Path
    policy_key: str
    sharp: frozenset[str]  # crop paths (relative to data_dir, posix) that may be shown
    gids: frozenset[int]  # identities that may be shown in videos

    @property
    def signature(self) -> str:
        """Changes with the policy and with who is allowed (part of the render cache key)."""
        text = f"{self.env}|{self.policy_key}|{sorted(self.gids)}"
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    def rel(self, path: str | Path) -> str:
        """``path`` relative to the data dir (posix); unchanged when it lies elsewhere."""
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.resolve().relative_to(self.data_dir.resolve())
            except ValueError:
                return p.as_posix()
        return p.as_posix()

    def is_sharp(self, path: str | Path | None) -> bool:
        """False for every path that is not a crop of an allowed tracklet (default deny)."""
        return bool(path) and self.rel(path) in self.sharp  # type: ignore[arg-type]


def female_probs(conn: sqlite3.Connection) -> dict[int, float]:
    """``tracklet id -> P(female)`` from the latest annotated head (empty without one)."""
    head = adb.latest_annotated_head(conn)
    if head is None or FEMALE not in head.index:
        return {}
    ids, probs, _ = adb.read_tracklet_attributes(conn, head.id, len(head.labels))
    col = head.index[FEMALE]
    return {int(t): float(p) for t, p in zip(ids, probs[:, col], strict=True)}


def build_view(policy: Showcase, env: str, data_dir: Path, conn: sqlite3.Connection) -> ShowView:
    """Decide every tracklet of ``env``: its crops are sharp only if the policy permits it."""
    female = female_probs(conn)
    sharp: set[str] = set()
    by_gid: dict[int, list[bool]] = defaultdict(list)
    rows = conn.execute("SELECT id, gid, best_crop_path, best_face_path FROM tracklets").fetchall()
    for r in rows:
        ok = policy.permits(env, r["gid"], female.get(r["id"]))
        if r["gid"] is not None:
            by_gid[r["gid"]].append(ok)
        if not ok:
            continue
        canon = (files.crop_path(data_dir, r["id"]), files.face_path(data_dir, r["id"]))
        sharp.update(files.rel_to_data(data_dir, p) for p in canon)
        sharp.update(p for p in (r["best_crop_path"], r["best_face_path"]) if p)
    gids = frozenset(g for g, oks in by_gid.items() if all(oks))
    return ShowView(env, data_dir, policy.key(), frozenset(sharp), gids)


def thumb(view: ShowView | None, path: str | Path | None, data_dir: Path, height: int = 220):
    """Data URI of a person crop; blurred unless ``view`` is None (showcase off) or allows it."""
    return thumb_uri(path, data_dir, height, blur=view is not None and not view.is_sharp(path))


def blurred_copy(view: ShowView, path: str | Path) -> Path:
    """A blurred JPEG of ``path`` under ``renders/app/showcase`` (for ``gr.Gallery`` items)."""
    src = Path(path)
    src = src if src.is_absolute() else view.data_dir / src
    st = src.stat()
    digest = hashlib.sha1(f"{view.rel(src)}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()
    out = files.render_path(view.data_dir, f"app/showcase/{digest[:16]}.jpg")
    if not out.exists():
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"tmp_{out.name}")
        cv2.imwrite(str(tmp), strong_blur(img))
        tmp.replace(out)
    return out


def gallery_path(view: ShowView | None, path: Path) -> str:
    """Path to hand to ``gr.Gallery``: the original, or a blurred copy when it may not be shown."""
    if view is None or view.is_sharp(path):
        return str(path)
    return str(blurred_copy(view, path))


def apply_to_cfg(
    cfg: Config,
    view: ShowView,
    highlight: int | None = None,
    gt_keep: list[int] | None = None,
    gt_path: str = "",
) -> Config:
    """Copy of ``cfg`` whose renders blur everybody but ``view.gids`` (the original is untouched).

    ``highlight`` is the followed person: when not allowed they are blurred too. With
    ``gt_path`` the ground-truth boxes are blurred as well, except the ids in ``gt_keep``.
    """
    out = cfg.model_copy(deep=True)
    r = out.render
    r.blur_others = True
    r.blur_strength = 1.0
    r.blur_keep_gids = sorted(view.gids)
    r.blur_highlight = highlight is not None and highlight not in view.gids
    if gt_path:
        r.privacy_gt = gt_path
        r.privacy_keep_gt = sorted(gt_keep or [])
    return out
