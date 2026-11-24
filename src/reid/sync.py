"""Two-way sync of environments between two workspaces (PC <-> Google Drive folder / rclone remote).

An environment is a folder. ``push`` copies local environments to the target, ``pull`` copies the
target's environments here. Per environment a *fingerprint* (sha256 of a consistent SQLite
snapshot, header counters masked, plus the sizes of the vector files) says whether its data
changed; ``<env>/.sync_state.json`` (local side only) remembers the fingerprint of the last sync
per target. :func:`decide` is a pure function over (source, destination, last-synced) fingerprints:

* identical -> nothing to do; destination missing -> create; only the source changed -> copy;
* only the destination changed, or both changed -> refused (data would be lost), unless ``force``,
  which first moves the destination aside as ``<env>.bak-<timestamp>``.

Files are moved through a small :class:`Transfer` interface. :class:`DirTransfer` is a plain folder
(a mounted Drive, a USB disk, the local workspace); :class:`RcloneTransfer` is a thin wrapper around
an ``rclone`` executable that is already installed (it is never installed here). The database is
always copied as a consistent snapshot (SQLite backup API); ``-wal``/``-shm`` files never travel.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from reid.storage import portable

STATE_FILE = ".sync_state.json"
DB_NAME = "reid.db"
_SKIP_NAMES = {"reid.db-wal", "reid.db-shm", "reid.db-journal", STATE_FILE}
_SKIP_SUFFIXES = (".part", ".tmp")
_MTIME_SLACK = 2.0  # seconds; Drive and FAT round modification times
_NAME_RE = portable._NAME_RE
_REMOTE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]+:")  # "gdrive:path" (>= 2 chars: not C:\)
DOCS = "docs/local_app.md (section 'Sync between your PC and Colab')"


class SyncError(RuntimeError):
    """A sync problem the user can act on (message says how)."""


# ----------------------------------------------------------------------------- fingerprints


def db_fingerprint(snapshot: Path, vector_sizes: dict[str, int]) -> str:
    """sha256 of a SQLite snapshot (change counter and version stamp masked) + vector sizes.

    The two masked header fields differ between a database and a copy of it, so masking makes the
    fingerprint depend on the content only. ``snapshot`` must be a consistent copy.
    """
    h = hashlib.sha256()
    with open(snapshot, "rb") as fh:
        head = bytearray(fh.read(100))
        head[24:28] = bytes(4)  # file change counter
        head[92:100] = bytes(8)  # version-valid-for + SQLite version number
        h.update(bytes(head))
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    for name in sorted(vector_sizes):
        h.update(f"\0{name}:{vector_sizes[name]}".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- decisions

Action = Literal["unchanged", "create", "copy", "overwrite", "conflict", "behind", "missing"]


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str

    @property
    def refused(self) -> bool:
        return self.action in ("conflict", "behind")


def decide(src: str | None, dst: str | None, base: str | None, *, force: bool = False) -> Decision:
    """What to do with one environment, from the fingerprints of source, destination and the
    last-synced state (``None`` = absent / never synced)."""
    if src is None:
        return Decision("missing", "not present at the source")
    if dst is None:
        return Decision("create", "new at the destination")
    if src == dst:
        return Decision("unchanged", "identical")
    if base is None:
        reason, action = "both sides have it and it was never synced from here", "conflict"
    elif dst == base:
        return Decision("copy", "changed at the source only")
    elif src == base:
        reason = "changed at the destination only (the other side is ahead: sync that way first)"
        action = "behind"
    else:
        reason, action = "changed on both sides since the last sync", "conflict"
    if force:
        return Decision("overwrite", f"{reason}; --force: destination kept as .bak-<timestamp>")
    return Decision(action, f"{reason}; nothing copied (use --force to overwrite, with a backup)")  # type: ignore[arg-type]


def is_synced_file(rel: str, with_renders: bool = False) -> bool:
    """Whether a path inside an environment travels (no WAL/SHM/temp files; renders optional)."""
    parts = rel.split("/")
    name = parts[-1]
    if name in _SKIP_NAMES or name.endswith(_SKIP_SUFFIXES) or parts[0].startswith(".import_"):
        return False
    return with_renders or parts[0] != "renders"


# ----------------------------------------------------------------------------- transfers


@dataclass(frozen=True, slots=True)
class FileInfo:
    size: int
    mtime: float


class Transfer(Protocol):
    """One side of a sync: a folder of environments."""

    label: str

    def list_envs(self) -> list[str]: ...
    def list_files(self, env: str, with_renders: bool = False) -> dict[str, FileInfo]: ...
    def snapshot_db(self, env: str, tmp: Path) -> Path | None: ...
    def get(self, env: str, rel: str, tmp: Path) -> Path: ...
    def put(self, env: str, rel: str, src: Path) -> None: ...
    def remove(self, env: str, rel: str) -> None: ...
    def move_aside(self, env: str, suffix: str) -> str: ...


def _valid_env(name: str) -> bool:
    return bool(_NAME_RE.match(name))


class DirTransfer:
    """A plain folder holding environments (local workspace, mounted Drive, USB disk)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.label = str(self.root)

    def list_envs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if _valid_env(p.name) and (p / DB_NAME).is_file()
        )

    def list_files(self, env: str, with_renders: bool = False) -> dict[str, FileInfo]:
        base = self.root / env
        out = {}
        for f in base.rglob("*") if base.is_dir() else []:
            rel = f.relative_to(base).as_posix()
            if f.is_file() and is_synced_file(rel, with_renders):
                st = f.stat()
                out[rel] = FileInfo(st.st_size, st.st_mtime)
        return out

    def snapshot_db(self, env: str, tmp: Path) -> Path | None:
        src = self.root / env / DB_NAME
        if not src.is_file():
            return None
        dst = tmp / f"{env}.snapshot.db"
        portable._backup_db(src, dst)
        return dst

    def get(self, env: str, rel: str, tmp: Path) -> Path:
        return self.root / env / rel

    def put(self, env: str, rel: str, src: Path) -> None:
        dst = self.root / env / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, part)
        shutil.copystat(src, part)
        part.replace(dst)
        if rel == DB_NAME:  # a stale WAL must never be replayed onto the new database
            for ext in ("-wal", "-shm", "-journal"):
                dst.with_name(DB_NAME + ext).unlink(missing_ok=True)

    def remove(self, env: str, rel: str) -> None:
        (self.root / env / rel).unlink(missing_ok=True)

    def move_aside(self, env: str, suffix: str) -> str:
        new = f"{env}.bak-{suffix}"
        (self.root / env).rename(self.root / new)
        return new


Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _run_rclone(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


class RcloneTransfer:
    """A folder on an rclone remote (``gdrive:reid_data/environments``). Thin subprocess wrapper.

    ``exe`` is an rclone that is already installed; ``runner`` is injectable for tests.
    """

    def __init__(self, target: str, exe: str = "rclone", runner: Runner | None = None) -> None:
        self.base = target.rstrip("/")
        self.exe = exe
        self.label = target
        self._run = runner or _run_rclone

    def _call(self, *args: str) -> str:
        res = self._run([self.exe, *args])
        if res.returncode != 0:
            raise SyncError(
                f"rclone {args[0]} failed ({res.returncode}): {res.stderr.strip()[-400:]}"
            )
        return res.stdout

    def _lsjson(self, path: str, *flags: str) -> list[dict[str, Any]]:
        res = self._run([self.exe, "lsjson", *flags, path])
        if res.returncode == 3:  # directory not found
            return []
        if res.returncode != 0:
            raise SyncError(f"rclone lsjson failed ({res.returncode}): {res.stderr.strip()[-400:]}")
        return json.loads(res.stdout or "[]")

    def list_envs(self) -> list[str]:
        items = self._lsjson(self.base, "-R", "--files-only", "--max-depth", "2")
        names = {
            i["Path"].split("/")[0]
            for i in items
            if i["Path"].endswith(f"/{DB_NAME}") and i["Path"].count("/") == 1
        }
        return sorted(n for n in names if _valid_env(n))

    def list_files(self, env: str, with_renders: bool = False) -> dict[str, FileInfo]:
        out = {}
        for i in self._lsjson(f"{self.base}/{env}", "-R", "--files-only"):
            if is_synced_file(i["Path"], with_renders):
                mt = i.get("ModTime", "")
                out[i["Path"]] = FileInfo(int(i["Size"]), _parse_time(mt))
        return out

    def snapshot_db(self, env: str, tmp: Path) -> Path | None:
        items = self._lsjson(f"{self.base}/{env}", "--files-only")
        by_name = {i["Name"]: i for i in items}
        if DB_NAME not in by_name:
            return None
        if by_name.get(DB_NAME + "-wal", {}).get("Size", 0) > 0:
            raise SyncError(
                f"{self.label}/{env} has an unmerged SQLite WAL file: stop the app that uses it "
                "(on Colab: stop the launcher cell or the runtime) and try again."
            )
        dst = tmp / f"{env}.snapshot.db"
        raw = tmp / f"{env}.raw.db"
        self._call("copyto", f"{self.base}/{env}/{DB_NAME}", str(raw))
        portable._backup_db(raw, dst)  # same normal form as DirTransfer (rollback journal mode)
        raw.unlink(missing_ok=True)
        return dst

    def get(self, env: str, rel: str, tmp: Path) -> Path:
        dst = tmp / "dl" / env / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._call("copyto", f"{self.base}/{env}/{rel}", str(dst))
        return dst

    def put(self, env: str, rel: str, src: Path) -> None:
        self._call("copyto", str(src), f"{self.base}/{env}/{rel}")

    def remove(self, env: str, rel: str) -> None:
        self._call("deletefile", f"{self.base}/{env}/{rel}")

    def move_aside(self, env: str, suffix: str) -> str:
        new = f"{env}.bak-{suffix}"
        self._call("moveto", f"{self.base}/{env}", f"{self.base}/{new}")
        return new


def _parse_time(text: str) -> float:
    from datetime import datetime

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def looks_like_remote(target: str) -> bool:
    """``name:path`` (rclone syntax), unless such a folder exists locally or it is ``C:\\``."""
    return bool(_REMOTE_RE.match(target)) and not Path(target).exists()


def make_transfer(
    target: str, *, which: Callable[[str], str | None] = shutil.which, runner: Runner | None = None
) -> Transfer:
    """A :class:`DirTransfer` for a folder, an :class:`RcloneTransfer` for ``name:path``."""
    if looks_like_remote(target):
        exe = which("rclone")
        if exe is None:
            raise SyncError(
                f"'{target}' looks like an rclone remote, but no `rclone` executable is on PATH. "
                f"Install and configure rclone yourself (see {DOCS}), or give a plain folder, "
                "e.g. a Google Drive folder mounted on this machine."
            )
        return RcloneTransfer(target, exe, runner)
    return DirTransfer(target)


# ----------------------------------------------------------------------------- state


def load_state(env_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((env_dir / STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}


def save_state(env_dir: Path, target: str, fp: str) -> None:
    state = load_state(env_dir)
    state.setdefault("targets", {})[target] = {"fingerprint": fp, "synced_at": time.time()}
    tmp = env_dir / (STATE_FILE + ".part")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(env_dir / STATE_FILE)


def _target_key(remote: Transfer) -> str:
    return str(Path(remote.label).resolve()) if isinstance(remote, DirTransfer) else remote.label


# ----------------------------------------------------------------------------- engine


@dataclass(slots=True)
class EnvReport:
    env: str
    decision: Decision
    copied_files: int = 0
    copied_bytes: int = 0
    backup: str | None = None
    done: bool = False  # data was actually written (False in a dry run or when refused)
    note: str = ""


@dataclass(slots=True)
class SyncReport:
    direction: str
    envs: list[EnvReport] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return any(e.decision.refused for e in self.envs)


def fingerprint_of(side: Transfer, env: str, tmp: Path, with_renders: bool = False) -> str | None:
    snap = side.snapshot_db(env, tmp)
    if snap is None:
        return None
    try:
        sizes = {
            r: i.size
            for r, i in side.list_files(env, with_renders).items()
            if r.startswith("vectors/")
        }
        return db_fingerprint(snap, sizes)
    finally:
        snap.unlink(missing_ok=True)


def _needs_copy(s: FileInfo, d: FileInfo | None) -> bool:
    return d is None or s.size != d.size or s.mtime > d.mtime + _MTIME_SLACK


def _copy_env(
    src: Transfer, dst: Transfer, env: str, tmp: Path, *, delete: bool, with_renders: bool
) -> tuple[int, int]:
    """Copy changed files, the database snapshot last. Returns (files, bytes)."""
    sfiles = src.list_files(env, with_renders)
    dfiles = dst.list_files(env, with_renders)
    n = total = 0
    for rel in sorted(sfiles):
        if rel == DB_NAME or not _needs_copy(sfiles[rel], dfiles.get(rel)):
            continue
        dst.put(env, rel, src.get(env, rel, tmp))
        n, total = n + 1, total + sfiles[rel].size
    for rel in dfiles:  # vector files must mirror the database; the rest only with --delete
        if rel not in sfiles and (delete or rel.startswith("vectors/")):
            dst.remove(env, rel)
    snap = src.snapshot_db(env, tmp)
    assert snap is not None
    dst.put(env, DB_NAME, snap)
    size = snap.stat().st_size
    snap.unlink(missing_ok=True)
    return n + 1, total + size


def sync(
    local: Transfer,
    remote: Transfer,
    local_root: Path,
    direction: Literal["push", "pull"],
    *,
    envs: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
    delete: bool = False,
    with_renders: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> SyncReport:
    """Sync environments between ``local`` (the workspace at ``local_root``) and ``remote``."""
    src, dst = (local, remote) if direction == "push" else (remote, local)
    key = _target_key(remote)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    src_envs, dst_envs = set(src.list_envs()), set(dst.list_envs())
    if envs:
        unknown = [e for e in envs if e not in src_envs | dst_envs]
        if unknown:
            raise SyncError(f"no such environment: {', '.join(unknown)}")
        src_envs, dst_envs = src_envs & set(envs), dst_envs & set(envs)
    report = SyncReport(direction)
    with tempfile.TemporaryDirectory(prefix="reid_sync_") as td:
        tmp = Path(td)
        for env in sorted(src_envs | (dst_envs if delete else set())):
            if env not in src_envs:  # only at the destination: --delete moves it aside
                bak = None if dry_run else dst.move_aside(env, stamp)
                rep = EnvReport(env, Decision("missing", "only at the destination (--delete)"))
                rep.backup, rep.done = bak, bak is not None
                rep.note = (
                    f"moved aside as {env}.bak-{stamp}" if not dry_run else "would be moved aside"
                )
                report.envs.append(rep)
                log(f"{env}: {rep.note}")
                continue
            sfp = fingerprint_of(src, env, tmp, with_renders)
            dfp = fingerprint_of(dst, env, tmp, with_renders) if env in dst_envs else None
            local_dir = local_root / env
            base = load_state(local_dir).get("targets", {}).get(key, {}).get("fingerprint")
            d = decide(sfp, dfp, base, force=force)
            rep = EnvReport(env, d)
            report.envs.append(rep)
            log(f"{env}: {d.action} ({d.reason})")
            if d.action == "unchanged" and not dry_run and sfp != base:
                save_state(local_dir, key, sfp)  # type: ignore[arg-type]
            if d.action in ("unchanged", "missing") or d.refused or dry_run:
                continue
            if d.action == "overwrite":
                rep.backup = dst.move_aside(env, stamp)
                log(f"{env}: destination kept as {rep.backup}")
            rep.copied_files, rep.copied_bytes = _copy_env(
                src, dst, env, tmp, delete=delete, with_renders=with_renders
            )
            check = fingerprint_of(dst, env, tmp, with_renders)
            if check != sfp:
                raise SyncError(f"{env}: verification failed after copy (fingerprints differ)")
            rep.done = True
            if (local_root / env).is_dir():
                save_state(local_root / env, key, sfp)  # type: ignore[arg-type]
    return report
