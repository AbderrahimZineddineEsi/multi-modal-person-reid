"""Helpers for driving the CLI from a notebook (Colab, Kaggle, Jupyter).

IPython's ``!cmd`` does not stop the notebook when the command fails, so a failed step shows up
later as a confusing error in another cell. :func:`sh` streams the command's output like ``!``
and raises when it exits with a non-zero code.
"""

from __future__ import annotations

import codecs
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


class CommandFailed(RuntimeError):
    """A shell command exited with a non-zero code."""


def _run_streamed(cmd: str | Sequence[object]) -> tuple[int, str]:
    """Run ``cmd`` streaming its output; returns ``(exit code, shown command)``."""
    shell = isinstance(cmd, str)
    args: str | list[str] = cmd if shell else [str(c) for c in cmd]
    shown = cmd if shell else shlex.join(args)
    print(f"$ {shown}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        args, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    assert proc.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = proc.stdout.fileno()
    while chunk := os.read(fd, 4096):  # chunks, not lines: keeps "\r" progress bars live
        sys.stdout.write(decoder.decode(chunk))
        sys.stdout.flush()
    sys.stdout.write(decoder.decode(b"", final=True))
    code = proc.wait()
    elapsed = time.perf_counter() - t0
    print(f"[{'done' if code == 0 else 'failed'} in {_fmt_duration(elapsed)}]", flush=True)
    return code, str(shown)


def sh(cmd: str | Sequence[object]) -> None:
    """Run ``cmd`` and stream its output (progress bars included).

    ``cmd`` is a shell string or an argument list (items are converted with ``str``, so
    ``Path`` objects can be passed directly). A non-zero exit raises :class:`CommandFailed`,
    which stops "Run all" at the step that actually failed. Nothing is returned, so a notebook
    cell ending with ``sh(...)`` does not display a stray value.
    """
    code, shown = _run_streamed(cmd)
    if code != 0:
        raise CommandFailed(f"exit code {code}: {shown}\nSee the output above for the cause.")


def try_sh(cmd: str | Sequence[object]) -> int:
    """Like :func:`sh` but never raises for a failing command: returns its exit code."""
    return _run_streamed(cmd)[0]


@dataclass
class Job:
    """One command of :func:`run_parallel`: its log file and extra environment variables."""

    name: str
    cmd: Sequence[object]
    log: str | os.PathLike[str]
    env: Mapping[str, str] = field(default_factory=dict)


def balance(items: Sequence[str], weights: Mapping[str, float], n: int) -> list[list[str]]:
    """Split ``items`` into ``n`` groups of about equal total weight (default weight 1).

    Greedy: the heaviest item first, each to the lightest group so far (ties go to the lower
    group index, which makes equal weights a round-robin). Items keep their input order inside a
    group. Empty groups are dropped, so ``n`` larger than ``len(items)`` is fine.
    """
    order = sorted(range(len(items)), key=lambda i: -weights.get(items[i], 1.0))  # stable
    groups: list[list[int]] = [[] for _ in range(max(1, n))]
    load = [0.0] * len(groups)
    for i in order:
        k = min(range(len(groups)), key=lambda g: (load[g], g))
        groups[k].append(i)
        load[k] += weights.get(items[i], 1.0)
    return [[items[i] for i in sorted(g)] for g in groups if g]


def _tail(path: Path, n: int) -> list[str]:
    """Last ``n`` non-empty lines of a log (a ``\\r`` progress bar counts as separate lines)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()][-n:]


def run_parallel(
    jobs: Sequence[Job],
    poll_s: float = 60.0,
    tail_lines: int = 2,
    fail_tail: int = 40,
    *,
    marker: str | None = None,
    raise_on_failure: bool = True,
) -> dict[str, int]:
    """Run ``jobs`` at the same time, each writing to its own log; print their progress.

    Every ``poll_s`` seconds each running job prints either the last ``tail_lines`` lines of its
    log or, with ``marker`` (e.g. ``"ETA |"``), only its latest line containing the marker. A
    failing job never stops the others: all jobs are waited for. When a job ends, the end of its
    log is printed once. Returns ``{job name: exit code}``; with ``raise_on_failure`` (default) a
    non-zero code raises :class:`CommandFailed` naming the failed jobs, after all jobs ended.
    An interrupt (stop button) terminates the jobs still running.
    """
    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, Path] = {}
    handles = []
    reported: set[str] = set()
    t0 = time.perf_counter()
    codes: dict[str, int | None] = {}
    try:
        for job in jobs:
            args = [str(c) for c in job.cmd]
            path = logs[job.name] = Path(job.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(path, "wb")  # noqa: SIM115 - closed in the finally block
            handles.append(fh)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", **job.env}
            print(f"$ [{job.name}] {shlex.join(args)}  (log: {path})", flush=True)
            procs[job.name] = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, env=env)
        while True:
            time.sleep(poll_s)
            codes = {n: p.poll() for n, p in procs.items()}
            stamp = _fmt_duration(time.perf_counter() - t0)
            for name in procs:
                code = codes[name]
                state = "running" if code is None else f"exit {code}"
                print(f"--- [{stamp}] {name} ({state})", flush=True)
                if code is None:
                    lines = _tail(logs[name], 400 if marker else tail_lines)
                    if marker:
                        lines = [ln for ln in lines if marker in ln][-1:] or lines[-1:]
                elif name not in reported:
                    reported.add(name)
                    lines = _tail(logs[name], tail_lines if code == 0 else fail_tail)
                else:
                    lines = []
                for line in lines:
                    print(f"    {line[:200]}", flush=True)
            if all(c is not None for c in codes.values()):
                break
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        for fh in handles:
            fh.close()
    result = {n: int(c) for n, c in codes.items() if c is not None}
    failed = [n for n, c in result.items() if c != 0]
    print(f"[{len(jobs)} jobs ended in {stamp}; failed: {', '.join(failed) or 'none'}]", flush=True)
    if failed and raise_on_failure:
        raise CommandFailed(
            "; ".join(f"{n} exited with code {result[n]} (log: {logs[n]})" for n in failed)
        )
    return result


def has_finished_run(data_dir: str | os.PathLike[str]) -> bool:
    """True if ``data_dir`` holds at least one processing run that completed."""
    from pathlib import Path

    from reid.storage import db, files

    path = files.db_path(Path(data_dir))
    if not path.exists():
        return False
    conn = db.connect(path)
    try:
        return any(r["status"] == "done" for r in db.list_runs(conn))
    finally:
        conn.close()


def _fmt_duration(seconds: float) -> str:
    """``42 s`` or ``3 min 05 s``."""
    m, s = divmod(round(seconds), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"
