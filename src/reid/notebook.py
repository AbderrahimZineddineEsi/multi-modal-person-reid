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
from collections.abc import Sequence


class CommandFailed(RuntimeError):
    """A shell command exited with a non-zero code."""


def sh(cmd: str | Sequence[object]) -> None:
    """Run ``cmd`` and stream its output (progress bars included).

    ``cmd`` is a shell string or an argument list (items are converted with ``str``, so
    ``Path`` objects can be passed directly). A non-zero exit raises :class:`CommandFailed`,
    which stops "Run all" at the step that actually failed. Nothing is returned, so a notebook
    cell ending with ``sh(...)`` does not display a stray value.
    """
    shell = isinstance(cmd, str)
    args: str | list[str] = cmd if shell else [str(c) for c in cmd]
    shown = cmd if shell else shlex.join(args)
    print(f"$ {shown}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
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
    if code != 0:
        raise CommandFailed(f"exit code {code}: {shown}\nSee the output above for the cause.")


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
