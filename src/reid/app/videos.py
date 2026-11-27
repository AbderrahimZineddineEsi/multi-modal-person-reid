"""Source videos may be missing: an environment processed on Colab stores Colab paths.

Everything that renders or plays video goes through here, so a missing file becomes a friendly
inline notice instead of a traceback. Browsing and search never need the videos.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from reid.app import components as C
from reid.app.state import AppContext
from reid.storage import db
from reid.viz.video_io import resolve_video


def missing_cameras(ctx: AppContext, only: Iterable[str] | None = None) -> list[str]:
    """Names of the cameras whose source video cannot be found on this machine.

    ``only`` limits the check to those camera ids (a clip needs just the cameras it shows).
    """
    wanted = None if only is None else set(only)
    with ctx.db() as conn:
        cams = db.list_cameras(conn)
    gone = []
    for cam in cams:
        if wanted is not None and cam.id not in wanted:
            continue
        try:
            resolve_video(cam.video_path, ctx.data_dir, cam.id)
        except FileNotFoundError:
            gone.append(cam.name)
    return gone


def availability_note(ctx: AppContext, only: Iterable[str] | None = None) -> str:
    """Notice when any camera video (of ``only``, default all) is missing, else ``""``."""
    gone = missing_cameras(ctx, only)
    if not gone:
        return ""
    shown = ", ".join(gone[:4]) + (f" and {len(gone) - 4} more" if len(gone) > 4 else "")
    return C.notice(C.MISSING_VIDEO_TITLE, f"{C.MISSING_VIDEO_HINT} Missing: {shown}.")


def render_or_note(render: Callable[[], str]) -> tuple[str | None, str]:
    """``(video path, "")`` on success, ``(None, notice)`` when a source video is missing."""
    try:
        return render(), ""
    except FileNotFoundError:
        return None, C.missing_video_notice()
