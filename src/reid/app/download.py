"""Download direct video links (http/https) into an environment's ``uploads/`` folder.

Streams to a ``.part`` file, enforces a size limit, checks the extension / content type and
reports progress. No gradio import; the opener is injectable for tests. No YouTube or other
page-type links: only URLs that serve a video file.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
MAX_BYTES = 4 * 1024**3
_PAGE_HOSTS = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "tiktok.com",
    "facebook.com",
    "instagram.com",
)
Progress = Callable[[int, int | None], None]


class DownloadError(ValueError):
    """The link cannot be downloaded as a video (message says why)."""


def parse_urls(text: str | None) -> list[str]:
    """One URL per line; blank lines and duplicates dropped."""
    seen: list[str] = []
    for line in (text or "").splitlines():
        u = line.strip()
        if u and u not in seen:
            seen.append(u)
    return seen


def check_url(url: str) -> urllib.parse.ParseResult:
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise DownloadError(f"{url}: only http(s) links are supported")
    host = (parts.hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in _PAGE_HOSTS):
        raise DownloadError(
            f"{url}: this is a video page, not a video file. Download the file yourself and "
            "upload it, or paste a direct link to an .mp4/.mov/... file."
        )
    return parts


def safe_name(url_path: str, header: str | None, content_type: str) -> str:
    name = ""
    if header:
        msg = Message()
        msg["content-disposition"] = header
        name = msg.get_filename() or ""
    name = Path(urllib.parse.unquote(name or Path(url_path).name)).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "video"
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        ext = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/webm": ".webm",
            "video/x-matroska": ".mkv",
            "video/x-msvideo": ".avi",
        }.get(content_type, "")
        name += ext
    return name


def unique_path(folder: Path, name: str) -> Path:
    target = folder / name
    k = 1
    while target.exists():
        target = folder / f"{Path(name).stem}_{k}{Path(name).suffix}"
        k += 1
    return target


def download_video(
    url: str,
    dest_dir: str | Path,
    *,
    max_bytes: int = MAX_BYTES,
    progress: Progress | None = None,
    opener: Callable[..., object] = urllib.request.urlopen,
    timeout: float = 30.0,
) -> Path:
    """Download ``url`` into ``dest_dir`` and return the file; raises :class:`DownloadError`."""
    parts = check_url(url)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "reid-app/2"})
    try:
        resp = opener(req, timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise DownloadError(f"{url}: cannot connect ({exc})") from exc
    with resp:  # type: ignore[attr-defined]
        headers = resp.headers  # type: ignore[attr-defined]
        ctype = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        ext_ok = Path(parts.path).suffix.lower() in VIDEO_EXTS
        if not (
            ctype.startswith("video/") or (ext_ok and ctype in ("", "application/octet-stream"))
        ):
            raise DownloadError(
                f"{url}: not a video file (content type {ctype or 'unknown'}, "
                f"extension {Path(parts.path).suffix or 'none'})"
            )
        length = headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else None
        if total is not None and total > max_bytes:
            raise DownloadError(
                f"{url}: {total / 1024**3:.1f} GB is over the {max_bytes / 1024**3:.0f} GB limit"
            )
        name = safe_name(parts.path, headers.get("Content-Disposition"), ctype)
        if Path(name).suffix.lower() not in VIDEO_EXTS:
            raise DownloadError(
                f"{url}: cannot tell the video format (use a link ending in .mp4, .mov, ...)"
            )
        target = unique_path(dest, name)
        part = target.with_name(target.name + ".part")
        got = 0
        try:
            with open(part, "wb") as fh:
                while chunk := resp.read(1 << 20):  # type: ignore[attr-defined]
                    got += len(chunk)
                    if got > max_bytes:
                        raise DownloadError(f"{url}: over the {max_bytes / 1024**3:.0f} GB limit")
                    fh.write(chunk)
                    if progress:
                        progress(got, total)
            if total is not None and got != total:
                raise DownloadError(f"{url}: incomplete download ({got} of {total} bytes)")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    return target
