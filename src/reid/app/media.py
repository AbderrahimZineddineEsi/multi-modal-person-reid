"""Image helpers for the UI: inline thumbnails, colour conversion and guarded URL download."""

from __future__ import annotations

import base64
import ipaddress
import socket
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from reid.viz.privacy import strong_blur

MAX_DOWNLOAD_BYTES = 8 * 2**20


@lru_cache(maxsize=2048)
def _thumb(path: str, mtime: float, height: int, blur: bool = False) -> str | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = height / max(h, 1)
    img = cv2.resize(img, (max(1, round(w * scale)), height), interpolation=cv2.INTER_AREA)
    if blur:  # part of the cache key: a sharp and a blurred thumbnail never share an entry
        img = strong_blur(img)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}" if ok else None


def thumb_uri(
    path: str | Path | None, data_dir: Path, height: int = 220, blur: bool = False
) -> str | None:
    """Small JPEG data URI of an image stored under ``data_dir`` (None if it is missing).

    Inlining keeps the cards independent of Gradio's file routes and works behind auth.
    ``blur`` makes the person unrecognisable (showcase mode, see :mod:`reid.app.showcase`).
    """
    if not path:
        return None
    p = Path(path)
    p = p if p.is_absolute() else data_dir / p
    try:
        return _thumb(str(p), p.stat().st_mtime, height, blur)
    except OSError:
        return None


def array_uri(img: np.ndarray, height: int = 220) -> str:
    """JPEG data URI of a BGR array (the search query preview)."""
    scale = height / max(img.shape[0], 1)
    small = cv2.resize(
        img, (max(1, round(img.shape[1] * scale)), height), interpolation=cv2.INTER_AREA
    )
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}" if ok else ""


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Gradio gives RGB(A) arrays; ``reid.search`` expects BGR like ``cv2.imread``."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if img.shape[2] == 4 else cv2.COLOR_RGB2BGR)


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # a redirect could point at a private host
        return None


def fetch_image(url: str, timeout: float = 10.0) -> np.ndarray:
    """Download an image (BGR). Only public http(s) hosts, 8 MB cap; raises ``ValueError``."""
    parts = urllib.parse.urlparse(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Enter an http(s) image URL.")
    if not _public_host(parts.hostname):
        raise ValueError("Only public hosts are allowed.")
    req = urllib.request.Request(url.strip(), headers={"User-Agent": "reid-app"})
    with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Image is larger than 8 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The URL did not return a readable image.")
    return img
