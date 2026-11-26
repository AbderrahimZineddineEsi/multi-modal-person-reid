"""Download PA-100K and PETA (without i-LID) for ``reid attributes fetch``.

Downloads resume (HTTP Range), retry and check the final size; finished files are skipped.
Licences: PA-100K is CC-BY 4.0; PETA is for research use only (its i-LID subset needs a separate
application and is skipped by the loaders). ``from_dir`` reuses zips or extracted folders that
are already present (for example a Kaggle input dataset) instead of downloading.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_DRIVE = "https://drive.usercontent.google.com/download?id={id}&export=download&resourcekey={rk}&confirm=t"
#: name -> (url, expected bytes, dataset folder)
SOURCES: dict[str, tuple[str, int, str]] = {
    "annotation.zip": (
        _DRIVE.format(id="0B5_Ra3JsEOyOOWdHcUx5eVVvMW8", rk="0-7UJyL42Ovu1-25R8sURpMA"),
        338_633,
        "pa100k",
    ),
    "data.zip": (
        _DRIVE.format(id="0B5_Ra3JsEOyOZ1BVN09XQ2xXbFU", rk="0-LMJGCxusx9IbrWk7tmvgrw"),
        450_818_381,
        "pa100k",
    ),
    "PETA.zip": (
        "https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1",
        235_869_860,
        "peta",
    ),
}
_UA = {"User-Agent": "Mozilla/5.0 (reid attributes fetch)"}


def download(url: str, dest: Path, expected: int, retries: int = 6, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``dest`` with resume and retries; verifies ``expected`` bytes."""
    if dest.is_file() and dest.stat().st_size == expected:
        log.info("%s already complete", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have > expected:
            part.unlink()
            have = 0
        req = urllib.request.Request(
            url, headers={**_UA, **({"Range": f"bytes={have}-"} if have else {})}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed URLs)
                if have and resp.status != 206:  # server ignored Range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    while block := resp.read(chunk):
                        f.write(block)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            log.warning("%s: attempt %d/%d failed (%s)", dest.name, attempt, retries, exc)
            time.sleep(min(30, 2**attempt))
            continue
        size = part.stat().st_size
        if size == expected:
            os.replace(part, dest)
            log.info("downloaded %s (%d bytes)", dest.name, size)
            return dest
        last = RuntimeError(f"size {size} != expected {expected}")
        log.warning("%s: %s", dest.name, last)
    raise RuntimeError(f"could not download {url}: {last}")


def extract(zip_path: Path, dest: Path) -> None:
    """Unzip once (a marker file records a finished extraction)."""
    marker = dest / f".extracted_{zip_path.name}"
    if marker.is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s", zip_path.name, dest)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    marker.write_text("ok\n")


def _find(root: Path, pattern: str, max_depth: int = 4) -> list[Path]:
    base = len(root.parts)
    return sorted(
        p for p in root.rglob(pattern) if len(p.parts) - base <= max_depth and p.is_file()
    )


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def use_existing(from_dir: Path, out: Path) -> dict[str, str]:
    """Extract zips found in ``from_dir`` (or link extracted folders) into ``out``."""
    got: dict[str, str] = {}
    for name, (_, _, ds) in SOURCES.items():
        zips = [p for p in _find(from_dir, "*.zip") if p.name.lower() == name.lower()]
        if not zips and name == "PETA.zip":
            zips = [p for p in _find(from_dir, "*.zip") if "peta" in p.name.lower()]
        if zips:
            extract(zips[0], out / ds)
            got[name] = "extracted"
    mats = _find(from_dir, "annotation.mat") if "annotation.zip" not in got else []
    if mats and (out / "pa100k").is_dir() and not (out / "pa100k").is_symlink():
        shutil.copy2(mats[0], out / "pa100k" / "annotation.mat")  # images came from data.zip
        got["annotation.mat"] = "copied"
    elif mats:
        _link(from_dir, out / "pa100k")
        got["pa100k"] = "linked"
    if "PETA.zip" not in got and _find(from_dir, "Label.txt"):
        _link(from_dir, out / "peta")
        got["peta"] = "linked"
    return got


def fetch(out: str | Path, from_dir: str | Path | None = None) -> Path:
    """Download (or reuse) and extract both datasets under ``out/{pa100k,peta}``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if from_dir is not None:
        got = use_existing(Path(from_dir), out)
        log.info("from %s: %s", from_dir, got)
    else:
        for name, (url, size, ds) in SOURCES.items():
            zp = download(url, out / ds / name, size)
            extract(zp, out / ds)
    return out
