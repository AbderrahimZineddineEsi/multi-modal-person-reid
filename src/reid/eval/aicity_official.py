"""Run NVIDIA's official AI City 2024 Track 1 evaluator (HOTA on floor positions) from here.

The evaluator code is NOT part of this repository (NVIDIA licence): :func:`fetch_official_eval`
downloads only its code from the dataset repo (never the 2.8 GB ``sample_file/`` folder) and
:func:`run_official_eval` runs its ``main.py`` as a subprocess and parses what it prints.

``main.py`` prints only the final HOTA, DetA, AssA, LocA (percent): the mean over the scenes of
each scene's mean over the alpha thresholds. It prints nothing per scene, so per-scene numbers
need one run per scene (:func:`run_official_per_scene`). Its file-path validator only accepts
``[A-Za-z0-9_-/.#]``; paths with other characters are copied to a temporary folder first.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
EVAL_SUBDIR = "MTMC_Tracking_2024/eval"
EVAL_PATTERNS = [
    f"{EVAL_SUBDIR}/*.py",
    f"{EVAL_SUBDIR}/trackeval/**",
    f"{EVAL_SUBDIR}/utils/**",
    f"{EVAL_SUBDIR}/*.md",
]
_SAFE_PATH = re.compile(r"^[a-zA-Z0-9_\-/.#]+$")
_METRIC_RE = re.compile(r"^\s*(HOTA|DetA|AssA|LocA):\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)\s*%", re.M)

# Published 2024 leaderboard (all 30 test scenes), for context only; online entries got a bonus.
#: Track 1 leaderboard of the 8th AI City Challenge (overview paper, arXiv:2404.09432, Table 1; 17
#: submissions, ranks 7-8 not listed there): rank -> (team, HOTA on all 30 test scenes).
LEADERBOARD_2024 = {
    1: ("Yachiyo", 71.94),
    2: ("SJTU-Lenovo", 67.22),
    3: ("Nota", 60.93),
    4: ("Fraunhofer IOSB", 60.88),
    5: ("UW-ETRI", 57.14),
    6: ("ARV", 51.06),
}


def fetch_official_eval(dest: str | Path) -> Path:
    """Download the evaluator code into ``dest`` (cached) and return the folder of ``main.py``."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    eval_dir = dest / EVAL_SUBDIR
    if (eval_dir / "main.py").exists() and (eval_dir / "trackeval").is_dir():
        return eval_dir
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID, repo_type="dataset", local_dir=str(dest), allow_patterns=EVAL_PATTERNS
    )
    if not (eval_dir / "main.py").exists():
        raise FileNotFoundError(f"official evaluator not found under {eval_dir} after download")
    return eval_dir


def write_scene_map(scenes: Mapping[str, Sequence[int]], path: str | Path) -> Path:
    """Write the scene map JSON: ``[{"scene_name": "scene_075", "camera_ids": [699, ...]}]``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"scene_name": n, "camera_ids": [int(c) for c in ids]} for n, ids in scenes.items()]
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return path


def parse_scores(stdout: str) -> dict[str, float]:
    """``HOTA``, ``DetA``, ``AssA``, ``LocA`` (percent) from the evaluator's stdout.

    The last occurrence of each wins. Raises ``ValueError`` if one of them is missing.
    """
    found = {m.group(1): float(m.group(2)) for m in _METRIC_RE.finditer(stdout)}
    missing = [k for k in ("HOTA", "DetA", "AssA", "LocA") if k not in found]
    if missing:
        raise ValueError(f"evaluator output has no {', '.join(missing)} line:\n{stdout[-2000:]}")
    return {k: found[k] for k in ("HOTA", "DetA", "AssA", "LocA")}


def _safe(path: Path, tmp: Path) -> Path:
    """``path`` itself if the evaluator accepts it, else a copy inside ``tmp``."""
    p = path.resolve()
    if _SAFE_PATH.match(str(p)):
        return p
    q = tmp / re.sub(r"[^A-Za-z0-9_.-]", "_", p.name)
    shutil.copyfile(p, q)
    return q


def run_official_eval(
    eval_dir: str | Path,
    pred: str | Path,
    gt: str | Path,
    scene_map: str | Path,
    num_cores: int = 4,
    output_dir: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, float]:
    """Run the official ``main.py`` and return ``{"HOTA", "DetA", "AssA", "LocA"}`` in percent.

    ``scene_map`` is the JSON written by :func:`write_scene_map`. Raises ``RuntimeError`` with
    the evaluator's stderr (and the end of its stdout) when it fails.
    """
    eval_dir = Path(eval_dir)
    with tempfile.TemporaryDirectory(prefix="aicityeval_") as tmpdir:
        tmp = Path(tmpdir)
        cmd = [
            sys.executable, "main.py",
            "--prediction_file", str(_safe(Path(pred), tmp)),
            "--ground_truth_file", str(_safe(Path(gt), tmp)),
            "--scene_2_camera_id_file", str(_safe(Path(scene_map), tmp)),
            "--num_cores", str(max(1, int(num_cores))),
        ]  # fmt: skip
        if output_dir is not None:
            out = Path(output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            if not _SAFE_PATH.match(str(out)):
                out = tmp / "out"
            cmd += ["--output_dir", str(out)]
        log.info("official evaluator: %s", " ".join(cmd))
        proc = subprocess.run(  # noqa: S603
            cmd, cwd=eval_dir, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"official evaluator failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}\n"
                f"stdout tail:\n{proc.stdout[-1500:]}"
            )
        scores = parse_scores(proc.stdout)
        if output_dir is not None and out != Path(output_dir).resolve():
            shutil.copytree(out, Path(output_dir), dirs_exist_ok=True)
        return scores


def run_official_per_scene(
    eval_dir: str | Path,
    pred: str | Path,
    gt: str | Path,
    scenes: Mapping[str, Sequence[int]],
    work_dir: str | Path,
    num_cores: int = 4,
) -> dict[str, dict[str, float]]:
    """One evaluator run per scene (the evaluator itself prints no per-scene numbers)."""
    work = Path(work_dir)
    out: dict[str, dict[str, float]] = {}
    for name, cams in scenes.items():
        sm = write_scene_map({name: cams}, work / f"scene_map_{name}.json")
        out[name] = run_official_eval(eval_dir, pred, gt, sm, num_cores)
    return out
