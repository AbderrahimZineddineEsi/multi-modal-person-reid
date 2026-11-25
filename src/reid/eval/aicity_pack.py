"""Pack a finished benchmark environment into one zip (Kaggle limits the number of output files).

After a scene succeeds, ``pack_scene`` writes ``<workspace>/envs/<env>.zip`` with
:func:`reid.storage.portable.export_env` (database, vectors, crops, 720p videos, evaluation
results, logs) and deletes the unpacked folder. The small files that scoring needs stay outside
the zip in ``<workspace>/results/<env>/`` (predictions, official GT, meta, evaluation JSON, run
stats, process log, the benchmark entry), so ``reid benchmark aicity24 score`` works without
unzipping. To use an environment: ``reid env import envs/<env>.zip --workspace W``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reid.eval.aicity_bench_io import read_benchmark
from reid.notebook import has_finished_run
from reid.storage.portable import export_env

ENVS_DIR = "envs"
RESULTS_DIR = "results"
RESULT_FILES = ("predictions.txt", "gt_official.txt", "meta.json", "calibration.json")


def zip_path(workspace: Path, name: str) -> Path:
    return workspace / ENVS_DIR / f"{name}.zip"


def results_path(workspace: Path, name: str) -> Path:
    return workspace / RESULTS_DIR / name


def packed_complete(workspace: Path, name: str) -> bool:
    """True when the environment's zip and its results folder exist."""
    return (
        zip_path(workspace, name).exists()
        and (results_path(workspace, name) / "predictions.txt").exists()
    )


def unpacked_complete(env: Path) -> bool:
    """True when ``env`` is unpacked with a finished run, predictions and gt.csv."""
    return (
        has_finished_run(env)
        and (env / "aicity" / "predictions.txt").exists()
        and (env / "gt.csv").exists()
    )


def is_complete(workspace: Path, name: str) -> bool:
    """A scene needs no more work (packed, or unpacked with ``--no-pack``)."""
    return packed_complete(workspace, name) or unpacked_complete(workspace / name)


def scene_files_dir(workspace: Path, name: str) -> Path | None:
    """Folder with the scene's ``predictions.txt`` etc.: ``results/<env>`` or ``<env>/aicity``."""
    for d in (results_path(workspace, name), workspace / name / "aicity"):
        if (d / "predictions.txt").exists():
            return d
    return None


def pack_scene(workspace: Path, name: str) -> Path:
    """Zip ``<workspace>/<name>`` into ``envs/``, fill ``results/<name>/``, delete the folder.

    Order matters for resuming after a crash: the zip appears atomically first, then the results
    folder (renamed into place), and only then is the unpacked folder removed.
    """
    env = workspace / name
    zp = zip_path(workspace, name)
    export_env(env, zp, with_videos=True)
    tmp = workspace / RESULTS_DIR / f"{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for fname in RESULT_FILES:
        if (env / "aicity" / fname).exists():
            shutil.copyfile(env / "aicity" / fname, tmp / fname)
    for p in sorted((env / "eval").glob("*.json")):
        shutil.copyfile(p, tmp / f"eval_{p.name}")
    for p in sorted((env / "cache").glob("run_*.json")):
        shutil.copyfile(p, tmp / p.name)
    if (env / "logs" / "process.log").exists():
        shutil.copyfile(env / "logs" / "process.log", tmp / "process.log")
    entry = read_benchmark(workspace).get("runs", {}).get(name, {})
    (tmp / "benchmark_entry.json").write_text(json.dumps(entry, indent=2, default=str))
    final = results_path(workspace, name)
    shutil.rmtree(final, ignore_errors=True)
    tmp.replace(final)
    shutil.rmtree(env)
    return zp
