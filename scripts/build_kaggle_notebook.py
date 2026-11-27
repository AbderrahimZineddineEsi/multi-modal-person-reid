# ruff: noqa: E501  (the notebook cell texts below are content, kept as they appear in Kaggle)
"""Generate ``notebooks/kaggle_aicity_benchmark.ipynb`` (outputs stripped).

The notebook runs the official AI City 2024 Track 1 benchmark on a Kaggle machine with two T4
GPUs and saves everything in ``/kaggle/working``. It contains no pipeline logic: every step is a
``reid`` command (or the small helpers of ``reid.notebook``). Edit this script, run
``python scripts/build_kaggle_notebook.py`` and commit the notebook; never edit the ``.ipynb``.

Cell tags: ``settings`` marks the parameter cell, ``kaggle-only`` the cells that only make sense
on Kaggle (pip installs, GPU listing).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/AbderrahimZineddineEsi/open-world-multimodal-multicam-reid"
NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_aicity_benchmark.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str, *tags: str) -> None:
    meta = {"tags": list(tags)} if tags else {}
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": meta,
            "outputs": [],
            "source": text.strip("\n"),
        }
    )


md("""
# AI City 2024 Track 1: official benchmark on Kaggle (2x T4)

This notebook runs the `reid` pipeline on whole **16-camera test scenes** of the AI City 2024
multi-camera tracking dataset (`nvidia/PhysicalAI-SmartSpaces`, CC-BY-4.0, synthetic people),
scores the result with **NVIDIA's official HOTA evaluator**, and keeps everything needed to show
the project later in `/kaggle/working`: databases, crops, 720p videos, predictions, scores,
timings and logs. It is the baseline: change `PRESET` and `SCENES` below to compare other setups.

It contains no pipeline logic of its own, every step is a `reid ...` command:

1. install the package and check the GPUs;
2. `reid benchmark aicity24 run`: per scene, download the 16 videos, process them (preset,
   720p processing, every 2nd frame), compute our own 2D metrics, export the predictions in the
   official format, keep 720p copies of the videos;
3. `reid benchmark aicity24 score`: run the official evaluator, write `RESULTS_<preset>.md`;
4. optional: render a short mosaic clip for the showcase.

**Before you start** (Kaggle notebook settings, right side panel):

- *Accelerator*: **GPU T4 x2**.
- *Internet*: **On** (needs a phone-verified account). It is used to install the package, download
  the dataset and the model weights.
- Run it with **Save Version > Save & Run All (Commit)**. It then runs without your browser and
  the content of `/kaggle/working` is saved as the version's output (limit 20 GB).

**Do a smoke run first** (`SMOKE = True`, the default): about 10 to 15 minutes on one scene with
2 cameras and 2 minutes of video. It exercises every step (download, processing, metrics, export,
720p copies, packing, official scoring) and prints a **projection** of the full run. Then set
`SMOKE = False` and run the real thing.

**Time**: roughly 3 to 3.5 hours of GPU for 4 scenes x 16 cameras on two T4s. This is an estimate
from the speed measured on one Colab T4 (34 fps of processed frames at 720p, frame stride 2), not
a measurement of this notebook end to end; the smoke run's projection and the live `ETA |` lines
of the real run refine it. A Kaggle session is limited to 12 hours.

**Crash safety**: a scene that fails (any step) is recorded and skipped, the other scenes go on,
every cell ends without an error, and each successful scene is packed into one zip (Kaggle limits
the number of output files). The run is resumable: finished scenes are skipped when you run it again.

Model weights are downloaded at run time and are licensed for non-commercial research use (see
the README).
""")

md("## 1. Settings")
code(
    """
import os

REID_REF = "main"  # branch or tag of the repository to install
PRESET = "balanced"  # "lite" | "balanced" | "accurate" | "max"
SCENES = ["075", "078", "082", "086"]  # test scenes 071-090 have 16 cameras (089 has 15)
FRAME_STRIDE = 2  # process every 2nd frame (the measured 34 fps used this)
KEEP_VIDEOS = "720"  # "720": keep 720p copies of the videos | "none"
SHOWCASE = True  # render a 60 s mosaic clip of the first scene at the end
SMOKE = True  # True: 1 scene, 2 cameras, 2 minutes + projection (10-15 min). False: the real run

WORKSPACE = "/kaggle/working/aicity24_bench"  # everything worth keeping (the version output)
RAW = "/kaggle/temp/raw" if os.path.isdir("/kaggle/temp") else "/tmp/aicity_raw"  # not saved

os.environ["REID_REF"] = REID_REF
""",
    "settings",
)

md("""
## 2. Install and check the machine

The install keeps Kaggle's preinstalled PyTorch. Its log goes to `/tmp/pip_install.log`; the cell
prints the last lines if something fails. Warnings about the dependency resolver for packages this
project does not use are harmless.
""")
code(
    f"""
%%bash
set -o pipefail
python -c "import torch; print('preinstalled torch', torch.__version__, '| CUDA', torch.version.cuda)"
pip install --progress-bar off "reid[models,eval] @ git+{REPO_URL}@${{REID_REF}}" \\
  > /tmp/pip_install.log 2>&1 || {{ tail -n 40 /tmp/pip_install.log; exit 1; }}
tail -n 3 /tmp/pip_install.log
""",
    "kaggle-only",
)
md("""
`ultralytics` pulls in `opencv-python` and `insightface` pulls in `onnxruntime`, which clash with
`opencv-python-headless` and `onnxruntime-gpu`. The cell removes the clashing wheels and
reinstalls the two we want (harmless if nothing clashed). If a later cell fails with an
`ImportError` about `numpy` or `cv2`, restart the session and run again from the top.
""")
code(
    """
%%bash
pip uninstall -y -q opencv-python onnxruntime
pip install -q --force-reinstall --no-deps opencv-python-headless "onnxruntime-gpu==1.32.0"
python -c "import cv2, onnxruntime; print('opencv', cv2.__version__, '| onnxruntime', onnxruntime.__version__)"
""",
    "kaggle-only",
)
code(
    """
from reid.notebook import sh

sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv")
sh("df -h /kaggle/working /kaggle/temp 2>/dev/null || df -h /kaggle/working /tmp")
sh(["reid", "info", "--preset", PRESET])
""",
    "kaggle-only",
)

md("""
## 3. Smoke run or full run

**`SMOKE = True`** (first run): one `reid benchmark aicity24 run --smoke` process on GPU 0 handles
the first scene with 2 cameras and the first 2 minutes, through every step, into
`WORKSPACE/smoke`. It scores that scene with the official evaluator and ends with a line
`PROJECTION | 4 scenes, 62 cameras, 2 GPUs: ~3 h 20 min (...)`, an **extrapolation from a 2-minute
sample** (measured speed and seconds per camera-minute of each step, scaled to the real run). Read
it, then set `SMOKE = False` and use **Save & Run All**.

**`SMOKE = False`**: the scenes are split across the visible GPUs (balanced by camera count) and
one `reid benchmark aicity24 run` process runs per GPU, pinned with `CUDA_VISIBLE_DEVICES`. Each
process handles its scenes one after the other: download (16 videos of about 185 MB), process,
2D metrics, export; then the 720p copies are encoded and the scene is packed into
`WORKSPACE/envs/<env>.zip` in the background while the next scene is already on the GPU. A scene
that fails is recorded (step and error) and the others continue.

Every process writes its own log to `WORKSPACE/run_gpu<i>.log` and a progress file
`WORKSPACE/progress_gpu<i>.json`. This cell prints the latest line like
`ETA | scene 075 camera 5/16 | 38% of this job | 31.2 fps (processed) | ~1 h 52 min left | ends ~14:40 UTC`
of each process every minute, and the end of a log when its process ends. It never raises: a
summary says which processes failed.
""")
code("""
import json
import subprocess
from pathlib import Path

# Only the standard library and reid.notebook (stdlib only) run in this kernel: the install
# upgraded numpy on disk while the kernel still holds the old one, so anything that imports
# numpy here fails. Everything else runs as `reid` commands (fresh processes).
from reid.notebook import Job, balance, run_parallel, sh, try_sh

RUN_WS = WORKSPACE + ("/smoke" if SMOKE else "")  # where this run writes its results
CAMERAS = {s: 15 if s == "089" else 16 for s in SCENES}
listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
n_gpu = max(1, sum(ln.startswith("GPU ") for ln in listing.splitlines()))
groups = [list(SCENES)] if SMOKE else balance(SCENES, CAMERAS, n_gpu)
print(f"{n_gpu} GPU(s); {'smoke run on GPU 0' if SMOKE else 'scenes per GPU'}: {groups}")


def job(i, scenes):
    cmd = ["reid", "benchmark", "aicity24", "run", "--scenes", ",".join(scenes), "--preset", PRESET,
           "--workspace", WORKSPACE, "--raw-dir", f"{RAW}/gpu{i}", "--keep-videos", KEEP_VIDEOS,
           "--frame-stride", FRAME_STRIDE, "--job", f"gpu{i}"]  # fmt: skip
    if SMOKE:
        cmd += ["--smoke", "--project-gpus", n_gpu]
    env = {"CUDA_VISIBLE_DEVICES": str(i)} if n_gpu > 1 else {}
    return Job(f"gpu{i}", cmd, f"{WORKSPACE}/run_gpu{i}.log", env)


jobs = [job(i, g) for i, g in enumerate(groups)]
try:
    codes = run_parallel(jobs, poll_s=60, marker="ETA |", raise_on_failure=False)
except Exception as exc:  # never end the notebook here: /kaggle/working must be saved
    print(f"The run loop stopped with {exc!r}; whatever finished is kept.")
    codes = {}
print("\\nProcess exit codes:", codes or "unknown")
SUMMARY = ["reid", "benchmark", "aicity24", "summary", "--workspace", RUN_WS, "--preset", PRESET,
           "--scenes", ",".join(SCENES[:1] if SMOKE else SCENES)]  # fmt: skip
try_sh(SUMMARY)
""")

md("""
### Smoke result

With `SMOKE = True` this shows the projection of the full run and the official score of the
2-camera sample (a score on 2 minutes of 2 cameras: it proves the chain works, it is not a result).
""")
code("""
if SMOKE:
    try:
        proj = json.loads(Path(RUN_WS, "projection.json").read_text())
        print("\\n".join(proj["lines"]))
    except Exception as exc:  # noqa: BLE001
        print(f"No projection available ({exc!r}): read the smoke log {WORKSPACE}/run_gpu0.log")
    print("\\nNow set SMOKE = False (settings cell) and use Save & Run All for the real run.")
else:
    print("SMOKE is False: nothing to show here.")
""")

md("""
## 4. Score with the official evaluator

`reid benchmark aicity24 score` downloads NVIDIA's evaluator code (not the 2.8 GB sample files),
runs it on all scenes that succeeded together and on each scene alone, and writes
`official_<preset>.json` and `RESULTS_<preset>.md` into the workspace. The official score is the
mean over the scenes of HOTA computed on floor positions (matching distance up to 2 m). Scenes
that failed have no predictions and are skipped, with a message. If no scene succeeded, scoring is
skipped. (In smoke mode the smoke command already scored its scene.)
""")
code("""
from IPython.display import Markdown, display

try:
    if SMOKE:
        print("SMOKE is True: the smoke run scored its own scene (see smoke/RESULTS_*.md).")
    else:  # `score` skips scenes without predictions and fails cleanly if there are none
        code_ = try_sh(["reid", "benchmark", "aicity24", "score", "--workspace", WORKSPACE,
                        "--preset", PRESET, "--num-cores", os.cpu_count() or 4])  # fmt: skip
        res = Path(WORKSPACE, f"RESULTS_{PRESET}.md")
        if code_ == 0 and res.exists():
            display(Markdown(res.read_text()))
        else:
            print(f"Scoring failed (exit code {code_}); scenes are kept; run `score` again.")
except Exception as exc:  # noqa: BLE001 - never end with an error before the files are saved
    print(f"Scoring step skipped: {exc!r}")
""")

md("""
## 5. Showcase clip (optional)

A mosaic of all cameras of the first successful scene, first 60 seconds, drawn from the stored
track points and the 720p copies of the videos. The scene's zip is unpacked into a temporary
folder for this (the zip itself stays untouched). The clip goes to `WORKSPACE/showcase/`. The
people in this dataset are synthetic. Skipped in smoke mode and when no scene succeeded.
""")
code("""
import shutil

try:
    zips = sorted(Path(WORKSPACE, "envs").glob(f"aicity24_*_{PRESET}.zip"))
    if SMOKE or not SHOWCASE or KEEP_VIDEOS == "none" or not zips:
        print("Showcase skipped (smoke mode, SHOWCASE False, no 720p copies or no scene).")
    else:
        tmp_ws = Path(RAW).parent / "showcase_ws"
        shutil.rmtree(tmp_ws, ignore_errors=True)
        sh(["reid", "env", "import", zips[0], "--workspace", tmp_ws])
        sh(["reid", "render", "--mosaic", "--start", 0, "--end", 60, "--data-dir",
            tmp_ws / zips[0].stem, "--out", Path(WORKSPACE) / "showcase"])  # fmt: skip
        shutil.rmtree(tmp_ws, ignore_errors=True)
except Exception as exc:  # noqa: BLE001
    print(f"Showcase skipped: {exc!r}")
""")

md("""
## 6. Summary
""")
code("""
try:
    try_sh(SUMMARY)
    sh(f"du -sh {WORKSPACE}; find {WORKSPACE} -type f | wc -l; ls {WORKSPACE}")
except Exception as exc:  # noqa: BLE001 - the last cell must not fail
    print(f"Summary unavailable: {exc!r}")
""")

md("""
## 7. Use the results

Everything is in `/kaggle/working/aicity24_bench` (the smoke run is in its `smoke/` subfolder):

| path | content |
|---|---|
| `envs/aicity24_<scene>_<preset>.zip` | one **environment** per scene, packed: `reid.db`, vectors, crops, `videos/` (720p), `gt.csv` (720p), `eval/` (our 2D metrics), `aicity/` (official predictions, official GT, calibration), `logs/`, `cache/run_<id>.json` (stage timings, GPU memory) |
| `results/aicity24_<scene>_<preset>/` | the small files that scoring needs, outside the zip: predictions, official GT, meta, evaluation JSON, run stats, process log, benchmark entry |
| `RESULTS_<preset>.md`, `official_<preset>.json` | the official scores, per scene and overall, with our 2D metrics and speed |
| `benchmark.json` | status (ok or failed with step and error), timings, fps, GPU, counts and commit per scene |
| `progress_gpu<i>.json`, `run_gpu<i>.log` | last ETA and the full log of each GPU process |
| `showcase/` | the optional mosaic clip |

To use it on your computer: download the version output (Output tab of the saved version) or
create a **New Dataset from the output**, extract it, then restore the environments and open the app:

```
reid env import aicity24_bench/envs/aicity24_075_balanced.zip --workspace envs_pc
reid env import aicity24_bench/envs/aicity24_078_balanced.zip --workspace envs_pc
reid app --workspace envs_pc
```

Each environment is self-contained (database, crops, 720p videos), so the app lists the scenes and
opens them without Kaggle. To score again or another preset later:
`reid benchmark aicity24 score --workspace aicity24_bench --preset <preset>` (it reads
`results/`, no unzipping needed).
""")


def write() -> None:
    nb = {
        "cells": [
            {**c, "id": f"c{i:02d}", "source": c["source"].splitlines(keepends=True)}
            for i, c in enumerate(cells)
        ],
        "metadata": {
            "kaggle": {
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {NB_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    write()
