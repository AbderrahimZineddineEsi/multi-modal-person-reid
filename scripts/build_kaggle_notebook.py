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

REPO_URL = "https://github.com/AbderrahimZineddineEsi/multi-modal-person-reid"
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

**Time**: about 3 to 3.5 hours of GPU for 4 scenes x 16 cameras on two T4s. This is an estimate
from the speed measured on one Colab T4 (34 fps of processed frames at 720p, frame stride 2), not
a measurement of this notebook end to end. A Kaggle session is limited to 12 hours.

Model weights are downloaded at run time and are licensed for non-commercial research use (see
the README).
""")

md("## 1. Settings")
code(
    """
import os

REID_REF = "kaggle-benchmark"  # branch or tag of the repository to install
PRESET = "balanced"  # "lite" | "balanced" | "accurate" | "max"
SCENES = ["075", "078", "082", "086"]  # test scenes 071-090 have 16 cameras (089 has 15)
FRAME_STRIDE = 2  # process every 2nd frame (the measured 34 fps used this)
KEEP_VIDEOS = "720"  # "720": keep 720p copies of the videos | "none"
SHOWCASE = True  # render a 60 s mosaic clip of the first scene at the end

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
## 3. Process the scenes

The scenes are split across the visible GPUs (balanced by camera count) and one
`reid benchmark aicity24 run` process runs per GPU, each pinned with `CUDA_VISIBLE_DEVICES`. Each
process handles its scenes one after the other: download (16 videos of about 185 MB), process,
2D metrics, export, then the 720p copies are encoded in the background while the next scene is
already on the GPU, and the downloaded originals are deleted. Every process writes its own log to
`WORKSPACE/run_gpu<i>.log`; this cell prints the end of each log every minute and stops with an
error if one of them fails.

The step is **resumable**: if the session stops, run the notebook again and finished scenes are
skipped (an unfinished scene is redone from its first missing step).
""")
code("""
import subprocess

from reid.notebook import Job, balance, run_parallel

CAMERAS = {s: 15 if s == "089" else 16 for s in SCENES}
listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
n_gpu = max(1, sum(ln.startswith("GPU ") for ln in listing.splitlines()))
groups = balance(SCENES, CAMERAS, n_gpu)
print(f"{n_gpu} GPU(s); scenes per GPU: {groups}")


def job(i, scenes):
    cmd = ["reid", "benchmark", "aicity24", "run", "--scenes", ",".join(scenes), "--preset", PRESET,
           "--workspace", WORKSPACE, "--raw-dir", f"{RAW}/gpu{i}", "--keep-videos", KEEP_VIDEOS,
           "--frame-stride", FRAME_STRIDE]  # fmt: skip
    env = {"CUDA_VISIBLE_DEVICES": str(i)} if n_gpu > 1 else {}
    return Job(f"gpu{i}", cmd, f"{WORKSPACE}/run_gpu{i}.log", env)


jobs = [job(i, g) for i, g in enumerate(groups)]
run_parallel(jobs, poll_s=60)
""")

md("""
## 4. Score with the official evaluator

`reid benchmark aicity24 score` downloads NVIDIA's evaluator code (not the 2.8 GB sample files),
runs it on all scenes together and on each scene alone, and writes `official_<preset>.json` and
`RESULTS_<preset>.md` into the workspace. The official score is the mean over the scenes of HOTA
computed on floor positions (matching distance up to 2 m).
""")
code("""
from pathlib import Path

from IPython.display import Markdown, display

sh(["reid", "benchmark", "aicity24", "score", "--workspace", WORKSPACE, "--preset", PRESET,
    "--num-cores", os.cpu_count() or 4])  # fmt: skip
display(Markdown(Path(WORKSPACE, f"RESULTS_{PRESET}.md").read_text()))
""")

md("""
## 5. Showcase clip (optional)

A mosaic of all cameras of the first scene, first 60 seconds, drawn from the stored track points
and the 720p copies of the videos (so it needs `KEEP_VIDEOS = "720"`). It is written to
`WORKSPACE/showcase/`. The people in this dataset are synthetic.
""")
code("""
if SHOWCASE and KEEP_VIDEOS != "none":
    env = Path(WORKSPACE) / f"aicity24_{SCENES[0]}_{PRESET}"
    sh(["reid", "render", "--mosaic", "--start", 0, "--end", 60, "--data-dir", env,
        "--out", Path(WORKSPACE) / "showcase"])  # fmt: skip
else:
    print("Showcase skipped (SHOWCASE is False or KEEP_VIDEOS is none).")
sh(f"du -sh {WORKSPACE} && ls {WORKSPACE}")
""")

md("""
## 6. Use the results

Everything is in `/kaggle/working/aicity24_bench`:

| path | content |
|---|---|
| `aicity24_<scene>_<preset>/` | one **environment** per scene: `reid.db`, vectors, crops, `videos/` (720p), `gt.csv` (720p), `eval/` (our 2D metrics), `aicity/` (official predictions, official GT, calibration), `logs/process.log`, `cache/run_<id>.json` (stage timings, GPU memory) |
| `RESULTS_<preset>.md`, `official_<preset>.json` | the official scores, per scene and overall, with our 2D metrics and speed |
| `benchmark.json` | timings, fps, GPU, counts and commit per scene |
| `run_gpu<i>.log` | full logs of the run |
| `showcase/` | the optional mosaic clip |

To use it on your computer: download the version output (Output tab of the saved version) or
create a **New Dataset from the output**, extract it, and run

```
reid app --workspace aicity24_bench
```

Each environment is self-contained (database, crops, 720p videos), so the app lists the scenes and
opens them without Kaggle. To score again or another preset later: `reid benchmark aicity24 score
--workspace aicity24_bench --preset <preset>`.
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
