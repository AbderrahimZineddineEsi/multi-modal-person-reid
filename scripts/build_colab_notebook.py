# ruff: noqa: E501  (the notebook cell texts below are content, kept as they appear in Colab)
"""Generate the two Colab notebooks (outputs stripped).

- ``notebooks/colab_demo.ipynb``: the LAUNCHER. Installs the package, mounts Drive and starts the
  app on ``DATA_DIR/environments``; footage is added from the app's Process tab (upload or URL).
- ``notebooks/colab_walkthrough.ipynb``: optional, reproduces the AI City benchmark step by step.

The notebook is generated so that its structure stays reviewable in one place. Edit this script,
then run ``python scripts/build_colab_notebook.py`` and commit both files.

Cell tags used by ``scripts/test_notebook.py``:
- ``colab-only``: skipped when the notebook is executed locally (pip installs, the app launch);
- ``settings``: the user settings cell; the local test injects its overrides right after it.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/AbderrahimZineddineEsi/open-world-multimodal-multicam-reid"
COLAB_URL = (
    "https://colab.research.google.com/github/AbderrahimZineddineEsi/open-world-multimodal-multicam-reid"
    "/blob/main/notebooks/colab_demo.ipynb"
)
NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
WALK_URL = COLAB_URL.replace("colab_demo", "colab_walkthrough")

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


md(f"""
# Optional: reproduce the benchmark (Colab walkthrough)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({WALK_URL})

**You do not need this notebook to use the platform**: `colab_demo.ipynb` only launches the app, and
you add footage from its Process tab. This one re-runs the whole AI City benchmark step by step.

This notebook installs the `reid` package and drives it through its command line. It contains no
pipeline logic of its own: every step is a `reid ...` command or a call to a documented package
function, so what you see here is what you get from a terminal.

What it does, in order:

1. checks the GPU and installs the package;
2. reads your settings (the **only cell you need to edit**);
3. mounts Google Drive (all results live under one `DATA_DIR`, so they survive the runtime);
4. gets footage: a labelled AI City 2024 scene, your own videos, or virtual cameras cut from one
   video;
5. processes the cameras into one shared identity space;
6. evaluates against the ground truth (AI City only);
7. renders a mosaic of all cameras and a clip that follows one person across cameras, and
   searches by image;
8. exports the environment as a zip for your own PC (`reid env export`);
9. launches the Gradio app.

Use a **T4 GPU runtime** (Runtime > Change runtime type), then **Runtime > Run all**. Every step
stops the notebook with a clear error if it fails, so the first red cell is the one to look at.
Face and body weights are downloaded at runtime and are licensed for non-commercial research use
(see the README).
""")

md("""
## 1. GPU and installation

First check that a GPU is attached. The `balanced` preset needs about 1 GB of VRAM and runs on a
free T4.
""")
code("!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", "colab-only")

md("""
Set the branch or tag to install with `REID_REF`.

The install keeps Colab's preinstalled PyTorch and takes about 2-4 minutes. Its log is written to
`/content/pip_install.log`; the cell prints the last lines if something fails. Warnings such as
*"pip's dependency resolver does not currently take into account all the packages..."* about
packages this project does not use (for example `bigframes`) are expected and harmless.
""")
code(
    """
import os

REID_REF = "main"  # branch or tag of the repository to install
os.environ["REID_REF"] = REID_REF
""",
    "colab-only",
)
code(
    f"""
%%bash
set -o pipefail
python -c "import torch; print('preinstalled torch', torch.__version__, '| CUDA', torch.version.cuda)"
pip install --progress-bar off "reid[models,app,eval] @ git+{REPO_URL}@${{REID_REF}}" \\
  > /content/pip_install.log 2>&1 || {{ tail -n 40 /content/pip_install.log; exit 1; }}
tail -n 3 /content/pip_install.log
python -c "import torch; print('torch after install', torch.__version__)"
""",
    "colab-only",
)
md("""
### Fixing the OpenCV / ONNX Runtime clash

`ultralytics` pulls in `opencv-python` and `insightface` pulls in `onnxruntime`. They ship the same
Python modules as `opencv-python-headless` and `onnxruntime-gpu`, which `reid` needs, so pip can
leave a broken mix of files behind. The cell below removes the clashing wheels and force-reinstalls
the two we want (without touching their dependencies). It is harmless if nothing clashed.

If a later cell fails with an `ImportError` mentioning `numpy` or `cv2`, use **Runtime > Restart
session** and run the notebook again from the top (Drive stays mounted, nothing is downloaded
twice).
""")
code(
    """
%%bash
pip uninstall -y -q opencv-python onnxruntime
pip install -q --force-reinstall --no-deps opencv-python-headless "onnxruntime-gpu==1.32.0"
python -c "import cv2, onnxruntime; print('opencv', cv2.__version__, '| onnxruntime', onnxruntime.__version__)"
""",
    "colab-only",
)

md("""
## 2. Settings

This is the only cell you need to edit. `SOURCE` picks the footage:

| `SOURCE` | Footage | Evaluation |
|---|---|---|
| `"aicity"` | AI City 2024 retail scene 075, 4 overlapping cameras, downloaded automatically | yes (ground truth) |
| `"own"` | your videos: put them in `MyDrive/reid_data/videos/` (every video there is one camera, in name order) | no |
| `"virtual"` | several "cameras" cut from one video, `MyDrive/reid_data/videos/store.mp4` by default | no |

Only the chosen source runs, so **Run all** is safe. Each source and preset is its own
**environment** (`MyDrive/reid_data/environments/<source>_<preset>`) with its own cameras and
people, so they never mix; the app lets you switch between them, create new ones and delete them.
Opening the notebook again later reuses a finished environment instead of processing again, unless
`REPROCESS = True`.
""")
code(
    """
from reid.notebook import has_finished_run, sh

SOURCE = "aicity"  # "aicity" | "own" | "virtual"
PRESET = "balanced"  # "lite" | "balanced" | "accurate" | "max"
STRIDE = 2  # process every STRIDE-th frame (2 halves the time; 1 = full frame rate)
REPROCESS = False  # True: process again even if this source/preset already has a finished run
EXPORT_WITH_VIDEOS = False  # True: the export zip also bundles the camera videos (can be GBs)

AICITY_MINUTES = 2  # (aicity) minutes of footage to keep
OWN_VIDEOS_DIR = "videos"  # (own) folder under DATA_DIR; every video in it is one camera
FACE_CAMS = []  # (own) camera ids with a clear view of faces, e.g. ["cam_01"]
VIRTUAL_VIDEO = "videos/store.mp4"  # (virtual) the single source video, under DATA_DIR
VIRTUAL_END = 120  # (virtual) seconds of the video to use

DRIVE_DATA_DIR = "/content/drive/MyDrive/reid_data"
LOCAL_DATA_DIR = "/content/reid_data"  # used if Drive is not mounted (lost with the runtime)

assert SOURCE in ("aicity", "own", "virtual"), f"unknown SOURCE {SOURCE!r}"
sh(["reid", "--help"])
""",
    "settings",
)

md("""
## 3. Where to keep the data

Everything (database, vectors, crops, renders, evaluation results, downloaded datasets) goes under
`DATA_DIR`. On Colab that should be a Drive folder so a disconnected runtime does not lose a
processing run. If you decline the Drive prompt, the notebook falls back to local storage with a
warning.
""")
code("""
from pathlib import Path

try:
    from google.colab import drive

    drive.mount("/content/drive")
    DATA_DIR = Path(DRIVE_DATA_DIR)
except Exception as exc:  # not on Colab, or the prompt was declined
    DATA_DIR = Path(LOCAL_DATA_DIR)
    print(f"WARNING: Drive not mounted ({exc!r}).")
    print(f"Using {DATA_DIR}: on Colab everything is lost when this runtime ends.")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("DATA_DIR =", DATA_DIR)
""")

md("""
## 4. Get the footage

The three cells below correspond to the three sources; only the one matching `SOURCE` does
anything. Each one writes `CAMERAS_YAML`, the camera set that the next steps use. A camera set
lists the videos; all cameras of a set are processed into one shared identity space, and
`start_ts` aligns them in time (0 on every camera means synchronized).

### (a) AI City 2024 scene with ground truth

`reid dataset fetch` downloads one retail scene (075) from the NVIDIA `PhysicalAI-SmartSpaces`
dataset on Hugging Face (CC-BY-4.0, synthetic people, no login needed). It picks 4 cameras whose
views overlap, keeps the first minutes, downscales to 720p, converts the ground truth to a uniform
format and writes the camera YAML with all cameras synchronized and no face cameras. Only the
ground truth file (about 50 MB) and the selected cameras are fetched; `--stream` makes ffmpeg read
just the first minutes instead of the whole videos. A scene already on Drive is not downloaded
again.
""")
code("""
import json

GT_CSV = None  # set by the AI City source only
if SOURCE == "aicity":
    AICITY_DIR = DATA_DIR / "datasets" / "aicity24_075"
    CAMERAS_YAML = DATA_DIR / "cameras_aicity24_075.yaml"
    GT_CSV = AICITY_DIR / "gt.csv"
    meta = AICITY_DIR / "meta.json"
    fetched = meta.exists() and CAMERAS_YAML.exists() and GT_CSV.exists()
    if fetched and json.loads(meta.read_text()).get("minutes") == AICITY_MINUTES:
        print(f"Already downloaded: {AICITY_DIR}")
    else:
        sh(["reid", "dataset", "fetch", "aicity24", "--scene", "075", "--cameras", 4,
            "--minutes", AICITY_MINUTES, "--height", 720, "--stream",
            "--out", AICITY_DIR, "--yaml", CAMERAS_YAML])  # fmt: skip
else:
    print(f"SOURCE is {SOURCE!r}: skipping the AI City download.")
""")

md("""
### (b) Your own videos from Drive

Every video in `DATA_DIR/OWN_VIDEOS_DIR` becomes one camera (`cam_01`, `cam_02`, ... in name
order). Put the videos of one place and one time span together: the more the views overlap in time
and space, the more the identity association has to work with. Cameras with a clear view of faces
go in `FACE_CAMS`. All cameras are assumed synchronized; edit `start_ts` per camera in the YAML if
they are not.
""")
code("""
from reid.datasets import write_camera_yaml

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
if SOURCE == "own":
    videos_dir = DATA_DIR / OWN_VIDEOS_DIR
    VIDEOS = sorted(p for p in videos_dir.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not VIDEOS:
        raise FileNotFoundError(f"no videos in {videos_dir}: upload them there first")
    CAMERAS_YAML = write_camera_yaml(DATA_DIR / "cameras_own.yaml", VIDEOS, face_cams=FACE_CAMS)
    print(CAMERAS_YAML.read_text())
else:
    print(f"SOURCE is {SOURCE!r}: skipping your own videos.")
""")

md("""
### (c) Virtual cameras from a single video

If you only have one video, `reid virtual-cams` derives several "cameras" from it (overlapping
crops, optional mirror and time offset) and writes the camera YAML. This exercises the whole
pipeline, but it does **not** show that the system generalizes across real viewpoints: the views
share lighting and the overlap is the same pixels.
""")
code("""
if SOURCE == "virtual":
    video = DATA_DIR / VIRTUAL_VIDEO
    if not video.exists():
        raise FileNotFoundError(f"{video} not found: upload the video there or edit VIRTUAL_VIDEO")
    VIRTUAL_DIR = DATA_DIR / "virtual_cams"
    CAMERAS_YAML = VIRTUAL_DIR / "cameras.yaml"
    sh(["reid", "virtual-cams", video, "--out-dir", VIRTUAL_DIR, "--n", 3, "--overlap", 0.5,
        "--mirror", 2, "--width", 960, "--end", VIRTUAL_END, "--yaml", CAMERAS_YAML])  # fmt: skip
else:
    print(f"SOURCE is {SOURCE!r}: skipping the virtual cameras.")
""")

md("""
## 5. Process the cameras

`reid process` detects and tracks people in every camera, builds tracklets (quality-gated body
samples, faces on face cameras), and associates the tracklets into global identities. The
`balanced` preset is the default for a T4: YOLO26m-pose detector, BoT-SORT tracker, CLIP-ReID body
model, buffalo_l faces. The first run also downloads the model weights.

Each source and preset gets its own environment folder `RUN_DIR`, so runs do not mix. If it
already holds a finished run, it is reused (set `REPROCESS = True` to process again; the new run is
added to the same environment, so known people keep their ids). A folder left by an interrupted run
(a crash or a disconnected runtime) is removed and processed from scratch. The output ends with the
GPU memory the run really needed (whole process: CUDA context, PyTorch and the ONNX face models)
and every command prints how long it took.
""")
code("""
import shutil

ENVIRONMENTS = DATA_DIR / "environments"  # one folder per environment; the app's workspace
ENV = f"{SOURCE}_{PRESET}"
RUN_DIR = ENVIRONMENTS / ENV
COMMON = ["--preset", PRESET, "--data-dir", RUN_DIR]  # passed to every command below

if has_finished_run(RUN_DIR) and not REPROCESS:
    print(f"{RUN_DIR} already has a finished run: reusing it (set REPROCESS = True to redo it).")
else:
    if RUN_DIR.exists() and not has_finished_run(RUN_DIR):
        print(f"Removing {RUN_DIR}: it only holds an interrupted run.")
        shutil.rmtree(RUN_DIR)
    sh(["reid", "process", "--cameras", CAMERAS_YAML, *COMMON,
        "--set", f"runtime.frame_stride={STRIDE}"])  # fmt: skip
""")
code("""sh(["reid", "info", *COMMON])""")

md("""
## 6. Evaluate (AI City only)

`reid evaluate` compares the run with the ground truth: per-camera tracking metrics, the combined
cross-camera IDF1, and held-out retrieval accuracy. The results are also saved under `RUN_DIR` and
shown in the app's Metrics tab.
""")
code("""
if GT_CSV is not None:
    sh(["reid", "evaluate", *COMMON, "--gt", GT_CSV])
else:
    print("No ground truth for this source: skipping the evaluation.")
""")

md("""
## 7. Render, follow one person, search

`reid render --mosaic` draws boxes and global ids on every camera, tiled into one mosaic video\n(`--camera all` would also write one video per camera).
`reid journey` cuts a clip that follows one identity from camera to camera; here it is the identity
seen in the most tracklets.
""")
code("""sh(["reid", "render", "--mosaic", "--scale", 0.5, *COMMON])""")
code("""
from reid.storage import db, files

conn = db.connect(files.db_path(RUN_DIR))
identities = db.list_identities(conn)
conn.close()
if not identities:
    raise RuntimeError(f"no identities in {RUN_DIR}: the processing step found nobody")
GID = max(identities, key=lambda i: i.n_tracklets).gid
print(f"{len(identities)} identities; following identity {GID}")
sh(["reid", "journey", "--gid", GID, *COMMON])
""")
code("""
from IPython.display import Video, display


def show(path, width=800):
    path = Path(path)
    if path.stat().st_size > 30e6:  # embed small files only; large ones would bloat the notebook
        size_mb = path.stat().st_size / 1e6
        print(f"{path} is {size_mb:.0f} MB: too large to embed; open it on Drive.")
        return
    display(Video(str(path), embed=True, width=width))


show(RUN_DIR / "renders" / "mosaic.mp4")
show(RUN_DIR / "renders" / f"journey_gid{GID}.mp4")
""")

md("""
`reid search` embeds a query image and looks it up in the identity space (body, face, or both; a
face match wins). Here the query is the best saved crop of the identity followed above, which is a
circular test meant only to show the output; use any photo of a person instead. The status is
`MATCH`, `AMBIGUOUS` (two identities too close) or `UNKNOWN`. The query is always embedded
with the models the environment was processed with, whatever `--preset` says.
""")
code("""
from IPython.display import Image

conn = db.connect(files.db_path(RUN_DIR))
crop = next(t.best_crop_path for t in db.get_tracklets(conn, gid=GID) if t.best_crop_path)
conn.close()
QUERY = RUN_DIR / crop

display(Image(str(QUERY), width=120))
sh(["reid", "search", "--image", QUERY, "--mode", "both", *COMMON])
""")

md("""
## 8. Use this environment on your PC

`reid env export` packs this environment (database, vectors, crops, faces, evaluation results;
not the renders, which can be regenerated) into one zip on Drive. Browsing, search, metrics and
admin then work on your own machine without a GPU. Set `EXPORT_WITH_VIDEOS = True` in the
settings cell to also bundle the camera videos (needed on the PC for clips and heatmaps; it can
add gigabytes). The zip's size is printed. For a continuous two-way link use `reid sync` instead (see `docs/local_app.md`).
""")
code("""
EXPORT_ZIP = DATA_DIR / "exports" / f"{ENV}.zip"
sh(["reid", "env", "export", "--data-dir", RUN_DIR, "--out", EXPORT_ZIP,
    *(["--with-videos"] if EXPORT_WITH_VIDEOS else [])])  # fmt: skip
print(
    f\"\"\"
On your PC (after downloading the zip from Drive in the browser):

  reid env import {EXPORT_ZIP.name} --workspace data/environments
  reid app --workspace data/environments --env {ENV}
\"\"\"
)
""")

md("""
## 9. Launch the app

The Gradio app has tabs for processing, exploring identities and journeys, searching, per-camera
views and metrics, plus an environment bar to switch between environments (or create one, e.g.
to process uploaded videos separately) and an Admin tab to merge, split, rename or delete people,
delete runs, or empty / delete an environment. `--share` opens a public `gradio.live` link, so it
requires a login: the cell generates a password and prints it (add `--read-only` to hide the
delete tools). The cell keeps running while the app is up; stop it with the stop button.
""")
code(
    """
import secrets

PASSWORD = secrets.token_urlsafe(9)
print(f"Login: user 'reid', password '{PASSWORD}'")
sh(["reid", "app", "--share", "--auth", f"reid:{PASSWORD}", "--preset", PRESET,
    "--workspace", ENVIRONMENTS, "--env", ENV])  # fmt: skip
""",
    "colab-only",
)


walkthrough = cells
cells = []

md(f"""
# Multi-camera person Re-ID: launch the platform

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

This notebook only **starts the app**. It does not process or evaluate anything. Run all cells
(**Runtime > Run all**, T4 GPU runtime), open the printed link, log in with the printed password,
then use the **Process** tab to upload a video or paste a video URL.

Your work lives on Google Drive (`MyDrive/reid_data/environments/<name>`, one folder per
environment), so it survives the runtime and can be synced with your PC: on the PC run
`reid sync push` before and `reid sync pull` after (see `docs/local_app.md`).
The benchmark walkthrough is in `colab_walkthrough.ipynb`.
""")
md("## 1. GPU and installation")
code("!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", "colab-only")
code(
    """
import os

REID_REF = "main"  # branch or tag of the repository to install
os.environ["REID_REF"] = REID_REF
""",
    "colab-only",
)
code(
    f"""
%%bash
set -o pipefail
python -c "import torch; print('preinstalled torch', torch.__version__, '| CUDA', torch.version.cuda)"
pip install --progress-bar off "reid[models,app,eval] @ git+{REPO_URL}@${{REID_REF}}" \\
  > /content/pip_install.log 2>&1 || {{ tail -n 40 /content/pip_install.log; exit 1; }}
tail -n 3 /content/pip_install.log
python -c "import torch; print('torch after install', torch.__version__)"
""",
    "colab-only",
)
md("""
`ultralytics` pulls in `opencv-python` and `insightface` pulls in `onnxruntime`, which clash with
`opencv-python-headless` and `onnxruntime-gpu`. The cell removes the clashing wheels and
reinstalls the two we want (harmless if nothing clashed). If a later cell fails with an
`ImportError` about `numpy` or `cv2`, use **Runtime > Restart session** and run again.
""")
code(
    """
%%bash
pip uninstall -y -q opencv-python onnxruntime
pip install -q --force-reinstall --no-deps opencv-python-headless "onnxruntime-gpu==1.32.0"
python -c "import cv2, onnxruntime; print('opencv', cv2.__version__, '| onnxruntime', onnxruntime.__version__)"
""",
    "colab-only",
)

md("""
## 2. Settings and Drive

`PRESET` is the default model preset of the Process tab (`lite`, `balanced`, `accurate`, `max`).
`ENV` is the environment opened first; the app's top bar creates, switches and deletes others.
""")
code(
    """
from reid.notebook import sh

PRESET = "auto"  # auto = from the GPU (balanced on a T4) | lite | balanced | accurate | max
ENV = "default"  # environment opened at start (created if missing)
DRIVE_DATA_DIR = "/content/drive/MyDrive/reid_data"
LOCAL_DATA_DIR = "/content/reid_data"  # used if Drive is not mounted (lost with the runtime)
""",
    "settings",
)
code("""
from pathlib import Path

try:
    from google.colab import drive

    drive.mount("/content/drive")
    DATA_DIR = Path(DRIVE_DATA_DIR)
except Exception as exc:  # not on Colab, or the prompt was declined
    DATA_DIR = Path(LOCAL_DATA_DIR)
    print(f"WARNING: Drive not mounted ({exc!r}).")
    print(f"Using {DATA_DIR}: on Colab everything is lost when this runtime ends.")
WORKSPACE = DATA_DIR / "environments"
WORKSPACE.mkdir(parents=True, exist_ok=True)
print("workspace =", WORKSPACE)
sh(["reid", "env", "list", "--workspace", WORKSPACE])
""")

md("""
## 3. Launch the app

`--share` opens a public `gradio.live` link, so a login is required: the cell generates a
password and prints it (add `--read-only` to hide the delete tools). The cell keeps running while
the app is up; stop it with the stop button. Before syncing to your PC, stop it so the database is
not being written.
""")
code(
    """
import secrets

PASSWORD = secrets.token_urlsafe(9)
print(f"Login: user 'reid', password '{PASSWORD}'")
sh(["reid", "app", "--share", "--auth", f"reid:{PASSWORD}", "--preset", PRESET,
    "--workspace", WORKSPACE, "--env", ENV])  # fmt: skip
""",
    "colab-only",
)
launcher = cells


def write(name: str, cs: list[dict]) -> None:
    nb = {
        "cells": [
            {**c, "id": f"c{i:02d}", "source": c["source"].splitlines(keepends=True)}
            for i, c in enumerate(cs)
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = NB_DIR / name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(cs)} cells)")


def main() -> None:
    write("colab_demo.ipynb", launcher)
    write("colab_walkthrough.ipynb", walkthrough)


if __name__ == "__main__":
    main()
