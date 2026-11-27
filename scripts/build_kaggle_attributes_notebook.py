# ruff: noqa: E501  (the notebook cell texts below are content, kept as they appear in Kaggle)
"""Generate ``notebooks/kaggle_attributes.ipynb`` (outputs stripped).

The notebook trains the pedestrian attribute head (default hyperparameters) on Kaggle (T4 x2, or one T4):
fetch PA-100K and PETA, embed every image once with the frozen CLIP-ReID model, train the MLP head
on the saved embeddings, report. It contains no logic of its own: every step is a
``reid attributes ...`` command. Edit this script, run
``python scripts/build_kaggle_attributes_notebook.py`` and commit the notebook; never edit the
``.ipynb``. Same helpers and cell tags as ``build_kaggle_notebook.py`` (``settings``,
``kaggle-only``).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/AbderrahimZineddineEsi/open-world-multimodal-multicam-reid"
NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_attributes.ipynb"

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
# Pedestrian attribute head on Kaggle (T4 x2)

This notebook trains a small **attribute head**: given the 1280-D body embedding that the `reid`
pipeline already stores for every person, it predicts 67 attributes (sex, age group, hat, glasses,
clothing types and colours, shoes, bags, ...) without running another image model.

In plain words:

1. **Fetch** the two datasets: **PA-100K** (100,000 pedestrian images, 26 attributes) and **PETA**
   (19,000 images in 10 sub-datasets, 65 attributes), both without any manual step.
2. **Embed** every image **once** with the frozen CLIP-ReID network (the same model and
   preprocessing as the pipeline). Training and validation images also get a horizontally flipped
   copy; test images do not. The vectors are saved in small resumable shards: if the session dies,
   running the cell again continues where it stopped.
3. **Train** an MLP head on the saved vectors with a masked, weighted loss: an attribute that one
   dataset does not annotate is ignored for that dataset (PA-100K and PETA label different things).
   Default hyperparameters; no tuning here.
4. **Report**: mA and instance metrics per dataset and per attribute, a "UI candidate" list, and a
   curve showing how much averaging the prediction over several images of one person helps.

**Before you start** (Kaggle settings, right panel): *Accelerator* **GPU T4 x2** (the embedding
uses both GPUs; one T4 also works, just slower),
*Internet* **On** (it downloads the datasets, the model weights and installs the package).
Use **Save Version > Save & Run All (Commit)** so that `/kaggle/working` is saved.

**Licences**: PA-100K is CC-BY 4.0. **PETA is for research use only**; its **i-LID subset is
skipped** (it needs a separate application). The model weights are for non-commercial research.

**Outputs** in `/kaggle/working/attributes`: `emb/` (embeddings, fp16 shards:
reusable), `head/head.npz` (the trained head), `report/` (`RESULTS_attributes_default.md`,
`metrics_default.json`, `training_default.png`, `voting_default.png`).

**Reuse saved embeddings** (for example to retrain with other hyperparameters): add this notebook's
previous version output as an *input* of the new notebook and set `EMB_FROM` below to the folder that
holds `pa100k/` and `peta/` (for example `/kaggle/input/<notebook-name>/attributes/emb`). The fetch and
embed steps are then skipped.
""")

md("## 1. Settings")
code(
    """
import os

REID_REF = "main"  # branch or tag of the repository to install
PRESET = "balanced"  # body model preset ("balanced" = CLIP-ReID ViT-B/16)
WORK = "/kaggle/working/attributes"  # everything worth keeping (the version output)
DATA = "/kaggle/temp/attr_data"  # downloaded datasets: scratch space, NOT saved (~120k files)
EMB_FROM = ""  # folder with pa100k/ and peta/ embeddings of an earlier run ("" = embed now)

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
pip install --progress-bar off "reid[models] @ git+{REPO_URL}@${{REID_REF}}" \\
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
# Only the standard library, IPython and reid.notebook run in this kernel: the install upgraded
# numpy on disk while the kernel still holds the old one. Everything else is a `reid` command.
from reid.notebook import sh

sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv")
sh(f"mkdir -p {WORK} && df -h {WORK}")
sh(["reid", "info", "--preset", PRESET])
""",
    "kaggle-only",
)

md("""
## 3. Fetch the datasets

Downloads PA-100K (451 MB, from the authors' Google Drive) and PETA (236 MB, Dropbox) into
`DATA` (`/kaggle/temp`: scratch space, not saved with the version, because ~120k image files would
exceed Kaggle's output file limit), with resume, retries and a size check, then unzips. Files that are already complete
are skipped, so running the cell again is cheap. Skipped when `EMB_FROM` is set.
""")
code("""
if EMB_FROM:
    print("EMB_FROM is set: the datasets are not needed.")
else:
    sh(["reid", "attributes", "fetch", "--out", DATA])
""")

md("""
## 4. Embed every image once

Runs the frozen CLIP-ReID model over about 120,000 images (plus flipped copies of train and val)
and saves fp16 shards in `WORK/emb`. With **two GPUs** (T4 x2) it starts one process per GPU,
each embedding every other shard (`--part 0/2` and `--part 1/2`), then a short final pass checks that
every shard is there and marks the embeddings complete. With one GPU it is a single process. The
logs print `ETA |` lines with images per second and the time left. **Resumable**: if the session
stops, run this cell again. With `EMB_FROM` set, the earlier embeddings are linked instead.
""")
code("""
if EMB_FROM:
    if not os.path.exists(f"{WORK}/emb"):
        os.symlink(EMB_FROM, f"{WORK}/emb")
    print(f"Using the embeddings in {EMB_FROM}")
else:
    from reid.notebook import Job, run_parallel

    embed_cmd = ["reid", "attributes", "embed", "--data", DATA, "--out", f"{WORK}/emb",
                 "--preset", PRESET]  # fmt: skip
    n_gpus = len(os.popen("nvidia-smi -L").read().strip().splitlines())
    if n_gpus >= 2:
        workers = str(max(1, (os.cpu_count() or 2) // n_gpus))  # share the CPUs (decoding)
        run_parallel(
            [
                Job(f"gpu{i}", [*embed_cmd, "--part", f"{i}/{n_gpus}", "--workers", workers],
                    f"{WORK}/logs/embed_gpu{i}.log", {"CUDA_VISIBLE_DEVICES": str(i)})
                for i in range(n_gpus)
            ],
            marker="ETA |",
        )  # fmt: skip
    sh(embed_cmd)  # one GPU: the whole job; two GPUs: verify every shard and mark complete
""")

md("""
## 5. Train the head

Default hyperparameters (hidden 512, dropout 0.3, AdamW lr 1e-3, weight decay 1e-4, batch 512,
up to 40 epochs, early stopping on validation mA). The test split is not read here.
""")
code("""
sh(["reid", "attributes", "train", "--emb", f"{WORK}/emb", "--out", f"{WORK}/head"])
""")

md("""
## 6. Report

Evaluates on the test splits: PA-100K on its 26 original attributes (comparable with published
numbers) and PETA on its own attributes with a person-disjoint split made here (50 / 10 / 40 % of
the persons, seed 0, so indicative only).
""")
code("""
from IPython.display import Image, Markdown, display

sh(["reid", "attributes", "report", "--emb", f"{WORK}/emb", "--head", f"{WORK}/head",
    "--out", f"{WORK}/report", "--tag", "default"])  # fmt: skip
display(Markdown(open(f"{WORK}/report/RESULTS_attributes_default.md").read()))
for name in ("training_default.png", "voting_default.png"):
    if os.path.exists(f"{WORK}/report/{name}"):
        display(Image(filename=f"{WORK}/report/{name}"))
""")

md("""
## 7. Output files
""")
code("""
sh(f"du -sh {WORK}/*; ls -la {WORK}/head {WORK}/report")
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
