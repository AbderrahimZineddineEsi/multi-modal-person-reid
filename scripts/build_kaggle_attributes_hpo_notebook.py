# ruff: noqa: E501  (the notebook cell texts below are content, kept as they appear in Kaggle)
"""Generate ``notebooks/kaggle_attributes_hpo.ipynb`` (outputs stripped).

The notebook tunes the hyperparameters of the pedestrian attribute head with Optuna, on the
embeddings saved by ``kaggle_attributes.ipynb`` (no re-embedding): one ``reid attributes hpo``
process per GPU sharing one study, then ``reid attributes hpo-final``. It contains no logic of
its own beyond locating the embeddings: every step is a ``reid`` command. Edit this script, run
``python scripts/build_kaggle_attributes_hpo_notebook.py`` and commit the notebook; never edit
the ``.ipynb``. Same helpers and cell tags as ``build_kaggle_attributes_notebook.py``
(``settings``, ``kaggle-only``).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/AbderrahimZineddineEsi/open-world-multimodal-multicam-reid"
NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_attributes_hpo.ipynb"

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
# Tuning the attribute head with Optuna (Kaggle, T4 x2)

The previous notebook trained the attribute head once, with hyperparameters chosen by hand
(hidden size 512, dropout 0.3, learning rate 1e-3, ...). This notebook **searches** for better
ones and measures honestly whether the search helped.

**What is hyperparameter optimisation (HPO)?** The weights of the head are learned by gradient
descent, but a few numbers are not: the size of the hidden layer, the dropout, the learning rate,
the weight decay, ... These are *hyperparameters*. HPO treats "train the head with these numbers,
look at the validation score" as a black-box function and tries to find the numbers that maximise
it. Each try is a **trial**.

**Why is it cheap here?** The CLIP-ReID backbone is frozen and every image was embedded once, so a
trial trains only a small MLP on saved vectors: about **20 seconds** on a T4 (the expensive part,
embedding 220,000 crops, is already done). In 25 minutes per GPU we can afford a hundred trials.

**What does TPE do?** The sampler is the Tree-structured Parzen Estimator. The first trials are
random. After that it splits the finished trials into a "good" group (best scores) and a "bad"
group, models where each group lives in the search space, and samples new parameters where
"good" is likely compared with "bad". So the search concentrates on promising regions instead of
trying values blindly (a grid would waste most trials on bad corners).

**What does pruning do?** A trial reports its validation score after every epoch. The **median
pruner** stops a trial whose score at some epoch is below the median of the earlier trials at the
same epoch (it ignores the first 8 trials and the first 3 epochs). Bad configurations are abandoned
after a few epochs, which leaves time for more trials.

**Why tune on validation and read test only once?** Every trial is picked because it scored well on
the validation split, so the best validation score is slightly optimistic (we selected the maximum
of many noisy numbers). If we also picked on test, the final test number would be optimistic too.
So the search sees **validation only**; `hpo-final` is the only step that reads the **test** split,
once, for two fixed configurations (the default and the best).

**Why several seeds?** Training has randomness (initial weights, batch order, dropout). Two runs of
the same configuration differ a little. `hpo-final` trains the default and the best configuration
with several seeds and reports mean +- standard deviation: **a gain smaller than the seed-to-seed
spread is noise**, not a result.

**What is searched** (epochs 40 and patience 8 are fixed):

- `hidden` (256 to 2048): width of the hidden layer, i.e. the capacity of the head.
- `dropout` (0 to 0.7): probability of zeroing hidden units while training; regularises (the default head overfits after epoch 6).
- `lr` (1e-4 to 5e-3, log scale): AdamW step size.
- `wd` (1e-6 to 1e-1, log scale): weight decay, the other regulariser.
- `batch` (256, 512, 1024): rows per gradient step.
- `peta_weight` (0.5 to 8, log scale): loss multiplier of PETA rows; PA-100K has about 9 times more rows, so this balances the two datasets.
- `weighting` (`ratio` or `none`): per-attribute positive / negative loss weights `exp(1-r)` / `exp(r)` from the positive ratio `r`, or all weights 1.
- `use_flips` (yes / no): also train on the horizontally flipped copies of the training images.
- `train_view` (yes / no): also train the three viewpoint labels (front / side / back).

**The viewpoint ablation.** Viewpoint is not a useful attribute to show, but learning it may help
(or hurt) the other attributes through the shared hidden layer. Because trials with and without
view labels must be comparable, the validation score and the final test numbers are computed on the
labels **without** `view_*`. `hpo-final` trains the best configuration with and without view labels
(several seeds each) and reports the difference.

**Before you start** (Kaggle settings, right panel): *Accelerator* **GPU T4 x2** (one process per
GPU shares the same study; one T4 also works), *Internet* **On** (installs the package), and
**add the previous notebook's output as an input**: *Add Input -> Your Work -> the attribute
notebook* (its `attributes/emb` folder holds the embeddings). Use **Save Version > Save & Run All
(Commit)** so that `/kaggle/working` is saved.

**Outputs** in `/kaggle/working/attributes_hpo`: `RESULTS_hpo.md`, `optimization_history.png`,
`param_importances.png`, `trials.csv`, `optuna_journal.log` (the study), `head_tuned/head.npz`
(best configuration, seed 0) and `report_tuned/` (the usual per-label report of that head).
""")

md("## 1. Settings")
code(
    """
import os

REID_REF = "main"  # branch or tag of the repository to install
WORK = "/kaggle/working/attributes_hpo"  # everything worth keeping (the version output)
TIMEOUT_MIN = 25  # minutes of search PER GPU process (the real limit)
TRIALS_PER_GPU = 400  # upper bound per process; the timeout stops the search first
SEEDS = 3  # seeds for the default vs best comparison and the viewpoint ablation
EMB = ""  # folder with pa100k/ and peta/ embeddings ("" = find it under /kaggle/input)

os.environ["REID_REF"] = REID_REF
""",
    "settings",
)

md("""
## 2. Install and check the machine

Only the `train` extra is installed (Optuna and pandas): PyTorch is preinstalled on Kaggle and
nothing here needs the detection, tracking or face models. The install log goes to
`/tmp/pip_install.log`; the cell prints its last lines if something fails.
""")
code(
    f"""
%%bash
set -o pipefail
pip install --progress-bar off "reid[train] @ git+{REPO_URL}@${{REID_REF}}" \\
  > /tmp/pip_install.log 2>&1 || {{ tail -n 40 /tmp/pip_install.log; exit 1; }}
tail -n 3 /tmp/pip_install.log
""",
    "kaggle-only",
)
code(
    """
# Only the standard library, IPython and reid.notebook run in this kernel: the install may have
# changed numpy on disk while the kernel still holds the old one. Everything else is a command.
from reid.notebook import Job, run_parallel, sh

check = (
    "import torch, optuna; print('torch', torch.__version__, '| CUDA', torch.version.cuda, "
    "'| optuna', optuna.__version__, '| GPUs', torch.cuda.device_count())"
)
sh(["python", "-c", check])
sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv")
sh(f"mkdir -p {WORK} && df -h {WORK}")
""",
    "kaggle-only",
)

md("""
## 3. Find the embeddings

Looks for the first folder under `/kaggle/input` that contains both `pa100k/train/index.json` and
`peta/train/index.json` (the embedding folder written by the previous notebook), following
symbolic links (Kaggle mounts inputs in different places depending on the input type). It prints
what is mounted under `/kaggle/input`, so you can see where an input went. Set `EMB` above to use
another folder.

**If no embeddings are found** (input not added, or the previous output was deleted), the cell does
not stop: it installs `gdown` (for the CLIP-ReID weights), downloads PA-100K and PETA into
`/kaggle/temp` and embeds them on every GPU, exactly like the first notebook (about 8 minutes on
T4 x2). The embeddings then go to `/kaggle/temp/emb` (not saved with this version).
""")
code("""
def find_embeddings(top="/kaggle/input"):
    \"\"\"First folder under ``top`` with pa100k/ and peta/ train indexes (follows symlinks).\"\"\"
    for dirpath, dirnames, _ in os.walk(top, followlinks=True):
        dirnames.sort()
        if all(os.path.exists(f"{dirpath}/{d}/train/index.json") for d in ("pa100k", "peta")):
            return dirpath
    return ""


print("Mounted under /kaggle/input (3 levels):")
for dirpath, dirnames, _ in os.walk("/kaggle/input", followlinks=True):
    depth = dirpath.count(os.sep) - "/kaggle/input".count(os.sep)
    print("  " * depth + (os.path.basename(dirpath) or dirpath) + "/")
    if depth >= 3:
        dirnames[:] = []
EMB = EMB or find_embeddings()
if not EMB:
    print("No embeddings found: fetching the datasets and embedding them here (~8 min).")
    DATA, EMB = "/kaggle/temp/attr_data", "/kaggle/temp/emb"
    sh("pip install -q gdown==5.2.2")
    sh(["reid", "attributes", "fetch", "--out", DATA])
    embed_cmd = ["reid", "attributes", "embed", "--data", DATA, "--out", EMB]
    n_gpus = len(os.popen("nvidia-smi -L").read().strip().splitlines())
    if n_gpus >= 2:
        workers = str(max(1, (os.cpu_count() or 2) // n_gpus))
        run_parallel(
            [
                Job(f"embed{i}", [*embed_cmd, "--part", f"{i}/{n_gpus}", "--workers", workers],
                    f"{WORK}/logs/embed_gpu{i}.log", {"CUDA_VISIBLE_DEVICES": str(i)})
                for i in range(n_gpus)
            ],
            marker="ETA |",
        )  # fmt: skip
    sh(embed_cmd)  # one GPU: everything; several: verify all shards and mark complete
print("Using the embeddings in", EMB)
""")

md("""
## 4. Search

One `reid attributes hpo` process per GPU (`CUDA_VISIBLE_DEVICES=i`, `--worker i`), all writing to
the same Optuna **journal file** `WORK/optuna_journal.log`, so they share one study: each process
asks the sampler for its next trial knowing every finished trial of the other one. The first trial
is always the default configuration (measured with the same objective, so it is the baseline).
Each finished trial prints one `TRIAL |` line (number, state, validation score, best so far,
best epoch, seconds, parameters); this cell shows the latest line of each GPU every minute. A
`PRUNED` trial was stopped early by the median pruner. The search stops after `TIMEOUT_MIN` minutes
per process (a running trial is finished first). With one GPU the command runs directly.
""")
code("""
hpo_cmd = ["reid", "attributes", "hpo", "--emb", EMB, "--out", WORK,
           "--trials", str(TRIALS_PER_GPU), "--timeout-min", str(TIMEOUT_MIN)]  # fmt: skip
n_gpus = len(os.popen("nvidia-smi -L").read().strip().splitlines())
if n_gpus >= 2:
    run_parallel(
        [
            Job(f"gpu{i}", [*hpo_cmd, "--worker", str(i)], f"{WORK}/logs/hpo_gpu{i}.log",
                {"CUDA_VISIBLE_DEVICES": str(i)})
            for i in range(n_gpus)
        ],
        marker="TRIAL |",
    )  # fmt: skip
else:
    sh(hpo_cmd)
""")

md("""
## 5. Final comparison on the test split

Run once, after the search. It loads the study, then trains the **default** and the **best**
configuration with `SEEDS` seeds each and scores them on the **test** splits (threshold 0.5 and
validation-tuned thresholds; mean +- std), trains the best configuration with and without
viewpoint labels, saves the best head (seed 0) as `head_tuned/head.npz` with its per-label report
in `report_tuned/`, and writes `RESULTS_hpo.md` (trial counts, best parameters, the two tables,
parameter importances) with the optimisation-history and importance plots and `trials.csv`.
It takes a few minutes (about 3 trainings per seed and configuration).
""")
code("""
from IPython.display import Image, Markdown, display

sh(["reid", "attributes", "hpo-final", "--emb", EMB, "--out", WORK, "--seeds", str(SEEDS)])
display(Markdown(open(f"{WORK}/RESULTS_hpo.md").read()))
for name in ("optimization_history.png", "param_importances.png"):
    if os.path.exists(f"{WORK}/{name}"):
        display(Image(filename=f"{WORK}/{name}"))
""")

md("""
## 6. Output files
""")
code("""
sh(f"du -sh {WORK}/*; ls -la {WORK} {WORK}/head_tuned {WORK}/report_tuned")
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
