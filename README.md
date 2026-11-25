# Multi-camera person re-identification with face confirmation

**Who walked where?** This system follows people across several cameras that do not see each other,
gives each person one identity, confirms identities with faces when a face is visible, and lets you
search the footage with a photo. It is the person re-identification part of a retail-surveillance
graduation project, rewritten as an installable package with a CLI, a web app and a free Colab path.

<p align="center">
  <img src="assets/showcase/hero.gif" width="760" alt="One worker followed from camera to camera with the same ID">
</p>

<p align="center">
  <img src="assets/showcase/identities.png" width="760" alt="The four workers, each with one identity across cameras">
</p>

*Eight clips of the same four warehouse workers, filmed from eight angles. The system gives each man one
identity: ID 1 is found in all 8 cameras and confirmed by his face; searching with a crop of each man
taken from one camera returns the right person (4 out of 4). Footage: Pexels, see [credits](assets/showcase/CREDITS.md).*

[![Open the app in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbderrahimZineddineEsi/multi-modal-person-reid/blob/v2/notebooks/colab_demo.ipynb)

---

## What it does

1. **Detects and tracks** people in each camera (YOLO26 or RF-DETR, BoT-SORT / ByteTrack / OccluBoost).
2. **Builds tracklets**: one piece of one person's path in one camera, with quality-gated body samples
   (pose visibility, detection confidence, size, not cut by the frame edge) and the best faces. A tracklet is cut automatically if the
   tracker swapped two people who crossed.
3. **Decides identity once per tracklet** with a self-adaptive gallery (Casao et al., 2023): body appearance
   (CLIP-ReID), face evidence (ArcFace) used as the strongest signal, and constraints such as "one person cannot
   be in two places at the same time". Unknown people are clustered into new identities; fragments of one
   person are merged at the end of a run, reversibly.
4. **Answers questions**: search by photo (body, face or both), each person's journey and time line, dwell
   times, a follow-this-person video that cuts from camera to camera, and tracking metrics.

Everything is stored locally (SQLite + NumPy + image files) in self-contained *environments*, one per set of
cameras, so nothing needs a cloud account.

## Results

**Labelled benchmark**: AI City Challenge 2024, retail scene 075 (synthetic, exact labels), 4 overlapping
cameras, 2 minutes, 10 people, free Colab T4, `balanced` preset:

| Identities for 10 people | People split / IDs mixing people | Cross-camera IDF1 | Held-out search Rank-1 | Speed | GPU memory |
|---|---|---|---|---|---|
| **10** | **0 / 0** | **88.3%** | **96%** | 34 fps | 1.0 GB |

IDF1 is limited by recall of small, distant people, not by identity errors (identity precision 97%).
How the numbers are measured, and why the earlier version's "85% accuracy" was not a real accuracy:
[technical report](docs/technical_report.md), [v1 → v2 report](docs/v1_to_v2_report.md).

**Real footage** (the showcase above, 8 cameras, 4 people): 4 identities with no person mixed into another,
3 of 4 men linked across every camera they appear in (the fourth is only visible in two cameras and gets one
identity there), search 4/4. Processed on a laptop-class GTX 1050 (3 GB) at 8.5 fps with the default `balanced` preset.

## The app

| | |
|---|---|
| ![Overview](assets/showcase/app_overview.png) | ![Explore a person](assets/showcase/app_explore.png) |
| ![Search by photo](assets/showcase/app_search.png) | ![Metrics](assets/showcase/app_metrics.png) |

Tabs: **Overview** (people and cameras), **Process** (upload videos or paste video links), **Explore**
(a person's journey, time line and follow clip), **Search** (by body, face or both), **Cameras**,
**Metrics** (speed per stage, GPU memory, evaluation) and **Admin** (merge, split, rename, delete,
environments). Public links require a login.

## Run it

### On Colab (free GPU)

Open the [launcher notebook](https://colab.research.google.com/github/AbderrahimZineddineEsi/multi-modal-person-reid/blob/v2/notebooks/colab_demo.ipynb),
choose a T4 runtime and run all cells. It installs the package, mounts Google Drive (your data persists
there) and prints a public link and a login for the app. Processing happens from the app.
[colab_walkthrough.ipynb](notebooks/colab_walkthrough.ipynb) reproduces the benchmark step by step.

### On your own computer

```bash
git clone -b v2 https://github.com/AbderrahimZineddineEsi/multi-modal-person-reid.git
cd multi-modal-person-reid
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e ".[models,app,eval]"     # GTX 10xx: install torch from the cu126 index first (pyproject.toml)
reid app --workspace data/environments      # opens the app at http://127.0.0.1:7860
```

The models are placed automatically: as much as fits goes on the GPU (detector first, then the body model),
the rest runs on the CPU, and an out-of-memory error moves a model to the CPU instead of stopping.
`--preset auto` (the default) picks the preset from the hardware:

| Preset | Models | GPU memory (measured) | Picked by `auto` when |
|---|---|---|---|
| `lite` | YOLO26n-pose, ByteTrack, OSNet, buffalo_sc | 0.26 GB | no GPU, or < 2 GB free |
| `balanced` | YOLO26m-pose, BoT-SORT, CLIP-ReID MSMT17, buffalo_l | 0.87 GB (1.0 GB on a T4 with faces on GPU) | a GPU with ≥ 2 GB free (incl. Colab T4) |
| `accurate` | RF-DETR-M + YOLO26s-pose, appearance-aware BoT-SORT, CLIP-ReID, buffalo_l | 0.82 GB | on request |
| `max` | RF-DETR-L + YOLO26m-pose, OccluBoost, CLIP-ReID, antelopev2 | 1.14 GB | on request |

Memory figures are whole-process peaks measured on a GTX 1050 (face models run on CPU on that card).

### Moving between Colab and your computer

Each environment is one folder, so it moves as a unit:

```bash
reid sync pull --workspace data/environments --remote <Drive folder or rclone remote>   # Colab -> PC
reid sync push --workspace data/environments --remote <...>                             # PC -> Colab
reid env export / reid env import                                                       # or as a zip
```

Sync refuses to overwrite an environment that changed on both sides (`--force` keeps a backup).
Details: [docs/local_app.md](docs/local_app.md).

### Reproduce the official benchmark (Kaggle, 2x T4)

[notebooks/kaggle_aicity_benchmark.ipynb](notebooks/kaggle_aicity_benchmark.ipynb) runs the `balanced` preset on four
full 16-camera test scenes of the AI City 2024 Track 1 dataset, scores them with NVIDIA's official HOTA evaluator and
saves databases, crops, 720p videos, predictions, scores and logs. Upload it to Kaggle, choose **GPU T4 x2** and
**Internet on**, and use *Save Version, Save & Run All (Commit)*. The same steps from a terminal:

```bash
reid benchmark aicity24 run --scenes 075,078,082,086 --preset balanced --workspace data/bench
reid benchmark aicity24 score --workspace data/bench --preset balanced   # writes RESULTS_balanced.md
reid app --workspace data/bench
```

Results are not published yet: they will be added here once the notebook has been run.

### Command line

```bash
reid process --cameras my_cameras.yaml            # detect, track, identify (camera set: see configs/cameras/)
reid search --image query.jpg --mode both
reid journey --gid 3                              # steps + a follow clip
reid render --mosaic                              # all cameras on one timeline
reid evaluate --gt gt.csv                         # with ground truth
reid info                                         # hardware, chosen preset and device plan
```

## Documentation

- [Technical report](docs/technical_report.md): architecture, every stage and its parameters, presets,
  evaluation protocol and results, limitations.
- [v1 → v2 report](docs/v1_to_v2_report.md): what the first version did, what was wrong and why, what changed,
  the measured impact, and the papers behind each part.
- [Using the app on a PC and syncing with Colab](docs/local_app.md).
- [Future work](docs/future_work.md): live phone / IP cameras, small-person detection, real labelled footage.

## Based on

- M. Casao, A. Azagra, A. C. Murillo, E. Montijano, *A Self-Adaptive Gallery Construction Method for Open-World
  Person Re-Identification*, Sensors 23(5):2662, 2023: gallery, softmax-ratio test, unknown-data manager.
- S. Li, L. Sun, Q. Li, *CLIP-ReID*, AAAI 2023 (body features, MSMT17 weights); J. Deng et al., *ArcFace*,
  CVPR 2019 (InsightFace face models); N. Aharon et al., *BoT-SORT*, 2022; Ultralytics YOLO26; Roboflow RF-DETR.
- M. Al Muzaddid, W. Beksi, *Person Re-Identification via Generalized Class Prototypes*, arXiv:2510.17043
  (search prototypes).
- Own contributions: tracklet-level identity with verify-before-update, face evidence used as must-link with
  margins, multi-camera constraints, reversible consolidation, appearance-change tracklet splitting.

## Licence and data

Code: AGPL-3.0 (required by Ultralytics and boxmot). Model weights keep their own licences; the InsightFace
face models are for non-commercial research use. Showcase footage: Pexels License, credits in
[assets/showcase/CREDITS.md](assets/showcase/CREDITS.md). Benchmark: NVIDIA AI City 2024 / PhysicalAI-SmartSpaces,
CC-BY-4.0. Face and body descriptors are biometric data: keep them local, and process people only with a
legal basis and their consent.

*Author: Abderrahim Zineddine. Part of a 4th-year graduation project (the theft-detection and VLM-orchestrator
parts were team contributions and are not in this repository).*
