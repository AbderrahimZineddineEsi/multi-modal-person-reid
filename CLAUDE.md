# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An installable Python package, `reid` (src layout, `src/reid`, version 0.2.0, AGPL-3.0), for multi-camera person
Re-ID with face association: detect, track, build tracklets, assign persistent identities, then search, analytics,
rendering and a Gradio app. All state is local (SQLite + numpy vectors + crops in one folder); there is no cloud
service. Branch `v2` is the rewrite of the v1 Kaggle notebooks (tag `v1-kaggle`). `team_materials/` is other people's
work, git-ignored: do not edit or commit it.

Read first when relevant: `docs/technical_report.md` (how it works, parameters, results), `docs/dev/PLAN.md`
(design decisions), `docs/dev/changes_log.md` (every change with measurements), `docs/v1_to_v2_report.md` (history),
`docs/local_app.md` (PC use, sync), `docs/future_work.md`.

## Commands

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[models,app,eval,dev]"     # extras: models, app, eval, faiss, dev
# Pascal GPU (GTX 10xx): install torch from the cu126 index first (see pyproject.toml)

pytest -q                                      # CPU, synthetic data, no weights (-m "not models" is the default)
pytest -q -m models                            # tests that load real weights
pytest -q tests/test_tracklets.py::test_name   # one test
ruff check . && ruff format .                  # line length 100; CI runs ruff check, ruff format --check, pytest

reid info                                      # hardware, device plan, effective config, data_dir stats
reid process --cameras configs/cameras/X.yaml [--preset auto|lite|balanced|accurate|max] [--data-dir D] [--set a.b=1]
reid render | search | journey | evaluate | dataset fetch aicity24 --out D | virtual-cams | app | env | sync
reid app --workspace data/environments         # Gradio app; --share needs a login (generated if no --auth)

python -m reid.config_yaml > configs/default.yaml    # regenerate after editing src/reid/config.py
python scripts/build_colab_notebook.py               # regenerate both notebooks in notebooks/
python scripts/test_notebook.py --out /tmp/nb.ipynb  # run colab_demo.ipynb locally (needs the dev extra)
```

Use `reid <command> --help` for flags. Commands that run models need weights (downloaded on first use) and a GPU
for reasonable speed; do not run them just to check something that unit tests cover.

## Architecture (needs several files to understand)

- **`pipeline.run(cfg, cameras, engine)`** processes cameras sequentially into one run and one shared identity space.
  Per camera: decode (`runtime.max_side` downscale, `frame_stride`), detect (+pose), `Tracker.update`, then
  `TrackletBuilder.process`, which returns tracklets as they close. Each finished tracklet is written to the
  database (with every track point and the best crop), then handed to `engine.on_tracklet`. After a camera,
  `engine.end_camera`; at the end `engine.end_run`.
- **`tracking/tracklets.py` `TrackletBuilder`**: every box becomes a stored point; only gated boxes (confidence,
  height, edge, keypoint visibility, every `sample_every` frames) are embedded; top 16 samples kept; faces on face
  cameras only; finalized after `track_lost_s` or at end of video; splits a tracklet where appearance changes
  abruptly (tracker ID switch).
- **`identity/associate.py` `IdentityEngine`** (Casao et al. 2023 at tracklet level). `on_tracklet`: face evidence
  first, then the softmax-ratio body test against quality-weighted gallery centroids, with cannot-links
  (`constraints.py`) excluded; verify-before-update gallery insertion (`gallery.py`); unmatched tracklets pool in
  `udm.py`. `end_camera`: UDM pass (DBSCAN). `end_run`: flush the pool, `consolidate.py` (constrained
  agglomerative clustering, face must-links, reversible through `tracklet_gid_history`), rebuild prototypes
  (`prototypes.py`). State is reloaded from the database on init, so runs accumulate.
- **Storage** (`storage/`): `db.py` (SQLite, schema v2, no ORM), `vectors.py` (named collections `body_samples`,
  `body_tracklet`, `face_tracklet`, `body_proto`; NumPy or faiss), `files.py` (paths under `data_dir`),
  `portable.py` (export/import/relink), `maintenance.py` (delete), `sync.py` (PC <-> Drive). Vectors are rebuilt
  from samples when gids change; do not edit one without the other. Paths in the database are relative to
  `data_dir`.
- **Models** (`models/`): wrappers per model behind `registry.build_models`. `planner.py` decides GPU/CPU per
  model from free VRAM and a measured cost table; `placement.py` handles OOM at run time (halve batch, then CPU).
  `--preset auto` picks lite (no GPU or < 2 GB) or balanced from the GPU.
- **Config**: pydantic models in `config.py` (unknown keys are errors). Precedence: defaults < preset
  (`configs/presets/*.yaml`, packaged as `reid/presets`) < `-c` file < `--set KEY=VALUE`.
- **App** (`app/`): `build_app(cfg)` assembles tabs from `app/tabs/*.py` (layout and wiring only, each exposes
  `build(ctx)`); logic stays in `reid.*`. `state.py` holds `AppContext`. An environment is a workspace subfolder
  with its own `reid.db`; the env bar re-points the shared context. Long jobs run in threads with concurrency 1;
  renders are cached by parameters plus an assignment fingerprint. Gradio 6: theme and css go to `launch()`.
- **Search/analytics/viz**: `search.py` (body, face, both; MATCH / AMBIGUOUS / UNKNOWN), `analytics.py` (pure
  functions over the DB), `viz/` (render, mosaic, follow-person, timeline, heatmaps), `eval/` (MOT metrics,
  held-out retrieval).

## Conventions that matter

- **Notebooks contain no pipeline logic.** They only install the package and call `reid` commands through
  `reid.notebook.sh`. Edit `scripts/build_colab_notebook.py` and regenerate; never edit the `.ipynb` by hand.
  `colab_demo.ipynb` is the launcher, `colab_walkthrough.ipynb` the benchmark walkthrough.
- **Generated files:** `configs/default.yaml` (from `config.py`; a test fails if stale) and `notebooks/*.ipynb`.
- **Log every behavioural change** in `docs/dev/changes_log.md` in the existing format
  (`[component] v1: ... -> v2: ... Why: ...`) with measurements, and update `docs/technical_report.md` when a
  default, algorithm or schema changes. Never claim a number that was not measured.
- Keep files near or under 300 lines, use type hints and docstrings, log with `logging` (no `print` in the package),
  no global mutable state. Every identity module docstring cites the paper section it implements and any deviation.
- **Pascal GPUs (GTX 1050 test machine, sm_61):** fp16 is off, and onnxruntime-gpu has no kernels, so face models run
  on CPU. Write code so a CPU-only machine still works.
- **`data/` is git-ignored** (videos, weights, databases, `*.mp4`, `*.pt`, `*.onnx`, `*.zip`). Never commit them.
- **Executed notebooks** (`notebooks/*_finished*.ipynb`) are git-ignored because their outputs contain the app
  password and share link. Do not commit notebooks with outputs.
- **Public showcase assets (README images, GIFs, screenshots) must show only men.** This is an owner
  requirement; check every frame before adding or publishing one.
- Tests are CPU-only on synthetic data (`tests/synth.py`, `tests/conftest.py`); anything that needs real weights
  gets `@pytest.mark.models`.
