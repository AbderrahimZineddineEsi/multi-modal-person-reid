# Using an environment on your own PC

Heavy processing (detection, tracking, embedding a whole scene) stays on Colab. The same app also
runs on a normal PC for light work: browsing people and journeys, searching by image, metrics and
admin. Light processing with the `lite` preset works locally too.

## 1. Install

```bash
git clone <repo> reid_project && cd reid_project
uv venv && source .venv/bin/activate
uv pip install -e ".[models,app,eval]"
```

Without a clone: `uv pip install "reid[models,app,eval] @ git+<repo-url>"`.

**Pascal GPUs (GTX 10xx, compute capability 6.1).** The default PyTorch wheels no longer ship
kernels for them. Install the CUDA 12.6 build before the package (see the comment in
`pyproject.toml`):

```bash
uv pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

`onnxruntime-gpu` has no kernels for Pascal either, so the face models (InsightFace) fall back
to the CPU automatically; fp16 is also switched off on such a card. With **no GPU at all** every
model runs on the CPU (`runtime.*_device: auto` resolves to `cpu`): `reid app` starts and search
works, only slower (a few seconds per query instead of ~0.2 s).

## 2. Move an environment from Colab (one zip)

This is the one-off route. For going back and forth, use the sync loop in section 5 instead.
The walkthrough notebook's section "Use this environment on your PC" writes
`MyDrive/reid_data/exports/<env>.zip` (set `EXPORT_WITH_VIDEOS = True` before running if you want
the videos inside; they can add gigabytes). Download the zip from the Drive web page, then:

```bash
reid env import ~/Downloads/aicity_balanced.zip --workspace data/environments
reid env list --workspace data/environments
```

By hand (on any machine): `reid env export --data-dir ENV_DIR --out FILE.zip [--with-videos]
[--with-renders]`. The export contains the database (a consistent copy), vectors, crops, faces,
evaluation results, cache and `env.json`, plus a `reid_export.json` manifest. `renders/` is skipped
because it can be regenerated. `import` takes `--name` to rename and `--force` to replace an
existing environment.

**Videos.** The database stores the video path from processing time (on Colab an absolute Drive
path). Three ways to make clips and heatmaps work on the PC:

1. export with `--with-videos`: import rewrites each camera to `videos/<camera_id><ext>` inside the
   environment;
2. you already have the videos: `reid env relink --data-dir data/environments/<env> --videos
   ~/my_videos` matches files by the stored file name, else `<camera_id>.*`, and reports what was
   relinked, missing or ambiguous (nothing is guessed);
3. copy them yourself into `<env>/videos/` (looked up by file name, or by `<camera_id>.*` when
   there is exactly one match).

## 3. Launch

```bash
reid app --workspace data/environments            # open http://127.0.0.1:7860
reid app --workspace data/environments --env aicity_balanced
```

The environment bar switches between environments. On selection the app reads the models the
environment was **processed with** (latest run) and uses them to embed search queries, whatever
`--preset` says: a `balanced` environment (CLIP-ReID, 1280-D) is searched with CLIP-ReID even if
the app was started with `lite` (OSNet, 512-D), otherwise the vectors would not even have the same
size. `reid search` does the same and prints a line on stderr when it overrides the preset.
Processing new footage still follows the preset you choose in the Process tab.

## 4. What runs locally

Measured peak GPU memory (whole process, GTX 1050, 3 GB, 1-minute AI City slice, face models on
the CPU so not counted):

| preset | GPU memory |
| --- | --- |
| lite | 0.26 GB |
| balanced | 0.87 GB |
| accurate | 0.82 GB |
| max | 1.14 GB |

All fit in 3 GB, so processing a short clip locally is possible; it is just slower than a T4 (and
face models run on the CPU). Use Colab for long footage.

| works without the videos | needs the videos |
| --- | --- |
| browsing people, tracklets and crops; journeys as a table; search by image (body and face); metrics and evaluation results; admin (merge, split, rename, delete) | journey and per-camera clips, mosaic renders, camera heatmaps and backgrounds, retrieval evaluation that re-reads crops from video |

Without a local video the clip commands stop with a message ("video not found ... use `reid env
relink`") instead of a traceback.

## 5. Sync between your PC and Colab

`notebooks/colab_demo.ipynb` is only a **launcher**: it installs the package, mounts Drive and
starts the app on `MyDrive/reid_data/environments`. It processes nothing by itself. To add footage
open the **Process** tab and upload a video or paste direct video links (one per line, up to 4 GB
each, no YouTube). Every video processed from the app is stored inside the environment
(`<env>/uploads/`) and the database keeps the relative path `uploads/<name>`, so the environment
still finds its videos after it moves. `notebooks/colab_walkthrough.ipynb` is the optional
notebook that reproduces the AI City benchmark.

**PC only.** `reid app --workspace data/environments`. Nothing else is needed.

**Colab only.** Open the launcher, work, close. Everything is on Drive.

**Switching.** An environment is a folder, `reid sync` copies it between two workspaces:

```bash
# on the PC, before going to Colab: send your environments to the Drive folder
reid sync push --workspace data/environments --remote <TARGET>
# open the Colab launcher (it sees the same data on Drive), work, stop the app cell
# back on the PC: fetch what changed on Colab
reid sync pull --workspace data/environments --remote <TARGET>
```

`<TARGET>` is one of:

* a **folder**: your Google Drive folder mounted on the PC by any means (Drive for desktop, a FUSE
  mount, a synced folder), or a USB disk, e.g. `~/GoogleDrive/reid_data/environments`;
* an **rclone path** `name:path`, e.g. `gdrive:reid_data/environments`. rclone is used only when
  the target looks like `name:path` *and* an `rclone` executable is already on your PATH;
  otherwise the command stops with a message. This project never installs it.

Options: `--env NAME` (repeatable) limits it to some environments, `--dry-run` shows the plan,
`--with-renders` also copies `renders/` (skipped by default, it can be regenerated), `--force`
and `--delete` as below.

**What it decides**, per environment (a fingerprint of a consistent SQLite snapshot plus the sizes
of the vector files; the last synced fingerprint per target is kept in `<env>/.sync_state.json` on
the local side):

| situation | result |
| --- | --- |
| identical | skipped |
| missing at the destination | copied |
| changed only at the source | copied (changed files only; the database last) |
| changed only at the destination (the other side is ahead) | refused: sync the other way first |
| changed on both sides since the last sync | refused, nothing is copied |
| refused, but `--force` | the destination environment is first moved aside as `<env>.bak-<timestamp>` on the destination side, then overwritten |
| only at the destination, with `--delete` | moved aside as `<env>.bak-<timestamp>` (never erased) |

The exit code is 3 when something was refused. The database is always copied as a consistent
snapshot (SQLite backup API); `-wal`/`-shm` files never travel, and the result is verified by
fingerprint after copying. Identity ids continue across machines because the whole environment
(database, vectors, crops, uploads) moves together. **Stop the app on the side you copy from**
(on Colab: stop the launcher cell) so the database is not being written; a pull from rclone
refuses to copy a database that still has an unmerged WAL file. A second PC has no sync state for
the target, so its first sync of an environment that already exists on both sides is refused until
you pick a direction with `--force` (the old copy is kept as a backup).

### Setting up rclone yourself (optional)

Install it from the official page <https://rclone.org/install/>, then run `rclone config` and add
a remote of type "Google Drive" (call it `gdrive`; follow the prompts, it opens a browser to log
in). Check with `rclone lsd gdrive:`, then use `--remote gdrive:reid_data/environments`.

### Zip alternative

Without any of the above: on one side `reid env export --data-dir <env> --out env.zip
[--with-videos]`, move the file by hand, and on the other `reid env import env.zip --workspace
<dir> [--force]` (section 2).
