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

## 2. Move an environment from Colab

The notebook's section "Use this environment on your PC" writes
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
