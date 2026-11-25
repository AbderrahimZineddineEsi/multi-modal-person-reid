> **Historical planning document** (written before the implementation). Some details changed later (e.g. appearance tracking in `accurate`/`max`, the Admin tab visible by default, `max` uses CLIP-ReID). The current design is in `docs/technical_report.md`; every change is in `docs/dev/changes_log.md`.

# v2 Rewrite Plan

This document is the single source of truth for implementation agents. Read this plan and `CLAUDE.md` first. Use `docs/dev/v1_review.md` for the v1 algorithm details (section numbers are referenced below, e.g. §1.12) and `docs/dev/research.md` for the model and dataset facts. Do **not** reopen the v1 notebooks unless the review is missing something.

## 0. Goals and decisions (fixed)

- This is a portfolio-quality, installable Python package. The Colab notebook only installs it and calls it, and contains no duplicated logic.
- **Storage is local only.** Everything lives under one `data_dir`; on Colab that is a Google Drive path, so data persists between runs. There is no Pinecone, MongoDB, or ImageKit.
- The system is **multi-camera first**: one run processes N camera videos into a single shared identity space.
- **All per-frame track boxes are persisted**, so any video can be re-rendered and any person followed across cameras after processing. Source videos are referenced by path and never copied.
- The UI is Gradio, split into small modules. UI files contain layout and wiring only; all logic lives in the package.
- Every v1 bug in review §2 must be fixed or made obsolete by the new design. No v1 bug may be ported.
- Hardware targets are a free Colab T4 (full runs) and a local GTX 1050 with 3 GB (smoke tests on short or downscaled clips). The local machine has system Python 3.14, so use `uv venv --python 3.12`. Torch for Pascal (sm_61) must come from the **cu126** wheel index; newer CUDA builds dropped Pascal. If CUDA fails, fall back to CPU and continue.

## 1. Repository layout

```
pyproject.toml            # package "reid" (src layout), python>=3.10, deps pinned with ranges; extras: [app], [dev], [eval]
configs/default.yaml      # every threshold, with comments (replaces the v1 config cells)
configs/cameras/*.yaml    # camera sets: store_virtual.yaml, <dataset>.yaml, phones.example.yaml
src/reid/
  config.py               # pydantic/dataclass Config loaded from YAML with overrides; CameraSpec
  types.py                # dataclasses: Detection, TrackPoint, Tracklet, FaceObs, Identity, SearchHit
  models/                 # one small class per model; lazy-loaded; device chosen in config
    detector.py pose.py body_reid.py face.py registry.py   # registry: build_models(cfg) -> Models
  tracking/
    tracker.py            # per-camera MOT wrapper (motion-only, or fed our embeddings; see §3.2)
    tracklets.py          # TrackletBuilder: accumulates points, quality-gated samples, faces; finalizes tracklets
  identity/
    associate.py          # tracklet -> identity assignment (online) + constraints
    consolidate.py        # offline constrained agglomerative clustering (replaces DBSCAN + retro merge)
    prototypes.py         # alpha-FPS (review §1.12), quality-weighted
  storage/
    db.py                 # SQLite schema + typed repository functions (no ORM)
    vectors.py            # VectorIndex: numpy (default) or faiss; save/load .npy + id map
    files.py              # crop/face/thumbnail paths under data_dir
  pipeline.py             # run(cfg, cameras) -> RunResult; orchestrates per-camera extraction + association
  search.py               # search_body(img), search_face(img), search_both(img) -> list[SearchHit] + status
  analytics.py            # per-identity journey, dwell times, camera transitions, counts, occupancy
  viz/
    colors.py render.py heatmap.py journey.py timeline.py
  app/                    # Gradio UI (see §6)
  cli.py                  # `reid process|render|search|journey|evaluate|virtual-cams|app`
scripts/                  # thin wrappers only if needed (prefer the CLI)
tests/                    # pytest, CPU-only, synthetic data, no model downloads; fast (<30 s)
notebooks/colab_demo.ipynb
docs/                     # technical_report.md (rewritten last), dev/ (this plan, review, research), future_work.md
assets/                   # README media and camera backgrounds
.github/workflows/ci.yml  # ruff + pytest on CPU
```

Code rules:
- Keep files under about 300 lines; split a file that grows past that.
- No global mutable state. Pass objects explicitly.
- Use type hints throughout and add docstrings to public functions.
- Lint with `ruff` using its default plus `I` rules.
- Log through `logging`, not `print`, inside the package.

## 2. Data model (`storage/db.py`, SQLite at `data_dir/reid.db`)

```
runs(id PK, created_at, config_json, status)
cameras(id TEXT PK, name, video_path, fps, width, height, start_ts REAL, is_face_cam INT,
        homography_json NULL, background_path NULL)
tracklets(id PK, run_id, camera_id, local_track_id, start_frame, end_frame, start_ts, end_ts,
          n_points, n_samples, mean_quality, gid NULL, assign_score NULL, assign_source TEXT,  -- 'body'|'face'|'consolidate'|'manual'
          best_crop_path, best_face_path NULL, face_quality NULL)
track_points(tracklet_id, frame, ts, x1, y1, x2, y2, conf, sampled INT)   -- EVERY frame; index (tracklet_id, frame) and (camera via tracklet)
identities(gid PK, created_at, label NULL, first_ts, last_ts, n_tracklets, has_face INT)
```

Vectors are stored under `data_dir/vectors/`:
- `body_samples`: up to K per tracklet, keyed by tracklet id.
- `body_tracklet`: one quality-weighted mean per tracklet.
- `face_tracklet`: one per tracklet with a face.
- `body_proto`: K=3 alpha-FPS prototypes per gid.

Indices are rebuilt from the samples whenever gids change, so they are never out of sync (this fixes v1 B5/B6/B13/B14/B15). Timestamps are `camera.start_ts + frame / fps`, using real fps read from the video (fixes B1). `start_ts` comes from the camera YAML; it defaults to 0 for all cameras, which means "synchronized".

## 3. Pipeline design

### 3.1 Per-camera extraction (`pipeline.py` + `tracking/`)

For each camera, in order:

1. Read frames with `cv2.VideoCapture`. Support `frame_stride` (process every n-th frame) and `max_frames`/`start_s`/`end_s` for smoke tests. Detection can use a resized frame (`det_imgsz`), with crops taken from the full-resolution frame.
2. Detect people (class 0, conf ≥ `det_conf`), then run the tracker, which gives local track IDs. Save a `TrackPoint` for **every tracked box**.
3. Quality gate, run on a subset only: each track is sampled at most every `sample_every` frames, and only when the box is large enough (`min_box_h`) and not truncated at the frame edge. Quality is keypoint visibility from pose (review §1.3, but use the pose output belonging to *this* box, not the first pose in the crop), multiplied by detection confidence. Samples below `min_quality` are skipped.
4. Embed accepted samples with batched body ReID. Keep the top `max_samples_per_tracklet` (default 16) samples by quality, plus the best crop image.
5. On face cameras, run face detection on accepted samples only, paired to the body box (top 40% region, pick the **best** face, which fixes B18). Face quality is det_score × size factor × frontalness (from landmarks). Keep the best `max_faces` faces per tracklet.
6. A tracklet is finalized when its track has been lost for `track_lost_s`, and every open tracklet is finalized at the end of the video (fixes B2/B8). Tracklets with fewer than `min_samples` accepted samples get `gid = NULL` but keep their track points; they are rendered as "unidentified".

### 3.2 Tracker

Per research.md, prefer one of the following, in order:
- (a) a boxmot tracker running in motion-only mode (ByteTrack, or OC-SORT-style); identity across occlusions is handled by tracklet association instead;
- (b) a tracker fed our own body embeddings, so that ReID is not computed twice (fixes W1).

Wrap the tracker behind `Tracker.update(dets, frame) -> list[(track_id, box, conf)]` so it can be swapped.

### 3.3 Identity association (`identity/`). REVISED 2026-11-24, see pipeline_decisions.md

The core stays **Casao et al. 2023's self-adaptive gallery** (source_papers.md, Paper A), with three changes: it is applied at the **tracklet level**, it uses **verify-before-update**, and it adds **multi-camera constraints**. Every module docstring cites the paper/section it implements, along with any deviation from it.

Interface (P2's pipeline calls it; P3 implements it):
```python
class IdentityEngine:
    def __init__(self, cfg: Config, conn, vectors: VectorStore): ...
    def on_tracklet(self, t: Tracklet) -> int | None   # t is already inserted in DB (t.id set), with samples/faces in memory; returns gid or None (pooled)
    def end_camera(self, camera_id: str) -> None        # UDM pass
    def end_run(self, run_id: int) -> RunIdentityReport # final UDM flush + consolidation + prototype rebuild
```
Persistence is owned by the engine:
- sample embeddings and their metadata go into a new `samples` table (key, tracklet_id, frame, quality, in_gallery) plus the vector collections;
- assigned gids are written with `db.set_gids` (which records history).

Steps:
1. **Tracklet descriptor.** Compute the quality-weighted mean of the tracklet's sample embeddings. For the face descriptor, take the quality-weighted mean of its best `max_faces` faces.
2. **Face first, with accumulated evidence** (this is the user's own contribution, made safe). Query the face vectors and aggregate per gid. Assign the gid when best ≥ `face_match` **and** best − second ≥ `face_margin` **and** at least `face_min_obs` (default 2) faces support it. The source becomes `face`. A face match whose gid contradicts the body decision creates a **must-link**, which is resolved in consolidation. It never triggers an immediate destructive merge.
3. **Body test (Casao §3).**
   - Compute the softmax over cosine similarities to visibility- and quality-weighted gallery centroids, with temperature `1/v` where v=0.1, the paper's default (v1 used 15).
   - Accept when `p1/p2 ≥ tau` (2.0) and cos ≥ `min_cos`.
   - Apply the test to the tracklet descriptor. Optionally, also require that a majority of the samples agree.
   - Exclude gids with a **cannot-link** to this tracklet. Cannot-links come from tracklets of the same camera that overlap in time (beyond `overlap_tolerance_s`), from face conflicts (< `face_conflict`), and, if `use_topology` is on, from transit times between cameras that are too short.
4. **Gallery update, done only after a confident assignment** (verify-before-update).
   - Insert the tracklet's samples one by one, subject to Casao's **diversity gate**: a sample `x` is inserted only if `D_i(x) ≥ D(C_i)` (min distance to the class ≥ the class's current mean min-distance).
   - If the gallery is over budget `m` (50), drop `argmax[γ·H(x)/log N − (1−γ)·D_i(x)]` with γ=0.6.
   - Never insert raw consecutive frames.
5. **Unknown Data Manager (Casao §3.3).** Unmatched tracklets go to the pool.
   - At `end_camera`, and every `udm_every` pooled tracklets, cluster the pooled **tracklet descriptors** using DBSCAN on cosine distance (`udm_eps`), honouring cannot-links.
   - Consider a cluster once it has ≥ `udm_min_samples` (20, the paper's value) total samples. Compare it with the k=3 nearest centroids: if its distance is > `udm_new_dist` (0.1), mint a new gid; otherwise merge it into the nearest gid.
   - `end_run` **flushes** the pool. Any remaining tracklet with ≥ `tracklets.min_samples` samples becomes its own gid, so nobody vanishes (fixes B8). Tracklets below that stay `gid=NULL`; they are kept and rendered grey.
   - Bootstrap, as in the paper: while there are fewer than `bootstrap_q` (20) identities, a well-sampled unmatched tracklet may mint directly.
6. **Consolidation (`end_run`, plus a UI button).**
   - Run constrained agglomerative clustering over **gid centroids** (average linkage, threshold `merge_threshold`).
   - Face must-links are enforced first; cannot-links are never violated.
   - The merged gid inherits the **oldest** id. Tracklets are relabelled with `db.set_gids(source='consolidate')` (history is kept, so the step is reversible), and the galleries of merged gids are combined and re-pruned to budget (fixes B7).
7. **Prototypes (`prototypes.py`).** The in-house α-FPS (review §1.12) keeps K=3. Normalize the quality and distance terms to [0,1] before combining them with `proto_alpha`. Also implement the paper variant from Al Muzaddid & Beksi 2025 (interpolation `x ← x + α(p − x)`) behind `proto_method: alpha_fps | gcp`. Prototypes are rebuilt for the affected gids after any change.

Add every new parameter to `AssociationCfg`/`ConsolidationCfg`, with the paper value as the default and a comment naming the paper. Then regenerate `configs/default.yaml`.

### 3.6 Model presets (`configs/presets/*.yaml`)

| Preset | Target | Detector | Pose gate | Tracker | Body ReID | Face |
|---|---|---|---|---|---|---|
| `lite` | local GTX 1050 (3 GB), CPU fallback | YOLO26n-pose (single model) | same | ByteTrack | OSNet x1.0 MSMT17 | buffalo_sc |
| `balanced` (default) | Colab T4 | YOLO26m-pose (single model) | same | BotSort, motion-only | CLIP-ReID ViT-B/16 MSMT17 | buffalo_l |
| `accurate` | Colab T4 | RF-DETR-M | YOLO26s-pose on crops | BotSort, motion-only | CLIP-ReID ViT-B/16 MSMT17 | buffalo_l |
| `max` | Colab T4 | RF-DETR-L | YOLO26m-pose on crops | OccluBoost, motion-only | PersonViT-B (if license OK) else CLIP-ReID MSMT17 | antelopev2 / AdaFace R100 (optional) |

When detector == pose model (single-model mode), keypoints come from the same forward pass, so each box has its **own** pose (fixes W12). The repo is AGPL-3.0, because ultralytics and boxmot are AGPL. Face weights are research-only, downloaded at runtime and never committed.

### 3.4 Search (`search.py`)

- **Body search**: embed the query, query `body_proto`, aggregate the best score per gid, then apply the softmax ambiguity check (review §1.13). The result status is `MATCH | AMBIGUOUS | UNKNOWN`.
- **Face search**: detect the largest face in the query, or embed a pre-cropped face with padding (review §1.14). Query `face_tracklet`, aggregate per gid, and apply a margin check.
- **"Both"**: run both searches; a face match takes priority and body is the fallback.
- Every hit returns gid, score, status, and the identity's journey (see analytics).

### 3.5 Analytics (`analytics.py`)

These are pure functions over the database:
- `journey(gid)`: the ordered tracklets across cameras, with times, durations and thumbnails, plus the gaps between them.
- `transitions()`: a camera→camera count matrix.
- `dwell_stats()`.
- `occupancy(camera, bin_s)`.
- `run_summary(run_id)`: people count, identified %, face-confirmed %, tracklets per identity (fragmentation), and processing fps per stage.

## 4. Visualization (`viz/`)

- `render.py`: `render_camera(cam, out_path, gid_filter=None, style=...)` re-reads the source video plus `track_points` and draws the following:
  - colored rounded boxes per gid, using a stable palette from `colors.py`; unidentified tracks are grey;
  - a label chip `ID 12 · face ✓`;
  - a fading trail of the last N foot points;
  - an optional top-bar HUD showing the camera name, timestamp and people-in-view count.

  Use the ffmpeg pipe for H.264 output, so the result plays in browsers and on GitHub. Support `scale` and `start_s/end_s` for short clips and GIF export.
- `render_mosaic(cameras, out_path, highlight_gid=None)`: a synchronized grid of all cameras on the shared timeline. The highlighted person is drawn bold and the other people are dimmed.
- `journey.py`: `follow_person(gid)` produces a clip that cuts between cameras following the person's tracklets in time order, with a small picture-in-picture of the camera mosaic and a progress timeline at the bottom.
- `timeline.py`: a Gantt-style chart of one identity's (or all identities') presence per camera over time (matplotlib/plotly).
- `heatmap.py`:
  - foot-point heatmaps from **all** track points (fixes B17 and v1's one-blob-per-sighting) over the camera background (the first frame by default);
  - a per-gid heatmap;
  - if a camera homography exists, a floor-plan heatmap and a trajectory map.

## 5. Virtual cameras (`reid virtual-cams`)

Build N "cameras" from one video using any combination of the following:
- spatial crops (left/right halves with configurable overlap);
- a horizontal mirror;
- a time offset;
- a scale or color jitter.

The command writes the new videos and a camera YAML. It is used to smoke-test with `store_cam3.mp4`, which is not committed. The docs must state that this proves the plumbing, not cross-view generalization.

## 6. Gradio app (`src/reid/app/`)

```
app/__init__.py      # build_app(cfg) -> gr.Blocks ; launch(cfg, share=False, auth=None)
app/state.py         # AppContext: cfg, db, models (lazy), service functions
app/theme.py         # gr.themes.* subclass + small css string (no 200-line CSS blobs)
app/components.py    # reusable: identity_card(gid), metric_tile(label, value), journey_html(...)
app/tabs/process.py  # choose camera set / upload videos, params accordion, progress, run summary
app/tabs/explore.py  # identity gallery grid -> select person -> journey, timeline, heatmap, follow-clip
app/tabs/search.py   # upload/URL image, body/face/both, results cards
app/tabs/cameras.py  # per-camera rendered videos, mosaic, heatmaps, transitions matrix
app/tabs/metrics.py  # run summary, stage timings, VRAM, eval results if present
app/tabs/admin.py    # re-cluster, rename/merge/split identity (manual), reset data_dir (with confirm)
```

Rules:
- Each tab exposes `build(ctx) -> None`, called inside `gr.Blocks`.
- Handlers are thin and only call `reid.*` functions.
- No tab file may exceed about 200 lines.
- `share=True` is opt-in. When sharing, require `auth`; the destructive admin actions are hidden unless `--admin` is passed (fixes B22).

## 7. Evaluation (`reid evaluate`, extra `[eval]`)

On the dev dataset chosen in research.md, report the following:
- per-camera MOT metrics: IDF1, MOTA, ID switches;
- **cross-camera** MTMC IDF1, after mapping gids to GT IDs over all cameras;
- the fragmentation rate;
- query retrieval Rank-1 and mAP, using **held-out** query crops drawn from GT boxes rather than from the system's own saved crops (fixes W4).

Results are written to `data_dir/eval/<run>.json` and shown in the Metrics tab and the README.

## 8. Colab notebook (`notebooks/colab_demo.ipynb`)

The cells are:
1. Set up the GPU and `pip install "reid[app] @ git+https://github.com/<user>/<repo>"`. A `REID_REF` variable pins a branch or tag.
2. Mount Drive and set `DATA_DIR=/content/drive/MyDrive/reid_data`. If the user declines Drive, use local `/content` with a warning.
3. Get demo footage: download the dataset subset, or use the user's videos on Drive.
4. `reid.pipeline.run(...)`, followed by a summary.
5. Render the videos and the follow-person clip, and display them inline.
6. Launch the app with `share=True` and a generated password.

Each cell has a short markdown cell above it. There is no logic beyond calls into the package.

## 9. Phases (each one is a separate agent task; after each phase run `ruff check` and `pytest` and do a smoke run, then stop and report)

| # | Phase | Depends on | Agent |
|---|---|---|---|
| P1 | Scaffold: pyproject, config + default.yaml, types, storage (db/vectors/files), CLI skeleton, CI, tests for storage; delete `archive/ reports/ presentations/`, the old notebooks and `design_discussion.md` (they are preserved by the `v1-kaggle` tag) | — | Sonnet high |
| P2 | Models + tracker + TrackletBuilder + per-camera extraction; `virtual-cams`; GPU smoke test on a store_cam3 clip | P1 | Sonnet high |
| P3 | Identity association + consolidation + prototypes + search + analytics; unit tests with synthetic embeddings (constraints, gid stability, must/cannot-link) | P1 (can run parallel to P2) | Sonnet high |
| P4 | viz: render, mosaic, follow-person, timeline, heatmaps | P2 + P3 | Sonnet medium |
| P5 | Gradio app per §6 | P3 + P4 | Sonnet high |
| P6 | Dataset download helper + evaluate + Colab notebook | P2 + P3 | Sonnet medium |
| P7 | (after user testing) README with media, technical_report rewrite, future_work.md (live phone cameras via RTSP/IP-webcam apps; compute constraints) | all | Sonnet medium + Opus review |
| P8 | **v1 → v2 report for the user** (`docs/v1_to_v2_report.md`), written by Opus after a fresh detailed reading of the v1 code (tag `v1-kaggle`) and the final v2 code. For each component it explains, precisely: what v1 did, what was wrong and why, what v2 does, the measured impact, and which paper it relates to. The user knows v1 deeply and must be able to understand and defend every change. | all | Opus |
