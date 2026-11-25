# Future work

Concrete next steps, in rough priority order. Effort: S = 1 to 2 days, M = 3 to 7 days, L = 2 weeks or more
(one developer, including tests and a real-footage check). Estimates are judgement, not measurements.

| # | Item | Effort | Needs |
|---|---|---|---|
| 1 | Live cameras (phone apps, IP cameras) | L (about 10 to 15 days) | RTSP/HTTP source, GPU reachable on the LAN or by tunnel |
| 2 | Better recall for small and distant people | M (3 to 5 days) | labelled footage with small people |
| 3 | Evaluation on real labelled footage | M (4 to 6 days) | MMPTrack and/or WILDTRACK, licence check |
| 4 | Face pipeline validation on real face cameras | S to M (2 to 4 days) | own recordings with consent |
| 5 | Stronger body and face models for `max` (PersonViT, AdaFace) | M (3 to 5 days) | licence review, weights, a T4 |
| 6 | Privacy features | M (4 to 6 days) | a decision on legal basis and retention |
| 7 | Several GPUs and batching several streams | M to L (5 to 10 days) | live ingestion (item 1) first |
| 8 | Packaging: Docker image | S (1 to 2 days) | NVIDIA container toolkit for GPU use |

## 1. Live cameras

**Goal.** Use phones or IP cameras as cameras of a run and see identities and journeys while people are still in the
store.

**Sources.**

- Phones: apps that expose a stream, for example IP Webcam (Android, MJPEG or RTSP over HTTP) or DroidCam
  (HTTP/MJPEG). The URL looks like `http://PHONE_IP:8080/video` or `rtsp://PHONE_IP:PORT/...`.
- IP cameras: RTSP, usually `rtsp://user:pass@host/stream`.
- OpenCV (`cv2.VideoCapture`) accepts such URLs, with the FFmpeg backend. Reconnect and timeouts must be handled
  by the code, because Wi-Fi streams drop. A subprocess `ffmpeg` pipe is the fallback when OpenCV is unreliable
  (it is already used for output in `viz/video_io.py`).

**Current state (from the code).**

- `pipeline.run` loops over cameras one after the other and reads each video to its end (`_process_camera`).
  It needs `CAP_PROP_FRAME_COUNT` for progress and an end of file to flush open tracklets.
- Camera sets point to files (`CameraSpec.video_path`), and the database stores a path. Time is
  `start_ts + frame / fps` with fps read from the file; a stream needs wall-clock time instead (`start_ts` as epoch
  seconds already exists).
- Already helpful: tracklets are finalized incrementally (`TrackletBuilder.process` returns closed tracklets every
  frame, `track_lost_s` 2.0 s) and each one goes to `engine.on_tracklet` at once. The identity engine keeps its state
  in the database and reloads it, and `end_run` is the only step that needs to see everything (consolidation,
  prototypes).
- Not ready: the engine, galleries and models live in one process; there is no frame queue; a slow frame is
  never dropped; the app's Process tab runs one blocking job.

**What to build.**

1. A `FrameSource` abstraction (file or URL) with timestamps taken at capture time, a bounded queue and a
   "drop oldest" policy, so latency stays bounded when the GPU is slower than the stream. Reconnect with back-off.
2. A streaming loop per camera: detect, track, tracklet builder, as today, but time from the clock.
   Process at a fixed target fps (`frame_stride` becomes time-based).
3. One worker process per camera for capture, detection and tracking; a single identity process that receives
   finished tracklets (descriptors, samples, faces) through a queue, so the gallery stays consistent. SQLite in WAL
   mode already allows one writer plus readers.
4. Periodic consolidation (every N minutes) instead of only `end_run`, with the existing cannot-link rules.
5. App: a "Live" tab that lists active cameras and shows the latest tracklets and identities; rendering from stored
   points keeps working for finished tracklets.
6. Tests: a fake stream (a video file replayed at real time through a local server) and a dropped-connection test.

**Constraints.**

- The GPU must be on the same LAN as the phones, or reachable by a tunnel (Tailscale, WireGuard). A free Colab T4
  cannot reach a phone behind a home router without one, and Colab sessions end after hours, so Colab is a poor fit
  for live use. A PC with a GPU (the planner already fits models to the GPU) is the realistic target.
- Budget: `balanced` ran at 34 fps on a T4 and 16.9 fps on a GTX 1050 on one process; with N cameras the fps is
  shared. Use `frame_stride` or `lite` for more cameras.
- A phone camera moves and changes exposure; the quality gate and appearance split were calibrated on fixed cameras.
- Security: stream URLs often carry passwords; store them in environment variables, not in the database or camera YAML.

## 2. Better recall for small and distant people

Detection recall caps IDF1 on the benchmark (people 80 to 120 px tall are found 39.7% of the time, 40 to 80 px
31.7% with `balanced`). The quality gate also needs a box of at least 96 px, so small people never get an identity
even when they are detected.

- Larger detector input (`detection.imgsz` 960 or 1280): trivial to try, costs speed. Measure first (S).
- Tiling / SAHI: slice each frame into overlapping tiles, detect, merge. Better for small people, 2 to 4 times the
  detector cost. Only worth it on the cameras that need it; make it a per-camera option (M).
- A lower `quality.min_box_h` with a separate rule for small boxes (they give weaker embeddings).
- Needs labelled footage with small people to measure. AI City scene 075 is enough for a first pass.

## 3. Evaluation on real labelled footage

Everything measured so far is one synthetic scene (AI City 2024 scene 075), with thresholds calibrated on it.

- **MMPTrack** (real, multi-camera, retail and other indoor scenes, labelled). Check the licence before downloading
  or redistributing; most such datasets are research-only. Add a loader next to `datasets/aicity24.py` that writes the
  same uniform GT CSV so `reid evaluate` works unchanged.
- **WILDTRACK** (real, 7 overlapping cameras, outdoor, with calibration; annotations at 2 fps). Good for cross-camera
  association. Needs a converter for its calibrated annotations.
- Report cross-camera IDF1, purity and fragmentation, and re-check the thresholds that were tuned on one scene
  (split 0.7, body match 0.78, consolidation 0.25).
- Evaluate `accurate` and `max`, which were never scored.

## 4. Face pipeline validation on real face cameras

The face path (SCRFD detection, ArcFace, face-first assignment, must-links, face conflicts) is tested on synthetic
evidence and sample images only. AI City people have no usable faces.

- Record short clips with consenting people at an entrance-like camera plus one or two body cameras.
- Measure face detection rate on accepted samples, how often a tracklet reaches `face_min_obs` 2, wrong face
  matches, and how many body-only identities the face evidence merges correctly.
- Tune `face_match` (0.55), `face_margin` (0.05), `face_conflict` (0.3) and `min_face_px` (32) on that data, not
  on guesses. Keep the recordings out of git (`data/` is ignored).

## 5. Stronger models for `max`

The `max` preset still uses CLIP-ReID because the planned PersonViT-B is not wired in.

- **PersonViT** for body ReID: check the code and weights licence, add a wrapper in `models/body_reid.py`, a registry
  entry, its dimension in `configs/presets/max.yaml`, and a VRAM entry in `planner.COSTS`.
- **AdaFace** (R100) for faces: weights are research-only like InsightFace's. Needs its own aligned-crop path.
- Change of body model changes the embedding dimension; environments processed earlier stay searchable because
  search reads the models of the environment, but they cannot be mixed in one run.
- Evaluate on real data (item 3); a gain on the synthetic scene is not informative, Rank-1 is already 96 to 98%.

## 6. Privacy

Faces and body embeddings are biometric data (GDPR, Algerian law 18-07). Today everything is local and shared links
need a login, nothing more.

- Retention: delete tracklets, crops, faces and vectors older than N days (extend `storage/maintenance.py`, run
  from the app or a cron command).
- Blurring: blur faces (and optionally bodies) in rendered videos and thumbnails by default, with an explicit
  switch for authorised users.
- Consent and opt-out: a delete-by-identity flow that already exists (`delete_identity`) plus a record of why and when.
- Store embeddings without crops where possible; encrypt the environment folder at rest.
- Needs a decision on the legal basis and on who may see what; the code can only enforce the policy.

## 7. Several GPUs and several streams

- The planner places models on one GPU. Extend `Hardware` to a list of GPUs and place the detector and the
  embedding models on different devices (v1 did this with `cuda:0` and `cuda:1`).
- Batch crops from several cameras into one embedding call (the body model already batches across tracks).
- Needs the worker design of item 1; without live streams the gain is small because offline cameras are processed
  one after the other.

## 8. Docker image

- A CUDA runtime base image (cu126 wheels for Pascal, a newer index for current GPUs), the package installed with
  the `models` and `app` extras, weights cached in a mounted volume, `reid app --host 0.0.0.0 --auth ...` as the
  entry point, and the environment folder as a volume.
- Check opencv and onnxruntime package clashes (see `pyproject.toml` `override-dependencies`).
- Needs the NVIDIA container toolkit on the host. CI could build the image on a tag.
