# Hyperparameter inventory (for the Optuna study)

**Summary.** Counts per tier: see the table at the end of this block. **T1 (tune, identity stage, replayable):**
`association.min_cos`, `association.tau`, `association.sample_vote`,
`association.udm_eps`, `association.udm_new_dist`, `association.udm_min_samples`, `association.bootstrap_q`,
`association.bootstrap_min_samples`, `association.bootstrap_join_cos`, `consolidation.merge_threshold`.
Eval data: AI City scene 075 (`configs/cameras/aicity24_075.yaml`, 0 face cameras, synthetic people, GT available)
is body-only, so every face parameter is T3 (cannot be measured there). The warehouse showcase has face cameras but
no GT file, so it is a sanity check only. The only recorded measurement for a T1 parameter is
`consolidation.merge_threshold` (changes_log line 216); all the others are v1 values, paper values or untested.

Origin values: `paper`, `v1`, `measured`, `engineering`, `unknown`. Sources: `src/reid/config.py` field
descriptions, `docs/dev/changes_log.md` (CL), `docs/technical_report.md` (TR), `docs/v1_to_v2_report.md`,
`docs/dev/v1_review.md`. "Casao" = Casao et al. 2023 (Sensors 23(5):2662); "AM&B" = Al Muzaddid & Beksi 2025
(arXiv:2510.17043). Presets (lite/balanced/accurate/max) override only the values noted; every other parameter is
identical across presets.

Notes on interactions: `tau` and `softmax_temp` are coupled (ratio = exp(temp * (s1 - s2)) >= tau, so the effective
cosine margin is ln(tau)/temp: 0.069 now). `softmax_temp` is also the temperature of the gallery entropy
(`identity/store.py:64`), so it is not a pure duplicate: tune `tau` and keep `softmax_temp` fixed. A tracklet is
pooled for the UDM when face and body both fail and it is not minted by the bootstrap, i.e. when there are already
>= `bootstrap_q` identities OR it has < `bootstrap_min_samples` samples (`identity/associate.py:241-249`). With
about 10 people on scene 075 (< `bootstrap_q` = 20) the bootstrap is active for the whole run, so
`bootstrap_join_cos` is live there and the UDM parameters only see short tracklets. `merge_threshold` also needs
`consolidation.enabled`.

## paths / runtime (all AVOID)

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| paths.data_dir | `data` | root of persisted state | engineering | runtime | AVOID | path |
| paths.weights_dir | `""` | weight cache | engineering | runtime | AVOID | path |
| runtime.detect_device | `auto` | detector device | engineering | runtime | AVOID | compute |
| runtime.embed_device | `auto` | pose/ReID/face device | engineering | runtime | AVOID | compute |
| runtime.device_plan | `auto` | GPU/CPU placement | engineering | runtime | AVOID | compute |
| runtime.vram_budget_gb | null | VRAM the planner may use | engineering | runtime | AVOID | compute |
| runtime.cpu_downsize | true | smaller inputs on CPU | engineering | runtime | AVOID | feature toggle |
| runtime.cpu_threads | 0 | torch CPU threads | engineering | runtime | AVOID | compute |
| runtime.fp16 | true (off on Pascal) | half precision | engineering | runtime | AVOID | compute |
| runtime.batch_size | 16 (lite 8) | crops per batch | engineering | runtime | AVOID | speed only |
| runtime.frame_stride | 1 | process every n-th frame | engineering | runtime | AVOID | speed/data selection |
| runtime.start_s / end_s / max_frames | 0 / null / null | clip selection | engineering | runtime | AVOID | data selection |
| runtime.max_side | 1920 | downscale longer side after decode | engineering | runtime | AVOID | speed trade-off, no recorded sweep |
| runtime.seed | 0 | RNG seed | engineering | runtime | AVOID | not a quality knob |
| runtime.log_level | INFO | logging | engineering | runtime | AVOID | not a quality knob |

## detection

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| detection.model | `yolo26m-pose.pt` (lite n, accurate rfdetr-medium, max rfdetr-large) | detector weights | engineering | detection | AVOID | preset choice, categorical |
| detection.conf | 0.25 | min detection confidence | v1 (Ultralytics default; CL line 69 "unchanged") | detection | T2 later | recall/precision of boxes, full rerun |
| detection.imgsz | 640 (max 704) | detector input size | v1 (640) | detection | T2 later | recall on small people, full rerun |
| detection.classes | [0] | COCO classes kept | engineering | detection | AVOID | structural |

## quality (sample gate)

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| quality.pose_model | null (accurate yolo26s-pose, max yolo26m-pose) | separate pose weights | engineering | detection | AVOID | preset choice |
| quality.pose_imgsz | 320 (max 384) | pose input size | engineering | detection | AVOID | preset choice |
| quality.pose_pad | 0.15 | crop padding for the pose model | engineering | detection | AVOID | only with two-model presets |
| quality.kp_conf | 0.3 | keypoint counts as visible above this | v1 (v1_to_v2_report 3.3) | tracklet | T2 later | gate, full rerun |
| quality.min_visible_ratio | 0.5 | min visible fraction of 17 keypoints | v1 (v1_to_v2_report 3.3) | tracklet | T2 later | gate, full rerun |
| quality.min_det_conf | 0.75 (lite 0.5) | min box confidence to be sampled | v1 (0.75); lite 0.5 is measured on one smoke clip (CL line 72) | tracklet | T2 later | sample count vs purity |
| quality.sample_every | 5 | sample a track every n processed frames | engineering (v1 sampled every frame; CL line 97 says not re-tuned) | tracklet | T2 later | sample density |
| quality.min_box_h | 96 | min box height (px) to be sampled | v1 (CL line 97) | tracklet | T2 later | resolution of crops, depends on `max_side` |
| quality.edge_margin_px | 2 | box this close to border = truncated | unknown (not in docs) | tracklet | T2 later | minor |
| quality.reject_truncated | true | skip truncated boxes | engineering | tracklet | AVOID | feature toggle |
| quality.min_quality | 0.35 | min visible_ratio x det_conf | engineering (v2 gate, v1_to_v2_report 3.3) | tracklet | T2 later | gate |

## tracking

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| tracking.tracker | botsort (lite bytetrack, max occluboost) | boxmot tracker | engineering | tracking | AVOID | preset choice, categorical |
| tracking.track_high_thresh | 0.5 | first-pass match confidence | v1 (library default, CL line 98) | tracking | T2 later | ID switches/fragmentation |
| tracking.track_low_thresh | 0.1 | second-pass lowest confidence | v1 (library default, v1_review line 87) | tracking | T2 later | as above |
| tracking.new_track_thresh | 0.6 | confidence to start a track | v1 (library default, v1_review line 87) | tracking | T2 later | as above |
| tracking.track_buffer | 30 | frames a lost track survives | v1 (library default, CL line 98) | tracking | T2 later | fragmentation |
| tracking.match_thresh | 0.8 | IoU-distance association | v1 (library default, CL line 98) | tracking | T2 later | ID switches |
| tracking.use_cmc | true | BoT-SORT camera-motion compensation | engineering | tracking | AVOID | feature toggle |
| tracking.appearance | false (accurate/max true) | feed ReID embeddings to tracker | engineering | tracking | AVOID | feature toggle, costs speed |
| tracking.appearance_thresh | 0.25 | tracker appearance cosine distance | paper (BoT-SORT default; CL line 184) | tracking | T3 fixed | only active with `appearance`; paper value |

## body_reid

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| body_reid.model | clipreid_msmt17 (lite osnet_x1_0_msmt17) | ReID model | engineering | embedding | AVOID | preset choice (v1 used Market-1501 weights) |
| body_reid.weights_url | null | weights mirror | engineering | embedding | AVOID | path |
| body_reid.neck_feat | after | CLIP-ReID feature layer | measured (CL line 66: `before` compresses cosines, would invalidate absolute thresholds) | embedding | T3 fixed | changing it invalidates every cosine threshold |
| body_reid.input_h / input_w | 256 / 128 | crop resize | v1 | embedding | T3 fixed | model input shape |
| body_reid.dim | 1280 (lite 512) | embedding dimension | v1 | embedding | AVOID | must match model |

## tracklets

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| tracklets.track_lost_s | 2.0 | close a tracklet after this gap (s) | engineering (CL line 76 describes, no sweep) | tracklet | T2 later | fragmentation vs merging of passes |
| tracklets.min_samples | 2 | fewer samples = no identity | engineering (also read by IdentityEngine: `associate.py:203`) | tracklet | T2 later | can only be raised in a replay |
| tracklets.max_samples_per_tracklet | 16 | best samples kept | engineering | tracklet | T2 later | descriptor quality vs size |
| tracklets.split_on_appearance | true | split on abrupt appearance change | engineering | tracklet | AVOID | feature toggle |
| tracklets.split_min_sim | 0.7 | split if mean-embedding cosine across best cut is below | measured (calibrated on scene 075 / CLIP-ReID: same person >= 0.88, different <= 0.58, CL line 183) | tracklet | T2 later | needs a rerun, already calibrated |
| tracklets.split_min_samples | 3 | min samples each side of a split | engineering | tracklet | T2 later | minor |
| tracklets.max_history | 400 | samples kept per open track | engineering (CL line 186, memory bound) | tracklet | AVOID | memory only |
| tracklets.min_points | 5 | drop tracks shorter than this (flicker) | engineering (CL line 76) | tracklet | T2 later | minor |
| tracklets.crop_pad | 0.0 | padding of saved crops | unknown | tracklet | AVOID | display only |
| tracklets.save_crops | true | write best crop | engineering | tracklet | AVOID | UI/storage |

## face (all T3: scene 075 has no face cameras, showcase has no GT)

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| face.enabled | true | master switch | engineering | embedding | AVOID | feature toggle |
| face.model | buffalo_l (lite buffalo_sc, max antelopev2) | InsightFace pack | engineering | embedding | AVOID | preset choice |
| face.det_size | 640 (lite 480) | face detector input | v1 (640) | embedding | AVOID | speed |
| face.det_conf | 0.7 | min face detection score | v1 (`FACE_CONF_THRESHOLD`) | embedding | T3 fixed | no face eval data |
| face.head_region | 0.4 | face centre in top fraction of body box | v1 | embedding | T3 fixed | structural pairing rule |
| face.max_faces | 5 | best faces kept per tracklet | engineering (TR line 177) | embedding | T3 fixed | no face eval data |
| face.min_face_px | 32 | min face side for quality > 0 | engineering (CL line 74) | embedding | T3 fixed | no face eval data |
| face.full_size_px | 112 | face side where size factor saturates | engineering (CL line 74) | embedding | T3 fixed | no face eval data |
| face.save_faces | true | write face crop | engineering | embedding | AVOID | storage |
| face.crop_margin | 0.15 | margin on saved face crop | unknown | embedding | AVOID | display only |

## association (identity stage)

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| association.min_cos | 0.78 | absolute cosine floor for a body match | v1 (`MIN_COS_SIM`; not in Casao; CL line 49) | identity | **T1 tune** | main merge/split trade-off, body-only, continuous |
| association.tau | 2.0 | top-2 softmax ratio for a body match | paper (Casao tau = 2) | identity | **T1 tune** | open-set decision; coupled with `softmax_temp`, tune one of them |
| association.softmax_temp | 10.0 | softmax inverse temperature | paper (Casao v = 0.1); v1 used 15 (CL line 33) | identity | T3 fixed | margin ln(tau)/temp is tuned through `tau`; also drives gallery entropy (store.py:64) |
| association.sample_vote | 0.5 | fraction of samples agreeing with tracklet gid | engineering (our addition, CL line 45) | identity | **T1 tune** | extra guard against mixed tracklets, 0 = off |
| association.udm_eps | 0.15 | DBSCAN eps (cosine distance) on pooled tracklets | v1 (Casao gives none, CL line 41) | identity | **T1 tune** | controls how pooled tracklets cluster |
| association.udm_min_samples | 20 | samples a pooled cluster needs | paper (Casao l = 20; v1 used 5) | identity | **T1 tune** | tracklets have <= 16 samples, so this means >= 2 pooled tracklets; strongly affects fragmentation |
| association.udm_new_dist | 0.1 | min distance to gallery above which a cluster mints a new id | paper (Casao alpha = 0.1) | identity | **T1 tune** | new-identity vs join |
| association.bootstrap_q | 20 | below this many ids, unmatched tracklets mint directly | paper (Casao Q = 20) | identity | **T1 tune** | with ~10 people on scene 075 bootstrap is always active, so it is effectively the whole minting rule there; also T3-ish, see note below |
| association.bootstrap_min_samples | 8 | samples needed to mint in bootstrap | unknown (not in docs or paper) | identity | **T1 tune** | gate on bootstrap minting |
| association.bootstrap_join_cos | 0.9 | bootstrap: join an existing gid above this cosine | paper (Casao epsilon = 0.9) | identity | **T1 tune** | bootstrap is active for the whole of scene 075 (< 20 people), so this decides join vs mint there |
| association.udm_k | 3 | candidate classes per sample | paper (Casao k = 3) | identity | T3 fixed | paper constant |
| association.udm_every | 20 | run UDM every n pooled tracklets | engineering (CL line 19) | identity | T3 fixed | scheduling, end_camera/flush run it anyway |
| association.gallery_budget | 50 | gallery size m per identity | paper (Casao m = 50) | identity | T3 fixed | paper constant |
| association.gallery_gamma | 0.6 | drop rule: entropy vs redundancy | paper (Casao 0.6; v1 0.5) | identity | T3 fixed | paper constant |
| association.diversity_ref | min | diversity gate reference | paper (Casao Eq. 6) | identity | T3 fixed | structural choice |
| association.proto_k | 3 | prototypes per identity | paper (AM&B N = 3; v1 K = 3) | identity | T3 fixed | prototypes only used for search/UI, not assignment |
| association.proto_method | alpha_fps | prototype algorithm | engineering | identity | AVOID | categorical, not used by assignment |
| association.proto_alpha | 0.5 | alpha_fps quality vs distance weight | v1 | identity | T3 fixed | prototypes only |
| association.gcp_alpha | 0.5 | gcp interpolation | unknown (paper does not state it) | identity | T3 fixed | prototypes only |
| association.face_match | 0.55 | face cosine for a known gid | v1 | identity | T3 fixed | no face eval data |
| association.face_margin | 0.05 | lead of best face gid over second | engineering (CL line 44) | identity | T3 fixed | no face eval data |
| association.face_conflict | 0.30 | face cosine below which gids are cannot-linked | engineering (CL line 18) | identity | T3 fixed | no face eval data |
| association.face_min_obs | 2 | faces needed for face evidence | engineering (CL line 43) | identity | T3 fixed | no face eval data; also fixed at run time (see replay) |
| association.overlap_tolerance_s | 0.0 | same-camera overlap tolerated before cannot-link | unknown | identity | T3 fixed | structural, 0 is the logical value |
| association.use_topology | true | topology cannot-links | engineering | identity | AVOID | feature toggle, no topology in scene 075 YAML |

Note on `bootstrap_q`: the code reads `len(bank.gids()) < bootstrap_q` (`associate.py:242`). With about 10 people it
never leaves bootstrap on scene 075, which hides the UDM; if the study should exercise the UDM, treat `bootstrap_q`
as a switch (set it low in a second study) rather than a smooth parameter.

## consolidation

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| consolidation.merge_threshold | 0.35 | stop merging above this 1 - cos distance | measured (CL line 216: 0.25 -> 0.35 on warehouse showcase, identical on scene 075 at IDF1 89.4%) | identity | **T1 tune** | direct split/merge dial; 0.25 vs 0.35 already show divergent behaviour on real data |
| consolidation.linkage | average | agglomerative linkage | engineering | identity | T3 fixed | structural, categorical |
| consolidation.use_faces | true | face must/cannot-links | engineering | identity | AVOID | feature toggle |
| consolidation.enabled | true | run consolidation | engineering | identity | AVOID | feature toggle |
| consolidation.include_existing | true | include identities already in data_dir | engineering | identity | AVOID | feature toggle |
| consolidation.min_samples | 2 | identities with fewer gallery samples not clustered | unknown | identity | T3 fixed | minor |

## search / render (not part of the identity pipeline metrics)

| name | default | what it controls | origin | stage | tier | reason |
|---|---|---|---|---|---|---|
| search.body_min_cos | 0.75 | min top-1 cosine for a body match | v1 (`th_emb`) | search | T3 fixed | query metrics are separate (held-out retrieval), out of scope |
| search.tau | 1.5 | ambiguity ratio | v1 (function default) | search | T3 fixed | as above |
| search.softmax_temp | 10.0 | softmax temperature | paper (Casao); v1 15 | search | T3 fixed | as above |
| search.top_k | 10 | neighbours before aggregation | unknown | search | T3 fixed | as above |
| search.face_min_cos | 0.55 | face match floor | v1 | search | T3 fixed | no face data |
| search.face_margin | 0.05 | face lead over second | engineering | search | T3 fixed | no face data |
| search.face_top_k | 5 | face neighbours | v1 | search | T3 fixed | no face data |
| search.face_pad | 1.0 | gray pad around face queries | v1 | search | T3 fixed | no face data |
| render.* (scale, trail_len, box_thickness, show_hud, crf, out_fps, unidentified_color, heat_sigma, heat_scale, heat_alpha, fill_alpha, dim_alpha, mosaic_tile_width, journey_width, gif_fps, gif_width) | see config.py | video / heatmap / GIF appearance | v1 for heat_sigma, heat_scale, heat_alpha; engineering for the rest | render | AVOID | UI only (16 fields) |

## Hardcoded constants that act as hyperparameters (not in config)

| location | value | what it controls | origin | stage | tier |
|---|---|---|---|---|---|
| `tracking/tracklets.py:36` | 17 keypoints | denominator of the visibility ratio | engineering (COCO pose) | tracklet | T3 fixed |
| `tracking/tracklets.py:301` | 0.1 | margin around the head-region crop for face detection | v1 (CL line 73 describes "top 40% + 10% margin") | tracklet | T3 fixed |
| `tracking/tracklets.py:304` | 8 px | min head crop side | unknown | tracklet | T3 fixed |
| `tracking/tracklets.py:255` | JPEG quality 90 | best-crop file quality | engineering | tracklet | AVOID |
| `models/face.py:100` | `det_thresh=0.5` | InsightFace internal detector threshold (before `face.det_conf` 0.7) | unknown | embedding | T3 fixed |
| `models/face.py:42,47,48` | 0.5 (yaw), 0.6 / 0.5 (pitch), 0.5 + 0.5 x pitch | face frontalness factor | engineering (CL line 74) | embedding | T3 fixed |
| `identity/gallery.py:34` | EPS 1e-9 | numeric guard | engineering | identity | AVOID |
| `identity/udm.py:93,102` | distance 2.0 | cannot-link distance fed to DBSCAN | engineering | identity | AVOID |
| `identity/udm.py:94` | DBSCAN `min_samples=1` | one tracklet is enough to form a cluster (sample count handled by `udm_min_samples`) | engineering (documented in udm.py docstring) | identity | T3 fixed |
| `identity/prototypes.py:40` | 1e-12 | min-max guard | engineering | identity | AVOID |
| `identity/store.py:28` | `PROTO_STRIDE` 1000 | prototype key = gid x 1000 + index | engineering | identity | AVOID |
| `models/planner.py:26,29` | `SAFETY_MARGIN` 0.12, `CONTEXT_MB` 100 | VRAM planning | engineering | runtime | AVOID |
| `models/planner.py:36` | `COSTS` table | measured VRAM per model | measured (CL line 214) | runtime | AVOID |
| `models/detector.py:17` | `RFDETR_PERSON_ID` 1 | class id | engineering | detection | AVOID |
| `tracking/tracker.py` | none | tracker parameters all come from `tracking.*` | n/a | tracking | n/a |
| `search.py` | none | thresholds all come from `search.*` | n/a | search | n/a |

No other magic thresholds were found in `identity/` (constraints.py compares only against configured values:
`overlap_tolerance_s`, `face_conflict`, topology `min_transit_s`).

## Tier counts

Counted as table rows (a few rows group several fields: runtime.start_s/end_s/max_frames, body_reid.input_h/w,
render.* with 16 fields in one AVOID row, so the field count is higher for AVOID).

| tier | rows |
|---|---|
| T1 tune | 10 |
| T2 later | 20 |
| T3 fixed | 38 |
| AVOID | 50 (render.* row stands for 16 fields) |

## Replay feasibility

**Verdict: mostly yes for body-only data (scene 075), not exactly for faces.** The identity stage can be re-run
from a database without detection, tracking or embedding, using a small replay driver. No such driver exists
today; `IdentityEngine.on_tracklet` cannot be called as-is on stored tracklets (it is idempotent and returns early
if samples already exist, `associate.py:198`: `if db.get_samples(self.conn, tracklet_id=t.id): return ...`).

What `on_tracklet` reads from a `Tracklet` (`associate.py:192-250`) and where it is persisted:

| needed | persisted? | where |
|---|---|---|
| `id`, `camera_id`, `start_ts`, `end_ts`, `run_id` | yes | `tracklets` table (`storage/db.py:42-60`) |
| per-sample body embeddings | yes | vector collection `body_samples`, key = `samples.id` (`identity/store.py:10-13`, `associate.py:204-205`) |
| per-sample quality | yes | `samples.quality` (`db.py:92-99`) |
| tracklet body descriptor (`body_embedding`) | yes | vector collection `body_tracklet` (`associate.py:226`); also recomputable from samples |
| tracklet face descriptor (`face_embedding`) | only if the run had >= `face_min_obs` faces | `face_tracklet` (`associate.py:227-228`) |
| per-face embeddings (`t.faces[i].embedding`, `.quality`) | **no** | used for the face support count in `_face_step` (`associate.py:153-173`) and `len(fobs)` for `_blocked` |
| samples of tracklets below `min_samples` | **no** | `on_tracklet` returns before persisting them (`associate.py:203-205`), so lowering `tracklets.min_samples` in a replay is impossible; raising it works |
| cameras and topology | yes (cameras table) / camera YAML | `db.py:33-44`; topology only in the YAML (`CameraSet`), not in the DB |

Missing or lossy items:
1. Per-face embeddings and face qualities are not stored, only one descriptor per tracklet, and only when it had
   >= `face_min_obs` faces. Face replay is therefore approximate (`face_min_obs`, `max_faces` cannot be varied;
   the support check `faces @ cents[best] >= face_match` cannot be reproduced). Irrelevant for scene 075.
2. Vectors and DB state are written back by the engine, so the replay needs a fresh copy of the data folder with
   `tracklets.gid/assign_*` cleared, `samples.in_gallery` cleared, `identities`, `tracklet_gid_history`,
   `body_proto` and the vector labels reset (all `body_*` labels to -1), and the embeddings of `body_samples`
   kept in memory (the engine inserts new sample rows, so keep the stored embeddings and qualities and rebuild
   `Tracklet.samples` objects, or add a replay path that skips the idempotency check and reuses sample ids).
3. Camera topology (`min_transit_s`) lives in the camera YAML, not in the DB; pass it to `IdentityEngine`.

**Replay order.** The engine is stateful and order dependent: tracklets arrive per camera, cameras one after
another (`pipeline.py` loops cameras sequentially; `commit` at `pipeline.py:215` inserts and calls `on_tracklet`
in closing order), with `end_camera` after each camera and `end_run` at the end (CLAUDE.md, `associate.py:296-346`).
Order is recoverable from the database: `tracklets.id` is autoincremented at insertion (`insert_tracklets`,
`db.py:282`), so ascending id equals closing order within a camera, and cameras appear as contiguous id ranges
(the camera order is the order of `min(id)` per camera). Do not use `get_tracklets` ordering, which is
`ORDER BY start_ts, id` (`db.py:389`), and do not use the `cameras` table order (`ORDER BY id`, `db.py:271`,
alphabetical). The `end_camera` call points are the id range boundaries. With several runs in one database,
group by `run_id` first (state accumulates across runs, `IdentityEngine` reloads state on init).
