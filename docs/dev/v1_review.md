# v1 Review and Porting Spec (Person Re-ID + Face Association)

Basis: the code cells of `module4_face_association.ipynb`, `module5_search_engine.ipynb`, `module6_dashboard.ipynb`, plus `docs/technical_report.md` and `docs/design_discussion.md`.
Citations: `m4 c5` = module 4, 5th **code** cell (markdown cells not counted); `m6 c5` etc. alike.
Cell maps:
- m4: c1 pip, c2 config, c3 models/cloud, c4 classes (SightingTracker, UnknownDataManager, GalleryManager), c5 helpers + hydration + main loop, c6 upload/persist.
- m5: c1 pip, c2 creds, c3 models/embed fns, c4 prototype viz (broken), c5 `search_person`, c6 `search_face`, c7/c8 URL tests, c9 body brute-force, c10 face brute-force, c11 prototype viz.
- m6: c1 pip, c2 creds, c3 models + `_make_botsort_tracker`, c4 classes/helpers, c5 persistence/pipeline/search/heatmap, c6 Gradio UI, c7 launch.

"Newer/better" verdicts: m6 is the newest notebook but is NOT uniformly better. Each difference is called out as `[m6 better]`, `[m4 better]` or `[neutral]`.

Verified outside the repo: boxmot 16.0.11 / 17.0.0 wheel source was read for CLIP-ReID preprocessing and output dim (see 1.4). The notebooks install boxmot unpinned, so the version actually used is unknown. See Bug B21.

---------------------------------------------------------------------------
# 1. PORTING SPEC
---------------------------------------------------------------------------

## 1.0 End-to-end data flow (one video, one camera)

```
frame (BGR) -> RT-DETR person det -> BotSort (track_id) -> conf>=0.75 gate -> raw bbox crop
  -> YOLO-pose gate (visible kpts/17 >= 0.5) -> CLIP-ReID 1280-D L2-normed emb  (batched per frame)
  [face cam: InsightFace on FULL frame -> face spatially paired to each body -> ArcFace 512-D -> Pinecone face lookup]
  -> per-track identity resolution (sticky local_to_global_map[tid] -> gid)
        gid known?  -> add crop to gallery[gid]
        face hit?   -> merge or forced_gid
        else        -> softmax/tau live classify vs gallery centroids -> else UDM pool (DBSCAN every 30 frames)
  -> SightingTracker keeps ONE record per track_id (best crop, counts, timestamps)
  -> END OF VIDEO ONLY: close all tracks, upload best crops (ImageKit), upsert face vectors (Pinecone),
     push sightings/faces (MongoDB), serialize gallery (m4: pickle + zip of crops; m6: Pinecone namespace)
  -> (m5/m6) prototype build: alpha-FPS K=3 per gid -> Pinecone `supermarket-reid`, default namespace
```

## 1.1 Models, devices, config constants

| Item | m4 (c2/c3) | m6 (c3/c5) | Notes |
|---|---|---|---|
| Detector | `RTDETR('rtdetr-l.pt')` on `cuda:0` | same | Called `detector(frame, classes=[0], verbose=False)`. Ultralytics defaults (conf 0.25, imgsz 640). Output `boxes.xyxy, .conf, .cls`. |
| Tracker | `BotSort(reid_weights='osnet_x0_25_msmt17.pt', device=cuda:0, half=False)`, one global instance | `_make_botsort_tracker()` (see 1.2), new instance per run | All other BotSort args are library defaults. |
| Pose | `YOLO('yolo26s-pose.pt')` | `YOLO('yolo11s-pose.pt')` | **Differs**; the tech report says YOLOv11s. Device `cuda:1` if 2 GPUs else `cuda:0`. |
| Body ReID | `ReID(weights='clip_market1501.pt', device=REID_DEVICE, half=False)` | same | The class is found by scanning `boxmot` for a class named `ReID` or `ReIDDetectMultiBackend`. |
| Face | `FaceAnalysis(name='buffalo_l', providers=[CUDA, CPU])`, `prepare(ctx_id=0, det_size=(640,640))`, only if `IS_FACE_CAMERA` | always loaded | SCRFD `det_10g` + ArcFace `w600k_r50`. `ctx_id=0` means GPU 0 regardless of `REID_DEVICE`. |
| Devices | `DET=cuda:0`; `POSE=REID=cuda:1` if `device_count>1` else `cuda:0` | same | |
| Cloud | ImageKit (private key only), Pinecone (2 indices), Mongo `supermarket_db.persons` | same + ImageKit/Pinecone clients named `imagekit_client`, `pinecone_*_index` | Secrets via Kaggle `UserSecretsClient`; m6 falls back to env vars, m4 to placeholders. |

Global constants (m4 c2; m6 exposes the same as `run_video_pipeline` params/sliders):

| Name | Value | Meaning |
|---|---|---|
| `IS_FACE_CAMERA` | True | Master switch for all face logic. m6 passes `is_face_camera`. |
| `CAMERA_ID` | m4 `"Face_Cam_Entrance"`; m6 `"cam_01"` | Free-text camera label, stored per sighting. m6 mutates the global from a textbox. |
| `VIDEO_START_TIME` | m4 `2026-05-05 09:00:00 UTC` (constant) | **m6 uses `datetime.now(UTC)` at processing time.** See B1. |
| `PINECONE_INDEX_NAME` / `PINECONE_FACE_INDEX_NAME` | `supermarket-reid` (1280, cosine) / `supermarket-faces` (512, cosine) | Must be created manually. |
| `REID_CONF_THRESHOLD` | 0.75 | Applied to the tracker output conf (`t[5]`). |
| `POSE_CONF_THRESHOLD` | 0.3 | Per-keypoint. |
| `MIN_VISIBLE_RATIO` | 0.5 | Fraction of 17 keypoints. |
| `FACE_CONF_THRESHOLD` | 0.7 | InsightFace `det_score`, strict `>`. |
| `FACE_MATCH_THRESHOLD` | 0.55 | Pinecone cosine. m4 uses `>`, m6 `>=`. |
| `TAU` | 2.0 | Top-2 softmax ratio. |
| `MIN_COS_SIM` | 0.78 | Absolute floor for live gallery match. |
| `MAX_IMAGES` | 50 | Per-identity gallery budget (m6 slider 10-100). |
| `UDM_MIN_SIZE` | 5 | DBSCAN `min_samples`, also the minimum pool size. |
| `UDM_EPSILON` | 0.15 | DBSCAN eps on `1-cos`, so cos >= 0.85. |
| Softmax temperature | 15.0 | Hard-coded in `classify_live_track`, `_optimize_budget` and `search_person`. |
| DBSCAN cadence | every 30 frames (`frame_count % 30 == 0`) | Also requires `len(pool) >= UDM_MIN_SIZE`. |
| Gallery cost weights | 0.5 uncertainty + 0.5 redundancy | |
| SightingTracker fps | m4 hard-coded 30; m6 read from video (`CAP_PROP_FPS or 30`) | `[m6 better]` |
| Upload threads | m4 10 (ThreadPool); m6 sequential | |
| Face spatial rule | face centre inside body box and `cy <= y1 + 0.4*(y2-y1)` | |
| Search | body `th_emb=0.75`, `tau=1.5` (m5 fn default) / 2.0 (m6 slider), top_k=10; face `0.55`, top_k=5 | |
| alpha-FPS | K=3, alpha=0.5 | |
| Heatmap | `sigma_small=4` (fn default) / 5 (UI), `heat_scale=0.10`, `alpha=0.6` | |
| Gallery persistence m6 | namespace `gallery_buffer`, max 500 gids probed, 50 embs/gid, 100/batch | |

## 1.2 Detection / tracking / BotSort workaround

Per-frame order (m4 c5 main loop, m6 `run_video_pipeline`):
1. `boxes = detector(frame, classes=[0])[0].boxes`. If none, **skip the whole frame without calling `tracker.update`**, so BotSort never ages its tracks on empty frames.
2. If `is_face_camera`: `faces_in_frame = face_app.get(frame)` (full frame, every frame, before tracking).
3. `dets = cat([xyxy, conf, cls]) -> numpy (N,6)`. `tracks = tracker.update(dets, frame)`. Rows are `[x1,y1,x2,y2,track_id,conf,cls,det_ind]`, so `t[4]` is the id and `t[5]` the conf.
4. Keep tracks with `conf >= REID_CONF_THRESHOLD`. Box is `int()`, `x1,y1` clamped to >= 0, `x2,y2` not clamped (numpy slicing clamps). Crop is `frame[y1:y2, x1:x2]`, the raw bbox with no padding. Skip if `crop.size == 0`.

**BotSort workaround (m6 c3 `_make_botsort_tracker`)**
- Why: `osnet_x0_25_msmt17.pt` download 404'd on Kaggle. m4 still requests it and will break the same way.
- What: it uses `Path("clip_market1501.pt")` as the tracker's ReID weights, since that file is known to download. It inspects `BotSort.__init__` and sets the first present of `model_weights`, `reid_weights` (as `str`), `weights`, or `reid_model` (= `reid_model.model`, an instance). It also sets `device=DET_DEVICE` if `device` exists, and `fp16=False` or `half=False`.
- Side effect: the tracker now runs a **second CLIP ViT-B/16 on cuda:0 for every detection above BotSort's low threshold, every frame**. This is on top of the pipeline's own CLIP pass on gated crops. m4's OSNet is similarly redundant. See W1.
- Library defaults that govern behaviour (from boxmot 16.x source; verify against the pinned version): `track_high_thresh 0.5`, `track_low_thresh 0.1`, `new_track_thresh 0.6`, `track_buffer 30` frames, `match_thresh 0.8`, `proximity_thresh 0.5`, `appearance_thresh 0.25`, `cmc_method "ecc"`, `with_reid True`.
- `tracker_template = _make_botsort_tracker` (factory); `run_video_pipeline` calls `tracker_template()` once per run `[m6 better]`: fresh track ids and state per video.

## 1.3 Quality gate (pose)

Input: list of raw crops (all gated tracks of the frame), `pose_model(crops, verbose=False)`.
```
for each result res:
    if res.keypoints is not None and len(res.keypoints.data) > 0:
        visible = count(kp.conf > POSE_CONF_THRESHOLD for kp in res.keypoints.data[0])   # FIRST detected person in the crop
    else: visible = 0
    ratio = visible / 17.0
    keep if ratio >= MIN_VISIBLE_RATIO      # >= 9 of 17 keypoints
    item.ratio = ratio; item.score = tracker conf
```
- `data[0]` is whichever person the pose model ranks first in the crop, not necessarily the target. The 5 head keypoints count equally, so a person seen from behind can pass.
- Quality for the best-crop choice is `conf * ratio`.
- Frames with no gated or pristine detections are skipped. In m6 this also skips the DBSCAN block (see B9).

## 1.4 Body embedding extraction

```python
def extract_reid_features(crops):  # crops: list[np.ndarray BGR HxWx3], variable sizes
    with torch.no_grad():
        e = reid_model(crops)  # np.ndarray or tensor, shape (N,1280)
        if isinstance(e, np.ndarray):
            e = torch.tensor(e)
        return F.normalize(e, p=2, dim=1).to(REID_DEVICE)  # cosine == dot
```
- m5 `embed_body` and m6 `embed_body_image` are the same with a single crop. m5's docstring says "512-d", which is wrong.
- **Weights**: `clip_market1501.pt`. This is the CLIP-ReID checkpoint (ViT-B/16 image encoder, SIE off, fine-tuned on Market-1501), auto-downloaded by boxmot from Google Drive id `1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7`. Original project: Syliz517/CLIP-ReID.
- **Where 1280 comes from** (boxmot `backbones/clip/make_model.py::forward`, eval, `NECK_FEAT="after"`): `torch.cat([bottleneck(cls_token_768), bottleneck_proj(proj_512)], dim=1)` = **768 + 512 = 1280**. Both parts are BatchNorm-neck outputs of the ViT cls token. The first is the pre-projection 768-D feature, the second the CLIP-projected 512-D feature. Then the notebook L2-normalises the whole 1280 vector.
- **Preprocessing (boxmot `ReID backend get_crops`, v16.0.11; the notebook calls the wrapper with a list of full crops, so the whole crop is the box)**:
  - `cv2.resize(crop, (W=128, H=256), INTER_LINEAR)`. This is a plain stretch with no aspect preservation.
  - `BGR->RGB`, `/255`, `(x-0.5)/0.5` per channel (CLIP models use mean=std=0.5; others ImageNet mean/std).
  - `half=False` (fp32).
- Caveat: v16/v17 `ReID.__call__(frame, dets)` takes two args, whereas the notebooks call `reid_model(crops)`. The notebook version is therefore a different boxmot release, or the call only works by luck. The rewrite should load CLIP-ReID directly (or pin boxmot) and implement the preprocessing above explicitly.
- Batched per frame over `pristine_data`. Embeddings stay on `REID_DEVICE` (GPU tensors are stored in the gallery).

## 1.5 Face detection, embedding, and spatial pairing (face cameras only)

- Detection: `face_app.get(frame)` on the full frame, `det_size=(640,640)`, library default `det_thresh=0.5`. The pipeline then filters on `det_score > 0.7`. Each face has `.bbox`, `.det_score`, `.normed_embedding` (512-D, already L2-normed, w600k_r50 on 5-point-aligned 112x112).
- Pairing `get_spatially_matched_face(body_box, faces)`:
  ```
  body_top = y1 + 0.4*(y2-y1)
  candidates = faces with x1 <= fcx <= x2 and y1 <= fcy <= body_top
  m4: return candidate with max det_score          [m4 better]
  m6: return FIRST candidate in list order
  ```
  No exclusivity: one face can pair to two overlapping bodies.
- Only runs for bodies that passed the conf and pose gates.
- Face crop for storage: `frame[max(0,fy1):fy2, max(0,fx1):fx2]` (tight bbox, no margin).
- `update_face` keeps, per track, the face with the highest `det_score` (detection confidence, not quality). It stores image, embedding and conf.
- Face lookup `search_face_pinecone(emb)`: `face_index.query(vector, top_k=1, include_metadata=True)`. Accept if `score > 0.55` (m6 `>=`), return `int(metadata['global_id'])`. m4 has no try/except (exception kills the run). m6 swallows every exception and returns None.
  - **Called every frame for every paired body** (network call in the hot loop).
- Pre-cropped face query embedding (m5 `embed_face_crop`, m6 same): place the chip in the centre of a gray(128) canvas of size `(h+2p, w+2p)` with `p = max(h,w)`, run `embed_face` (detect, take highest `det_score`, return `normed_embedding`). Both m5 and m6 do this. For a square chip this makes a 3x canvas.

## 1.6 GalleryManager (m4 c4; m6 c4 near-identical)

Data: `classes: dict[gid] -> {'embs': [tensor1280 on REID_DEVICE], 'ratios': [float], 'scores': [float], 'filepaths': [Path]}`, `gids: list` (insertion order), `max_images`, `current_max_id`.
- `mint_new_class()`: `current_max_id += 1` (m6 additionally skips ids already in `classes`) `[m6 better]`. Calls `_init_class`, which makes dir `DB_IMAGES_DIR/global_id_{gid:03d}`.
- `mint_class_at(gid)`: creates an empty class at a specific gid (face-forced) and bumps `current_max_id`.
- `get_weighted_centroids()`: for each non-empty class, `normalize(sum(emb_i * ratio_i))`. Weights are the **pose visibility ratio**, not the det score. Returns `(tensor[G,1280], active_gids)`. Recomputed from scratch on every call (no cache).
- `classify_live_track(emb, tau, exclude_gids)` (m6 adds a `min_cos` param):
  ```
  if no classes: return (None,0,0)
  sims = centroids @ emb                       # cosine
  for gid in exclude_gids: sims[gid] = -inf
  p = softmax(sims * 15)                       # excluded -> 0
  sort desc; max_prob=p[0]; best_sim=sims[top]; best_gid
  if best_sim == -inf: return (None,0,0)
  if best_sim < MIN_COS_SIM: return (best_gid, 0.0, max_prob)   # ratio 0 => caller treats as unknown
  ratio = p[0]/(p[1]+1e-9) if G>1 else inf
  return (best_gid, ratio, max_prob)
  ```
  Equivalent test: `ratio = exp(15*(s1-s2))`, so the accept condition is `s1-s2 >= ln(tau)/15` (tau=2 -> 0.0462, tau=1.5 -> 0.027) AND `s1 >= 0.78`. It is a margin test. If all other classes are excluded, `p2=0` and ratio is huge, so it accepts on the 0.78 floor alone.
- `add_observation(gid, item)`: append emb/ratio/score; write `frame_{frame:06d}_trk_{tid:03d}.jpg` to `DB_IMAGES_DIR/global_id_{gid:03d}/` (**a JPEG write per observation, every frame**); append path; call `_optimize_budget(gid)`.
- `_optimize_budget(gid)` (runs only if `len(embs) > max_images`; removes exactly ONE entry per call):
  ```
  E = stack(embs); S = E@E.T; fill_diag(-1); redundancy = S.max(dim=1)          # sim to nearest other crop
  C = weighted_centroids (ALL classes)
  if C.shape[0] > 1: P = softmax(E@C.T * 15, dim=1); H = -sum(P*log(P+1e-9)); uncertainty = H / log(C.shape[0])
  else: uncertainty = 0
  cost = 0.5*uncertainty + 0.5*redundancy ; drop argmax(cost)
  unlink file if it exists and 'working' in path (never deletes /kaggle/input files)
  pop index from embs/ratios/scores/filepaths   (guards for length desync)
  ```
  Entropy normaliser was `log(emb_dim)` and is now `log(n_classes)` (fixed). Cost ignores crop quality and recency, and can drop the crop just added.

## 1.7 Identity resolution decision logic (m4 c5 loop; m6 `run_video_pipeline` identical modulo params)

State: `local_to_global_map: dict[track_id -> gid]` (sticky, whole video), `claimed` = per-frame set of gids, `unresolved` = per-frame list.

```
for each pristine detection d (embedding e_i):
    item = {emb, ratio, score, img=crop, track_id, frame}
    gid = L2G.get(tid)
    sighting_tracker.update_track(tid, frame, score, ratio, crop, gid, box)      # always, before resolution
    if IS_FACE_CAMERA:
        f = spatially_matched_face(box, faces)
        if f and f.det_score > FACE_CONF:
            sighting_tracker.update_face(tid, face_crop, f.normed_embedding, f.det_score)
            ftg = search_face_pinecone(f.normed_embedding)        # gid or None, vs vectors from PREVIOUS runs only
            if ftg:
                if gid and gid != ftg: execute_identity_merge(ftg, gid); gid = ftg     # irreversible, one-frame evidence
                elif not gid:          item['forced_gid'] = ftg
    gid = L2G.get(tid)                      # re-read
    if gid is not None:
        if gid not in claimed: claimed.add(gid); gallery.add_observation(gid, item)   # ALWAYS adds, no similarity check
        else: del L2G[tid]; unresolved.append(item)                                   # duplicate-gid conflict in this frame
    else: unresolved.append(item)

for item in unresolved:
    if item.forced_gid is not None:
        mint_class_at(forced) if absent; L2G[tid]=forced; claimed.add(forced)
        gallery.add_observation(forced,item); sighting.assign_global_id(tid,forced)
        m4: create_person_document(forced)               (m6 defers doc creation to save step)
    elif gallery.num_classes()==0:
        gid = mint_new_class(); L2G[tid]=gid; claimed.add; add_observation; assign; m4: create_person_document
    else:
        gid, ratio, _ = classify_live_track(emb, TAU, exclude_gids=claimed)
        if gid is not None and ratio >= TAU: L2G[tid]=gid; claimed.add; add_observation; assign
        else: udm.add(item)

every 30 frames (if len(udm.pool) >= 5): actions = udm.run_dbscan(gallery)
    for (act, gid, cnt, tids): MINT -> create_person_document (m4 only); for tid in tids: assign_global_id + L2G[tid]=gid
```
- Entropy appears ONLY in `_optimize_budget`, not in the live decision. The live decision is the softmax-ratio margin plus the cosine floor.
- The VIP/forced path never goes through the UDM in current code: forced items are handled directly in the unresolved loop. The `FACE_VIP_BYPASS` branch inside `run_dbscan` is **dead code** (see B11).
- `forced_gid` is not checked against `claimed`.
- No temporal smoothing, voting, or re-evaluation: first assignment wins for the life of the track.

## 1.8 UnknownDataManager (DBSCAN minting)

`pool: list[item]` (each item holds the full crop array and a GPU embedding).
```
run_dbscan(gallery):
  if len(pool) < min_samples: return []
  D = clip(1 - (E@E.T), 0, 2); labels = DBSCAN(eps=0.15, min_samples=5, metric='precomputed').fit(D)
  for each label != -1:
      items = cluster; tids = unique track_ids
      if any item.forced_gid: gid = mode(forced); mint_class_at(gid) if absent; add all items; action FACE_VIP_BYPASS; continue   # dead path
      mean = normalize(mean(embs))
      if gallery has classes: sims = weighted_centroids @ mean; best = argmax
            if (1 - best_sim) <= eps (i.e. cos >= 0.85): add all items to that gid; action MERGE; continue
      gid = mint_new_class(); add all items; action MINT
  pool = [items with label -1]        # noise stays forever, no expiry
```
- No softmax or ambiguity check on MERGE. No check that the target gid is currently claimed by a visible track. No per-cluster track-count minimum, so 5 frames of ONE track can mint a person.
- No final flush at end of video (see B8).

## 1.9 execute_identity_merge (face-driven retroactive merge)

**m4 c5** `execute_identity_merge(target_gid, old_gid)` (globals):
1. Remap every `local_to_global_map` entry old->target. `sighting_tracker.assign_global_id` per such tid (active sightings only; all tracks are active until end of video).
2. If `old_gid in gallery.classes`:
   - `mint_class_at(target)` if absent.
   - Pop old class; remove from `gids`.
   - For each old entry whose file exists: copy (path contains `input`) or move into `global_id_{target}/` and append emb/ratio/score/path to the target. Entries whose files are missing are **dropped**.
   - `while len(target) > max_images: _optimize_budget(target)`.
3. Mongo: `old_doc = find_one(old)`. If it exists: `update_one({global_id: target}, {$push: {sightings: {$each: old.sightings}, faces: {$each: old.faces}}})` with **no upsert**, then `delete_one(old)`.
4. Pinecone faces: `face_index.query(vector=[0.0]*512, top_k=100, filter={global_id: old})`, then `update(id, set_metadata={..., global_id: target})` for each match. Wrapped in try/except. Likely rejected by Pinecone because of the all-zero cosine query (see B6).

**m6 c4** `execute_identity_merge(target, old, gallery_ref, local_map_ref, sighting_ref)`: in-memory only.
- Early return if `old not in classes or target == old`.
- Same gallery move, same path rule.
- Remap local map and active sightings.
- **No Mongo update, no Pinecone face update, no budget re-prune** `[m4 better]` (see B5, B7).

## 1.10 SightingTracker (m4 c4, m6 c4; m6 takes `video_start_time`)

`active_sightings: dict[tid] -> record`; `completed_sightings: list`; `active_faces`, `completed_faces`.
- `_frame_to_ts(frame) = video_start_time.timestamp() + frame/fps` (float epoch seconds). Frame numbers are 1-based (`frame_count` increments before use).
- `update_track(tid, frame, conf, ratio, img, gid, bbox)`:
  - First call creates `{camera_id (GLOBAL CAMERA_ID), track_id, global_id, entry_ts, exit_ts, duration_sec 0, frame_start, frame_end, n_detections 1, sum_det_score, sum_skeleton_ratio, best_crop_quality = conf*ratio, best_crop_img, best_crop_bbox [x1,y1,x2,y2], best_crop_frame}`.
  - Later calls update `exit_ts`, `frame_end`, `duration_sec`, `n_detections`, the sums, and replace the best crop/bbox/frame when `conf*ratio` is higher.
  - Only pristine (quality-gated) frames update it.
- `update_face(tid, face_img, emb, conf)`: keep the highest-conf face per track.
- `assign_global_id(tid, gid)`: overwrite the sighting's gid. The last write wins.
- `close_track(tid)`: **only called for all remaining tracks AFTER the video loop ends** (the tech report claims it is called when BotSort drops a track, which is false). If `global_id is None` the sighting AND its face are silently discarded. Otherwise it computes `avg_det_score`, `avg_skeleton_ratio`, deletes the sums, appends to `completed_sightings`, and appends a face record `{best_face_img, best_face_emb, best_face_conf, global_id, track_id, camera_id, timestamp=exit_ts}`.

## 1.11 Cloud upload and schemas

**ImageKit** (m4 c6 `upload_sighting`/`upload_face`, m6 `save_results_to_cloud`):
- Body: `cv2.imencode('.jpg')` of `best_crop_img`, file `body_g{gid:03d}_t{tid}_{int(time.time())}.jpg`, folder `/supermarket_reid/global_id_{gid:03d}/`. Result `sighting['best_crop_url']` (empty string on any exception, which is swallowed). `best_crop_img` is then deleted from the dict.
- Face: `face_g{gid:03d}_t{tid}_{int(time.time())}.jpg`, same folder.
- m4 uses ThreadPool(10); m6 is a sequential loop.

**Pinecone `supermarket-faces`** (512, cosine): one vector per completed face, id `face_g{gid:03d}_{int(time.time())}` (**collides within the same second**, see B10). Metadata `{global_id:int, face_url, camera}`. Upserted ONLY at the end of a run, so faces from the current video are invisible to its own face lookups.

**MongoDB `supermarket_db.persons`**, one doc per gid:
```
{global_id:int, first_seen_ts:float, last_seen_ts:float,
 faces:[{url, confidence, timestamp, camera_id}],
 sightings:[{camera_id, track_id, entry_ts, exit_ts, duration_sec, frame_start, frame_end, n_detections,
             best_crop_quality, best_crop_bbox:[x1,y1,x2,y2], best_crop_frame, avg_det_score, avg_skeleton_ratio, best_crop_url}]}
```
- Sightings are pushed with `$push` and `$set last_seen_ts = exit_ts of the pushed sighting` (not a max; order-dependent).
- m4: docs are created at mint time (`create_person_document`) via `insert_one` when absent. Final pushes use `update_one` without upsert, so a gid with no doc silently loses its data.
- m6: `save_results_to_cloud` creates docs for all gids present in completed sightings or faces before pushing. First-seen is `min(entry_ts)`. `[m6 better]`
- Face push in m4 happens after Pinecone upsert in the same try, so a failed Pinecone call means the Mongo face entry is also skipped.

**Pinecone `supermarket-reid`** (1280, cosine):
- Default namespace `""`: prototype vectors, id `global_{gid}_proto_{k}` (k=1..3), metadata `{global_id, prototype_idx}`.
- m6 namespace `gallery_buffer`: full gallery (see 1.16).
- m4 pickle (`identity_buffer.pkl`): `{gid: {embeddings:[np (1280)], ratios, det_scores, filepaths:[str]}}`, plus a zip of `saved_identities/`.

## 1.12 alpha-FPS prototype algorithm (defined only in m6 c5; m5 calls it but never defines it, see B12)

```python
def alpha_fps(emb[N,D] (L2-normed), q[N] = det_score*ratio, K=3, alpha=0.5):
    if N <= K: return list(range(N))
    sel = [argmax(q)]
    dist = 1 - emb @ emb[sel[0]]                    # distance to nearest selected
    for _ in range(K-1):
        comb = alpha*q + (1-alpha)*dist; comb[sel] = -1
        nxt = argmax(comb); sel.append(nxt)
        dist = min(dist, 1 - emb @ emb[nxt])
    return sel
```
- Scale mismatch: `q` is in [0,1] and typically 0.4-0.9 with small spread. `dist` is `1-cos` and typically 0.0-0.3, so quality dominates and the diversity term is weak.
- Prototype build (`save_results_to_cloud` tail, `rebuild_prototypes`): `delete_all(namespace="")` then, per gid with embs, upsert the K prototypes (the raw embedding of the selected crops). `rebuild_prototypes` loads the gallery from Pinecone first. Both are destructive (delete then write, non-atomic).

## 1.13 Body search (m5 c5 `search_person`, m6 `search_body`)

```
q = embed_body(img) -> 1280 normed
res = index.query(vector=q, top_k=10, include_metadata=True)           # default namespace = prototypes
if no matches or res[0].score < th_emb(0.75): UNKNOWN
id_scores = {gid: first(=max) score per gid}; sort desc
if >=2 gids: p = softmax([s1,s2]*15); ratio = p0/(p1+1e-9); if ratio < tau: AMBIGUOUS(best_guess_gid)   # i.e. s1-s2 < ln(tau)/15
doc = persons.find_one({global_id:int(best)}); m5: ERROR if missing; m6: MATCH with {} if missing
MATCH {global_id, score, mongodb_data}
```
- Pinecone returns numeric metadata as float. m5 returns the raw `best_gid` (float); m6 casts `int()` `[m6 better]`.
- Only the top-2 identities are compared.
- If the same real person is split across two gids (fragmentation), this check returns AMBIGUOUS by construction.
- Defaults differ: function `tau=1.5`, UI slider 2.0.

## 1.14 Face search (m5 c6 `search_face`, m6 `search_face_query`)

`embed_face(img)` (or `embed_face_crop` if `is_crop`): none -> `NO_FACE`. Then `face_index.query(top_k=5)`; `top.score < 0.55 -> UNKNOWN`; `MATCH {global_id, score, det_score (m5), match_face_url (m6), mongodb_data}`. Only `top_k=1` is used, with no margin check. m5 c8 (URL test) and c10 (brute force) use `embed_face_crop` with a fallback to `embed_face`.

m6 `run_search`: Body and/or Face per radio. If both match, **the face gid silently overrides the body gid** for details and heatmaps. The URL path does `cv2.imdecode(arr,-1)` then `COLOR_BGR2RGB`, which fails for PNGs with alpha or grayscale.

## 1.15 Heatmap generation (m6 c5 only)

- Background: `CAMERA_BG_DIR = /kaggle/input/reid-testing-samples`, file `bg_{camera_id}.png` (**PNG**). The repo's `assets/camera_backgrounds/` holds `bg_cam_03.jpg` and `bg_cam_03_mirrored.jpg`, so naming and extension do not match the code or CLAUDE.md.
- `_bbox_centers_from_sightings`: for each sighting with a 4-element `best_crop_bbox`, take the centre `(cx,cy)` plus the box. **One point per sighting.**
- `build_all_camera_heatmaps(sightings)`: group by `sighting.camera_id`; one `build_person_heatmap` per camera.
- `build_person_heatmap(sightings, camera_id, sigma_small=4, heat_scale=0.10, alpha=0.6, draw_centers=True)`:
  1. Load background (W,H). Otherwise a dark fallback canvas `max(640, 1.05*max_x2) x max(480, 1.05*max_y2)`.
  2. Small grid `Hs=max(32,int(H*0.1))`, `Ws` likewise. For each point, map `(cx,cy)` proportionally into the small grid and stamp a filled disc of radius `max(2, round(sigma*1.1))` with `+=1`. The plain Python loops are O(points*r^2).
  3. `GaussianBlur(k=max(3, (sigma*6)|1), sigma)`; `resize` to (W,H) with INTER_LINEAR; `heat/(heat.max()+1e-9)`; `COLORMAP_JET`; `addWeighted(bg, 1-alpha, heat, alpha)`.
  4. Marker: red circle plus yellow (BGR 0,255,255) rectangle at **`pts[-1]`** (the last in DB order, not the latest in time).
- Assumes bbox coordinates are in the same pixel space as the background image. Frame size is not stored (see B17).
- Returns `(rgb, info_str)`. `build_all_camera_heatmaps` returns `[(cam_id, rgb, info)]`. The UI shows them in a `gr.Gallery`.

## 1.16 Gallery persistence in Pinecone (m6 c5)

- `save_gallery_to_pinecone(gallery)`: `delete(delete_all=True, namespace='gallery_buffer')` (errors swallowed), then upsert per embedding: id `gal_g{gid:03d}_{j:03d}` (j = index in the class list), values = embedding, metadata `{global_id:int, ratio, det_score, filepath:str}`; batches of 100.
- `load_gallery_from_pinecone(gallery, max_gid_search=500, max_embs_per_gid=50)`:
  1. `describe_index_stats` -> `vector_count` for the namespace (0/missing -> return 0).
  2. Probes the deterministic ids `gal_g{gid:03d}_{j:03d}` for gid 1..500, j 0..49, via `fetch(ids=chunk100, namespace=...)`, and early-exits once `total_fetched >= vector_count`.
  3. Groups by `metadata.global_id`; defaults ratio 0.5, score 0.5; filepath string kept as-is (these paths are NOT valid after a kernel restart).
  4. Sets `gallery.gids`, `gallery.classes`, `current_max_id = max gid`.
- Limits: gids > 500 or j >= 50 are never loaded. Then the next save wipes them (B14).
- `run_video_pipeline(use_existing_data=True)` loads the gallery and records `loaded_gids` (UI badge "Existing"/"New").

## 1.17 Evaluation brute-force tests (m5 c9, c10)

- Body (c9): `local_db = <saved_identities dir>`, iterate `rglob("*.jpg")` (every crop the pipeline saved). `true_gid = int(parent_dir.split('_')[-1])`. `search_person(query)` with defaults (th 0.75, tau 1.5). Counters: Correct (MATCH and gid == true), Wrong (MATCH and gid != true), Ambiguous, Unknown; per-query ms and similarity. Failures are written to `eval_body_failures/{failed_misclassified,rejected_unknown}`. Metric is `correct / total` (so rejections count as errors). Reported: 85.35% on 1,317 crops; 54 wrong, 74 ambiguous, 9 unknown; avg 183.8 ms.
- Face (c10): for every person in Mongo with `faces[]`, download each face URL, `embed_face_crop`, Pinecone top_5, accept if `>= 0.55`, correct if `gid == person.global_id`. Counters: total, correct, wrong, no_face, unknown. No reported number except an anecdotal 0.942 similarity.

---------------------------------------------------------------------------
## 1.18 m4 vs m6 difference table (checklist for the port)

| Aspect | m4 | m6 | Better |
|---|---|---|---|
| Timestamps | fixed `VIDEO_START_TIME`, fps=30 hard-coded | `now()` + real fps | Mixed: m4 deterministic but fps wrong; m6 fps right but start time meaningless (B1) |
| Tracker ReID weights | osnet (404-prone) | clip_market1501 via signature-sniffing wrapper | m6 works; both redundant (W1) |
| Tracker lifecycle | one global | new per run | m6 |
| Pose model | yolo26s-pose | yolo11s-pose | n/a (verify which exists) |
| Face-body pairing | highest det_score | first in list | m4 |
| Face Pinecone errors | crash | swallow silently | neutral (both bad) |
| Face match compare | `>` | `>=` | neutral |
| `mint_new_class` | may collide with hydrated ids | skips existing | m6 |
| `classify_live_track` min cos | global | param | m6 |
| Merge: Mongo + Pinecone face + budget prune | yes (with flaws) | no, in-memory only | m4 |
| Person-doc creation | at mint time | at save time for all gids | m6 |
| Gallery persistence | pickle + zip of crops | Pinecone namespace (embeddings only) | m6 for portability; loses the crop files |
| Cloud upload | threaded | sequential | m4 (speed) |
| Search gid type | raw float | `int()` | m6 |
| Prototype build | none in m5 (broken) | `save_results_to_cloud`/`rebuild_prototypes` | m6 |
| Heatmaps | none | per-camera | m6 |
| Hydration | pickle, drops entries whose files are missing | Pinecone, keeps embeddings with dead paths | m6 keeps more, but merge then drops them (B5) |

---------------------------------------------------------------------------
# 2. BUGS
---------------------------------------------------------------------------

## 2.A Status of the items from `docs/design_discussion.md`

| # | Item | Status in current code |
|---|---|---|
| 1 | `_frame_to_ts` uses wall clock | **Fixed in m4** (anchored to `VIDEO_START_TIME`). **Regressed in m6**: anchor is `datetime.now()` at processing time (B1). |
| 2 | `forced_gid` ignored when gallery empty | **Fixed** in both (forced branch is checked first). |
| 3 | Entropy normalised by `log(512)` | **Fixed** in both (`log(n_classes)`). |
| 4 | VIP bypass rarely fires | **Moot but different**: forced items skip the UDM entirely, so the DBSCAN bypass is dead code and the "collect enough crops before minting" requirement (F) is not implemented (B11). |
| 5 | Merge KeyError on stale target | **Fixed** in m4 (guard `mint_class_at`) and m6. A related data-loss bug remains (B4). |
| 6 | Sightings in RAM until end | **Still present** (accepted by author). All tracks stay open until the end. |
| 7 | Face conf 0.5 low | **Addressed** (0.7). |
| 8 | Unused `imgs` key | **Fixed** (removed). |
| 9 | Face index manual | **Still manual.** The body index has the same requirement. |

## 2.B Confirmed defects (verified against current code)

**B1. Timestamps are meaningless for multi-camera analytics** (m4 c2 `VIDEO_START_TIME`, m6 `run_video_pipeline`)
- m6: `VIDEO_START_TIME = datetime.now(utc)` is set at run time, and the UI has no start-time input.
- m4: a hard-coded constant, so every camera's video starts at 2026-05-05 09:00:00 UTC, and `SightingTracker(fps=30)` is hard-coded.
- Scenario: videos from two cameras recorded at the same real time but processed an hour apart get timestamps an hour apart (m6), or any video that is not exactly 30 fps gets wrong durations (m4).
- Fix: per-camera `start_time` supplied as input (or from file metadata/NTP) and `fps` read from the video. Store both on the camera/run record.

**B2. Sightings never close until end of video** (m4 c5 end, m6 end of `run_video_pipeline`; contradicts the tech report 5.5)
- `close_track` is only called in a final loop. A "sighting" is a whole BotSort track lifetime with no gap logic.
- Scenario: a person at the camera for 20 s, gone 10 min, back with a new track id -> two sightings (fine). But a track kept alive through ID switches or occlusions contains several people or poses in one record, with a single best crop and a single gid.
- Fix: close on track loss (`not in tracker output for N frames`) and persist incrementally.

**B3. Sticky, unverified track->gid mapping poisons the gallery** (m4 c5 step 5)
- Once `L2G[tid]` is set, every later crop of that track is `add_observation`'d with no similarity check. A wrong early assignment, or a BotSort ID swap between two people, keeps writing the wrong person's embeddings into the wrong class (and the "self-adaptive" centroid drifts).
- Fix: verify each frame (or track-level vote) that `emb` is still consistent with `gid`'s centroid; if not, flag the track for re-resolution and stop updating the gallery.

**B4. Merge can lose data, and uses single-frame evidence** (m4 `execute_identity_merge`)
- `update_one({global_id: target}, $push)` has no upsert, then `delete_one(old)`. If the target document does not exist (stale face vector after `RESET_DATABASE`, which clears Mongo but not Pinecone), the old doc's sightings and faces are deleted and not stored anywhere.
- Merge fires on a single frame with a face score just above 0.55, and is irreversible (files moved, docs deleted). One bad face match fuses two real people.
- It does not check whether `old_gid` already owns confirmed face vectors (two enrolled people being merged).
- It does not check whether the target is already claimed by another visible track this frame, so two people end up with one gid and flip-flop through the `claimed` logic.
- Fix: require N consistent face hits (or a margin over the second-best gid) before merging; check target doc exists (upsert); refuse/flag when both gids have face evidence; make merges soft (alias table `gid -> canonical_gid`) rather than destructive.

**B5. m6 merge is not persisted and loses hydrated embeddings** (m6 c4 `execute_identity_merge`)
- No Mongo or face-vector update. Old doc and `supermarket-faces` vectors keep `old_gid` while the gallery and prototypes use `target_gid`. A body search returns the target (whose doc lacks the old sightings), while a face search returns the old gid. The `old_gid` doc becomes an orphan.
- The merge copies only entries whose `filepath` exists. Gallery entries loaded from Pinecone carry dead `/kaggle/working/...` paths from earlier sessions, so **all hydrated embeddings of `old_gid` are silently discarded**.
- Fix: carry embeddings regardless of file existence; persist merges (alias table).

**B6. Pinecone zero-vector query for face re-labelling** (m4 `execute_identity_merge`)
- `face_index.query(vector=[0.0]*512, filter=...)` on a cosine index is expected to be rejected by Pinecone (all-zero dense vector on a cosine index), and the broad `except` hides the failure. Not verified live.
- It also caps at `top_k=100`.
- Fix: `index.list`/`fetch` by id prefix or keep an id list per gid (`delete`/`update` by metadata filter via the data-plane API).

**B7. m6 merge never re-prunes the budget** (m6 `_optimize_budget`, `execute_identity_merge`)
- `_optimize_budget` removes ONE entry per call. After a merge a class can hold ~100 entries and each subsequent `add_observation` adds one and removes one, so it stays over budget indefinitely (m4 loops `while len > max_images`).
- Fix: loop until within budget.

**B8. No final DBSCAN flush; short-lived people disappear** (m4 c5, m6 pipeline)
- DBSCAN only runs when `frame_count % 30 == 0` AND `len(pool) >= 5` AND the frame survived the earlier `continue`s (in m6 any frame with no pristine detection skips it).
- At end of video, the pool is not clustered. A track that never reached a confident live match and never formed a 5-sample cluster has `global_id=None` and **`close_track` drops its sighting**, so the person is not stored at all.
- Fix: run a final clustering pass with relaxed `min_samples`, assign leftover tracks (even low-confidence) a gid or an explicit "unresolved" sighting record.

**B9. DBSCAN scheduling is skipped on empty frames** (m4 `continue` before DBSCAN for no boxes/no tracks; m6 for no pristine too)
- Fix: schedule by pool size/time, not by modulo of frame index.

**B10. Face vector id collision** (m4 c6 `upload_face`, m6 `save_results_to_cloud`)
- `id = f"face_g{gid:03d}_{int(time.time())}"`. With 10 upload threads (m4), two face records of the same gid inside one second overwrite each other in Pinecone while both remain in Mongo.
- Fix: uuid or `{gid}_{run_id}_{track_id}`.

**B11. Dead VIP-bypass branch; requirement F unmet** (m4/m6 `run_dbscan`, step 6)
- Forced items never enter the UDM, so `FACE_VIP_BYPASS` never fires. A person recognised by face on the first frame gets a gallery class from ONE (possibly poor) crop immediately.
- Fix: decide the intended behaviour and delete either branch. If requirement F is wanted, hold forced items until >= k good crops, then mint.

**B12. Module 5 is not runnable as shipped** (m5 c4, c11)
- Both cells call `alpha_fps`, which is not defined in m5 (only in m6 c5), so they raise `NameError`. m5 also contains no code that upserts prototypes to Pinecone despite the markdown "Run this once to populate `supermarket-reid`". The only builders are m6 `save_results_to_cloud` and `rebuild_prototypes`.
- Fix: one `prototypes.build()` function in the package, used by both.

**B13. Gallery persistence in m6 silently truncates** (m6 `load_gallery_from_pinecone`)
- The budget slider allows up to 100 entries per gid, but only j 0..49 are probed. Gids > 500 are never probed. After a load, the next save runs `delete_all` on the namespace and writes only what was loaded, **permanently removing** the extra vectors.
- Fix: use `list(prefix=...)` / metadata query, or store one document per gid.

**B14. "Load existing" unchecked + Save wipes the world** (m6 `save_results_to_cloud`)
- With `use_existing_data=False`, the gallery in RAM contains only the new video. Save still does `delete_all` on `gallery_buffer` and on the default namespace, replacing all old prototypes while Mongo, ImageKit and the face index keep the old people.
- Fix: never delete-all as part of a normal save; upsert per gid.

**B15. Save is not idempotent** (m6 `save_results_to_cloud`)
- It mutates `completed_sightings` in place (`del best_crop_img`, `pop global_id`). A second click skips sightings (now `global_id=None`) but re-uploads and re-pushes faces (duplicates). The Pinecone wipe and rewrite are also non-atomic, so a crash between delete and upsert leaves empty indices.

**B16. Face lookup per frame, per body, with no caching** (m4 c5)
- A remote Pinecone call every frame for each paired body. Latency scales with people x frames, and in m4 any network error aborts the run.
- Fix: look up once per track until a face of better quality arrives (or on a cadence), with batching, retries and local caching.

**B17. Heatmap background and frame-size mismatch** (m6 `load_camera_background`)
- Code expects `bg_{camera_id}.png` in `/kaggle/input/reid-testing-samples/`. The repo has `.jpg` files under `assets/camera_backgrounds/`.
- Frame width/height are not stored with sightings, so bbox coordinates are assumed to be in the same pixel space as the background. If video resolution differs from the background, points land in the wrong place.
- Fix: store per-camera `(W,H)` in a camera registry, and scale.

**B18. m6 `get_spatially_matched_face` returns the first, not the best, face** (see 1.5). Regression versus m4.

**B19. `last_seen_ts`/`first_seen_ts` not maintained** (m4 c6 / m6 save)
- `last_seen_ts` is set to the `exit_ts` of whichever sighting was pushed last (can go backwards). After a merge, the target's `first_seen_ts` is not min'ed with the old doc's.

**B20. gid collisions across stores** (m4 hydration)
- `current_max_id` comes only from the pickle. Mongo docs and face vectors with higher gids (pickle older than DB) will receive new, unrelated people under the same gid (new sightings pushed into someone else's doc). m4 `mint_new_class` also does not skip existing ids (m6 does, but only vs the loaded gallery).
- m4 hydration also drops identities whose crop files cannot be resolved, so they get re-minted as duplicates while Pinecone/Mongo still hold them.

**B21. Environment is not reproducible** (all notebooks)
- Unpinned `pip install` for boxmot/ultralytics/insightface/pinecone/imagekitio. Model files named but not versioned (`yolo26s-pose.pt` may not exist in some Ultralytics releases).
- The notebooks call `reid_model(crops)`, which does not match the `ReID.__call__(frame, dets)` of boxmot 16/17 that was inspected. m6's `_make_botsort_tracker` sniffs four possible constructor signatures, which shows API drift is real.

**B22. Security and privacy** (m6 c7, c6)
- `demo.launch(share=True)` publishes a public `.gradio.live` URL with no authentication, including the "Reset All Databases" button (wipes Mongo and Pinecone). Face crops are served from public ImageKit URLs. Biometric data (face embeddings, face chips) is stored in third-party clouds with no retention, consent or deletion path.

**B23. Smaller defects**
- m4 face crop uses `frame[max(0,fy1):fy2, ...]`; if the box is fully off-frame it yields an empty array that is stored as best face and later skipped silently.
- m6 `build_results_html` and `get_person_details` use `datetime.fromtimestamp` (server local time) while stored timestamps are UTC epoch.
- m6 `rebuild_prototypes` builds a `GalleryManager(max_images=100)` unnecessarily.
- `run_search` reports only the face gid when body and face disagree.
- m6 `search_face_pinecone` swallows all exceptions, so an outage looks like "unknown face".
- Tech-report inconsistencies: names YOLOv8 and YOLOv11, says tracker ReID is OSNet, says `close_track` is called on track loss, says heatmap shows "all positions the person occupied" when it plots one point per sighting.

---------------------------------------------------------------------------
# 3. ALGORITHMIC / ARCHITECTURAL WEAKNESSES
---------------------------------------------------------------------------

**W1. Redundant ReID computation.** BotSort runs its own ReID (OSNet in m4; a second full CLIP ViT-B/16 in m6) on every detection each frame, and the pipeline then runs CLIP-ReID again on gated crops. In m6 that is two ViT-B/16 forward passes per person-frame.
- Alternatives: `with_reid=False` (static CCTV: IoU + Kalman is enough), or compute the embedding once per detection and pass it to the tracker (boxmot's `update(dets, img, embs)` where supported). Disable ECC camera-motion compensation on fixed cameras.

**W2. Per-frame identity decisions, no track-level evidence.** Each frame is classified independently using a margin test against centroids. Then the first success sticks forever (B3).
- Alternative: accumulate a per-track embedding (quality-weighted mean of the last N good crops); decide once per track with a margin test (and re-check periodically); allow track re-assignment while the gid is "provisional".

**W3. Self-adaptive gallery is fed every frame.** Each tracked frame writes a JPEG and a 1280-D vector, then the O(N^2) budget pruner runs every frame once a class has 50 entries. Consecutive frames are near-duplicates, so the gallery is effectively a few seconds of one track.
- Alternative: add at most one crop per track per time/pose bucket (or when novelty `1-max_sim > delta` and quality is high); keep embeddings in memory and write only the chosen crops.
- `_optimize_budget` ignores quality and recency, and `get_weighted_centroids` is recomputed per call (cache per class, update incrementally).

**W4. Circular evaluation.** The body test queries with crops that the pipeline itself saved (the very images whose embeddings built the gallery, and three of which ARE the prototypes, giving similarity ~1.0). The labels are the pipeline's own cluster ids, not ground truth. Avg similarity of 0.925 shows near-duplicate queries. The face test is the same: stored chips against vectors computed from the same frames.
- The 85.35% therefore measures self-consistency, not re-identification accuracy. Counting Ambiguous/Unknown as errors also mixes rejection with accuracy.
- Alternative: hand-labelled identities across disjoint time segments/cameras; report Rank-1/mAP (re-id standard), plus FAR/FRR at the operating threshold; split gallery/query by track and by time.

**W5. Thresholds tuned on one or two videos, with inconsistent semantics.** `MIN_COS_SIM 0.78`, DBSCAN `cos>=0.85`, search `0.75`, face `0.55`, `tau 2.0 vs 1.5`, softmax temperature 15 appear in three places, BotSort defaults untouched. No calibration data (impostor/genuine similarity distributions), no per-camera adaptation. Repo samples are `store_cam1_mirrored.mp4` (face) and `store_cam3.mp4`.
- Alternative: one calibration module that measures genuine/impostor distributions per camera/model and sets thresholds from a target FAR; single config object.

**W6. Cross-camera logic is appearance-only.** See section 5: no camera topology, no travel-time windows, no overlap handling, no per-camera exclusivity. Gallery matches are global.
- Alternative: candidate gating by camera graph and plausible transit time, a "same gid cannot be in two non-overlapping places at once" constraint, and association on tracklets rather than frames.

**W7. Face is "ground truth" without confidence handling.** One face per track (best `det_score`, not quality: no blur/size/pose measure). `top_k=1` with no margin to the second identity, per-frame lookups, and faces from the current video cannot match each other (they are upserted at the end). Two tracks of the same unknown person in one run are never face-merged within the run.
- Alternative: keep an in-memory face gallery during the run, score face quality (size, frontality, blur), aggregate per track, match with a margin, and only then merge.

**W8. Clustering of unknowns.** DBSCAN on per-frame embeddings with `min_samples=5` lets five consecutive frames from one track mint a person. Noise is retained forever and re-clustered every 30 frames (O(n^2) and unbounded RAM). Cluster->gallery MERGE ignores ambiguity.
- Alternative: cluster tracklets (track-level embeddings), require a minimum duration or >= 2 independent tracks for minting from low-confidence material, or allow provisional identities that are merged later.

**W9. Fragmentation is not measured, and search punishes it.** Multiple gids per person are inevitable, yet `AMBIGUOUS` fires whenever top-2 gids are close, which fragmentation guarantees. There is no offline merge/consolidation step.
- Alternative: periodic gallery consolidation (agglomerative clustering of centroids with cannot-link constraints from co-visibility), and report per-person identity fragments in search.

**W10. RAM and I/O growth.** Full best crops for every track, the whole UDM pool (full crops + GPU embeddings, never expiring), faces, plus a JPEG per observation on disk. Results are only persisted at end (a crash loses everything).
- Alternative: stream results per closed track to a store; keep only embeddings and small thumbnails; cap the pool by age.

**W11. Hard dependency on three cloud services in the hot path.** Pinecone is called per frame (face) and ImageKit/Mongo/Pinecone are required for even a local demo. 1280-D embeddings x (50 gallery + 3 prototypes) x persons is tiny; a local FAISS/NumPy index or pgvector plus SQLite/Postgres plus filesystem/S3 would be faster, offline-capable, and cheaper. Index dims are baked in.
- Alternative: define `VectorStore`, `MetadataStore`, `BlobStore` interfaces with local default implementations; cloud adapters optional.

**W12. Detection/quality gate limitations.** The pose gate counts head keypoints and uses the first pose in the crop (multi-person crops). It ignores occlusion by other people, truncation at frame edges, blur and scale. Raw-bbox crops include background and neighbours, and are stretched to 128x256. The RT-DETR conf (0.75) is applied to tracker output, not raw detections. Frames with no detection do not tick the tracker.
- Alternative: crop with padding/aspect-preserving resize (matching CLIP-ReID training), mask or reject overlapping boxes (IoU > x), add sharpness/size criteria, and always call `tracker.update` (with empty dets).

**W13. Appearance-only body ReID has known failure modes.** Uniforms or same clothing, clothing change across days, lighting changes between cameras. `clip_market1501` was trained on Market-1501 (outdoor, few cameras), not retail CCTV, and there is no domain adaptation.
- Alternative: evaluate other backbones on your own data; weigh body evidence by camera pair; use face and other soft biometrics when available.

**W14. Code structure.** Three notebooks with copy-pasted divergent logic (m4/m5/m6), global state (`CAMERA_ID`, `gallery`, `local_to_global_map` as globals in m4), no tests, no typed data model. m4 merge uses globals, m6 passes references.
- Alternative: a package with a single `Pipeline` class; pure functions for decision logic (testable with synthetic embeddings); config dataclass; one CLI plus an optional UI.

**W15. Time-based concerns.** Frame-driven DBSCAN schedule, fps hard-coded in m4, no idle-track expiry in `local_to_global_map` (grows for the whole video).

---------------------------------------------------------------------------
# 4. WHAT THE STORED DATA CONTAINS
---------------------------------------------------------------------------

| Store | Per-person | Per-sighting (per track) | Per-frame |
|---|---|---|---|
| MongoDB `persons` | `global_id`, `first_seen_ts`, `last_seen_ts`, `faces[]`, `sightings[]` | `camera_id`, `track_id`, `entry_ts`, `exit_ts`, `duration_sec`, `frame_start`, `frame_end`, `n_detections`, `avg_det_score`, `avg_skeleton_ratio`, `best_crop_quality`, `best_crop_bbox`, `best_crop_frame`, `best_crop_url` | **Nothing** |
| ImageKit | `/supermarket_reid/global_id_XXX/` | one `body_g..._t{tid}_{unix}.jpg` (best crop) | nothing |
| ImageKit (faces) | same folder | one `face_g..._t{tid}_{unix}.jpg` per track that had a paired face | nothing |
| Pinecone `supermarket-faces` | vectors per face with `{global_id, face_url, camera}` | 1 per track with a face | nothing |
| Pinecone `supermarket-reid` default ns | 3 alpha-FPS prototype vectors per gid | n/a | n/a |
| Pinecone `gallery_buffer` ns (m6) | up to 50 gallery embeddings per gid with `{ratio, det_score, filepath}` | n/a | n/a |
| Pickle + crops zip (m4) | gallery embeddings/ratios/scores/paths + crop JPEGs (ephemeral Kaggle disk) | n/a | n/a |

Answers:
- **Per-frame trajectory stored? No.** The only spatial datum per sighting is **one** bbox (`best_crop_bbox`) and its frame (`best_crop_frame`). There is no polyline, no per-frame boxes, no foot points, and the heatmap is one blob per sighting.
- **Full videos stored? No.** Not by the pipeline, and no `video_id`/filename in any sighting (only `track_id`, which is local to a run and can repeat across runs and cameras). One cannot re-open the source frames of a sighting from the DB.
- **Only best crops?** Yes: one best body crop per sighting (highest `conf*ratio`), one best face per sighting (highest `det_score`), plus the pruned embedding gallery (<= 50 per person, no timestamps or cameras attached to those embeddings).
- Not stored: frame width/height, fps, camera start time, camera geometry/floor-plan mapping, per-sighting embeddings, which gallery crops came from which sighting (filepaths are `frame_{n}_trk_{t}` locally only), identity-merge history, or any "unresolved" tracks (dropped, B8).
- Sighting `global_id` is "the last assigned gid at track end". If a track changed identity mid-way, the history is lost. Merges rewrite gids in Mongo for m4 only (B5).
- Timestamps are unreliable for cross-camera ordering (B1).

**Implication for a "track a person across cameras" feature:**
1. A coarse timeline (camera, entry_ts, exit_ts, one thumbnail) can be built today from `sightings[]`, but only if timestamps are made trustworthy (B1).
2. A map/trajectory view needs new persisted data: per-track samples (e.g. every 0.5-1 s: `ts, frame, bbox or foot-point`), frame size per camera, `run_id`/`video_id`, camera registry with pixel->floor-plan mapping (homography), start time per video, and the track->gid history including merge/alias records.
3. Keep embeddings per track (not just per person) so identities can be re-clustered offline.

---------------------------------------------------------------------------
# 5. MULTI-CAMERA BEHAVIOR TODAY
---------------------------------------------------------------------------

**Configuration**
- m4: edit `CAMERA_ID`, `IS_FACE_CAMERA`, `INPUT_VIDEO` in c2 and re-run the notebook. One video per run, one camera per run.
- m6: UI fields "Camera ID" textbox and "Face Camera" checkbox plus video upload/path. It sets the module-global `CAMERA_ID`, which `SightingTracker` reads when creating each sighting. `_last_run` stores only the LAST run's gallery and tracker, so two videos cannot be processed and then saved together; each run must be saved before the next.
- Neither takes camera topology, position, resolution, or start time.

**Multiple videos in one run?** No. A kernel can run sequentially, but m4 would reuse its global tracker, `local_to_global_map` and UDM across videos (track ids continue, UDM pool carries over). m6 builds fresh objects per run and re-hydrates the gallery from Pinecone.

**How gid consistency across cameras is achieved**
1. **Gallery hydration**: m4 loads the previous run's `identity_buffer.pkl` (picked via `rglob` on `/kaggle/input`, first match, nondeterministic order) and resolves crop files via the `saved_identities` dataset. m6 loads embeddings from the `gallery_buffer` namespace.
2. A person in camera 2 is then recognised by the live softmax-margin test against ALL gids' centroids (floor 0.78, margin ln(2)/15), or by DBSCAN MERGE (cos >= 0.85) of unresolved crops.
3. **Face cameras**: face lookup in `supermarket-faces` (vectors from previous runs) overrides body-based gids and triggers `execute_identity_merge` (the "self-correcting loop" in the design notes). Non-face cameras never create or use face evidence.
4. Mongo `persons` is the shared record keyed by gid, so sightings from all cameras land in the same doc (`camera_id` per sighting).

**What it does NOT do**
- No camera topology, adjacency, transit-time windows, or entry/exit zones. Time is never used in matching.
- No cross-camera same-time exclusivity: `claimed` is per frame within one video only. The same gid can be present in two cameras at once, and nothing flags it. Simultaneous cameras processed in separate runs cannot interact with each other at all (no live shared state), and later runs only see what was saved.
- No per-camera embedding normalisation or calibration, so illumination and viewpoint differences hit the same thresholds.
- Timestamps across cameras are only comparable if all videos shared the same (assumed) start time (m4 constant) or never (m6).
- Identity merge is the only retroactive correction mechanism. It only fires on face cameras and only vs face vectors from earlier runs.
- A rewrite should treat cameras as first-class entities (registry with id, resolution, start time, face-capable flag, topology) and process tracklets from all cameras into one global association step, with time gating.
