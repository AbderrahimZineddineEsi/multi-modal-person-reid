# From v1 to v2: what changed, what was wrong, and why

This report compares the original project (**v1**, the Kaggle notebooks kept at git tag `v1-kaggle`) with the
rewritten package (**v2**, branch `v2`). It was written after re-reading the v1 code cell by cell
(`notebooks/module4_face_association.ipynb`, `module5_search_engine.ipynb`, `module6_dashboard.ipynb` at
`v1-kaggle`) and the final v2 code.

It is written for the author of v1. You already know what your v1 does; the goal here is that you can also explain
**every** v2 change: what v1 did, what was wrong with it (and how we know), what v2 does instead, what it changed in
the results, and which paper it comes from. Where v2 itself made mistakes during the rewrite, they are listed too
(section 9), because you will be asked about them.

Notation: `m4`, `m5`, `m6` are the three v1 notebooks. `B#` / `W#` are the bug and weakness ids of the detailed
review in `docs/dev/v1_review.md`. File names without a folder are in `src/reid/`. Numbers come from runs on AI City
2024 scene 075 (section 8) unless stated otherwise.

---

## Contents

1. [The short version](#1-the-short-version)
2. [The papers behind v1, and how faithfully each part follows them](#2-the-papers-behind-v1)
3. [Stage by stage: detection, tracking, quality gate, embeddings](#3-stage-by-stage-perception)
4. [The core change: deciding identity per tracklet, not per frame](#4-the-core-change-tracklets)
5. [Identity memory: gallery, unknowns, faces, constraints, repair](#5-identity-memory)
6. [Search and prototypes](#6-search-and-prototypes)
7. [Storage, time, multi-camera, app, reproducibility](#7-storage-time-multi-camera-app)
8. [Evaluation: why 85.35% was not a real number, and what we measure now](#8-evaluation)
9. [Mistakes made during the rewrite, and how they were found](#9-mistakes-made-during-the-rewrite)
10. [What v1 got right and v2 kept](#10-what-v1-got-right)
11. [Limitations that remain](#11-limitations-that-remain)
12. [Talking points](#12-talking-points)

---

## 1. The short version

v1 had the right ingredients: a detector, a tracker, a pose-based quality gate, CLIP-ReID body embeddings, ArcFace
faces, Casao et al.'s self-adaptive gallery, an Unknown Data Manager with DBSCAN, a "face is ground truth" healing
idea, and α-FPS prototypes for search. Its problems were mostly **where and when decisions were taken**, not which
models were used:

| # | v1 behaviour | Consequence | v2 behaviour |
|---|---|---|---|
| 1 | Identity decided **per frame**, and the first decision for a track **stuck forever** (`local_to_global_map`) | One bad frame, or a tracker ID switch, wrote the wrong person into a gallery for the rest of the track (B3, W2) | Identity decided **once per tracklet**, on 16 quality-weighted samples; a sample vote must agree |
| 2 | **Every frame** of an assigned track was added to the gallery (no diversity gate) | Galleries filled with near-duplicates of a few seconds of one track; O(N²) pruning every frame (W3) | Only verified tracklets' samples enter, through Casao's diversity gate (verify-before-update) |
| 3 | One face frame ≥ 0.55 **irreversibly merged** two identities (files moved, docs deleted) | One wrong face match fused two people permanently (B4-B6) | Face evidence needs ≥ 2 faces and a margin; it becomes a must-link; merges are recorded and reversible |
| 4 | Faces were uploaded to Pinecone **only at the end of the video** | Inside one video, two sightings of the same new person could never be matched by face (W7) | Face descriptors enter the local store as each tracklet finishes, so later tracklets of the same run can use them |
| 5 | No constraint said "one person cannot be in two places at once" (except within a single frame) | Two people seen together could share a gid | Cannot-links: same camera at the same time, impossible camera-to-camera transit, conflicting faces |
| 6 | Unknown crops (single frames) clustered every 30 frames, never flushed | 5 consecutive frames of one track could mint a person; short-lived people vanished at the end (B8, W8) | Unknown **tracklets** clustered with the paper's parameters; the pool is flushed at the end of the run |
| 7 | Timestamps: fps hard-coded to 30, one fixed start time for every camera | Cross-camera times and durations were meaningless (B1) | `start_ts + frame / fps` with the file's real fps, per camera |
| 8 | "Evaluation" re-queried the system's own saved crops against its own labels | 85.35% measured self-consistency, not accuracy (W4) | Ground-truth evaluation on AI City 2024: tracking IDF1, cross-camera IDF1, purity, held-out retrieval |
| 9 | Three cloud services in the processing loop (Pinecone per frame, MongoDB, ImageKit), three notebooks with copied logic | Not runnable without accounts; m4/m5/m6 diverged (B12, W11, W14) | One installable package, local SQLite + NumPy store, CLI + modular app, ~200 automated tests |

Measured result on a labelled 4-camera, 2-minute retail scene (10 people, T4 GPU, `balanced` preset):
**10 identities for 10 people, none split, none mixing two people, cross-camera IDF1 88.3%, held-out search Rank-1
96%, 34 fps, 1.02 GB of GPU memory.** v1 could not be run on this data (cloud dependencies, Kaggle paths), so there is
no head-to-head v1 number; section 8 explains what can and cannot be compared.

---

## 2. The papers behind v1

Your notebooks cite no papers, so the sources were identified from the techniques (search and full-text reading,
`docs/dev/source_papers.md`).

| v1 technique | Source | v1 vs the paper | v2 |
|---|---|---|---|
| Self-adaptive gallery: budget 50, drop the sample with the worst redundancy + uncertainty | **Casao, Azagra, Murillo, Montijano, "A Self-Adaptive Gallery Construction Method for Open-World Person Re-Identification", Sensors 23(5):2662, 2023** | Same structure. Weights 0.5/0.5 instead of the paper's γ = 0.6 (entropy) / 0.4 (diversity). **Missing**: the paper's diversity gate (a sample is added only if it increases the class's diversity) | Paper weights (γ = 0.6) and the diversity gate (section 5.1) |
| Softmax-ratio test p1/p2 ≥ τ, τ = 2, against weighted centroids | Casao et al., Sec. 3.3 | Same τ. Temperature ×15 instead of the paper's ×10 (v = 0.1) | Paper temperature ×10, applied to tracklet descriptors |
| Visibility-weighted centroids, skeleton gate r ≥ 0.5 | Casao et al., technique 4 (OpenPose joints there, YOLO-pose here) | Faithful in spirit; v1 picked the first pose in a crop (wrong person when two overlap, W12) | Kept; weights = visibility × detection confidence; the pose belongs to the right box |
| Unknown Data Manager with DBSCAN | Casao et al., Sec. 3.3: clusters need l = 20 samples, compared with k = 3 nearest classes, new class if distance > 0.1, bootstrap until Q = 20 classes | Looser: 5 samples, merge if cos ≥ 0.85, no k-nearest test, no bootstrap rule, no final flush | Paper parameters, on tracklets, with a final flush (section 5.2) |
| α-FPS prototypes (K = 3) | Name from **Al Muzaddid & Beksi, "Person Re-Identification via Generalized Class Prototypes", arXiv:2510.17043** | Only the name: your formula (quality + distance, greedy) is your own, and the paper's algorithm is different (centroid + interpolated farthest points) | Your version kept, fixed (section 6); the paper's version added as the `gcp` option |
| Face as ground truth, retroactive identity healing | **No paper**: your own idea | Implemented unsafely (single frame, irreversible) | Kept as the project's own contribution, made evidence-based and reversible (section 5.3) |
| CLIP-ReID body model | Li, Sun, Li, "CLIP-ReID", AAAI 2023 | Market-1501 weights | MSMT17 weights of the same model (section 3.4) |
| ArcFace faces (InsightFace buffalo_l) | Deng et al., ArcFace, CVPR 2019; SCRFD detector | Same | Same, plus face quality scoring |

Casao et al. work on a **single camera, frame by frame**, with OSNet features and no faces. Everything multi-camera,
every face rule, the tracker and the prototypes are outside that paper, in v1 and in v2. A 2026 paper,
**OmniGlimpse (arXiv:2610.36785)**, formalises the principle v2 adopts in section 4 ("only verified observations
update the memory") and reports +9.1 IDF1 for it over online baselines; it is cited as supporting evidence, not as a
source we implemented.

---

## 3. Stage by stage: perception

### 3.1 Detection

**v1** (m4 cell 6): RT-DETR-L (`rtdetr-l.pt`, April 2023) on every frame, person class only. If a frame had no
detection the loop did `continue` **before** calling the tracker, so the tracker's motion model was not advanced on
empty frames (W12).

**v2** (`models/detector.py`, presets in `configs/presets/`):

| Preset | Detector | Why |
|---|---|---|
| `lite` | YOLO26n-pose | Runs on a GTX 1050 at ~29 fps |
| `balanced` (default) | YOLO26m-pose, **one model for boxes and keypoints** | Removes the separate pose model; COCO AP 52.5 vs RT-DETR-L 53.0 (equal), half the latency |
| `accurate` | RF-DETR-M (Mar 2025) + YOLO26s-pose on crops | +1.7 COCO AP over RT-DETR-L |
| `max` | RF-DETR-L + YOLO26m-pose | +3.5 AP |

The tracker is now updated on every processed frame, including empty ones. Frames larger than 1920 px are downscaled
right after decoding (`runtime.max_side`), which matters for 4K phone videos.

*Impact*: detection quality is about the same as v1 for `balanced` (the AP numbers are general COCO numbers; no model
publishes a person-only figure). The measured weak point is **recall of small people** (80-120 px tall: 39.7% found by
`balanced`), see section 11.

### 3.2 Tracking

**v1**: BotSort with its **own** re-ID network running on every detection of every frame: OSNet x0.25 in m4,
a second full CLIP-ReID ViT-B/16 in m6 (`_make_botsort_tracker`). The pipeline then ran CLIP-ReID **again** on the
gated crops. In m6 that is two ViT-B/16 passes per person per frame (W1).

**v2** (`tracking/tracker.py`): boxmot 26 trackers (ByteTrack, BoT-SORT, OccluBoost). Two modes:
- **motion-only** (`lite`, `balanced`): no appearance network in the tracker at all;
- **appearance** (`accurate`, `max`): the body embedding of every detection, computed once by our CLIP-ReID, is
  *passed* to BoT-SORT/OccluBoost (boxmot's `Detections(embeddings=...)`), and the same embeddings are reused as
  tracklet samples. So the tracker gets appearance without a second network.

Why motion-only is still acceptable for `balanced`: the tracklet splitter (section 4.3) catches the identity swaps a
motion-only tracker makes when people cross. This was learned the hard way (section 9.1). Measured on the 2-minute
scene: motion-only + split IDF1 89.4% at 16.9 fps (GTX 1050); appearance + split 90.4% at 9.6 fps.

### 3.3 Quality gate

**v1** (m4 step 3): crops of tracks with confidence ≥ 0.75 were sent to YOLO-pose; the visible-keypoint ratio
(`conf > 0.3`) had to be ≥ 0.5. Two problems: `res.keypoints.data[0]` is whichever person the pose model found
**first** in the crop, which in a crowded crop can be the neighbour (W12); and every crop of every frame went through
it, so the gate ran ~30 times per second per person.

**v2** (`tracking/tracklets.py`):
- the keypoints belong to the box: in single-model presets they are the detector's own keypoints for that detection;
  in two-model presets the pose whose box contains the target's centre is chosen;
- cheap gates first, in this order: detection confidence, box height ≥ 96 px, box not cut by the frame edge, then
  keypoint visibility ≥ 0.5 and quality (visibility × confidence) ≥ 0.35;
- **sampling**: a track is sampled at most once every 5 processed frames. Consecutive frames are almost identical, so
  this loses nothing and divides the pose/embedding work by 5.

### 3.4 Body embeddings

**v1**: CLIP-ReID ViT-B/16 with **Market-1501** weights (`clip_market1501.pt`, through boxmot).

**v2**: the same CLIP-ReID ViT-B/16, with the official **MSMT17** weights, re-implemented in `models/clipreid_net.py`
(boxmot 26 dropped the CLIP-ReID weights; our implementation loads the official checkpoint with zero missing keys).
Same 1280-D feature (BN-neck of the 768-D class token + 512-D projection).

*Why MSMT17*: Market-1501 has 6 outdoor cameras and 1,501 people; MSMT17 has 15 indoor and outdoor cameras and
4,101 people. Cross-domain performance is the real issue for a store system (CLIP-ReID Market → Duke drops from 89.6
to 50.2 mAP), and the larger, more varied training set is the better bet for unseen cameras. Same model size, same
speed, same dimension. `lite` uses OSNet x1.0 MSMT17 (512-D) instead.

Embeddings are computed in batches across tracks, in fp16 on T4-class GPUs (with `torch.autocast`, see section 9.3).

### 3.5 Faces

**v1** (m4): on face cameras, `face_app.get(frame)` ran on the **whole frame, every frame**; a face was paired with a
body if its centre fell in the top 40% of the box; the best face per track was the one with the highest detection
score; then a **Pinecone query per frame per paired body** (B16).

**v2** (`tracklets.py`, `models/face.py`):
- face detection only for accepted body samples of face cameras, on the head region crop, so one face cannot be
  paired with two bodies and the cost drops by ~5× (sampling) plus the region size;
- face **quality** = detection score × size (0 at 32 px, 1 at 112 px) × frontalness (from the 5 landmarks), instead of
  the detection score alone, which happily picks a sharp profile view;
- up to 5 best faces kept per tracklet; no network call anywhere.

---

## 4. The core change: tracklets

This is the change that everything else depends on, so it is worth understanding precisely.

### 4.1 What v1 did, frame by frame

For every frame and every gated track (m4 cell 10, steps 5-6):

1. `gid = local_to_global_map.get(track_id)`;
2. if the track already had a gid → `gallery.add_observation(gid, crop)` (no check);
3. if it had none → `classify_live_track`: softmax ×15 over the gallery centroids; accept if best cosine ≥ 0.78 and
   p1/p2 ≥ 2, otherwise the crop goes to the unknown pool;
4. once accepted, the mapping is stored and **never re-examined** for the rest of the track.

Three consequences:
- **Sticky errors (B3).** One unlucky frame (motion blur, an occluding person) decides the identity of a track that
  may last a minute, and every following frame is written into that wrong gallery, pulling its centroid towards the
  wrong person. The error then spreads to future decisions.
- **ID switches poison the gallery.** If the tracker swaps two people, the track keeps its gid and the second
  person's crops are written into the first person's gallery.
- **Noise.** A single frame's embedding is noisy; v1 took its most important decision on exactly that.

### 4.2 What v2 does

A **tracklet** is one continuous track of one person in one camera. The `TrackletBuilder` collects, during the
track's life, every box (stored, for drawing and evaluation) and every 5th processed frame's gated crop with its
embedding. A tracklet is closed when the track has been lost for 2 s of video time, or at the end of the video.
Only then is its identity decided, **once**, by `IdentityEngine.on_tracklet` (`identity/associate.py`):

1. **descriptor** = quality-weighted mean of the tracklet's best 16 samples (averaging 16 views cancels most of the
   per-frame noise);
2. **face first**, if the tracklet has accumulated face evidence (section 5.3);
3. **body test** (Casao): softmax ×10 over the gallery centroids, accept if p1/p2 ≥ 2 and cosine ≥ 0.78, **and** at
   least half of the individual samples vote for the same gid (`sample_vote`), with impossible gids excluded
   (section 5.4);
4. **verify-before-update**: only now are the samples offered to that gid's gallery, through the diversity gate;
5. otherwise the tracklet goes to the Unknown Data Manager (section 5.2).

The price is latency: a decision comes when the tracklet ends (2 s after the person leaves or is lost), not on the
first frame. For a system that answers "who went where" after the fact, and for a demo, that is the right trade.

### 4.3 Splitting a tracklet when the tracker swaps two people

A tracklet-level design has one weak point: if the tracker swaps two people, the tracklet contains both, and one
decision covers both. You found exactly this in testing ("person 1" was a mix of two differently dressed people).
v2 handles it in `tracklets.py`:

- the builder keeps the chronological history of all embedded samples, and for every box the largest overlap with
  another track in the same frame;
- at finalisation it searches the change point `k` that minimises the cosine similarity between the mean embedding
  of the samples before `k` and after `k` (at least 3 samples each side); if that similarity is below **0.7**, the
  tracklet is cut, recursively;
- the cut is placed at the frame where the box overlapped another person the most (where they crossed), and each
  piece becomes its own tracklet with its own samples, best crop and faces.

The 0.7 threshold was **calibrated with ground truth**: on AI City, two halves of the same person's tracklet have
similarity ≥ 0.88 (median 0.96); segments of two different people ≤ 0.58 (median 0.17); the halves of the two mixed
tracklets found were 0.05 and 0.04. A new evaluation metric counts tracklets that contain more than one real person.

*Impact* (2-minute scene, `balanced`): mixed tracklets 1 → 0, identities 11 → 10 (the duplicate disappears),
people split over two gids 2 → 0, cross-camera IDF1 82.4% → 89.4%, search Rank-1 92% → 96%, no speed cost.

---

## 5. Identity memory

### 5.1 Self-adaptive gallery (Casao)

`identity/gallery.py`, per identity, at most m = 50 samples.

| | v1 (`GalleryManager`) | v2 (`GalleryBank`) | Paper |
|---|---|---|---|
| What is inserted | every frame of an assigned track | a confirmed tracklet's samples (≤ 16, best quality first) | samples of confident decisions |
| Insertion test | none | **diversity gate**: insert only if the sample's distance to its nearest neighbour in the class is ≥ the class's smallest pairwise distance (Eq. 5-6) | same |
| Drop rule (over budget) | `0.5 × entropy + 0.5 × redundancy` | `argmax[0.6 × H/log N − 0.4 × D]` | γ = 0.6 |
| Prune after a merge | one sample per call (stayed over budget in m6, B7) | loops until within budget | — |
| Centroid weights | pose visibility | visibility × detection confidence | visibility |

*Why the gate matters*: without it, the 50 slots of a gallery fill with 50 almost identical frames from the first
two seconds of a track, and the pruning then mostly removes near-duplicates. With it, a gallery collects different
views (front, back, side, other cameras), which is what makes it useful for matching later sightings.

Note on the entropy normaliser: the paper prints `log(1/N)`, which is negative and would invert the term; the text
says the entropy is normalised to [0, 1], so v2 (like your v1, after its own fix) uses `log N`.

### 5.2 Unknown Data Manager

`identity/udm.py`.

| | v1 (`UnknownDataManager`) | v2 | Paper |
|---|---|---|---|
| Pooled items | single **frame** crops | **tracklet** descriptors | samples |
| Schedule | every 30 frames (skipped on empty frames, B9) | every 20 pooled tracklets and at the end of each camera | — |
| Cluster size to act | 5 frames (one track can mint a person, W8) | 20 samples (l = 20) | l = 20 |
| Compare with gallery | nearest weighted centroid, merge if cos ≥ 0.85 | k = 3 most frequent classes among the samples' nearest centroids; new identity if min distance > 0.1 | k = 3, α = 0.1 |
| Constraints | none | cannot-linked pairs never clustered together | — |
| End of video | not flushed: unresolved people **dropped** (B8) | flushed: everyone ends with an identity or is explicitly unidentified | — |
| Bootstrap | the first frame of the first person minted a gid | until Q = 20 identities exist, a well-sampled (≥ 8 samples) tracklet mints directly or joins at cos ≥ 0.9 | Q = 20, ε = 0.9 |

### 5.3 Faces as ground truth (your idea), made safe

**v1** (`execute_identity_merge`): a single frame whose face matched Pinecone at ≥ 0.55 caused, immediately:
reassignment of all local tracks, moving the old identity's files into the face identity's folder, pushing and
deleting MongoDB documents, and relabelling face vectors (through a zero-vector query, which Pinecone is expected to reject on
cosine indexes and whose failure was hidden by a broad `except`, B6). Problems:
- **single-frame evidence, no margin**: one look-alike frame fuses two people (B4);
- **irreversible**: files moved, documents deleted, no history;
- **data loss** when the target document did not exist (after a reset of MongoDB but not Pinecone);
- **ineffective inside one video**: faces were upserted to Pinecone after the video ended (W7).

**v2** (`associate.py`, `constraints.py`, `consolidate.py`):
- a tracklet uses face evidence only with **≥ 2 faces** of sufficient quality, combined into one descriptor;
- the best face identity must score ≥ 0.55 **and** lead the next *different-looking* identity by 0.05 (a runner-up that
  is itself a face match of the winner is a fragment of the same person and does not count as competition);
- a face assignment that the body evidence strongly contradicts is not written into the body gallery; it records a
  **must-link** instead;
- two identities whose faces clearly disagree (cos < 0.3) get a **cannot-link**: they can never be merged;
- the actual merging happens in the consolidation step (5.5), which records every relabelling in
  `tracklet_gid_history`, so any merge can be undone;
- face descriptors are stored as each tracklet finishes, so face evidence works inside one run.

What is kept: the face still overrides the body when the evidence is good, and still triggers the repair of body
fragments. That is your contribution, and the README presents it as such.

Honest status: the face path is tested with synthetic data and InsightFace sample images, not yet on a real
face-camera video (section 11).

### 5.4 Constraints: one person cannot be in two places

**v1** had one constraint, and it was a good one: inside a single frame, two tracks could not take the same gid
(`claimed_gids_this_frame`). It did not extend over time: a gid could be given to a track that overlapped in time
with another track of the same gid on the same camera, as long as they were never processed in the same frame.

**v2** generalises your rule (`identity/constraints.py`):
- **same camera, overlapping in time** → cannot be the same person;
- **topology** (optional): two cameras with a known minimum walking time, and a gap shorter than that → cannot be the
  same person;
- **face conflict** (5.3);
- **manual split** in the app → stored, so automatic consolidation never re-merges it.

These rules apply **within one run** (one recording session with its own clock). Section 9.2 explains why.

### 5.5 Consolidation (repairing fragmentation)

Neither v1 nor the paper repairs fragmentation: if one person ends up with two gids, both live forever. v2 adds a
final step at the end of each run (`identity/consolidate.py`): agglomerative clustering (average linkage) of the
identity centroids, stopping at cosine 0.75, applying face must-links first and never violating a cannot-link (also
transitively). The merged identity keeps the oldest gid; every relabelling is recorded and reversible. The app
exposes manual merge, split, rename, re-cluster and delete.

*Impact*: on the evaluation scene, the tracker alone produces 44-49 tracklets for 10 people (a person is cut into
many pieces by occlusions and camera changes). Per-camera IDF1 of the raw tracker on the busy cameras is 56-65%; with
identity association it is 80-94% (Colab run, cam_0701 56.0 → 80.5, cam_0711 65.1 → 94.3). That +20-30 points is the
identity layer re-joining fragments.

---

## 6. Search and prototypes

### 6.1 Prototypes (α-FPS)

**v1** (`alpha_fps`, defined only in m6, so m5 crashed with `NameError`, B12): start from the best-quality sample,
then repeatedly pick the sample maximising `0.5 × quality + 0.5 × distance` to the nearest picked one. The two terms
are on different scales: quality values sat in 0.4-0.9 with little spread, distances in 0-0.3. The sum was therefore
driven by quality, and the "farthest point" part had almost no effect: the 3 prototypes tended to be the 3 sharpest
crops, often from the same moment.

**v2** (`identity/prototypes.py`): the same algorithm with both terms min-max normalised to [0, 1] at each step, so
quality and diversity really weigh 50/50, and one function used everywhere. The algorithm of the paper whose name
you used (centroid + farthest points pulled towards the nearest prototype, `x ← x + α(p − x)`) is available as the
`gcp` option, so the two can be compared.

### 6.2 Search

| | v1 (`search_person`, `search_face`) | v2 (`search.py`) |
|---|---|---|
| Body | Pinecone top-10, best score per gid, UNKNOWN below 0.75, AMBIGUOUS if softmax ×15 ratio of the top two < 1.5 | same logic on local prototypes, temperature 10 (paper) |
| Face | top-1, no margin | best per gid, AMBIGUOUS if the runner-up is within 0.05 |
| Both | face silently replaced the body answer | a face MATCH wins, otherwise the body answer; the result says which modality decided |
| Missing gid document | ERROR | cannot happen (one store) |

Search queries are always embedded with the model the environment was processed with (an environment processed with
CLIP-ReID cannot be searched with OSNet features).

---

## 7. Storage, time, multi-camera, app

### 7.1 Storage

**v1**: gallery in RAM and a pickle (with absolute Kaggle file paths that had to be re-resolved on reload, B20),
sightings and faces in MongoDB, crops in ImageKit, vectors in Pinecone, prototypes in a Pinecone namespace that
"Save" deleted and rewrote (B13-B15). Only the best box of each sighting was stored.

**v2**: everything in one folder per environment: SQLite (`runs`, `cameras`, `tracklets`, `track_points` for **every
box of every processed frame**, `identities`, `samples`, `tracklet_gid_history`, `identity_cannot_links`), NumPy vector
collections, crop and face JPEGs. Runs accumulate; reloading reproduces the same decisions (tested). Works offline,
free, on Colab with Drive, and the same folder opens on a PC.

Why every box: drawing videos, measuring tracking metrics and computing dwell time all need the full trajectory, not
one box per sighting.

### 7.2 Time

**v1**: `VIDEO_START_TIME` was one constant for all cameras (m4) or "now" (m6), and fps was hard-coded to 30 (B1).
Durations of a 25 fps video were 20% wrong, and two cameras could not be placed on one timeline.

**v2**: `timestamp = camera.start_ts + frame_index / fps`, fps read from the file, `start_ts` per camera in the camera
YAML. This is what makes the mosaic, the follow-a-person clip, transit-time constraints and journeys possible.

### 7.3 Multi-camera

**v1** processed one camera per run; "multi-camera" meant running the notebook again on another video with the
gallery carried over. Matches across cameras were appearance-only and global (W6).

**v2** processes a **camera set** in one run (sequentially, into one shared identity space), with per-camera time,
cannot-links, optional topology, consolidation across cameras, a synchronized mosaic, and a follow clip that cuts from
camera to camera.

### 7.4 App

**v1** (m6): a 1,958-line notebook cell that re-embedded copies of m4 and m5 logic, published with `share=True` and
**no login**, including a "Reset All Databases" button (B22).

**v2** (`app/`): modular (one file per tab, shared theme and HTML components), password required for public links,
environments (isolated sets of cameras and people), admin tools with confirmation, cached renders, metrics including
per-stage timing and real GPU memory. The notebooks and the app call the same package functions, so there is no
copied logic.

### 7.5 Reproducibility

**v1**: unpinned `pip install`s, boxmot API drift handled by sniffing constructor signatures, Kaggle paths, secrets.
**v2**: pinned versions, a CLI, a Colab notebook that only calls the CLI, a test suite (unit tests on synthetic
embeddings for every decision rule), AGPL-3.0 license (required by Ultralytics and boxmot).

---

## 8. Evaluation

### 8.1 Why 85.35% was not a real accuracy

v1's evaluation (m5 cells 9-10) took the best crops **saved by the pipeline**, queried them against prototypes built
**from those same crops**, and counted a hit when the answer matched the label **the pipeline itself had assigned**.
Three of the queries per person literally were the prototypes (similarity ≈ 1.0; the average similarity of 0.925
reflects this). So 85.35% measured how consistent the system was with itself. A system that merged two real people
into one gid would score *higher*, because the merged gid is still self-consistent. There was no ground truth, so
accuracy could not be measured; the 85.35% should not be quoted as accuracy.

### 8.2 What v2 measures

`reid evaluate` on a labelled dataset (`src/reid/eval/`):
- **per-camera tracking** (IDF1, MOTA, ID switches, precision, recall) for the system identities and for the raw tracker;
- **cross-camera IDF1** in the CityFlow style (one accumulator over all cameras, global ids), the standard metric of
  multi-camera tracking challenges;
- **fragmentation and purity**: people split over several gids, gids mixing several people, and tracklets mixing
  several people (added after your swap report);
- **held-out retrieval**: 5 query boxes per person cropped from the video at moments the system did not sample,
  Rank-1, Rank-5, mAP.

Dataset: **AI City Challenge 2024, Track 1** (NVIDIA, CVPR workshop), synthetic retail/warehouse scenes rendered with
Omniverse, exact labels for every person in every frame, CC-BY-4.0. We use scene 075: 4 overlapping cameras, 2
minutes, 10 people, 19,146 labelled boxes.

### 8.3 Results (scene 075, 2 minutes, frame stride 2)

| Configuration | Cross-camera IDF1 | Identities (10 people) | Split | Mixed | Rank-1 | Speed | GPU memory |
|---|---|---|---|---|---|---|---|
| `balanced`, before the tracklet splitter | 82.4% | 11 | 2 | 1 | 92% | 16.7 fps (GTX 1050) | — |
| `balanced`, final, local | 89.4% | 10 | 0 | 0 | 96% | 16.9 fps (GTX 1050) | 0.87 GB* |
| `balanced` + appearance tracking | 90.4% | 10 | 0 | 0 | 96% | 9.6 fps (GTX 1050) | — |
| **`balanced`, final, Colab T4** | **88.3%** | **10** | **0** | **0** | **96%** | **34 fps** | **1.02 GB** |

\* face models run on CPU on the GTX 1050 (old GPU architecture), so they are not counted there; on the T4 they run on
the GPU and are included.

Reading the table: identity is now essentially perfect on this scene (purity 100%, completeness 97%); what limits
IDF1 is **recall** (IDR 81%): small and distant people are not detected (section 11).

### 8.4 What cannot be compared

v1 and v2 were never run on the same data: v1 needs Pinecone/MongoDB/ImageKit accounts and Kaggle paths, and v1's
own number is not an accuracy (8.1). The v2 numbers are on one synthetic scene with 10 people; they show the system
works and where it fails, not how it ranks against published methods. A fair next step is real labelled footage
(e.g. MMPTrack, licence permitting) and more scenes.

---

## 9. Mistakes made during the rewrite

These were made by the v2 rewrite, not by v1, and were found by your testing or by the new metrics.

### 9.1 Motion-only tracking swapped people who crossed

v2 first removed appearance from the tracker entirely (to avoid v1's duplicate network, W1), assuming the tracklet
layer would compensate. It cannot: a tracker swap creates one tracklet containing two people. Your v1 did not have
this problem as visibly because its BotSort used OSNet appearance. You noticed it in the app ("person 1" was two
people, plus a duplicate). Found, measured (new tracklet-purity metric) and fixed with the splitter and the optional
appearance mode (4.3, 3.2).

### 9.2 Re-processing the same videos duplicated everyone

The "same camera at the same time" rule was applied across runs. Re-processing the same videos (same camera names,
timestamps from 0 again) made every new sighting "overlap" the old sighting of the same person, which forbade that
person's gid: all 10 people got a second identity, and search Rank-1 fell to 16% because every person existed twice.
Fixed by applying time rules only within a run; verified (a second run of the same clip creates 0 new identities).

### 9.3 Crash on the T4 in half precision

On GPUs with fast fp16 (T4) the CLIP-ReID model was converted with `net.half()`, which also converted its LayerNorm
layers that must stay in fp32, so the first Colab run crashed. The GTX 1050 never uses fp16, so local tests did not
see it. Fixed with `torch.autocast`; a test now forces fp16 and checks fp16 and fp32 embeddings agree.

### 9.4 Smaller ones

The `max` preset's face model (antelopev2) could not load (nested zip folder); RF-DETR printed logs into the JSON
output; GPU memory was under-reported (PyTorch only, missing the CUDA context and ONNX Runtime); rendering was slower
than processing on Colab (an expensive resize mode, fixed with a 5× faster one). All fixed, each with a test or a
measurement, listed in `docs/dev/changes_log.md`.

---

## 10. What v1 got right

- **The architecture choice**: a self-adaptive gallery with an open-world unknown manager is a sound basis for a
  system where new people keep arriving. v2 keeps it and cites it.
- **Pose-based quality gating** with visibility-weighted centroids (from the paper) is kept.
- **The softmax-ratio ambiguity test** for both association and search is kept.
- **The one-gid-per-frame constraint** (`claimed_gids_this_frame`) was the seed of v2's cannot-links.
- **Faces as the strongest evidence, used to repair body fragments**: your own idea, kept as the project's
  contribution.
- **Prototypes for search** (your α-FPS): kept, with the scale problem fixed.
- **Storing per-sighting entry/exit times, best crop and its box**: the right data for analytics, now complete.

---

## 11. Limitations that remain

- **Small-person recall.** With `balanced`, people 80-120 px tall are found 39.7% of the time and 40-80 px 31.7%.
  This, not association, caps IDF1. Levers: a larger detector input size, the `accurate`/`max` presets, cameras
  closer to the scene.
- **Faces not yet validated on real footage.** The AI City people are synthetic with no usable faces; the face path is
  tested with synthetic evidence and sample images. Your phone recordings will be the first real test.
- **Thresholds calibrated on one scene** (split 0.7, association 0.78, consolidation 0.75). One false split was seen
  (one person cut in two, re-joined by the identity layer).
- **Appearance tracking costs about 40% of the speed** on the GTX 1050; it is on only in `accurate` and `max`.
- **Offline, not live.** Videos are processed as files; live phone/IP cameras are future work
  (to be described in `docs/future_work.md`).
- **Privacy.** Faces and body descriptors are biometric data. v2 keeps everything local and requires a login for
  shared links, but a real deployment would need consent, retention limits and a legal basis (GDPR / Algerian law
  18-07 on personal data).

---

## 12. Talking points

Short answers you can give in an interview:

- *"What did you change from the first version?"* The identity decision moved from single frames to whole tracklets,
  and the gallery only learns from verified decisions. That removed the sticky errors of v1 and is consistent with
  recent work (OmniGlimpse 2026).
- *"Which papers?"* Casao et al., Sensors 2023 for the self-adaptive gallery, the softmax-ratio test and the Unknown
  Data Manager, implemented with the paper's parameters; CLIP-ReID (AAAI 2023) and ArcFace (CVPR 2019) for features;
  Al Muzaddid & Beksi 2025 for class prototypes. The face-driven repair and the multi-camera constraints are my own.
- *"How do you know it works?"* Ground-truth evaluation on AI City 2024: 10 identities for 10 people, none merged,
  none split, cross-camera IDF1 88%, held-out search Rank-1 96%, 34 fps on a free Colab T4 with 1 GB of GPU memory.
  The earlier 85% figure was self-consistency, not accuracy, and I replaced it.
- *"What went wrong?"* Removing appearance from the tracker caused identity swaps when people crossed; I found it in
  testing, added a metric that measures it, calibrated a tracklet splitter on ground truth, and it went to zero.
- *"What would you do next?"* Better small-person detection, real labelled footage with faces, and live camera input.
