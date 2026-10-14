# 🛒 Multi-Modal Person Re-ID & Face Association Pipeline
## Complete Technical Report

> **Project:** Real-time person re-identification and face association for retail analytics
> **Environment:** Kaggle (2× NVIDIA Tesla T4 GPUs, 16 GB VRAM each)
> **Language:** Python 3.12 | **Framework:** PyTorch

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [AI Models & Specifications](#2-ai-models--specifications)
3. [Cloud Infrastructure](#3-cloud-infrastructure)
4. [Configuration & Thresholds](#4-configuration--thresholds)
5. [Module 4: Video Processing Pipeline](#5-module-4-video-processing-pipeline)
6. [Module 5: Search Engine](#6-module-5-search-engine)
   - [6.3 Movement Heatmaps](#63-movement-heatmaps-person-search)
7. [Module 6: Interactive Dashboard](#7-module-6-interactive-dashboard)
8. [Evaluation Results](#8-evaluation-results)
9. [Known Limitations & Design Decisions](#9-known-limitations--design-decisions)

---

## 1. System Overview

### 1.1 What This System Does

This pipeline watches security camera footage from a supermarket and:
1. **Detects** every person in every frame
2. **Tracks** them as they move through the scene (temporary ID)
3. **Identifies** them by matching their body appearance and/or face against a persistent database (permanent global ID)
4. **Records** their visit: when they entered, how long they stayed, best body/face photo
5. **Enables search**: given a photo, find who this person is in the database

### 1.2 Two Modalities Working Together

| Modality | Strength | Weakness | Role |
|----------|----------|----------|------|
| **Body Re-ID** | Works at any distance/angle, doesn't need face visibility | Fails with similar clothing (e.g. store uniforms) | Primary identification |
| **Face Recognition** | Extremely accurate, identity ground truth | Requires close-up, front-facing, well-lit face | Override & correction |

**Key insight:** Face recognition acts as **ground truth**. When a face is recognized, it can *retroactively correct* a wrong body-based assignment (Identity Merge).

### 1.3 Architecture Diagram

```
Video Frame
    │
    ├──► RT-DETR (Person Detection)
    │        │
    │        ▼
    │    BotSort (Multi-Object Tracking) ──► track_id (temporary)
    │        │
    │        ▼
    │    YOLOv8-Pose (Quality Gate) ──► reject bad crops
    │        │
    │        ▼
    │    CLIP-ReID (Body Embedding) ──► 1280-D vector
    │        │
    │        ▼
    │    ┌─────────────────────────────────────────────┐
    │    │         IDENTITY RESOLUTION ENGINE          │
    │    │                                             │
    │    │  Gallery Match ──► Softmax Confidence        │
    │    │  Face Override ──► Pinecone Face Lookup      │
    │    │  Unknown Pool ──► DBSCAN Clustering          │
    │    │                                             │
    │    │  Output: global_id (permanent)              │
    │    └─────────────────────────────────────────────┘
    │
    └──► InsightFace (Face Detection + Embedding)
              │
              ▼
         Face-Body Spatial Pairing ──► 512-D face vector
```

---

## 2. AI Models & Specifications

### 2.1 RT-DETR-L (Person Detection)

| Property | Value |
|----------|-------|
| **Full Name** | Real-Time DEtection TRansformer (Large) |
| **Architecture** | CNN backbone (HGNet-v2) + Efficient Hybrid Encoder + Transformer decoder |
| **Parameters** | ~32–36 million |
| **FLOPs** | ~100–140 GFLOPs |
| **Model Size** | ~130 MB |
| **Input Resolution** | 640×640 |
| **VRAM** | ~2–4 GB (inference) |
| **Key Feature** | **NMS-free** — no post-processing bottleneck, consistent latency |
| **GPU Assignment** | `cuda:0` |

**Why RT-DETR over YOLO for detection?** RT-DETR's transformer decoder eliminates Non-Maximum Suppression (NMS), which is a serial bottleneck. This gives more predictable latency, critical for real-time tracking.

We filter for `classes=[0]` (COCO class 0 = person), so only humans are detected.

### 2.2 YOLOv11s-Pose (Skeleton Quality Gate)

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv11 Small with Pose head |
| **Keypoints** | 17 (COCO format: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| **Input** | Cropped body image |
| **GPU Assignment** | `cuda:1` |

**Purpose:** This model does NOT contribute to identification. It serves purely as a **quality gate**. For each body crop, it estimates which of the 17 skeleton keypoints are visible. The system calculates:

```
visible_ratio = (keypoints with confidence > 0.3) / 17
```

If `visible_ratio < 0.5` (less than 50% of skeleton visible), the crop is **rejected**. This prevents garbage data (e.g., a partially visible arm, a blurry silhouette) from entering the gallery and corrupting identity embeddings.

### 2.3 BotSort (Multi-Object Tracker)

| Property | Value |
|----------|-------|
| **Full Name** | Bag of Tricks for SORT |
| **Type** | Tracking-by-detection |
| **ReID Backbone** | OSNet_x0_25 (MSMT17 pretrained) |
| **Association** | IoU + ReID appearance fusion |

**How BotSort works:**
1. Receives person detections from RT-DETR each frame
2. Uses a **Kalman Filter** to predict where existing tracks should be
3. **Camera Motion Compensation** adjusts predictions if the camera moves
4. Computes a **cost matrix** combining:
   - **IoU** (spatial overlap between predicted and detected boxes)
   - **Appearance similarity** (lightweight ReID features via OSNet)
5. Uses the Hungarian algorithm to match detections to tracks
6. Assigns a `track_id` that persists as long as the person is continuously visible

> ⚠️ **Important:** `track_id` is **temporary** — it only lives while the person is in view. The moment they leave and return, they get a new `track_id`. Our system maps `track_id → global_id` (permanent) using the Identity Resolution Engine.

### 2.4 CLIP-ReID (Body Re-Identification)

| Property | Value |
|----------|-------|
| **Architecture** | CLIP ViT-B/16 fine-tuned for ReID |
| **Training Dataset** | Market-1501 |
| **Parameters** | ~86 million (frozen CLIP backbone + learnable tokens) |
| **Embedding Dimension** | **1280-D** (our Pinecone index dimension) |
| **Training Strategy** | 2-stage: (1) optimize learnable text tokens, (2) fine-tune visual encoder |
| **GPU Assignment** | `cuda:1` |

**What it does:** Takes a cropped body image → outputs a 1280-dimensional vector that encodes the person's appearance (clothing color, texture, body shape, accessories). Two crops of the same person should produce vectors with high cosine similarity (>0.85).

**Output normalization:** All embeddings are L2-normalized: `F.normalize(emb, p=2, dim=1)`, so cosine similarity = dot product.

### 2.5 InsightFace — SCRFD (Face Detection)

| Property | Value |
|----------|-------|
| **Model File** | `det_10g.onnx` (part of `buffalo_l` pack) |
| **Architecture** | SCRFD (Sample & Computation Redistribution for Face Detection) |
| **Input Resolution** | 640×640 |
| **Output** | Face bounding boxes + landmarks + detection confidence |

### 2.6 InsightFace — ArcFace (Face Recognition)

| Property | Value |
|----------|-------|
| **Model File** | `w600k_r50.onnx` (part of `buffalo_l` pack) |
| **Architecture** | ResNet-50 trained with ArcFace loss |
| **Training Dataset** | WebFace600K (600,000 identities) |
| **Embedding Dimension** | **512-D** |
| **Total Pack Size** | ~326 MB (`buffalo_l`) |
| **VRAM** | ~1–2 GB (entire pack) |

**Together:** SCRFD finds faces in the frame → ArcFace extracts a 512-D identity vector. This vector is so discriminative that even partial faces (sunglasses, masks) can still be matched.

### 2.7 GPU Distribution

```
┌──────────────────────────────────────┐
│           cuda:0 (T4 #1)             │
│  • RT-DETR-L (detection)             │
│  • BotSort (tracker)                 │
│  • InsightFace SCRFD+ArcFace (faces) │
├──────────────────────────────────────┤
│           cuda:1 (T4 #2)             │
│  • YOLOv11s-Pose (quality gate)      │
│  • CLIP-ReID (body embeddings)       │
└──────────────────────────────────────┘
```

---

## 3. Cloud Infrastructure

### 3.1 Pinecone (Vector Database)

Two separate indices for the two modalities:

| Index | Dimension | Metric | Contents |
|-------|-----------|--------|----------|
| `supermarket-reid` | 1280 | Cosine | Body prototypes (3 per person via α-FPS) |
| `supermarket-faces` | 512 | Cosine | Face embeddings (all captured faces) |

**Why Pinecone?** Sub-millisecond approximate nearest neighbor search over millions of vectors, with metadata filtering.

### 3.2 MongoDB Atlas

Stores structured metadata per person:

```json
{
    "global_id": 29,
    "first_seen_ts": 1746432000.0,
    "last_seen_ts": 1746432185.0,
    "faces": [
        {"url": "https://ik.imagekit.io/.../face_g029_t4.jpg", "timestamp": 1746432185.0, "camera_id": "cam_01"}
    ],
    "sightings": [
        {
            "camera_id": "Face_Cam_Entrance",
            "track_id": 4,
            "entry_ts": 1746432010.0,
            "exit_ts": 1746432185.0,
            "duration_sec": 175.0,
            "n_detections": 312,
            "avg_det_score": 0.87,
            "avg_skeleton_ratio": 0.72,
            "best_crop_url": "https://ik.imagekit.io/.../body_g029_t4.jpg"
        }
    ]
}
```

### 3.3 ImageKit (CDN)

Hosts body crops and face chips as JPEGs. Organized by identity:
```
/supermarket_reid/
    global_id_001/
        body_g001_t2_1746432000.jpg
        face_g001_t2_1746432000.jpg
    global_id_002/
        ...
```

---

## 4. Configuration & Thresholds

Every threshold in the system is tunable. Here are the defaults and what they control:

### 4.1 Detection & Quality

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `REID_CONF_THRESHOLD` | **0.75** | Minimum RT-DETR confidence to even consider a detection |
| `POSE_CONF_THRESHOLD` | **0.3** | Per-keypoint confidence threshold in YOLOv8-Pose |
| `MIN_VISIBLE_RATIO` | **0.5** | Minimum fraction of 17 keypoints that must be visible |

### 4.2 Identity Resolution

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `TAU` | **2.0** | Softmax ratio threshold — how confident the system must be to assign an identity |
| `MIN_COS_SIM` | **0.78** | Absolute minimum cosine similarity to even consider a gallery match |
| `MAX_IMAGES` | **50** | Maximum crops per identity in the self-adaptive gallery (budget) |

### 4.3 Face Recognition

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `IS_FACE_CAMERA` | **True** | Master switch: enables/disables all face logic for this camera |
| `FACE_CONF_THRESHOLD` | **0.7** | Minimum InsightFace detection score to trust a face |
| `FACE_MATCH_THRESHOLD` | **0.55** | Minimum Pinecone cosine similarity to accept a face match |

### 4.4 Unknown Data Manager (DBSCAN)

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `UDM_EPSILON` | **0.15** | DBSCAN distance threshold: `1 - cosine_similarity` |
| `UDM_MIN_SIZE` | **5** | Minimum crops needed to form a cluster |

### 4.5 Session Control

| Parameter | Default | Effect |
|-----------|---------|--------|
| `RESET_DATABASE` | **False** | When `True`: wipes MongoDB documents, deletes local crops and pickle. **Does NOT touch Pinecone or ImageKit** (manual deletion required) |

---

## 5. Module 4: Video Processing Pipeline

This is the core of the system. Here is the exact frame-by-frame logic.

### 5.1 Gallery Hydration (Pre-Processing)

Before processing any video, the system can load a previously saved `identity_buffer.pkl` file. This pickle contains:
- All body embeddings per global ID
- Detection confidence scores
- Skeleton visibility ratios
- File paths to saved crops

**Hydration logic:**
```
For each gid in saved_data:
    For each entry (embedding, ratio, score, filepath):
        If filepath resolves to an actual file → keep it
        If filepath is broken → skip this entry
    If gid has zero valid entries → skip this gid entirely
    Otherwise → load into gallery
```

This ensures the arrays (`embs`, `ratios`, `scores`, `filepaths`) stay **synchronized** — a critical bugfix that prevents `IndexError` crashes during budget optimization.

### 5.2 Frame Processing Loop

For every single frame in the video:

#### Step 1: Person Detection
```python
results = detector(frame, classes=[0], verbose=False)  # RT-DETR, persons only
```

#### Step 2: Multi-Object Tracking
```python
dets = [xyxy, confidence, class]  # stack detections
tracks = tracker.update(dets, frame)  # BotSort assigns track_ids
```

#### Step 3: Confidence Filter
Only tracks with `confidence >= REID_CONF_THRESHOLD (0.75)` proceed. This removes low-confidence ghosts.

#### Step 4: Pose Quality Gate
```python
visible_count = sum(1 for x, y, c in keypoints if c > 0.3)
ratio = visible_count / 17.0
if ratio < 0.5: REJECT  # less than 50% skeleton visible
```

#### Step 5: Body Embedding
```python
embeddings = extract_reid_features([crop1, crop2, ...])  # batch CLIP-ReID
# Each embedding is L2-normalized, 1280-D, on cuda:1
```

#### Step 6: Face Detection & Spatial Pairing (Face Cameras Only)
```python
faces_in_frame = face_app.get(frame)  # SCRFD finds all faces
```

For each tracked person, we check if any detected face **spatially belongs** to their body:

```python
def get_spatially_matched_face(body_box, faces):
    bx1, by1, bx2, by2 = body_box
    for f in faces:
        face_cx = (f.bbox[0] + f.bbox[2]) / 2
        face_cy = (f.bbox[1] + f.bbox[3]) / 2
        # Face center must be inside body box AND in the top 40%
        if bx1 <= face_cx <= bx2 and by1 <= face_cy <= by1 + 0.4 * (by2 - by1):
            return f
    return None
```

**Why top 40%?** A person's face is always in the upper portion of their body bounding box. This prevents matching a face floating near someone's waist (which would be another person behind them).

#### Step 7: Face Pinecone Lookup
If a face is matched to a body AND `face_det_score > 0.7`:
```python
face_truth_gid = search_face_pinecone(face_embedding)
```
This queries the `supermarket-faces` index. If cosine similarity ≥ 0.55, we get back the known global ID.

#### Step 8: Identity Resolution

This is where the magic happens. For each person in the frame:

**🔵 Case A: Track already has a global_id**
```
if gid not in claimed_gids_this_frame:
    ✅ Add observation to gallery
    Mark gid as "claimed" for this frame
else:
    ❌ CONFLICT: Another track already claimed this gid
    → Remove mapping, send to unresolved
```

> 🔴 **The Duplicate Guard:** `claimed_gids_this_frame` is a `set()` that prevents two different people visible in the same frame from being assigned the same global ID. If Track A is already GID 5, Track B cannot also be GID 5 in the same frame — Track B gets sent to the unknown pool instead.

**🔵 Case B: Face override (forced_gid)**
If the face lookup returned a known identity:
```
if face recognized as GID 29 BUT body was previously assigned GID 26:
    → execute_identity_merge(target=29, old=26)
    → GID 26 is permanently folded into GID 29
    → All history, embeddings, sightings migrated

if face recognized as GID 29 AND track has no body assignment yet:
    → item['forced_gid'] = 29  (honored in the unknown resolution step)
```

**🔵 Case C: Gallery classification (body matching)**
```python
best_gid, softmax_ratio, max_prob = gallery.classify_live_track(
    emb, tau=TAU, exclude_gids=claimed_gids_this_frame
)
```

The classification process:
1. Compute **weighted centroids** for each known identity (embeddings weighted by skeleton ratio)
2. Calculate cosine similarity between the query and all centroids
3. **Exclude** any GIDs already claimed this frame (the duplicate guard)
4. Apply **temperature-scaled softmax** (temperature = 15.0) to get probabilities
5. Compute **softmax ratio** = P(best) / P(second_best)
6. If `best_sim < MIN_COS_SIM (0.78)`: reject (too dissimilar)
7. If `softmax_ratio >= TAU (2.0)`: confident match → assign
8. Otherwise: ambiguous → send to UDM

**🔵 Case D: Unknown Data Manager (UDM)**
Crops that can't be confidently assigned go to the UDM pool. Every 30 frames, if the pool has ≥ 5 items:

```python
# DBSCAN clustering on cosine distance
dist_matrix = 1.0 - cosine_similarity_matrix
labels = DBSCAN(eps=0.15, min_samples=5, metric='precomputed').fit(dist_matrix).labels_
```

For each cluster found:
- **Face VIP Bypass:** If any item in the cluster has a `forced_gid` from face recognition, the entire cluster inherits that identity (majority vote if multiple faces)
- **Merge:** If the cluster centroid is close enough to an existing gallery identity (distance ≤ ε), merge into that identity
- **Mint:** If no match, create a brand new global ID

Items labeled as noise (label = -1) stay in the pool for the next DBSCAN run.

### 5.3 The Self-Adaptive Gallery (`GalleryManager`)

The gallery stores up to `MAX_IMAGES = 50` crops per identity. When a new observation pushes it over budget, the **worst** crop is dropped using a combined cost function:

```
drop_cost = 0.5 × redundancy + 0.5 × uncertainty
```

Where:
- **Redundancy** = max cosine similarity to any OTHER crop of the same person (high = this crop looks just like another one we already have)
- **Uncertainty** = normalized entropy over softmax similarities to ALL gallery centroids (high = this crop is ambiguous, looks like multiple people)

The crop with the **highest drop cost** is deleted — it's both redundant AND confusing.

### 5.4 Identity Merge (`execute_identity_merge`)

When face recognition detects that two GIDs are actually the same person:

```
execute_identity_merge(target_gid=29, old_gid=26)
```

This does:
1. **Gallery:** Move all embeddings, ratios, scores from GID 26 → GID 29
2. **Files:** Move image crops from `global_id_026/` → `global_id_029/`
3. **Local map:** Update all `track_id → 26` mappings to `track_id → 29`
4. **Sighting tracker:** Update active sightings pointing to 26 → 29
5. **MongoDB:** Migrate sightings array, delete old document

### 5.5 Sighting Tracking

The `SightingTracker` maintains per-track metadata:
- **Entry/exit timestamps** (frame number → UTC via `VIDEO_START_TIME + frame/fps`)
- **Detection count** and running sums for average confidence and skeleton ratio
- **Best body crop** (highest `conf × ratio`)
- **Best face crop** (highest `det_score`)
- **Best face embedding** (for Pinecone upload)

When a track disappears (BotSort stops reporting it), `close_track()` is called:
- Computes averages
- Moves the sighting to `completed_sightings` (if it has a global_id)

### 5.6 Cloud Upload (Post-Processing)

After the video finishes, all completed sightings are uploaded in parallel (10 threads):

1. **Body crop → ImageKit** → get URL → store in sighting
2. **Face crop → ImageKit** → get URL
3. **Face embedding → Pinecone** `supermarket-faces` with metadata `{global_id, face_url, camera}`
4. **Sighting → MongoDB** `$push` to person's sightings array
5. **Gallery → Pickle** serialized as `identity_buffer.pkl` for future hydration

---

## 6. Module 5: Search Engine

### 6.1 Body Search (α-FPS Prototypes)

Since each person can have up to 50 crops, storing all of them in Pinecone is wasteful. We use **α-FPS (Adaptive Farthest Point Sampling)** to select K=3 prototypes that maximize both **quality** and **diversity**.

**Algorithm:**
```
1. Pick the crop with highest quality score (conf × ratio) → Prototype 1
2. Compute distances from every remaining crop to Prototype 1
3. For each subsequent prototype:
   combined_score = α × quality + (1-α) × distance_to_nearest_prototype
   Pick the crop with highest combined_score
4. Update distances (min of current distance and distance to new prototype)
```

With `α = 0.5`, this balances quality and diversity equally. The result: 3 representative views of the person (e.g., front, side, back) that are all high-quality.

These 3 prototypes are upserted to `supermarket-reid` with IDs like `global_29_proto_1`.

**Search flow:**
```
Query image → CLIP-ReID → 1280-D vector
    → Pinecone query (top_k=10)
    → Group results by global_id (take MAX score per ID)
    → Softmax ambiguity check (temperature=15.0)
        If ratio < tau: "AMBIGUOUS"
        If best_score < threshold: "UNKNOWN"
        Otherwise: "MATCH" → fetch MongoDB document
```

### 6.2 Face Search (Direct)

Face embeddings are already highly discriminative, so **no prototypes are needed**. The search queries `supermarket-faces` directly.

**Pre-cropped face handling:** If the query is an already-cropped face chip (from ImageKit), the face detector can't find a face because the image is 100% face with no background context. Fix: pad the crop into a 3× larger gray canvas:

```python
def embed_face_crop(img_bgr):
    h, w = img_bgr.shape[:2]
    pad = max(h, w)
    canvas = np.full((h + 2*pad, w + 2*pad, 3), 128, dtype=np.uint8)
    canvas[pad:pad+h, pad:pad+w] = img_bgr
    return embed_face(canvas)
```

### 6.3 Movement Heatmaps (Person Search)

After a person is identified through a body or face search, the dashboard generates a **spatial movement heatmap** that visualises all the positions the matched person occupied across every camera where they were seen.

#### How It Works

**Step 1 — Collect positions from sightings**

Each completed sighting stores a `best_crop_bbox` field: the exact bounding box `[x1, y1, x2, y2]` (in full-frame pixel coordinates) of the best body crop. The helper `_bbox_centers_from_sightings()` converts these into center points:

```python
cx = (x1 + x2) / 2.0
cy = (y1 + y2) / 2.0
```

All sightings that have no `best_crop_bbox` are simply skipped — they contribute nothing to the heatmap.

**Step 2 — Group by camera**

`build_all_camera_heatmaps()` splits the sighting list by `camera_id`. Each camera gets its own independent heatmap. This is important: the bounding box pixel coordinates only make sense relative to the frame dimensions of the camera that recorded them.

**Step 3 — Render on camera background**

For each camera, `build_person_heatmap()` attempts to load a **reference background frame** from disk:

```
{CAMERA_BG_DIR}/bg_{camera_id}.png
```

If found, the heatmap is overlaid on the real camera view, letting you immediately see *where in the store* the person was standing. If no background exists, a plain dark canvas (`640×480` minimum, scaled to the furthest detected position) is used as a fallback.

**Step 4 — Gaussian density estimation**

Heatmap rendering uses a two-pass approach to avoid the computational cost of full-resolution Gaussian evaluation:

1. A **small downscaled canvas** (`height × heat_scale`, `width × heat_scale`, default `heat_scale = 0.10`) receives a circular stamp (`radius = round(σ × 1.1)`) at each center point.
2. A Gaussian blur is applied to the small canvas: `cv2.GaussianBlur(kernel=(k, k), sigmaX=σ)`.
3. The result is **upscaled** back to full camera resolution with `cv2.INTER_LINEAR` interpolation.
4. The heatmap is normalised to `[0, 255]` and colourised with `cv2.COLORMAP_JET` (blue = sparse, red = dense).
5. The final overlay blends the background and heatmap: `cv2.addWeighted(bg, 1−α, heat_color, α, 0)`.

**Step 5 — Last-position marker**

The most recent sighting's bounding box is drawn on top of the heatmap:
- A **red filled circle** at the center point.
- A **cyan rectangle** tracing the full bounding box.

This gives a clear "last known position" anchor on top of the density map.

#### Configuration Parameters

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `heatmap_sigma` | **5** | Gaussian spread radius in the downscaled space. Higher = smoother, larger blobs. |
| `heatmap_alpha` | **0.6** | Blend ratio of the heatmap overlay. `0.0` = background only; `1.0` = heatmap only. |

Both are exposed as sliders in the Person Search tab of the dashboard.

#### Output

The function returns a list of `(camera_id, rgb_ndarray, info_str)` tuples — one entry per camera that had at least one sighting with a valid bbox. The Gradio UI renders these as a captioned gallery (`gr.Gallery`), so each camera's heatmap is shown as a separate panel with the camera ID as its label.

```
Example info string:
"cam_01 — 47 position(s) (camera `cam_01` — 1920×1080)"
```

If no sightings contain bounding box data (e.g. the person was found only via face search against very old records), the gallery remains empty and an explanatory message is shown instead.

#### Dependency: Bounding Box Storage

The heatmap feature relies on the `best_crop_bbox` field being populated in every sighting. This field is set during video processing by `SightingTracker.update_track()` whenever the current detection offers a higher-quality crop than the previous best. Since bounding boxes are stored in MongoDB as part of the sighting subdocument, they persist across sessions — a heatmap generated from a search months after original processing will still show the correct real-world positions.

---

## 7. Module 6: Interactive Dashboard

A Gradio web UI with 3 tabs:

| Tab | Features |
|-----|----------|
| **🎬 Video Processing** | Upload video or enter dataset path, toggle face camera, save/discard results, reset DB, adjust ALL thresholds via sliders, progress bar |
| **🔍 Person Search** | Upload image or paste URL, choose Body/Face/Both, toggle "pre-cropped face", rebuild prototypes button, shows match + full person details + **movement heatmap per camera** |
| **📋 Database Browser** | Two sub-tabs: **Browse All** (bulk card view, faces first then body sightings) and **Person Detail** (dropdown, visit history table, body/face galleries) |

Runs with `demo.launch(share=True)` for a public URL on Kaggle.

---

## 8. Evaluation Results

### 8.1 Body Search (Brute Force Test)

Tested against 1,317 saved body crops:

| Metric | Value |
|--------|-------|
| ✅ **Accuracy** | **85.35%** (1,124 / 1,317) |
| ❌ Wrong ID | 54 crops |
| ⚠️ Ambiguous (rejected) | 74 crops |
| ⚠️ Unknown (rejected) | 9 crops |
| **Avg Speed** | 183.8 ms/query |
| **Avg Similarity** | 0.925 |
| **Min Similarity** | 0.753 |

> The 74 "Ambiguous" rejections are the safety net working correctly — the system refused to guess when unsure, preventing false positives.

### 8.2 Face Search

Direct face matching via Pinecone achieves near-perfect accuracy when faces are properly detected. The 0.942 cosine similarity score on test queries confirms the discriminative power of ArcFace embeddings.

---

## 9. Known Limitations & Design Decisions

### 9.1 Data Consistency After Reset
`RESET_DATABASE = True` clears MongoDB and local files but **does not touch Pinecone or ImageKit**. This means face embeddings in Pinecone may reference GIDs that no longer exist in MongoDB. **Manual action required** to fully reset: delete Pinecone index entries and ImageKit folder separately.

### 9.2 RAM-First Processing
All sightings are accumulated in memory during video processing and uploaded in bulk at the end. This is optimized for Kaggle/Colab where persistent storage is limited, but means a kernel crash mid-processing loses all data.

### 9.3 Single-Frame Identity Lock
The `claimed_gids_this_frame` set prevents the same GID from being assigned to two tracks in one frame. However, if BotSort fragments one person into two tracks temporarily, one of them gets sent to the UDM — this is the intended safety behavior.

### 9.4 Face Camera Toggle
`IS_FACE_CAMERA = False` completely disables all face logic (no SCRFD, no ArcFace, no Pinecone face queries). This is designed for cameras where faces are never visible (overhead, far-away), saving GPU time.

### 9.5 Embedding Dimension Mismatch
Body embeddings are 1280-D, face embeddings are 512-D. They live in completely separate Pinecone indices and are **never compared to each other**. The only link between a body and a face is the shared `global_id` in metadata.

### 9.6 Heatmap Accuracy Depends on BBox Storage
The movement heatmap uses `best_crop_bbox` from the sighting record in MongoDB. If an older sighting was recorded before the bounding box field was added to the schema, that sighting will be silently skipped and will not appear on the heatmap. Heatmaps from mixed old/new data may therefore under-represent early appearances of a person.

### 9.7 Camera Background Registration
For the heatmap overlay to be meaningful, a background image must exist at `{CAMERA_BG_DIR}/bg_{camera_id}.png`. This file must match the exact resolution of the recorded footage. If the background is missing or mis-sized, the heatmap falls back to a plain dark canvas and positional accuracy is lost. Best practice: capture a background frame from each camera before deployment and save it with the correct naming convention.
