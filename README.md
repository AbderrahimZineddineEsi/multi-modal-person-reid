# Multi-Modal Person Re-ID & Face Association Pipeline

Real-time person **re-identification (Re-ID)** and **face association** for retail surveillance analytics.

This repository contains the **Re-ID part** of a larger Intelligent Retail Surveillance graduation project
(automated shoplifting detection and a conversational VLM orchestrator are separate team contributions — see
`team_materials/`, which is intentionally excluded from git).

**Author:** Abderrahim Zineddine
**Environment:** Kaggle (2× NVIDIA Tesla T4, 16 GB VRAM each) · Python 3.12 · PyTorch
**Evaluation:** 85.35% body-search accuracy on 1,317 crops · ~184 ms average query latency

---

## What the system does

Given supermarket camera footage, the pipeline:

1. **Detects** every person in every frame (RT-DETR)
2. **Tracks** them within a camera view (BotSort → temporary `track_id`)
3. **Quality-gates** the crop using pose estimation (YOLOv11-Pose), rejecting partial bodies
4. **Embeds** body appearance (CLIP-ReID, 1280-D) and face identity (InsightFace ArcFace, 512-D)
5. **Resolves** a persistent global ID (`gid`) per person, using face recognition as ground truth
6. **Records** each visit: entry/exit time, duration, camera, and the best body & face crop
7. **Enables search**: give an image, find the matching identity and its movement history

### Two modalities working together

| Modality | Strength | Weakness | Role |
|----------|----------|----------|------|
| **Body Re-ID** | Works at any distance/angle, no face needed | Fails with similar clothing / uniforms | Primary identification |
| **Face Recognition** | Extremely accurate | Needs a close, front-facing, well-lit face | Ground truth & correction |

**Key idea — retroactive identity healing:** when a face is recognized, the system can *merge* a
previously (wrongly) assigned body-based ID into the true face identity and repair all of its history.

```
Video Frame
   ├─► RT-DETR (detection) ─► BotSort (tracking) ─► track_id
   │        └─► YOLOv11-Pose (quality gate) ─► reject bad crops
   ├─► CLIP-ReID  ─► 1280-D body embedding ─┐
   └─► InsightFace ─► 512-D face embedding   │
                                             ▼
                          Identity Resolution (face-as-ground-truth,
                          self-adaptive gallery, DBSCAN unknown minting)
                                             ▼
                 Pinecone (vectors) + MongoDB Atlas (metadata) + ImageKit (CDN)
                                             ▼
                          Search engine + Gradio dashboard
```

---

## Repository layout

```
notebooks/      The three Re-ID modules (run in order on Kaggle)
docs/           Full technical report + design discussion notes
reports/        Exported PDF reports
presentations/  Slide decks for the Re-ID part
assets/         Diagrams, demo images, per-camera background frames
archive/        Superseded notebook versions and dev scripts (kept for reference)
team_materials/ Teammates' parts (shoplifting detector, VLM orchestrator) — git-ignored
```

### Pipeline modules

| Module | Notebook | Role |
|--------|----------|------|
| **4** | `notebooks/module4_face_association.ipynb` | Video processing: detect, track, embed, resolve identities, retroactive merge, sync to cloud |
| **5** | `notebooks/module5_search_engine.ipynb` | α-FPS body prototypes + direct face search, query by body/face crop |
| **6** | `notebooks/module6_dashboard.ipynb` | Gradio dashboard: video processing, search, movement heatmaps, database browser |

**Run order:** Module 4 → Module 5 → Module 6. Module 4 produces `identity_buffer.pkl`, the
Pinecone indices and the MongoDB documents that Modules 5 & 6 consume.

---

## Models & infrastructure

| Component | Model / Service | Purpose |
|-----------|-----------------|---------|
| Detection | **RT-DETR-L** | NMS-free person detection |
| Tracking | **BotSort** | Multi-object tracking (temporary IDs) |
| Quality gate | **YOLOv11s-Pose** | Skeleton visibility filter |
| Body Re-ID | **CLIP-ReID** (Market-1501) | 1280-D appearance embedding |
| Face detect | **InsightFace SCRFD** | Face localization |
| Face recognition | **InsightFace ArcFace** (`buffalo_l`) | 512-D identity embedding |
| Vector DB | **Pinecone** | `supermarket-reid` (body) + `supermarket-faces` (face) |
| Metadata | **MongoDB Atlas** | Persons, sightings, faces, timestamps |
| CDN | **ImageKit** | Hosts best body/face crops |

Cloud credentials are read from **Kaggle Secrets** at runtime (`UserSecretsClient`), with safe
placeholder fallbacks — no keys are stored in this repository.

---

## Results

Body search stress test (1,317 saved crops):

| Metric | Value |
|--------|-------|
| Accuracy | **85.35%** (1,124 / 1,317) |
| Wrong ID | 54 crops |
| Ambiguous (safely rejected) | 74 crops |
| Unknown (safely rejected) | 9 crops |
| Avg. query latency | 183.8 ms |
| Avg. similarity | 0.925 |

---

## Running

These notebooks were developed and run on **Kaggle** (GPU T4 ×2). To reproduce:

1. Import `notebooks/module4_face_association.ipynb`, attach a supermarket video dataset and
   (optionally) a previously saved `identity_buffer.pkl`.
2. Add Kaggle Secrets: `PINECONE_API_KEY`, `MONGO_URI`, `IMAGEKIT_URL_ENDPOINT`,
   `IMAGEKIT_PUBLIC_KEY`, `IMAGEKIT_PRIVATE_KEY`.
3. Run Module 4, then Module 5 (search), then Module 6 (dashboard).

For a local run, replace the `/kaggle/input` and `/kaggle/working` paths in the configuration cells.

---

## Documentation

- `docs/technical_report.md` — complete architecture, thresholds, evaluation and design decisions
- `docs/design_discussion.md` — design review notes (correctness, edge cases, fixes)
- `reports/ReID_Technical_Report.pdf` / `reports/ReID_Dashboard.pdf` — rendered reports
- `archive/README.md` — what was superseded and why

## License

Coursework / graduation project. Please contact the author before reuse.
