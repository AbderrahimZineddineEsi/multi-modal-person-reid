# Research: models, licenses, datasets for the Re-ID rewrite

Date: 2026-11-23. Facts below were checked on the web, on PyPI/GitHub/HF APIs, or by reading the boxmot 26.0.0 wheel.
Anything marked **(unverified)** came from memory or a secondary source. Latency numbers are vendor T4 TensorRT FP16 figures and are not comparable across vendors.
Target: Colab T4 (15 GB) for full runs, GTX 1050 (3 GB) for smoke tests. Public portfolio repo, so licenses matter.

## 0. Key findings (read first)

1. **License picture.** v1 already depends on AGPL-3.0 code: Ultralytics 8.4.218 (AGPL-3.0) and boxmot 26.0.0 (AGPL-3.0+). A public repo that uses them should itself be AGPL-3.0 (or at least state this). The only clean permissive stack is RF-DETR (Apache-2.0) + own ReID + own tracker glue.
2. **InsightFace weights (buffalo_l, antelopev2) are non-commercial research only.** The code is MIT, the models are not. See A5.
3. **boxmot 26.0.0 (2026-10-27) no longer ships CLIP-ReID.** Its weight catalog (`boxmot/reid/core/catalog.py`) contains only ResNet50, MLFN, HACNN, MobileNetV2, OSNet and LMBN files. There is no `clip_market1501.pt` and no clip backbone. The v1 notebooks install boxmot unpinned, so v1 now breaks on a fresh install. The BotSort API also changed (`reid=ReIDConfig(...)`, no `reid_weights=`).
4. **Best design for tracking.** Run our own ReID model once per crop and hand the embeddings to the tracker through `Detections.with_embeddings(...)`. The tracker then skips its internal ReID (or turn ReID off with `use_embeddings=False`). This removes the double embedding cost, and CLIP-ReID or SOLIDER can be loaded from their own repos.
5. **faiss-cpu is not needed below 100k vectors.** NumPy brute force is exact and fast enough (A6).
6. **Dev dataset.** NVIDIA/AI City 2024 MTMC (HF `nvidia/PhysicalAI-SmartSpaces`, CC-BY-4.0, ungated, downloadable per scene or per camera). It is synthetic. MMPTrack is the better real-data option, but it needs a signed agreement by email (Part B).

## Part A: Models

### A1. Person detector (v1: Ultralytics `rtdetr-l.pt`, ~53 AP, 32M params, AGPL via ultralytics)

| Model | COCO AP50:95 | Params | Speed (T4 TRT FP16, bs1) | Package / weights | License |
|---|---|---|---|---|---|
| RT-DETR-L (v1) | ~53.0 (unverified, from memory) | ~32M | ~6-7 ms (unverified) | `ultralytics`, auto-download | AGPL-3.0 (ultralytics). Original lyuwenyu repo is Apache-2.0 |
| RT-DETRv2 R18 / R34 / R50 / R101 | 47.9-48.1 / 49.9 / 51.9 (M*) or 53.4 (L) / 54.3 | 20 / 31 / 36-42 / 76M | n/a | HF `transformers`: `PekingU/rtdetr_v2_r18vd` (Apache-2.0); repo lyuwenyu/RT-DETR (Apache-2.0) | Apache-2.0 |
| D-FINE S/M/L/X | L 54.0, X 55.8 (paper) | L 31M, X 62M | n/a | HF `transformers`, e.g. `ustc-community/dfine-small-coco` (Apache-2.0); repo Peterande/D-FINE (Apache-2.0) | Apache-2.0 |
| DEIMv2 (DINOv3 backbone) S/M/L/X | 50.9 / 53.0 / 56.0 / 57.8 | 9.7 / 18.1 / 32.2 / 50.3M | 5.8 / 8.8 / 10.5 / 13.8 ms (hardware not stated) | No pip package. HF `Intellindust/DEIMv2_*` or Google Drive. Needs the repo (PyTorch 2.4/2.5) | **Non-commercial only** ("DEIMv2 License", I read LICENSE.md). Fine for research. Not permissive |
| RF-DETR (`rfdetr` 1.12.1) Nano / S / M / L | 48.4 / 53.0 / 54.7 / 56.5 | 30.5 / 32.1 / 33.7 / 33.9M | 2.3 / 3.5 / 4.4 / 6.8 ms | `pip install rfdetr`, `RFDETRSmall()` auto-downloads COCO weights, Python >=3.10 | **Apache-2.0** for N/S/M/L. XL/2XL (58.6 / 60.1 AP) and Atto/Femto/Pico need `rfdetr[plus]`, license PML 1.0 (not Apache) |
| YOLO26 (ultralytics, released Jan 2026, stable) n/s/m/l/x | 40.9 / 48.6 / 53.1 / 55.0 / 57.5 (e2e head: -0.4 to -0.8) | 2.4 / 9.5 / 20.4 / 24.8 / 55.7M | 1.7 / 2.5 / 4.7 / 6.2 / 11.8 ms | `ultralytics`, `YOLO("yolo26m.pt")`. NMS-free head via `nms=False` | AGPL-3.0 (or paid Enterprise) |

Sources: ultralytics docs (YOLO26), RF-DETR README, DEIMv2 README and LICENSE.md, HF model cards. YOLO27 appears in the docs only as an unreleased preview. YOLO26 exists and is released.

Notes:
- RF-DETR Small matches RT-DETR-L on AP (53.0) at about half the vendor latency, with a permissive license.
- COCO class ids differ: Ultralytics uses 0 for person. RF-DETR uses the 1-indexed COCO ids from `COCO_CLASSES` **(verify before filtering)**.
- Smoke test on 3 GB: RF-DETR Nano or YOLO26n fit easily.

**Recommendation**
- Same size, better: **RF-DETR Small** (53.0 AP, Apache-2.0, `pip install rfdetr`). If the repo is AGPL anyway, YOLO26m (53.1) is equivalent and keeps the Ultralytics API.
- Bigger, better: **RF-DETR Large** (56.5 AP, 6.8 ms, 33.9M, still Apache-2.0). It is cheaper than the XL/2XL models, whose license is not Apache.
- Smoke: RF-DETR Nano (48.4).

### A2. Pose / quality gate (v1: `yolo11s-pose.pt` (m6) or `yolo26s-pose.pt` (m4), second model after the detector)

| Model | COCO-kpt AP50:95 | Params | T4 TRT | Load | License |
|---|---|---|---|---|---|
| YOLO11s-pose (v1) | not on the docs page I fetched | ~9.9M (unverified) | n/a | ultralytics | AGPL-3.0 |
| YOLO26n/s/m/l/x-pose (e2e) | 57.2 / 63.0 / 68.8 / 70.4 / 71.6 | 2.9 / 10.4 / 21.5 / 25.9 / 57.6M | 1.8 / 2.7 / 5.0 / 6.5 / 12.2 ms | `YOLO("yolo26s-pose.pt")` | AGPL-3.0 |
| RF-DETR Keypoint Preview | 71.8 | 40.7M | 9.7 ms | `from rfdetr import RFDETRKeypointPreview` | Apache-2.0, **preview status** |

Single model for detection + keypoints (person-only):
- Pose models are person-only by construction (17 COCO keypoints). Their results also carry `boxes`, so one pass gives boxes, confidences and keypoints.
- **Crowd quality: I found no published person-box recall on CrowdHuman for YOLO26-pose or YOLO11-pose.** The docs give only keypoint AP. Pose heads are known to be weaker than dedicated detectors on heavy occlusion, but I have no number to cite. Treat "acceptable" as unproven.
- Cheap test on our own data: run YOLO26m-pose and RF-DETR-S on the dev videos and compare person recall against GT boxes. Keep the two-model path as a config switch.
- The detector + pose-on-crop path has a side benefit: the pose model sees the cropped person, so the keypoint-visibility gate is more reliable when people overlap.

**Recommendation**
- Same size, better: YOLO26s-pose (63.0 AP) in place of YOLO11s-pose, ideally as the single model if the recall test passes (AGPL).
- Permissive/bigger: RF-DETR detector + `RFDETRKeypointPreview` (71.8 AP) as the single model. It is a preview, so pin the version.
- Smoke: YOLO26n-pose (2.9M).

### A3. Multi-object tracker (v1: boxmot `BotSort` + internal OSNet-x0.25-MSMT17 / CLIP)

boxmot **26.0.0**, released 2026-10-27 (PyPI). Python >=3.10,<3.14. License **AGPL-3.0+**. Earlier releases: 25.0.0 (Sep 2026), 23.0.0 (Aug), 22.0.0 (Jul), so the API churns fast. Pin the version.

README benchmark (HOTA / MOTA / IDF1; detector and weights not stated on the page):

| Tracker | MOT17-ablation | SportsMOT val | Uses ReID by default |
|---|---|---|---|
| occluboost | 71.10 / 78.50 / 85.28 | 83.17 / 97.48 / 89.36 | yes (new) |
| **botsort** | 69.68 / 78.23 / 82.33 | 76.93 / 98.11 / 78.30 | yes |
| boosttrack | 69.25 / 75.91 / 83.20 | 76.32 / 97.08 / 77.82 | yes |
| strongsort | 68.05 / 76.19 / 80.76 | 79.80 / 97.31 / 80.27 | always |
| deepocsort | 67.95 / 75.83 / 80.54 | 79.51 / 97.94 / 79.59 | yes |
| bytetrack | 67.68 / 78.04 / 79.16 | 67.93 / 97.25 / 76.90 | no |
| hybridsort | 67.31 / 74.09 / 78.87 | 81.14 / 98.07 / 81.88 | yes |
| ocsort | 66.44 / 74.55 / 77.90 | 76.34 / 96.60 / 75.64 | no |
| sfsort | 62.65 / 76.87 / 69.18 | 75.73 / 98.39 / 72.99 | no |

Gaps are 1-2 HOTA between the top four, so the ranking is detector- and dataset-dependent. These are MOT17/SportsMOT numbers, not retail.

API in 26.0.0 (from boxmot docs):
- `tracker.update(detections: np.ndarray (N,6) or Detections, frame) -> np.ndarray`. Output columns: x1,y1,x2,y2,id,conf,cls,det_ind.
- **Own embeddings:** `tracker.update(detections.with_embeddings(embs_float32_NxD), frame)`. Embeddings must be row-aligned with the detections. Attached embeddings skip the internal encoder. A plain NumPy input cannot carry embeddings.
- **No internal ReID:** for BotSort, DeepOcSort, HybridSort, BoostTrack and OccluBoost set `use_embeddings=False` in the typed config. StrongSort cannot disable ReID. ByteTrack and OcSort are motion-only. With ReID off, supplied embeddings are ignored.
- Camera-motion compensation still needs the frame image.
- Internal encoder: `reid=ReIDConfig(model="osnet-x0-25-msmt17", device="cpu", ...)`.

**Recommendation**
- Same size, better: keep **BotSort** (best-validated, top on the OBB benchmark) but feed it our own embeddings (finding 4). That removes the second CLIP/OSNet pass and the broken `reid_weights` call.
- Bigger/better: **occluboost** (best HOTA/IDF1 on MOT17 and SportsMOT) or **boosttrack** as a config switch. occluboost is new, so check stability.
- Smoke: **bytetrack** (no ReID, no model).
- Permissive alternative: Roboflow's `trackers` package exists, but I did not check its license or API.

### A4. Body ReID (v1: CLIP-ReID ViT-B/16 on Market-1501, 1280-D)

| Model | Metric (mAP / R1) | Weights and loading | License |
|---|---|---|---|
| CLIP-ReID ViT-B/16, **Market** (v1) | n/a here | Was `clip_market1501.pt` in boxmot. **Gone from boxmot 26.0.0** | MIT (repo) |
| **CLIP-ReID ViT-B/16, MSMT17** | 3rd-party table: 75.8 / 89.7. Repo README: SIE-OLP with re-ranking 86.7 / 91.1 (unverified for base model) | Google Drive links in github.com/Syliz517/CLIP-ReID (MSMT17, Market, Duke; ViT-CLIP-ReID, -SIE-OLP, CNN). Not pip. The `clip_msmt17.pt` name does not exist in boxmot | MIT |
| SOLIDER Swin-T / S / B | MSMT17 67.4/85.9, 76.9/90.8, 77.1/90.7. Market 91.6/96.1, 93.3/96.6, 93.9/96.9 (no re-ranking) | github.com/tinyvision/SOLIDER-REID, Google Drive | MIT |
| KPR (keypoint-promptable ReID, ECCV 2024) | Occluded-Duke 67.1/79.8 (ImageNet Swin) and 75.1/84.3 (SOLIDER). Market 89.6/95.9 and 93.2/96.6. Occluded-ReID cross-domain (Market-trained) 79.1/85.4 and 82.6/84.8. **MSMT17 not evaluated** | github.com/VlSomers/keypoint_promptable_reidentification, weights on Google Drive, source install (conda, Python 3.10, torch 1.13). Needs keypoints (PifPaf labels in the paper) | Hippocratic License HL3 (ethical-source, not OSI-permissive). Check before use |
| OSNet x0.25/x1.0 MSMT17 (boxmot) | n/a here | boxmot 26.0.0 catalog: `osnet_x0_25_msmt17.pt`, `osnet_x1_0_msmt17.pt`, `lmbn_n_*`, etc. | AGPL (boxmot), OSNet weights from torchreid |
| PersonViT (2024) | MSMT17 fine-tuned weights released (numbers and license **unverified**) | repo | unverified |

Notes:
- Everything above was trained on person ReID datasets, so none is a true zero-shot model. A secondary review reports that zero-shot DINOv2 gets only 0.3-4.7% mAP and that Market-trained OSNet collapses on MSMT17. This supports using MSMT17-trained weights (MSMT17 has indoor/outdoor, many cameras and lighting changes) for cross-domain retail. It is my expectation, not a measurement: measure on the dev set.
- I found no verified 2025-2026 ReID model with released weights that beats SOLIDER or CLIP-ReID on MSMT17. DINOv3/SigLIP person ReID searches returned nothing usable.
- KPR is the only occlusion-specific candidate. It adds a keypoint-extraction dependency and an unusual license, so it fits a bigger-model optional path.
- Embedding dim changes with the model (CLIP-ReID 1280-D in v1). Keep the dim in config because Pinecone indices are dim-fixed.

**Recommendation**
- Same size, better: **CLIP-ReID ViT-B/16 MSMT17** (same architecture and 1280-D output in v1; load with the CLIP-ReID repo code, not boxmot). Keep the v1 preprocessing from `v1_review.md` §1.4.
- Bigger, better: **SOLIDER Swin-B** (77.1 mAP MSMT17, MIT), or Swin-S as the cheaper variant. Fits T4. KPR as an optional occlusion experiment.
- Smoke: OSNet x0.25 MSMT17 from boxmot.

### A5. Face detection and recognition (v1: InsightFace `buffalo_l` = SCRFD-10GF + ArcFace R50 WebFace600K)

**License (checked in the InsightFace README and model_zoo README):**
- Code is MIT. "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." Both manual downloads and `insightface` library auto-downloads follow this policy.
- Update dated 2025-11-24 in the README: for open-sourced recognition models (e.g. the buffalo_l package) contact recognition-oss-pack@insightface.ai for licensing.
- `antelopev2` and `buffalo_*` are under the same policy. The PyPI package `insightface` is at 2.1.
- **What it means for the demo repo:** hosting the code is fine. Do not commit or redistribute the weights (download at runtime). Say "non-commercial research use" prominently in the README. A portfolio or academic demo is research-style use. A hosted public app with a business purpose is a risk. I am not giving legal advice. If the repo must be commercially clean, use one of the options below.

| Model | Accuracy | Params / load | License |
|---|---|---|---|
| buffalo_l (v1): SCRFD-10GF + R50 WebFace600K | model_zoo table: 91.25 / 90.29 / 94.70 / 93.16 / 74.96 (columns not checked) | `FaceAnalysis(name="buffalo_l")`, ~326 MB | non-commercial research |
| antelopev2: SCRFD-10GF + R100 Glint360K | R100 Glint360K row in the same table: 90.66 / 89.49 / 94.29 / 93.43 / 72.53 | `FaceAnalysis(name="antelopev2")`, 407 MB | non-commercial research |
| AdaFace R18/R50/R100 (WebFace4M), R100 (WebFace12M) | R100 WebFace12M: IJB-C TAR@FAR=1e-4 97.66, TinyFace R1 72.29, IJB-S surv-to-single R1 71.35. R50 WebFace4M IJB-C 96.98 | Repo mk-minchul/AdaFace, manual download to `pretrained/`, 112x112 **BGR** input, no pip | Code MIT. Weights are trained on WebFace, which has its own non-commercial terms (secondary source, unverified). Treat as non-commercial |
| AuraFace v1 (R100, `fal/AuraFace-v1`) | LFW 99.65, CFP-FP 95.19, AgeDB 96.10, CALFW 94.70, CPLFW 90.93. No IJB numbers | ONNX, `snapshot_download(...)` then `FaceAnalysis(name="auraface")`. The pack bundles an InsightFace detector (SCRFD), so swap the detector before calling it commercially clean | Apache-2.0 (recognition weights) |

- AdaFace is designed for low-quality surveillance faces (TinyFace and IJB-S are its benchmarks). It would suit this project, but it is not drop-in: it needs a separate detector and 5-point alignment, and its inputs are BGR.
- I did not find a newer verified face model with permissive weights and better surveillance accuracy. (YuNet, the OpenCV detector, was mentioned by a third party as permissive. Not checked.)

**Recommendation**
- Same size, better: keep SCRFD detection, switch the recognizer to **antelopev2 R100** (+ about 4% on the zoo table) if non-commercial is acceptable.
- Bigger, better: **AdaFace R100 WebFace12M** for low-quality crops, with SCRFD for detection and alignment. Gate it behind an A/B on the dev set (face is "ground truth" in the pipeline, so errors there are expensive).
- Commercially clean path: AuraFace + a permissive detector. Expect lower accuracy.
- Smoke on 3 GB: `buffalo_s` / `buffalo_sc` (MBF, 16-159 MB).

### A6. Vector search (local backend, <100k vectors)

- Size: 100k x 1280 float32 = 512 MB. 100k x 512 (faces) = 205 MB. Both fit in RAM on Colab and on the local machine.
- Plain NumPy `X @ q` on L2-normalized vectors is exact cosine. One query is ~0.13 GFLOP for 1280-D, so a few ms to tens of ms on CPU (my estimate, not measured here). Batched queries are one GEMM. Use float16 storage for 2x less RAM if needed.
- `faiss-cpu` `IndexFlatIP` is also exact, so it mainly gives multi-threaded speed. Approximate indexes (HNSW, IVF) only pay off beyond several hundred thousand vectors.
- **Wheels:** `faiss-cpu` 1.16.0 on PyPI (MIT, Python >=3.10) ships `cp310-abi3` wheels for Linux x86_64 (`manylinux_2_27/2_28`, 19 MB), Linux aarch64, musllinux, macOS, and per-version Windows wheels (cp310-cp314). The abi3 tag covers Python 3.12 on Colab's Linux x86_64. The abi3 Linux wheel needs glibc >= 2.28 (Colab's Ubuntu image is newer, from memory).
- **Recommendation:** default `NumpyIndex` behind a small `VectorIndex` interface (add, search, delete). Add a `FaissIndex` as an optional backend. Deletion and incremental update are simpler in NumPy.

## Part B: Multi-camera datasets (overlapping indoor/retail views)

| Dataset | Cameras / scenes | Duration, res | GT global IDs | License | Size and access | Subset? |
|---|---|---|---|---|---|---|
| **AI City 2024 MTMC** (HF `nvidia/PhysicalAI-SmartSpaces`, `MTMC_Tracking_2024`) | 953 cams, 90 synthetic scenes (Omniverse), ~10 cams/scene; scenes 071-080 contain a retail space plus a storage room (README) | 212 h total, 1080p, 30 fps, MP4 H.264 | **Yes**: MOT text `camera_id obj_id frame_id xmin ymin w h xworld yworld`, `obj_id` consistent across cameras | CC-BY-4.0 | 216.95 GB total. HF API reports `gated: False`, so no registration. One camera video is ~211 MB | **Yes**. Layout `MTMC_Tracking_2024/{train,val,test}/scene_XXX/camera_YYYY/{video.mp4,calibration.json}` plus per-scene `ground_truth.txt` (70 MB for scene_041, whole scene) |
| AI City 2025 / 2026 MTMC (same repo) | warehouse, 23 / 28 scenes, 504 / 353 cams (2026 includes 2 real test scenes) | 42 h / 28.7 h | JSON per frame, `object_id` (3D boxes) | CC-BY-4.0 | 3.31 TB / 2.94 TB with depth, ~70 / ~50 GB without | per-scene folders with `videos/`, `ground_truth.json`, `calibration.json` |
| **MMPTrack** (Microsoft, WACV 2023) | 23 cams over 5 scenes (retail 6, then 4, 4, 4, 5: lobby, industry, cafe, office), 28 actors; every camera overlaps at least one other, calibrated | 9.6 h total (5 h train, 1.5 h val, rest test), 640x320, 15 fps | **Yes**: per-view boxes (`rgb_<frame>_<cam>.json`) with person ids and top-down footpoints (`topdown_<frame>.csv`: id,x,y). Test GT hidden | "Research purpose only", signed user agreement | **No public link.** Download the Terms docx from iccv2021-mmp.github.io/subpage/dataset.html, sign and email iccv2021mmp@outlook.com ("MMPTracking Challenge data request"). Total size not published. Helper code github.com/iccv2021-mmp/mmp-tracking-helper | Unknown. Request per scene by email |
| EPFL multi-camera pedestrians (CVLab) | Laboratory 4 cams (4 or 6 people), Terrace 4 cams, Passageway 4 cams, Campus 3 cams, Basketball 4 cams | Laboratory ~2.5 min, Terrace ~3.5 min, 25 fps; resolution not on the page | GT only for Laboratory (6 people), Terrace seq 1, Passageway seq 1. Global-ID semantics **unverified** | "Research purposes" (CVLab copyright) | Direct links per video/calibration/GT on the page (epfl.ch/labs/cvlab/data/data-pom-index-php/), sizes not listed, no registration mentioned | Yes, per file |
| WILDTRACK (EPFL) | 7 GoPro cams (4 Hero3, 3 Hero4), **outdoor** plaza, not indoor | 1920x1080, 60 fps, released frames at 10 fps, 400 annotated frames (2 fps) | boxes linked by position id (0-691199) across views; ~20 people/frame | not stated on the page | `Wildtrack_dataset_full.zip` on documents.epfl.ch plus per-camera videos on Google Drive, size not listed, no registration | Per-camera videos yes |
| MTMMC (CVPR 2024) | 16 RGB+thermal cams, campus and factory | 1080p, 23 fps (third-party listing), 3M frames, 3,669 ids | yes | **unverified**; one listing says a signed access agreement is required | project page sites.google.com/view/mtmmc; I could not confirm size or procedure | unknown |
| MEVA | 29 public cameras (AWS) or 38 (paper), indoor+outdoor, overlapping and non-overlapping | public 328 h (516 GB, 4259 clips) + 4.6 h UAV | activity annotations, **not** person-identity MTMC GT | CC-BY-4.0 | `aws s3 ls --no-sign-request s3://mevadata-public-01/` (no account) | Yes, per clip. Not suitable for ID evaluation |

Notes:
- DukeMTMC and similar retracted datasets are not listed.
- I did not verify per-scene camera counts or retail labels for the 2024 train/val/test splits beyond the README line about scenes 071-080. List `test/` on HF to see which scenes exist: `curl https://huggingface.co/api/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2024/test`.
- HF page says "test GT is public" for 2024 and the server submission is disabled, so all splits are usable for dev.

### Recommendation: ONE dev dataset

**AI City 2024 MTMC, one scene, 3-4 cameras.**
- Why: ungated, CC-BY-4.0 (safe to cite and to publish a tiny derived sample if the license text is kept), one scene is 10 cameras, a single camera is ~211 MB, and cross-camera `obj_id` is in plain MOT text.
- Procedure: `huggingface-cli download nvidia/PhysicalAI-SmartSpaces --repo-type dataset --include "MTMC_Tracking_2024/val/scene_041/camera_036[1-4]/*" "MTMC_Tracking_2024/val/scene_041/ground_truth.txt"`. That is about 0.85 GB for 4 cameras plus the 70 MB GT. Only Hugging Face file listings were checked, so the include pattern itself is untested. For the retail scene, look in `test/` for scenes 071-080 (the README says these include a retail room).
- Caveats: synthetic people (ReID domain gap, so use it to validate pipeline logic and global-ID association, not to tune absolute body-ReID thresholds). 1080p 30 fps video, so subsample frames. Scene 071/camera_0649 is corrupt and should be skipped.
- Add MMPTrack retail (6 cams, real people, 640x320) as the real-data check once the agreement is signed. It is research-only and cannot be redistributed, so the repo must not contain it.

### Metrics and tooling
- Official AI City metric: HOTA on 3D world locations (sim zero beyond 2 m). Eval script: `MTMC_Tracking_2024/eval/main.py` in the same HF repo. It is the reference for the ground-truth text format.
- **Single-camera tracking:** `motmetrics` 1.4.0 (MIT) for MOTA/IDF1 and `TrackEval` 1.3.0 (MIT, PyPI `trackeval`) for HOTA/IDF1. Both on PyPI.
- **Cross-camera association (what this project cares about):** build one MOT-style table of (camera, frame, gid, box). Treat each camera-frame as a separate frame key and compute IDF1 with `motmetrics` over the whole set, so the global ID must be consistent across cameras. Report also per-camera IDF1 and the share of GT identities that map to exactly one predicted `gid` (identity purity). This is my suggested protocol, not a standard published one.
- Also report the v1-style crop retrieval metrics (Rank-1 / mAP over saved best crops) to compare with the v1 85.35% body accuracy figure.

## Summary of picks

| Slot | Same size, better | Bigger, better | Smoke (3 GB) | License note |
|---|---|---|---|---|
| Detector | RF-DETR Small (53.0 AP) | RF-DETR Large (56.5) | RF-DETR Nano | Apache-2.0 |
| Pose gate | YOLO26s-pose (63.0), test as single model | RF-DETR Keypoint Preview (71.8, preview) | YOLO26n-pose | AGPL / Apache |
| Tracker | boxmot 26.0.0 BotSort + own embeddings | occluboost / boosttrack | bytetrack | AGPL-3.0+ |
| Body ReID | CLIP-ReID ViT-B/16 MSMT17 | SOLIDER Swin-B (+ KPR experiment) | OSNet x0.25 MSMT17 | MIT (HL3 for KPR) |
| Face | antelopev2 | AdaFace R100 WebFace12M | buffalo_s | non-commercial; AuraFace Apache-2.0 for a clean path |
| Vector index | NumPy | FAISS (optional) | NumPy | MIT |
