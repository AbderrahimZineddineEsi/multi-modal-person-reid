# Model comparison (extends `research.md` Part A)

Date: 2026-11-23. Checked against primary sources: Ultralytics docs (YOLO11, YOLO26, pose, RT-DETR), the READMEs, LICENSE files and GitHub/HF/PyPI APIs of RF-DETR, D-FINE, RT-DETRv2, DEIMv2, mmpose (RTMPose, RTMO), ViTPose, boxmot, ByteTrack, OC-SORT, BoT-SORT, Deep OC-SORT, StrongSORT, BoostTrack, Hybrid-SORT, CLIP-ReID, SOLIDER, TransReID, PersonViT, torchreid, InsightFace (SCRFD, model_zoo, arcface_torch), AdaFace, TopoFR, YuNet, and the papers (CLIP-ReID, PromptSG, PersonViT, ViTPose, RTMO, StrongSORT, AdaFace, TopoFR, CLIP-FGDI, Asperti 2025, "Person Re-ID in 2025").
Anything not read in a primary source is marked **(unverified)**. Numbers I computed are marked **(est.)**. `Δ` = absolute points / relative % against the v1 row of the same table.

## Read first

- **No vendor publishes inference VRAM for any candidate.** Every VRAM cell is an estimate (formula below), not a measurement. Measure with `torch.cuda.max_memory_allocated()` before locking a stack.
- **Corrections to Part A.** (1) The InsightFace "91.25 / 90.29 / 94.70 / 93.16 / 74.96" row is MR-ALL and ethnicity splits, not WIDER FACE or IJB-C. buffalo_l's IJB-C(E4) is 97.25. (2) YOLO26 AP used in Part A was the NMS head. The Ultralytics speeds are for the end-to-end (`nms=False`) head, so the paired AP is 47.8 / 52.5 / 54.4 (s/m/l). (3) RT-DETR-L (HGNetv2-L) is now verified: 53.0 AP, 32M, 110 GFLOPs, 114 FPS (RT-DETR README); Ultralytics docs add 8.0 ms and 52.7 AP under TensorRT 10.11. (4) YOLO11s-pose now has numbers: 58.9 AP, 9.9M, 2.6 ms.
- **VRAM is not the binding constraint on a T4.** Every stack below needs 2-3 GB (est.). Limits are latency, licenses and what weights exist.
- **Cross-domain evidence is thin.** Nobody publishes Market->MSMT17 for the transformer ReID models. Section 4b lists everything I could source and says what is not comparable.

### VRAM formula (est.), used in every table

`VRAM = P*bw/1000 + B*6*E*ba/1e9 + 0.15` GB, shown as `fp16 / fp32`.
P = params in M, bw = 2 or 4 bytes. B = batch. E = elements of the largest feature map (conv nets: stride-2 stem output, e.g. 320x320x64 = 6.6M for YOLO m/l at 640; ViT: tokens x 4d MLP width, e.g. 129x3072 = 0.4M). 6 = tensors alive at once in inference. ba = 2 or 4 bytes. 0.15 = cuDNN/cuBLAS workspace and allocator slack.
Detector E: 3.3M (<12M params), 6.6M (12-30M), 9.8M (>30M), scaled by (res/640)^2. Batch: detectors/pose-on-frame bs1, crop models (ReID, pose-on-crop, face rec) bs32 for ReID, bs8 for others. Excludes the CUDA context (0.4-0.6 GB per process). Expect `nvidia-smi` to read 1.5-2x higher in eager PyTorch.
Speed (est.) for crop models: `ms/crop = GFLOPs/12` (T4 FP16 effective ~12 TFLOPs, calibrated on Ultralytics m/l rows: 68 GFLOPs / 4.7 ms = 14.5 TFLOPs) plus a 1.5 ms launch floor per batch.

## 1. Person detector (COCO val2017 AP50:95, 640 px unless noted)

Speed = T4 TensorRT FP16 bs1, as each vendor states it; vendors measure differently, so compare latency only within a vendor. AP comes from the same benchmark, but training recipes differ (see notes).

| Model | Release | Params M | VRAM GB fp16/fp32 (est.) | T4 TRT ms | AP50:95 | Δ vs v1 | License | Load |
|---|---|---|---|---|---|---|---|---|
| **RT-DETR-L (v1)** | 2023-04 | 32 | 0.33 / 0.51 | 8.8 (114 FPS paper); 8.0 (Ultralytics, TRT 10.11) | 53.0 | - | AGPL via ultralytics; upstream Apache-2.0 | `RTDETR("rtdetr-l.pt")` |
| YOLO11s | 2024-09 | 9.4 | 0.21 / 0.27 | 2.5 | 47.0 | -6.0 / -11.3% | AGPL-3.0 | `YOLO("yolo11s.pt")` |
| YOLO11m | 2024-09 | 20.1 | 0.27 / 0.39 | 4.7 | 51.5 | -1.5 / -2.8% | AGPL-3.0 | `yolo11m.pt` |
| YOLO11l | 2024-09 | 25.3 | 0.28 / 0.41 | 6.2 | 53.4 | +0.4 / +0.8% | AGPL-3.0 | `yolo11l.pt` |
| YOLO26s | 2026-01 | 9.5 | 0.21 / 0.27 | 2.5 | 47.8 e2e (48.6 NMS) | -5.2 / -9.8% | AGPL-3.0 | `yolo26s.pt` |
| YOLO26m | 2026-01 | 20.4 | 0.27 / 0.39 | 4.7 | 52.5 e2e (53.1) | -0.5 / -0.9% | AGPL-3.0 | `yolo26m.pt` |
| YOLO26l | 2026-01 | 24.8 | 0.28 / 0.41 | 6.2 | 54.4 e2e (55.0) | +1.4 / +2.6% | AGPL-3.0 | `yolo26l.pt` |
| RF-DETR Nano | 2025-03 (pip) | 30.5 | 0.25 / 0.36 | 2.3 @384 | 48.4 | -4.6 / -8.7% | Apache-2.0 | `pip install rfdetr==1.12.1`; `RFDETRNano()` |
| RF-DETR Small | 2025-03 (pip) | 32.1 | 0.29 / 0.43 | 3.5 @512 | 53.0 | 0.0 / 0.0% | Apache-2.0 | `RFDETRSmall()` |
| RF-DETR Medium | 2025-03 (pip) | 33.7 | 0.31 / 0.48 | 4.4 @576 | 54.7 | +1.7 / +3.2% | Apache-2.0 | `RFDETRMedium()` |
| RF-DETR Large | 2025-03 (pip) | 33.9 | 0.36 / 0.57 | 6.8 @704 | 56.5 | +3.5 / +6.6% | Apache-2.0 | `RFDETRLarge()` |
| D-FINE-S | 2024-10 | 10 | 0.21 / 0.27 | 3.49 | 48.5 (O365: 50.7) | -4.5 / -8.5% | Apache-2.0 | HF `ustc-community/dfine-small-coco` or Peterande/D-FINE |
| D-FINE-M | 2024-10 | 19 | 0.27 / 0.38 | 5.62 | 52.3 (O365: 55.1) | -0.7 / -1.3% | Apache-2.0 | `...dfine-medium-coco` (id unverified) |
| D-FINE-L | 2024-10 | 31 | 0.33 / 0.51 | 8.07 | 54.0 (O365: 57.3) | +1.0 / +1.9% | Apache-2.0 | `...dfine-large-coco` (id unverified) |
| RT-DETRv2-S (R18) | 2024-07 | 20 | 0.27 / 0.39 | 4.6 (217 FPS) | 48.1 | -4.9 / -9.2% | Apache-2.0 | HF `PekingU/rtdetr_v2_r18vd`; lyuwenyu/RT-DETR |
| RT-DETRv2-M (R50-m) | 2024-07 | 36 | 0.34 / 0.53 | 6.9 (145 FPS) | 51.9 | -1.1 / -2.1% | Apache-2.0 | repo ckpt `rtdetrv2_r50vd_m_7x_coco_ema.pth` |
| RT-DETRv2-L (R50) | 2024-07 | 42 | 0.35 / 0.55 | 9.3 (108 FPS) | 53.4 | +0.4 / +0.8% | Apache-2.0 | HF `PekingU/rtdetr_v2_r50vd` (id unverified) |
| DEIMv2-S | 2025-09 | 9.7 | 0.21 / 0.27 | 5.78 (hw not stated) | 50.9 | -2.1 / -4.0% | **DEIMv2 License: non-commercial only** | HF `Intellindust/DEIMv2_DINOv3_S_COCO`; needs repo code |
| DEIMv2-M | 2025-09 | 18.1 | 0.27 / 0.38 | 8.80 (hw not stated) | 53.0 | 0.0 / 0.0% | non-commercial | repo `.pth` (Drive) |
| DEIMv2-L | 2025-09 | 32.2 | 0.33 / 0.51 | 10.47 (hw not stated) | 56.0 | +3.0 / +5.7% | non-commercial | repo `.pth` (Drive) |

Notes
- **Person-class AP / CrowdHuman AP, MR-2: not published for any candidate.** I read the Ultralytics, RF-DETR, D-FINE, RT-DETRv2 and DEIMv2 pages; none reports a person-only or CrowdHuman number, and a web search found none (DEYO and a DETR crowd paper report CrowdHuman for other models only). Measure on the dev videos with pycocotools, class `person` only, before trusting COCO AP for retail crowding.
- RF-DETR numbers are Roboflow's in-house T4 TRT runs at each variant's native resolution (384-704). DEIMv2 does not state hardware or input size. D-FINE: TRT 10.4, bs1, FP16; input size not stated. RT-DETRv2: FPS converted to ms (1000/FPS).
- D-FINE "O365" AP is Objects365 pretraining; the page warns those checkpoints may carry Objects365 terms. RF-DETR XL/2XL and Atto/Femto/Pico are PML 1.0 (`rfdetr[plus]`), excluded.
- DEIMv2 LICENSE.md (read): rights granted "solely for Non-Commercial Purposes"; no Commercial Use. Fine for a research demo, not for a product.
- RF-DETR class ids are 1-indexed COCO; Ultralytics uses 0 for person. Verify before filtering.
- **Timeline:** 2023-04 RT-DETR (53.0 AP, 32M, 8.8 ms) -> 2024-07 RT-DETRv2 -> 2024-09 YOLO11 -> 2024-10 D-FINE -> 2025-03 RF-DETR (DINOv2 backbone, NAS) -> 2025-09 DEIMv2 (DINOv3) -> 2026-01 YOLO26 (NMS-free). The same 53 AP now costs 3.5 ms (RF-DETR-S) or 4.7 ms (YOLO26m, 20M) instead of 8.8 ms.

## 2. Pose / quality gate (COCO-keypoints val AP50:95, OKS)

"Det?" = also detects persons (single-model option). Box AP of the pose models is **not reported** by Ultralytics, RF-DETR or the RTMO README.

| Model | Release | Params M | VRAM GB fp16/fp32 (est.) | Speed | AP(pose) | Det? | Δ vs v1 | License | Load |
|---|---|---|---|---|---|---|---|---|---|
| **YOLO11s-pose (v1)** | 2024-09 | 9.9 | 0.21 / 0.27 | 2.6 ms T4 TRT | 58.9 | yes, box AP n/r | - | AGPL-3.0 | `YOLO("yolo11s-pose.pt")` |
| YOLO11m-pose | 2024-09 | 20.9 | 0.27 / 0.39 | 4.9 ms T4 TRT | 64.9 | yes | +6.0 / +10.2% | AGPL-3.0 | `yolo11m-pose.pt` |
| YOLO26s-pose | 2026-01 | 10.4 | 0.21 / 0.27 | 2.7 ms T4 TRT | 63.0 | yes | +4.1 / +7.0% | AGPL-3.0 | `yolo26s-pose.pt` |
| YOLO26m-pose | 2026-01 | 21.5 | 0.27 / 0.39 | 5.0 ms T4 TRT | 68.8 | yes | +9.9 / +16.8% | AGPL-3.0 | `yolo26m-pose.pt` |
| YOLO26l-pose | 2026-01 | 25.9 | 0.28 / 0.41 | 6.5 ms T4 TRT | 70.4 | yes | +11.5 / +19.5% | AGPL-3.0 | `yolo26l-pose.pt` |
| RF-DETR Keypoint Preview | n/r (after 2025-11, unverified) | 40.7 | 0.33 / 0.50 | 9.7 ms T4 TRT @576 | 71.8 | yes | +12.9 / +21.9% | Apache-2.0, preview | `RFDETRKeypointPreview` (`rfdetr`) |
| RTMO-s (one-stage) | 2023-12 | 9.9 | 0.21 / 0.27 | 8.9 ms V100 ORT | 67.7 | yes (no detector needed) | +8.8 / +14.9% | Apache-2.0 | mmpose / `rtmlib` `RTMO` |
| RTMO-m | 2023-12 | 22.6 | 0.27 / 0.40 | 12.4 ms V100 ORT | 70.9 | yes | +12.0 / +20.4% | Apache-2.0 | same |
| RTMO-l | 2023-12 | 44.8 | 0.36 / 0.56 (est., bs1) | 19.1 ms V100 ORT | 72.4 | yes | +13.5 / +22.9% | Apache-2.0 | same |
| RTMPose-s + RTMDet-nano | 2023-01 | 5.5 + 1.0 | 0.21 / 0.27 (crops bs8) | pose 1.39 ms, 1660Ti TRT FP16 | 68.5 (pipeline) | no, needs detector | +9.6 / +16.3% | Apache-2.0 | mmpose / `rtmlib` |
| RTMPose-m + RTMDet-m | 2023-01 | 13.6 + 24.7 | 0.25 / 0.36 (pose only) | pose 2.29 ms, 1660Ti TRT | 75.7 (pipeline); 73.2 with RTMDet-nano | no | +16.8 / +28.5% | Apache-2.0 | same |
| RTMPose-l + RTMDet-m | 2023-01 | 27.7 + 24.7 | 0.36 / 0.57 (pose only) | pose 3.46 ms, 1660Ti TRT | 76.6 (pipeline) | no | +17.7 / +30.0% | Apache-2.0 | same |
| ViTPose-S (MAE) | 2022-04 | ~24 (unverified) | 0.25 / 0.34 (crops bs8) | n/r | 73.8 | no | +14.9 / +25.3% | Apache-2.0 | ViTPose repo (OneDrive); HF `usyd-community/vitpose-plus-small` (33.1M, plus variant) |
| ViTPose-B (MAE) | 2022-04 | 86 | 0.38 / 0.61 (crops bs8) | 944 fps, A100 bs64, 256x192 | 75.8 | no | +16.9 / +28.7% | Apache-2.0 | HF `usyd-community/vitpose-base-simple` (85.9M) |

Notes
- **Comparability.** YOLO, RTMO and RF-DETR-KP are one-stage, evaluated on all of val2017 with their own boxes. RTMPose uses "pipeline AP" with the named detector (README). ViTPose AP uses externally supplied detector boxes **(detector AP unverified)**. RTMPose was trained on AIC+COCO and RTMO rows listed are COCO-only (body7 variants reach 68.6 / 72.6 / 74.8). Gaps under ~2 AP are not decisive.
- Latency hardware differs: T4 (Ultralytics, RF-DETR), V100 + ONNX Runtime (RTMO), GTX 1660 Ti (RTMPose), A100 (ViTPose). Not comparable across rows.
- Crowd robustness of pose-head detection: no published CrowdHuman or crowd-recall number for YOLO-pose or RF-DETR-KP. RTMO has CrowdPose results (RTMO-l 73.2 AP, body7 83.8) from its README.
- For a keypoint-visibility gate, the 5-10 AP gap between families matters less than recall in crowds. Test YOLO26s/m-pose as single model against RF-DETR + RTMPose-m on dev video.
- **Timeline:** 2022-04 ViTPose -> 2023-01 RTMPose -> 2023-12 RTMO (one-stage) -> 2024-09 YOLO11-pose (58.9-69.5) -> 2026-01 YOLO26-pose (63.0-71.6, NMS-free) -> RF-DETR Keypoint Preview (71.8, Apache).

## 3. Tracker (boxmot 26.0.0, 2026-10-27, AGPL-3.0; `pip install boxmot==26.0.0`)

Only the **boxmot ablation column is same-detector**: MOT17 ablation split, YOLOX detections, LMBN ReID, one eval harness (`comparison.md`, README). The paper columns use different detectors and private detections and are not comparable to each other; read them as ceilings. Order is HOTA / MOTA / IDF1 in the ablation and test columns (README order differs per project; I reordered).

| Tracker | Release | MOT17 ablation (boxmot) | Δ HOTA vs v1 | MOT17 test (paper) | DanceTrack test or MOT20 test (paper) | Published FPS | ReID needed? | License | Load |
|---|---|---|---|---|---|---|---|---|---|
| **BotSort + ReID (v1)** | 2022-06 | 69.68 / 78.23 / 82.33 | - | 65.0 / 80.5 / 80.2 (no ReID 64.6 / 80.6 / 79.5) | MOT20 63.3 / 77.8 / 77.5 | n/r | yes, optional | MIT (repo); AGPL via boxmot | `boxmot.BotSort(reid=ReIDConfig(...))` |
| BotSort, no ReID | 2022-06 | n/r by boxmot | - | 64.6 / 80.6 / 79.5 | MOT20 62.6 / 77.7 / 76.3 | n/r | no (`use_embeddings=False`) | same | same |
| OccluBoost | 2026-10 | **71.10 / 78.50 / 85.28** | +1.42 / +2.0% | n/r (no paper found) | n/r | n/r | yes, motion-only mode exists | AGPL | `boxmot` `occluboost` |
| BoostTrack | 2024-08 (unverified) | 69.25 / 75.91 / 83.20 | -0.43 / -0.6% | 65.4 / 80.5 / 80.2 | MOT20 63.0 / 76.4 / 76.5 | n/r | optional | MIT (repo) | `boosttrack` |
| BoostTrack++ (original repo only) | 2024-08 (unverified) | not in boxmot | - | 66.6 / 80.7 / 82.2 | MOT20 66.4 / 77.7 / 82.0 | n/r | optional | MIT | github vukasin-stanojevic/BoostTrack |
| StrongSORT (++) | 2022-02 | 68.05 / 76.19 / 80.76 | -1.63 / -2.3% | 63.5 / 78.3 / 78.5 (++ 64.4 / 79.6 / 79.5) | DanceTrack++ 55.6 / 91.1 / 55.2; MOT20++ 62.6 / 73.8 / 77.0 | 7.5 / 7.1 (paper) | always | GPL-3.0 (repo); AGPL via boxmot | `strongsort` |
| Deep OC-SORT | 2023-02 | 67.95 / 75.83 / 80.54 | -1.73 / -2.5% | 64.9 / 79.4 / 80.6 | DanceTrack 61.3 / 92.3 / 61.5; MOT20 63.9 / 75.6 / 79.2 | n/r | optional | MIT (repo) | `deepocsort` |
| ByteTrack | 2021-10 | 67.68 / 78.04 / 79.16 | -2.00 / -2.9% | 63.1 / 80.3 / 77.3 | DanceTrack 47.7 / 89.6 / 53.9; MOT20 61.3 / 77.8 / 75.2 | 29.6 / 13.7 (V100, incl. YOLOX-X) | no | MIT (unverified, repo moved) | `bytetrack` |
| Hybrid-SORT (+ReID) | 2023-08 | 67.31 / 74.09 / 78.87 | -2.37 / -3.4% | 63.6 / 79.3 / 78.4 (ReID 64.0 / 79.9 / 78.7) | DanceTrack 62.2 / 91.6 / 63.0 (ReID 65.7 / 91.8 / 67.4) | 27.8 / 15.5 (README, hw n/s) | optional | MIT (repo) | `hybridsort` |
| OC-SORT | 2022-03 | 66.44 / 74.55 / 77.90 | -3.24 / -4.6% | 63.2 / 78.0 / 77.5 | DanceTrack 55.1 / 89.4 / 54.2; MOT20 62.4 / 75.9 / 76.4 | ~29 (paper) | no | MIT (repo) | `ocsort` |

Notes
- **FPS.** No tracker-only T4 number is published. The cached-eval path excludes detector and ReID time. Published FPS include the detector and are not comparable. Motion-only association (Kalman + Hungarian) on tens of boxes is far below the detector cost **(est., unmeasured)**.
- **ReID cost = the ReID model's VRAM.** With our own embeddings (`Detections.with_embeddings`, `use_embeddings=False` otherwise) the tracker adds ~0 GPU memory. Boxmot's internal encoder: OSNet-x1.0 ~0.28 GB (bs16, fp16, est.), x0.25 ~0.2 GB. v1's CLIP-ReID encoder is not in boxmot 26.0.0 (Part A).
- The ranking is dataset-dependent: on SportsMOT val (boxmot README) HOTA is OccluBoost 83.17, HybridSORT 81.14, StrongSORT 79.80, BotSort 76.93, ByteTrack 67.93. Validate on our dev scene.
- Not found: MOT17-test, DanceTrack or MOT20 numbers for OccluBoost.
- **Timeline:** 2021-10 ByteTrack -> 2022 OC-SORT, BoT-SORT, StrongSORT -> 2023 Deep OC-SORT, Hybrid-SORT -> 2024 BoostTrack(++) -> 2026-10 OccluBoost in boxmot 26.0.0 (+1.4 HOTA on the ablation, no published paper).

## 4. Body ReID

### 4a. Same-domain (each model trained and tested on that dataset; no re-ranking unless noted)

Metric: mAP / Rank-1. v1 was trained on Market-1501 only, so it has no same-domain MSMT17 score; for MSMT17 the Δ is taken against CLIP-ReID ViT-B/16 MSMT17 (73.4). ms/crop = (est.) T4 FP16, batch 32, formula above.

| Model | Release | Params M | Emb dim | VRAM GB fp16/fp32 (est., bs32) | ms/crop (est.) | Market mAP / R1 | MSMT17 mAP / R1 | Δ vs v1 (Market mAP; MSMT mAP) | License | Load |
|---|---|---|---|---|---|---|---|---|---|---|
| **CLIP-ReID ViT-B/16, Market (v1)** | 2022-11 | ~86 (unverified) | 1280 (768+512, per code) | 0.48 / 0.80 | 1.9 (23 GFLOPs est.) | 89.6 / 95.5 (SIE-OLP 90.5 / 95.4) | n/a | - | MIT | Syliz517/CLIP-ReID, Google Drive; not boxmot |
| CLIP-ReID ViT-B/16, MSMT17 | 2022-11 | ~86 | 1280 | 0.48 / 0.80 | 1.9 | n/a (cross-domain 4b) | 73.4 / 88.7 (SIE-OLP 75.8 / 89.7) | - ; ref | MIT | same repo, MSMT17 link |
| CLIP-ReID RN50 | 2022-11 | ~38 (unverified) | 3072 (2048+1024, per code) | 0.53 / 0.92 | 0.7 | 89.8 / 95.7 | 63.0 / 84.4 | +0.2 / +0.2%; -10.4 / -14.2% | MIT | same repo |
| OSNet x1.0 | 2019-05 | 2.2 | 512 | 0.35 / 0.54 | <0.2 | 82.6 / 94.2 | 43.8 / 74.9 | -7.0 / -7.8%; -29.6 / -40.3% | MIT (torchreid); AGPL (boxmot) | `osnet_x1_0_msmt17.pt` in boxmot 26.0.0 |
| OSNet-AIN x1.0 | 2019-10 | 2.2 | 512 | 0.35 / 0.54 | <0.2 | n/r (cross only, 4b) | n/r | - | MIT (torchreid) | torchreid model zoo (Drive) |
| TransReID (ViT-B/16, SIE+OLP) | 2021-02 | ~86+ (unverified) | 768 (JPM concat larger, unverified) | 0.48 / 0.80 | ~3 (stride-12 patches, est.) | 88.9 / 95.2 | 67.4 / 85.3 | -0.7 / -0.8%; -6.0 / -8.2% | MIT | damo-cv/TransReID (Drive) |
| SOLIDER Swin-T | 2023-03 | ~28 (unverified) | 768 (unverified) | 0.67 / 1.18 | 0.8 (9 GFLOPs est.) | 91.6 / 96.1 | 67.4 / 85.9 | +2.0 / +2.2%; -6.0 / -8.2% | MIT | tinyvision/SOLIDER-REID (Drive) |
| SOLIDER Swin-S | 2023-03 | ~50 (unverified) | 768 (unverified) | 0.71 / 1.27 | 1.5 | 93.3 / 96.6 | 76.9 / 90.8 | +3.7 / +4.1%; +3.5 / +4.8% | MIT | same |
| SOLIDER Swin-B | 2023-03 | ~88 (unverified) | 1024 (unverified) | 0.94 / 1.73 | 2.5 | 93.9 / 96.9 | 77.1 / 90.7 | +4.3 / +4.8%; +3.7 / +5.0% | MIT | same |
| PromptSG ViT-B/16 | 2024-03 | ~86 (unverified) | unverified | ~0.5 / 0.8 | ~2 | 94.6 / 97.0 | 87.2 / 92.6 | +5.0 / +5.6%; +13.8 / +18.8% | none found | github YzXian16/PromptSG; **no weights found** |
| PersonViT ViT-S/16 | 2024-08 | ~22 (unverified) | 384 (unverified) | 0.31 / 0.47 | ~0.5 | 92.9 / 96.8 | 74.3 / 89.2 | +3.3 / +3.7%; +0.9 / +1.2% | code Apache-2.0; weights "cc" tag | HF `lakeAGI/PersonViTReID` (`msmt.vits...pth`) |
| **PersonViT ViT-B/16** | 2024-08 | ~86 (unverified) | 768 (unverified) | 0.48 / 0.80 | 1.9 | 95.0 / 97.6 | **80.8 / 92.0** | +5.4 / +6.0%; +7.4 / +10.1% | code Apache-2.0; weights "cc" tag; LUPerson data terms unverified | same, `msmt.vitb...pth` |
| KPR (SOLIDER init; needs keypoints) | 2024 (ECCV) | n/r | n/r | n/r | n/r | 93.2 / 96.6 (ImageNet Swin 89.6 / 95.9) | not evaluated | +3.6 / +4.0% | Hippocratic HL3 | VlSomers/keypoint_promptable_reidentification (Part A) |
| LightMBN-n (boxmot catalog) | 2021 | 9.15 | 3584 | 0.48 / 0.80 | ~0.4 (4.87 GFLOPs @384x128) | 91.5 / 96.3 (authors, no re-rank) | n/r | +1.9 / +2.1% | AGPL (boxmot) | boxmot `lmbn_n_*` |

Notes
- **No candidate with a released checkpoint beats PersonViT-B on MSMT17.** PBR-ReID (DINOv3-B, 79.7 / 91.2 MSMT17, WACV 2027) has code but **no trained weights** (DINOv3 gated), so it is not a candidate. CLIP-SCGI (claims 88.2 / 92.9) and PromptSG's large jump (+13.8 mAP over CLIP-ReID on MSMT17) are paper claims I could not reproduce or find weights for **(unverified)**.
- **Combined multi-dataset / LUPerson.** SOLIDER and PersonViT are LUPerson-pretrained (self-supervised) then fine-tuned on one dataset; their released ReID heads are single-dataset, not combined-dataset models. The only multi-source ReID weights I verified are torchreid's OSNet-AIN multi-source models (4b). CLIP-FGDI trains multi-source; code is public, weights **(unverified)**.
- Embedding dims for transformer models other than CLIP-ReID were not read from code. Keep the dim in config (Pinecone indexes are dim-fixed).
- Same-domain scores are comparable only when preprocessing, resolution (256x128 vs 384x128) and re-ranking match. Rows above are no-re-ranking numbers from each paper/README.
- **Timeline:** 2019 OSNet -> 2021 TransReID -> 2022-11 CLIP-ReID (73.4 MSMT17) -> 2023-03 SOLIDER (77.1) -> 2024-03 PromptSG (claims 87.2) -> 2024-08 PersonViT (80.8) -> 2026-01 benchmark paper (arXiv 2601.20598) shows zero-shot CLIP/DINOv2/PE score under 1% mAP and SigLIP2 under ~6% on MSMT17/Market-1501, so supervised ReID heads are still required.

### 4b. Cross-domain (train on one set, test on another). mAP / Rank-1 unless stated

| Model (trained on) -> test | mAP | R1 | Source and caveat |
|---|---|---|---|
| CLIP-ReID ViT-B/16 (Market) -> DukeMTMC | 50.2 | 68.9 | Asperti 2025 (Sensors), official code, own runs |
| same -> CUHK03 / ENTIRe-ID / Airport | 38.6 / 38.9 / 20.0 | 40.4 / 38.4 / 22.4 | same |
| CLIP image-encoder baseline ViT-B/16 (Market) -> Duke | 44.8 | 64.3 | same (prompt learning adds +5.4 mAP) |
| CLIP-ReID RN50 (Market) -> Duke | 21.0 | 36.8 | same |
| CLIP-ReID (MSMT17-trained, 87.5M trainable = ViT-B) -> Market / Duke / CUHK03 | 50.6 / 58.3 / 39.8 | not reported | arXiv 2601.20598, mAP only, checkpoint not identified **(unverified)** |
| OSNet-x1.0 (Market?) -> MSMT17 | 3.4 | n/r | same paper; its caption and table disagree on the training set **(unverified)** |
| CLIP-ReID ViT-B/16* (M+CS+C3 -> MSMT17) | 26.6 | 53.1 | CLIP-FGDI paper, leave-one-out, retrained from open code |
| CLIP-ReID ViT-B/16* (MS+CS+C3 -> Market) | 68.8 | 84.4 | same |
| TransReID* (M+CS+C3 -> MSMT17 / MS+CS+C3 -> Market) | 23.2 / 59.9 | 46.3 / 79.8 | same |
| plain ViT-B* (-> MSMT17 / -> Market) | 20.5 / 59.2 | 42.7 / 78.3 | same |
| CLIP-FGDI (-> MSMT17 / -> Market) | 31.1 / 79.4 | 59.4 / 91.3 | same; code only, weights **(unverified)** |
| OSNet-AIN x1.0 (Market -> Duke / Duke -> Market) | 30.5 / 30.6 | 52.4 / 61.0 | torchreid MODEL_ZOO, 256x128 |
| OSNet-AIN x1.0 (MSMT17 -> Market / -> Duke) | 43.3 / 52.7 | 70.1 / 71.1 | same; plain OSNet-x1.0: 37.5 / 45.3 mAP, 66.6 / 66.0 R1 |
| OSNet-AIN x1.0 multi-source (D+M+C3 -> MSMT17) | 16.2 | 40.2 | same |
| KPR (Market) -> Occluded-ReID | 79.1 | 85.4 | Part A; occlusion test set, not comparable |

Findings
- **Market -> MSMT17 for CLIP-ReID, SOLIDER, PersonViT, PromptSG, TransReID, boxmot models: not published.** Closest evidence is the leave-one-out table (training set differs, and our v1 trained on Market alone).
- On Market -> Duke, CLIP-ReID ViT-B/16 (50.2 mAP) is well above OSNet-AIN (30.5), but resolution, backbone and training recipe differ, so only the direction is informative. **Cross-domain loss is large for every model** (Market 89.6 mAP -> Duke 50.2 for v1, -39 points, -44%).
- Training on MSMT17 (more cameras, indoor and outdoor) is the better bet for new cameras (Part A expectation); the only direct support is MSMT17->Market 50.6 mAP vs Market->Duke 50.2 in different papers, which is not a controlled comparison. Measure on the dev scene.

## 5. Face

### 5a. Detection (WIDER FACE val, AP Easy / Medium / Hard)

Speed from the SCRFD README is at VGA 640x480; GPU rows state no hardware, mobile-family rows are CPU single-thread. Not T4.

| Model | Release | Params M | VRAM GB fp16/fp32 (est., bs1) | Speed | E / M / H | Δ Hard vs v1 | License | Load |
|---|---|---|---|---|---|---|---|---|
| **SCRFD-10G_KPS (buffalo_l, antelopev2) (v1)** | 2021 (unverified) | 4.23 | 0.19 / 0.24 | 5.0 ms | 95.40 / 94.01 / 82.80 | - | non-commercial weights; code MIT | `FaceAnalysis("buffalo_l")` (`insightface` 2.1) |
| SCRFD-2.5G_KPS (buffalo_m) | 2021 | 0.82 | 0.17 / 0.19 | 4.3 ms | 93.80 / 92.02 / 77.13 | -5.67 / -6.8% | same | `buffalo_m` |
| SCRFD-500M_KPS (buffalo_s, buffalo_sc) | 2021 | 0.57 | 0.16 / 0.17 | 3.6 ms (CPU) | 90.97 / 88.44 / 69.49 | -13.31 / -16.1% | same | `buffalo_s` / `buffalo_sc` |
| SCRFD-34G | 2021 | 9.80 | ~0.2 / 0.3 | 11.7 ms | 96.06 / 94.92 / 85.29 | +2.49 / +3.0% | same | onnx from OneDrive link in README (not in a pack) |
| RetinaFace-R50 | 2019 | 29.5 | 0.29 / 0.43 | 21.7 ms | 94.92 / 91.90 / 64.17 | -18.63 / -22.5% | MIT code | InsightFace RetinaFace |
| YuNet 2023mar (OpenCV zoo) | 2023-03 | ~0.08 (unverified) | ~0.15 / 0.16 | CPU, n/r | 88.44 / 86.56 / 75.03 | -7.77 / -9.4% | MIT | `cv2.FaceDetectorYN`, `face_detection_yunet_2023mar.onnx` |

The YuNet README prose gives 83.4 / 82.4 / 70.8 and its table 88.44 / 86.56 / 75.03 (different protocol or version). YuNet outputs 5 landmarks (unverified here); check alignment compatibility before swapping.

### 5b. Recognition (IJB-C TAR@FAR=1e-4; low quality = TinyFace R1 / IJB-S surveillance-to-single R1)

IJB-C numbers come from three harnesses (InsightFace zoo "E4", arcface_torch, AdaFace script); differences under ~0.3 are noise. Params and GFLOPs for R50 / R100 / MBF are arcface_torch values (6.3 / 12.1 / 0.45, MACs-style); params 43.6 / 65.2 / 1.2 are **(unverified)**, consistent with the 166 MB / 261 MB ONNX sizes.

| Model | Release | Params M | VRAM GB fp16/fp32 (est., bs8) | Speed (GFLOPs; est. ms/crop) | IJB-C @1e-4 | TinyFace R1 / IJB-S R1 | Δ IJB-C vs v1 | License | Load |
|---|---|---|---|---|---|---|---|---|---|
| **buffalo_l R50 WebFace600K (v1)** | n/r | ~43.6 | 0.31 / 0.48 | 6.3; 1.0 | 97.25 | not published | - | non-commercial research | `FaceAnalysis("buffalo_l")` |
| buffalo_s MBF WebFace600K | n/r | ~1.2 | 0.23 / 0.31 | 0.45; 0.1 | 95.02 | not published | -2.23 / -2.3% | same | `buffalo_s` |
| antelopev2 R100 Glint360K | n/r | ~65.2 | 0.36 / 0.56 | 12.1; 2.0 | 97.55 (arcface_torch Glint360K R100 row; the ONNX itself is not separately scored) | not published | +0.30 / +0.3% | same | `FaceAnalysis("antelopev2")` |
| AdaFace R50 WebFace4M | 2022-04 | ~43.6 | 0.31 / 0.48 | 6.3; 1.0 | 96.98 | not published | -0.27 / -0.3% | MIT code; WebFace terms (unverified) | repo, Drive; 112x112 **BGR** |
| AdaFace R100 WebFace4M | 2022-04 | ~65.2 | 0.36 / 0.56 | 12.1; 2.0 | 97.39 | 72.02 / 70.42 | +0.14 / +0.1% | same | same |
| **AdaFace R100 WebFace12M** | 2022-04 | ~65.2 | 0.36 / 0.56 | 12.1; 2.0 | **97.66** | **72.29 / 71.35** | +0.41 / +0.4% | same | same |
| TopoFR R50 Glint360K | 2024-10 | ~43.6 | 0.31 / 0.48 | 6.3; 1.0 | 97.27 | not published | +0.02 / +0.0% | code Apache-2.0; Glint360K terms (unverified) | modelscope/facechain `face_module/TopoFR`, Drive |
| TopoFR R100 Glint360K | 2024-10 | ~65.2 | 0.36 / 0.56 | 12.1; 2.0 | 97.60 | not published | +0.35 / +0.4% | same | same |
| TopoFR R200 Glint360K | 2024-10 | n/r | n/r | 23.5; 3.5 | 97.84 | not published | +0.59 / +0.6% | same | same |
| AuraFace v1 R100 | 2024-08 | ~65 (unverified) | ~0.36 / 0.56 | ~12; 2.0 | not published (LFW 99.65, CFP-FP 95.19, CPLFW 90.93 per Part A) | not published | n/a | **Apache-2.0** (commercial-trained data) | HF `fal/AuraFace-v1`; bundled detector is SCRFD |

Notes
- Low quality: AdaFace is the only family with TinyFace / IJB-S numbers. The InsightFace and TopoFR packs have none. AdaFace's validation scripts (`validation_lq`) can score buffalo_l and antelopev2 on our own crops.
- UniFace++: not found (searches return the ICCV 2023 UniFace loss paper only). No verified 2025-2026 model with released weights, permissive license and better surveillance accuracy; TopoFR is the newest with weights.
- AdaFace pairs with SCRFD detection plus 5-point alignment; it is not a drop-in `FaceAnalysis` pack.
- **Timeline:** 2019 RetinaFace -> 2021 SCRFD / ArcFace WebFace600K packs (buffalo_l) -> 2022-04 AdaFace (low-quality margin, TinyFace 72.3) -> 2023-03 YuNet -> 2024-08 AuraFace (Apache) -> 2024-10 TopoFR (IJB-C 97.60 R100). Gains on IJB-C since v1 are +0.1 to +0.6 points; the real differentiator is low-quality robustness (AdaFace) and licensing (AuraFace).

## 6. VRAM budget stacks (end-to-end, 1080p stream)

Assumptions: T4 with CUDA context 0.5 GB and 0.3 GB headroom added once; model VRAM from the tables (est.). Detector/pose via TensorRT/ONNX FP16, ReID and face recognition in PyTorch FP16 batched. 8 people per frame. Pose-on-crops at 320 px for YOLO. Face: 1 crop per frame, upper-body crop through SCRFD then recognition. CPU decode + preprocess ~8 ms per frame not overlapped (conservative). `FPS = 1000 / (GPU ms + 8)`. Two ReID cadences: every frame (worst case) and every 5th frame with per-track caching. Latencies: published T4 TRT where available, else est. Eager PyTorch end to end will be ~0.5x of these numbers.

| Stack | Detector | Pose / gate | Tracker | Body ReID | Face | Σ VRAM (est.) | T4 FPS (ReID every frame / every 5th) |
|---|---|---|---|---|---|---|---|
| **Lite** (3 GB GTX 1050, fp32) | YOLO26n-pose single model (det + 17 kp), 1.8 ms | same model | ByteTrack (no ReID) | OSNet x1.0 MSMT17, 2.8 ms per 8 crops | buffalo_sc (SCRFD-500M + MBF), 2.0 + 1.6 ms | 0.24 + 0.35 + 0.17 + 0.16 = 0.92; + 0.8 = **1.7 GB** (fp32; Pascal fp16 is slow, unverified) | ~62 / ~71 on T4; on a 1050 about 6x slower GPU time (est.) -> ~15 |
| **Balanced** (Colab T4) | RF-DETR Medium, 4.4 ms | YOLO26s-pose on crops, 5.5 ms | BotSort + own embeddings | CLIP-ReID ViT-B/16 MSMT17, 16.8 ms per 8 crops (SOLIDER-S if license/dim change is acceptable) | antelopev2 (SCRFD-10G + R100), 3.0 + 3.5 ms | 0.31 + 0.21 + 0.40 + 0.19 + 0.28 = 1.39; + 0.8 = **2.2 GB** | ~24 / ~36 |
| **Max** (15 GB T4) | RF-DETR Large, 6.8 ms | RTMPose-l on crops via `rtmlib`, 7.0 ms | BotSort or OccluBoost + own embeddings | PersonViT ViT-B/16 MSMT17 (SOLIDER Swin-B if the weights' "cc" tag is a problem), 16.8 ms | SCRFD-10G + AdaFace R100 WebFace12M, 3.0 + 3.5 ms | 0.36 + 0.36 + 0.48 + 0.19 + 0.28 = 1.67; + 0.8 = **2.5 GB** | ~22 / ~31 |

- Planning figure: double the sums (4.4 / 5.0 GB Balanced / Max) to cover eager-PyTorch overhead. Max still leaves ~9 GB: use it for 4-6 simultaneous streams (batch the ReID) or an ensemble such as PersonViT-B + SOLIDER-B embeddings, not for a bigger single model (no larger permissive candidate exists; RF-DETR XL/2XL are PML 1.0).
- Detector input is the 1080p frame resized to 640-704. People shorter than ~40 px after resizing lose recall; running RF-DETR-L at 960 costs roughly 2x detector time (est.).
- Licenses: Lite and Balanced pose models and YOLO26 are AGPL (ultralytics, boxmot); InsightFace, AdaFace and TopoFR weights are research-only. License-clean variant: RF-DETR (Apache) + RTMPose or RF-DETR Keypoint Preview (Apache) + ReID CLIP-ReID or SOLIDER (MIT) + face AuraFace (Apache) with YuNet (MIT) for detection; boxmot is AGPL, so use a separate MIT tracker repo (OC-SORT, BoT-SORT or Deep OC-SORT) or accept AGPL.

## 7. What is comparable

- **Directly comparable:** detector AP50:95 (COCO val2017, single scale; recipes differ). Pose AP50:95 of YOLO, RTMO, RF-DETR-KP (one-stage, same val set). Tracker HOTA/MOTA/IDF1 inside the boxmot ablation column. Same-domain ReID mAP / R1 within a dataset when resolution and re-ranking match. IJB-C within ~0.3. WIDER FACE AP inside the SCRFD README.
- **Not comparable:** latency across vendors (T4 TRT 10.4 vs RF-DETR in-house vs V100 ORT vs 1660 Ti vs A100) and across input sizes (384-704 for RF-DETR, 640 for the rest). RTMPose and ViTPose AP (top-down, detector-dependent, extra training data) against one-stage AP. Tracker paper columns (different detectors, private detections, MOT17 test vs ablation vs DanceTrack). ReID same-domain scores against cross-domain ones, and cross-domain numbers from different training sets (Asperti, CLIP-FGDI, torchreid, 2601.20598). TinyFace / IJB-S exist only for AdaFace. All VRAM and "est." ms are heuristics.

## Unverified list

RT-DETRv2/D-FINE HF ids other than the small ones in Part A; RF-DETR Keypoint Preview release date; BoostTrack release month; ByteTrack license; params of CLIP-ReID, TransReID, SOLIDER, PersonViT, ViTPose-S, R50/R100/MBF/YuNet; embedding dims other than CLIP-ReID; ViTPose detector boxes; PromptSG and CLIP-SCGI claims and weights; CLIP-FGDI weights; license of LUPerson and of PersonViT weights ("cc" tag); license of WebFace, Glint360K and MS1MV2 data for AdaFace and TopoFR weights; Pascal FP16 rate; every est. cell.
