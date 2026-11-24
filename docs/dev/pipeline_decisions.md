# v2 Pipeline Decisions: papers vs v1 implementation vs proposal

Sources:
- `source_papers.md` (which papers v1 is based on)
- `v1_review.md` (what v1 actually does, bugs B*, weaknesses W*)
- `model_comparison.md` (numbers; all VRAM figures there are estimates)

## 1. Algorithm-by-algorithm

Legend: ✅ matches the paper · ⚠️ deviates from the paper · ❌ bug · 🆕 the student's own idea

| # | Component | Paper says | v1 did | Verdict | v2 proposal |
|---|---|---|---|---|---|
| 1 | Self-adaptive gallery (Casao et al. 2023) | budget m=50; drop `argmax[γ·H/log N − (1−γ)·D]`, γ=0.6 | budget 50; drop = 0.5·redundancy + 0.5·uncertainty | ✅ structure, ⚠️ weights 0.5 vs 0.6 | **Keep it as the identity memory.** Use γ=0.6 from the paper (configurable). |
| 1b | Diversity gate before insertion | add a sample only if `D_i(x) ≥ D(C_i)` (it must add diversity) | **missing**: every frame is inserted, and the O(N²) pruner runs every frame (W3) | ⚠️ omitted, and it is the cause of near-duplicate galleries | Add the gate. Insert only the tracklet's selected samples, never raw consecutive frames. |
| 2 | Softmax-ratio test | `p1/p2 ≥ τ`, τ=2, temperature v=0.1 (×10) | τ=2, temperature ×15 | ⚠️ temperature differs | Paper default ×10 (configurable). Apply it to the **tracklet** embedding, not to single frames (W2). |
| 3 | Visibility-weighted centroids | weights r = fraction of visible joints | same (`get_weighted_centroids`) | ✅ | Keep. Also weight by our quality score. |
| 4 | Skeleton quality gate | unknown-pool samples need r ≥ 0.5 (OpenPose) | 9/17 keypoints visible (YOLO-pose); the gate is applied to everything; it uses the *first* pose in the crop (W12) | ✅ spirit, ❌ wrong pose picked when crops contain several people | Keep. Use the pose that belongs to *this* box. Add box-size and edge-truncation checks. |
| 5 | Unknown Data Manager (DBSCAN) | clusters need ≥ l=20 samples; compare with k=3 nearest centroids; new class if distance > 0.1; probabilistic decisions only after Q=20 classes | min 5 samples, eps 0.15, merge if cos ≥ 0.85; no final flush (B8); skipped on empty frames (B9) | ⚠️ looser, so 5 frames of one track can mint a person (W8); ❌ short-lived people vanish | Keep the UDM but run it on **unknown tracklets** with the paper's parameters. Add cannot-link rules (same camera at the same time). Flush at the end of the video. |
| 6 | α-FPS prototypes | Al Muzaddid & Beksi (arXiv 2510.17043): start from the centroid, pick the farthest point, interpolate it toward the nearest prototype `x ← x + α(p − x)`; no quality term | quality-weighted greedy farthest-point sampling, `α·q + (1−α)·dist`, K=3; module 5 cannot run (B12) | 🆕 only the name comes from the paper; scales are unnormalized | Keep the in-house variant with q and dist normalized to [0,1]. Offer the paper's interpolation variant as an option and compare both in evaluation. |
| 7 | Face as ground truth / identity healing | no paper | one face match ≥ 0.55 in **one frame** triggers an irreversible merge (B4); the module 6 merge only happens in memory (B5) | 🆕 novel, but implemented unsafely | Keep the idea (it is the project's main contribution), but require **accumulated evidence**: best-K faces per tracklet, a margin over the 2nd-best identity, then a must-link. Merges are reversible through re-clustering with history. This matches the 2026 "verify-before-update" line of work (OmniGlimpse +9.1 IDF1). |
| 8 | Multi-camera association | Casao: single camera | appearance-only, global gallery, timestamps broken (B1) | ❌ | Correct per-camera timestamps; cannot-link for same-camera/same-time pairs; optional transit-time topology. |
| 9 | Tracker re-ID | — | BotSort runs its own re-ID on top of CLIP-ReID (W1) | inefficient | Motion-only tracker; body embeddings computed once on the sampled frames. |
| 10 | Evaluation | Casao: open-world metrics on standard data | search tested on the system's own saved crops, against its own labels (W4), so the 85.35% figure is circular | ❌ | IDF1/HOTA per camera and across cameras on AI City 2024 ground truth, plus held-out retrieval. |

**Net design:** Casao's self-adaptive gallery remains the core identity memory, so the project's lineage stays citable ("based on Casao et al. 2023"). v2 adds three extensions:
- (a) It runs at the **tracklet level**: a tracklet's identity is decided first, and only then are its selected samples allowed into the gallery (verify-before-update).
- (b) Multi-camera constraints.
- (c) Face evidence acts as a must-link with margin, accumulated over time, plus a reversible consolidation pass.

This replaces the PLAN.md §3.3 idea of dropping the gallery entirely. The gallery stays, with agglomerative consolidation as the offline repair step.

## 2. Model choices (numbers from model_comparison.md)

Same metric within each table. Δ is measured against the v1 model.

| Slot | v1 (date) | Option A: same size, better | Option B: best available | Notes |
|---|---|---|---|---|
| Detector (COCO AP) | RT-DETR-L 2023-04: 53.0 AP, 8.8 ms T4 | RF-DETR-M 2025-03: **54.7 (+1.7)**, 4.4 ms, Apache | RF-DETR-L: **56.5 (+3.5)**, 6.8 ms | YOLO26m 2026-01: 52.5 (−0.5), 4.7 ms. No candidate publishes person-only or crowd AP, so measure on dev data. |
| Pose gate (COCO-kp AP) | YOLO11s-pose 2024-09: 58.9 | YOLO26s-pose 2026-01: **63.0 (+4.1)** | YOLO26m-pose **68.8 (+9.9)**; RTMO-l 72.4 | **Single-model option:** YOLO26m-pose as detector + pose removes one model (its box AP is unpublished). |
| Tracker (boxmot MOT17 HOTA) | BotSort+ReID 2022: 69.7 | ByteTrack (no re-ID): 67.7 (−2.0) | OccluBoost 2026-10: **71.1 (+1.4)**, no paper | Motion-only + tracklet association compensates. Validate OccluBoost on dev data. |
| Body ReID (MSMT17 mAP) | CLIP-ReID-Market 2022-11 (Market 89.6; never trained on MSMT17) | CLIP-ReID **MSMT17** weights: 73.4, same model/dim/speed | PersonViT-B 2024-08: **80.8 (+7.4 vs CLIP-MSMT)**, same ViT-B size, weights license unclear ("cc"); SOLIDER Swin-S 76.9 (MIT) | Cross-domain loss is large for every model (v1: Market 89.6 → Duke 50.2 mAP). MSMT17 training is the better bet for unseen cameras. |
| Face rec (IJB-C TAR@1e-4) | buffalo_l R50: 97.25 | keep (gains elsewhere are only +0.1–0.6) | AdaFace R100 WebFace12M: 97.66 (+0.4), the **only one with low-quality CCTV numbers** (TinyFace 72.3) | All good face weights are research-only; AuraFace is the Apache option. |

VRAM is not the limiting factor. The full stacks are estimated at 1.7 GB (Lite), 2.2 GB (Balanced) and 2.5 GB (Max), roughly ×2 in eager PyTorch. Even the Max stack uses about 5 GB of the T4's 15 GB. The real constraints are speed, licenses and weight availability. The spare VRAM can go to batching several camera streams.

Recommended defaults, all switchable via config:
- **Detector + pose:** YOLO26m-pose as a single model; RF-DETR-M + YOLO26s-pose is the "accurate" preset. Both are compared on AI City.
- **Tracker:** BotSort motion-only (OccluBoost as an option).
- **Body re-ID:** CLIP-ReID MSMT17 as the default; PersonViT-B as an option, after its license is checked.
- **Face:** buffalo_l; AdaFace as an option.
- **Lite preset (GTX 1050):** YOLO26n-pose, ByteTrack, OSNet MSMT17, buffalo_sc.
