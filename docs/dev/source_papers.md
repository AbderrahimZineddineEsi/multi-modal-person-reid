# Likely source papers for the v1 Re-ID pipeline

Method: web search (2026-11), then reading the abstract or full text of each hit. Items marked (abstract only) were not read in full.
Compared against `docs/dev/v1_review.md` §1.3, 1.6-1.9, 1.12-1.13 and `docs/technical_report.md` §5.3.

## Verdict

| # | Technique | Source | Confidence |
|---|-----------|--------|-----------|
| 1 | Self-adaptive gallery, budget 50, drop = redundancy + entropy | Casao et al., Sensors 2023 | HIGH |
| 3 | Softmax ratio p1/p2 >= tau (tau=2) vs weighted centroids | Casao et al. | HIGH |
| 4 | Skeleton visibility gate (+ visibility-weighted centroids) | Casao et al. (idea); 17-kp COCO/YOLO-pose threshold is the student's | HIGH (idea) |
| 5 | Unknown Data Manager + DBSCAN to mint identities | Casao et al. | HIGH |
| 2 | alpha-FPS prototypes | Al Muzaddid and Beksi, arXiv 2510.17043 (name only) | MEDIUM (name) / LOW (formula) |
| 6 | Face as ground truth, retroactive "identity healing" merge | No source found; student's own design | n/a |

Paper 1 (Casao) is almost certainly "the" main paper. Paper 2 (alpha-FPS) is the likely second. CLIP-ReID and ArcFace/SCRFD are the model papers (below) and not pipeline papers.

## Paper A (primary): Casao, Azagra, Murillo, Montijano
**"A Self-Adaptive Gallery Construction Method for Open-World Person Re-Identification"**, Sensors 23(5):2662, Feb 2023 (Univ. Zaragoza).
Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10007633/ , https://www.mdpi.com/1424-8220/23/5/2662 , doi:10.3390/s23052662.
Features: OSNet body descriptors, OpenPose joints (no face). Exact definitions as read from the full text:

- **Technique 1 (budget + drop cost).** Each class holds at most m = 50 features. When over budget, drop
  `x* = argmax_{x in C_i} [ gamma * H(x)/log(1/N) - (1-gamma) * D_i(x) ]`, gamma = 0.6.
  `D_i(x) = min_{x_j in C_i\{x}} (1 - x^T x_j)` is intra-class diversity.
  `H(x) = -sum_i p_i(x) log p_i(x)` is the Shannon entropy of the class-probability vector.
  Since `1 - D = max cosine to another crop`, this is the student's "redundancy + uncertainty" with different weights
  (student 0.5/0.5, paper 0.6 entropy / 0.4 redundancy). The student's normaliser is log(n_classes), same as the paper's log(1/N) in magnitude.
  Extra paper rule the student lacks: a new sample is only added if `D_i(x_q) >= D(C_i)` (it must add diversity to the class).
- **Technique 3 (softmax ratio).** `p_i(x) = exp(xbar_i^T x / v) / sum_j exp(xbar_j^T x / v)`, temperature v = 0.1 (i.e. multiply by 10; the student uses 15).
  Accept if `max_i p_i / max_{j != i} p_j >= tau`, tau = 2 (the student's default is also 2.0). Rejected samples "go into the pool of unknown data".
- **Technique 4 (skeleton visibility).** Centroid weights `r_ij = s_ij / s_T` (fraction of joints visible/confident), `xbar_i = sum r_ij x_ij / sum r_ij`.
  This is exactly the student's `get_weighted_centroids` (weights = pose visibility ratio). Unknown-pool samples need joint ratio r >= 0.5
  ("at least half of a person"); the student's 9/17 = 0.53 matches that spirit. The student applies the gate to everything, not only unknowns.
- **Technique 5 (DBSCAN / UDM).** Unknown Data Manager, DBSCAN chosen because "the number of classes is unbounded". Clusters with >= l = 20 samples are compared with
  the k = 3 nearest centroids; if distance > alpha = 0.1 a new class is created, else the cluster merges into the nearest class.
  Student: min_samples 5, eps 0.15, merge if cos >= 0.85 (an eps-like threshold of 0.15 vs the paper's 0.1), mints otherwise. Same structure, looser parameters, no k=3 centroid approximation.
  Paper also bootstraps: probabilistic decisions only start after Q = 20 classes; candidates join on cos > 0.9.
- **Not in the paper:** face, retroactive merging, prototypes/alpha-FPS, multi-camera, tracker.
- Related thesis (Casao, Zaragoza): https://zaguan.unizar.es/record/169179/files/texto_completo.pdf (not read).

Overlap is so specific (m = 50, tau = 2, entropy over centroid softmax, redundancy via nearest same-class crop, DBSCAN UDM, visibility-weighted centroids) that I rate it HIGH.
Note for the port: the student's `redundancy` sign and entropy normaliser are consistent with the paper; the weak spots flagged in v1_review §1.6 (cost ignores quality, can drop the crop just added)
are the paper's too, apart from the "diversity gate before adding" which the student omitted.

## Paper B (secondary, name match): Al Muzaddid and Beksi
**"Person Re-Identification via Generalized Class Prototypes"**, arXiv:2510.17043 (v1 Oct 2025, v2 15 Apr 2026), UT Arlington.
Link: https://arxiv.org/abs/2510.17043 (html: https://arxiv.org/html/2510.17043v2).
- Algorithm 1 (alpha-FPS): P starts as the class centroid; repeatedly choose the point farthest from P, then interpolate it toward its nearest prototype:
  `x <- x + alpha (p - x)` where p is the closest prototype. Motivation: plain FPS picks class-boundary outliers that hurt mAP. Training-free, N prototypes (no fixed K = 3), no quality term.
- Results with PHA features: CUHK03-NP 86.8 R1 / 84.8 mAP; Market-1501 95.9 / 95.8; MSMT17 85.4 / 81.5. No CLIP-ReID.
- **Match to technique 2: the name only.** The student's `score = alpha*q + (1-alpha)*dist_to_nearest_selected` with q = det_score*visibility, seeded by argmax(q), is a
  different formula: a quality-weighted classic FPS. It looks like the student's own adaptation (classic FPS + a quality prior). If the student read this paper, it was in the last ~2 months of the v1 timeline
  (v1 Oct 2025), so the timing is borderline for "about 1 year ago"; the same name could also come from an LLM suggestion. Confidence MEDIUM that it is the cited source of the name, LOW that the formula comes from it.
- Classic FPS (PointNet++, Qi et al., NeurIPS 2017; Eldar et al. 1997) is the true origin of the greedy max-min loop.

## Model papers (not pipeline sources, listed for the report)
- CLIP-ReID: Li, Sun, Li, "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels", AAAI 2023, arXiv:2211.13977.
- ArcFace: Deng et al., CVPR 2019 (arXiv:1801.07698). SCRFD: Guo et al., ICLR 2022 (arXiv:2105.04714). OSNet: Zhou et al., ICCV 2019 (used by Paper A, not by the student).
- BoT-SORT (Aharon et al., 2022, arXiv:2206.14651), RT-DETR (Zhao et al., CVPR 2024), YOLO-pose (Ultralytics) for the detection/pose/tracking parts.

## Technique 6 (face as ground truth, identity healing): no source
No paper found that retroactively merges body-only identities once a face match arrives. The closest searches returned:
- "Face Helps Person Re-Identification: Multi-modality Person Re-Identification Based on Vision-Language Models" (IEEE, 2024) https://ieeexplore.ieee.org/document/10744476/ , a learned face+body fusion, not a merge procedure (abstract only).
- Culturally-Aware Benchmark for ReID in Modest Attire, arXiv:2412.18874: ArcFace+CLIP-ReID distance fusion did not beat body-only when faces are occluded. A caution for a face-first rule.
Conclusion: this step (v1_review §1.9) is the student's own invention, an engineering union of Paper A's gallery with face recognition. The v1 weaknesses (one-frame evidence triggers an irreversible merge, §1.7) have no support in the literature.

## Related earlier work (context, lower confidence)
- Anchor-guided clustering + spatio-temporal ID re-assignment, Huang et al., arXiv:2304.09471, AI City 2023 Track 1 winner, IDF1 95.36 (offline, abstract only). No gallery, no DBSCAN described, no pose gate. Confidence LOW as a source; useful for multi-camera.
- Casao et al. cite the broader open-world / incremental ReID line (e.g. "Continuous Adaptation of Multi-Camera Person Identification Models through Sparse Non-redundant Representative Selection", arXiv:1607.00417), which is the conceptual ancestor of redundancy-based selection. Not read in detail.

## Newer follow-ups (2024-2026) relevant to online MCPT / indoor / retail
All found via search; results quoted from abstracts only unless stated.

| Work | Year | Idea | Reported result |
|------|------|------|-----------------|
| Xie et al., "A Robust Online Multi-Camera People Tracking System With Geometric Consistency and State-aware Re-ID Correction", CVPRW AI City 2024 https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Xie_A_Robust_Online_Multi-Camera_People_Tracking_System_With_Geometric_Consistency_CVPRW_2024_paper.pdf | 2024 | Online MCPT, geometry + appearance, re-ID correction on ID mismatches | AI City 2024 Track 1 leaderboard (number not read) |
| MCBLT, arXiv:2412.00692 | 2024-25 | 3D multi-camera tracking in long indoor (warehouse/retail) videos | HOTA 81.22 on AICity'24; IDF1 95.6 on WildTrack |
| Model Optimization for Multi-Camera 3D Detection and Tracking, arXiv:2602.00450 | 2026 | Optimising MC 3D detection/tracking | not read |
| Veng et al., "Calibration-Free 3D Multi-Camera People Tracking for Indoor Environment", arXiv:2607.22731 (CITA 2026) | 2026 | YOLOX+BoT-SORT, OSNet, HRNet pose, VGGT geometry, agglomerative clustering with appearance+geometry cost | HOTA 53.13 on AICity'24 without calibration |
| OmniGlimpse, arXiv:2610.36785 | 2026 | Online multi-view identity memory: propose in adaptive space, verify in a frozen canonical space, only verified observations update memory (the principled fix for the v1 "always add" flaw) | +9.1 IDF1 / +22.1 CVIDF1 on M3Track, +20.3 / +26.2 on MvMHAT over best online baseline |
| Fernando et al., "Generative AI Integrated Multimodal Framework for Low-Latency Multi-Camera Person ReID", arXiv:2609.14419 | 2026 | Early-exit cascade: body embedding first, face/VLM attributes only for ambiguous queries, adaptive reliability weights (a softer version of face-first) | resolves 60.7% (Duke) / 68.4% (Market) of queries without semantic stage |
| Yao et al., "Identity-Conditioned Score Fusion for Open-Set Person ReID", arXiv:2610.07366 | 2026 | Training-free per-identity score fusion of face, gait, body shape | up to 8.8% absolute reduction in false non-identification rate (clothes-changing sets) |
| 9th AI City Challenge (ICCV 2025W), arXiv:2508.13564, Track 1 | 2025 | Multi-camera multi-object tracking, synthetic | offline winner Team65 ZV HOTA 69.91; top online entry (TeamQDT, rank 3) HOTA 28.75, so online lags badly |
| Retail ReID study, arXiv:2505.00772 | 2025 | Evaluation of ReID in retail stores | false detections cause false re-ID; methods "need improvement" |
| Generalized Class Prototypes, arXiv:2510.17043 (Paper B) | 2025-26 | alpha-FPS prototypes | see above |

Gaps: no 2024-26 paper found that directly improves Casao's gallery (budgeted, entropy+redundancy) for online retail MCPT. Worth a follow-up search for "Casao" and "open-world re-identification" citations on Google Scholar / Semantic Scholar, which the web search tool could not query.

## Practical suggestions for the v2 port
1. Cite Casao et al. 2023 as the design basis; adopt its diversity gate before insertion and its l >= 20 / k = 3 centroid cluster test.
2. Replace the one-frame face merge with an evidence-accumulation rule (OmniGlimpse-style verify-before-update; Fernando et al. cascade).
3. Treat alpha-FPS as an in-house variant; normalise q and dist to the same scale (v1_review §1.12) or cite Muzaddid and Beksi only for the interpolation form.
4. Legal note: face-linked identity across cameras is biometric processing; review GDPR/biometric statutes before retail deployment.
