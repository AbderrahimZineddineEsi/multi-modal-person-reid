"""Evaluation of a processed ``data_dir`` against labelled ground truth (PLAN section 7).

``evaluate(data_dir, gt_path)`` reports, for one run:

* per-camera MOT metrics (IDF1, MOTA, ID switches, precision, recall), once with the system gid
  as hypothesis id and once with the tracklet id (tracker only);
* the cross-camera **CityFlow-style combined IDF1**: one ``motmetrics`` accumulator over all
  cameras, frame keys (camera, frame), GT ids global, hypothesis ids = gids, so a gid that spans
  cameras must match ONE GT person globally (details in :mod:`reid.eval.mot`);
* fragmentation / purity: GT people split over several gids, gids that merge several people;
* held-out body retrieval Rank-1 / mAP from GT crops (:mod:`reid.eval.retrieval`; fixes the
  circular v1 evaluation W4).

The GT is the uniform CSV written by ``reid dataset fetch`` (:mod:`reid.datasets.gt`). GT boxes
are only compared on frames the run actually processed (``runtime.frame_stride`` etc.) and
GT boxes shorter than ``min_gt_height`` pixels are ignored (nobody can detect them; set 0 to
keep all). Results go to ``data_dir/eval/<timestamp>.json`` with the config and preset.
Needs the ``[eval]`` extra (motmetrics, pandas).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from reid.datasets.gt import read_gt
from reid.eval import mot
from reid.eval.hypothesis import (
    infer_preset,
    latest_run,
    load_hypothesis,
    processed_frame_mask,
    run_config,
)
from reid.eval.report import save_result, to_markdown
from reid.storage import db, files

log = logging.getLogger(__name__)

__all__ = ["evaluate", "save_result", "to_markdown"]


def scale_gt_to_cameras(gt, cams: dict, data_dir: Path):
    """Scale GT boxes to the stored camera size (``runtime.max_side`` may have downscaled it).

    GT boxes are in the coordinates of the camera's source video; its size is read with OpenCV
    and compared with the camera size recorded in the database. Equal sizes (the usual case,
    and every database made before ``max_side`` existed) leave the boxes untouched.
    """
    import cv2

    from reid.viz.video_io import resolve_video

    gt = gt.copy()
    for cid, cam in cams.items():
        m = gt["camera"] == cid
        if not m.any():
            continue
        try:
            cap = cv2.VideoCapture(str(resolve_video(cam.video_path, data_dir)))
            vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except FileNotFoundError:
            log.warning("%s: source video not found, GT boxes are not rescaled", cid)
            continue
        if vw <= 0 or vh <= 0 or (vw, vh) == (cam.width, cam.height):
            continue
        sx, sy = cam.width / vw, cam.height / vh
        log.info("%s: scaling GT boxes %dx%d -> %dx%d", cid, vw, vh, cam.width, cam.height)
        for a, k in (("x1", sx), ("x2", sx), ("y1", sy), ("y2", sy)):
            gt.loc[m, a] = gt.loc[m, a] * k
    return gt


def evaluate(
    data_dir: str | Path,
    gt_path: str | Path,
    *,
    iou: float = 0.5,
    run_id: int | None = None,
    min_gt_height: float = 40.0,
    retrieval: bool = True,
    queries_per_id: int = 5,
    min_query_height: float = 80.0,
    preset: str | None = None,
    models: Any = None,
) -> dict[str, Any]:
    """Evaluate run ``run_id`` (default: the latest) of ``data_dir`` against ``gt_path``.

    Returns a JSON-serializable dict (not yet saved; see :func:`save_result`). ``preset`` only
    labels the report (default: inferred from the run's model choices).
    """
    data_dir = Path(data_dir)
    conn = db.connect(files.db_path(data_dir))
    try:
        row = latest_run(conn, run_id)
        rid = int(row["id"])
        cfg = run_config(row)
        cfg.paths.data_dir = str(data_dir)
        rt = cfg.runtime
        cams = {c.id: c for c in db.list_cameras(conn)}
        gt = read_gt(gt_path)
        cameras = sorted(c for c in cams if c in set(gt["camera"]))
        if not cameras:
            raise ValueError(
                f"no camera of {data_dir} ({sorted(cams)}) appears in the GT ({gt_path}); "
                f"GT cameras: {sorted(gt['camera'].unique())}"
            )
        gt = scale_gt_to_cameras(gt[gt["camera"].isin(cameras)], cams, data_dir)
        keep = (gt["y2"] - gt["y1"]) >= min_gt_height
        for cam in cameras:
            m = gt["camera"] == cam
            keep &= ~m | processed_frame_mask(gt["frame"], rt, cams[cam].fps)
        gt = gt[keep]
        hyp = load_hypothesis(conn, rid)
        hyp = hyp[hyp["camera"].isin(cameras)]

        pc = mot.per_camera(gt, hyp, iou, cameras, "hid")
        pct = mot.per_camera(gt, hyp, iou, cameras, "tid")
        acc = mot.build_accumulator(gt, hyp, iou, cameras, "hid", combined=True)
        pairs = mot.pair_counts(acc)
        acc_t = mot.build_accumulator(gt, hyp, iou, cameras, "tid", combined=True)
        n_gids = int(hyp.loc[hyp["gid"].notna(), "gid"].nunique())
        result: dict[str, Any] = {
            "timestamp": time.time(),
            "data_dir": str(data_dir),
            "gt": str(gt_path),
            "run_id": rid,
            "preset": preset or infer_preset(cfg),
            "config": cfg.model_dump(),
            "protocol": {
                "iou": iou,
                "min_gt_height": min_gt_height,
                "frame_stride": rt.frame_stride,
                "cameras": cameras,
                "combined": "one accumulator, frame key = (camera, frame), global GT ids and gids",
            },
            "counts": {
                "gt_ids": int(gt["gid"].nunique()),
                "gt_boxes": len(gt),
                "hyp_boxes": len(hyp),
                "tracklets": int(hyp["tracklet_id"].nunique()),
                "unidentified_tracklets": int(hyp.loc[hyp["gid"].isna(), "tracklet_id"].nunique()),
                "identities": n_gids,
            },
            "per_camera": pc,
            "per_camera_tracklet": pct,
            "mtmc": mot.summarize(acc),
            "fragmentation": mot.fragmentation(pairs, int(gt["gid"].nunique()), n_gids),
            "tracklet_purity": mot.tracklet_purity(mot.pair_counts(acc_t), hyp),
        }
        if retrieval:
            from reid.eval.retrieval import run_retrieval

            result["retrieval"] = run_retrieval(
                conn,
                data_dir,
                cfg,
                rid,
                gt,
                mot.majority_gt(pairs),
                {c: cams[c].video_path for c in cameras},
                {c: (cams[c].width, cams[c].height) for c in cameras},
                per_id=queries_per_id,
                min_height=min_query_height,
                models=models,
            )
    finally:
        conn.close()
    return result
