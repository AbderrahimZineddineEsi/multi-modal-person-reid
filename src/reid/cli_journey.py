"""``reid journey``: print one identity's journey and render its follow clip.

    reid journey --data-dir ENV --gid 3 --cameras cam_01,cam_02 --speed 4 --max-seconds 60 --plan

``--plan`` prints what the clip would contain and the estimated render time without rendering.
Every option goes through :class:`reid.viz.clip_plan.ClipOptions`, the same object the Clips tab
of the app uses, so both agree on the plan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from reid.storage import db, files
from reid.viz.clip_plan import ClipOptions, ClipPlan


def _options(args: argparse.Namespace, cfg, cam_ids: dict[str, str]) -> ClipOptions:
    """Command-line values -> options; camera ids and names are both accepted."""
    cams = None
    if args.cameras:
        wanted = [c.strip() for c in args.cameras.split(",") if c.strip()]
        cams = tuple(cam_ids.get(c, c) for c in wanted)
    return ClipOptions.from_cfg(
        cfg,
        cameras=cams,
        start_s=args.start,
        end_s=args.end,
        speed=args.speed,
        max_seconds=args.max_seconds,
        blur_others=bool(args.blur_others) or cfg.render.blur_others,
        show_others=not args.no_other_boxes,
        thumbs=args.thumbs > 0,
        max_thumbs=args.thumbs,
        **({"width": args.width} if args.width else {}),
        **({"fps": args.fps} if args.fps else {}),
    )


def _print_plan(plan: ClipPlan, est_s: float) -> None:
    from reid.viz.clip_cost import fmt_seconds

    out = sys.stdout.write
    if plan.empty:
        out("plan: nothing to render (no sighting in the selected cameras and window)\n")
        return
    lay = plan.layout
    for i, s in enumerate(plan.segments, 1):
        gap = "first" if s.gap_s is None else f"+{s.gap_s:.1f} s skipped"
        out(
            f"  {i:>2}. {s.camera_name:<18} {s.t0:8.1f} s -> {s.t1:8.1f} s  "
            f"{s.n_frames:>6} frames  {gap}\n"
        )
    out(
        f"plan: {len(plan.segments)} cuts, {plan.out_frames} frames at {plan.fps:g} fps = "
        f"{plan.out_seconds:.1f} s clip covering {plan.source_seconds:.1f} s of footage "
        f"({plan.speed:g}x)"
    )
    out(f", {lay.width}x{lay.height}\n" if lay else "\n")
    if plan.truncated:
        out(f"      capped: the whole journey would be {plan.full_out_seconds:.1f} s\n")
    out(f"estimated render time: ~{fmt_seconds(est_s)}\n")


def cmd_journey(args: argparse.Namespace) -> int:
    """``reid journey --gid G [--out F] [--gif] [--no-video] [--plan] [clip options]``."""
    from reid import analytics
    from reid.cli import _build_config
    from reid.viz import clip_cost
    from reid.viz.clip_plan import plan_clip
    from reid.viz.journey import render_clip

    cfg = _build_config(args)
    logging.basicConfig(level=cfg.runtime.log_level, format="%(levelname)s %(name)s: %(message)s")
    conn = db.connect(files.db_path(cfg.data_dir))
    try:
        steps = analytics.journey(conn, args.gid)
        if not steps:
            sys.stderr.write(f"reid journey: identity {args.gid} has no tracklets\n")
            return 1
        for s in steps:
            gap = "" if s.gap_s is None else f"  (gap {s.gap_s:+.1f} s)"
            sys.stdout.write(
                f"{s.camera_name:<16} {s.start_ts:8.1f} s -> {s.end_ts:8.1f} s "
                f"[{s.duration_s:5.1f} s]{gap}\n"
            )
        if args.no_video and not args.plan:
            return 0
        names = {c.name: c.id for c in db.list_cameras(conn)}
        opts = _options(args, cfg, names)
        plan = plan_clip(conn, args.gid, opts)
        if args.plan:
            _print_plan(plan, clip_cost.estimate(plan, clip_cost.load_model(cfg.data_dir)))
            return 0 if not plan.empty else 1
        if plan.empty:
            sys.stderr.write("reid journey: no sighting in the selected cameras and window\n")
            return 1
        from reid.viz.render import export_gif

        out = (
            Path(args.out)
            if args.out
            else files.render_path(cfg.data_dir, f"journey_gid{args.gid}.mp4")
        )
        try:
            res = render_clip(conn, cfg, args.gid, out, opts, plan=plan)
        except FileNotFoundError as exc:
            sys.stderr.write(f"reid journey: {exc}\n")
            return 1
        clip_cost.record(cfg.data_dir, plan, res.seconds, res.frames)
        sys.stdout.write(f"{out}\n")
        if args.gif:
            g = export_gif(out, out.with_suffix(".gif"), cfg.render.gif_fps, cfg.render.gif_width)
            sys.stdout.write(f"{g}\n")
    finally:
        conn.close()
    return 0


def _journey_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gid", type=int, required=True, help="identity to follow")
    p.add_argument("--out", help="output mp4 (default data_dir/renders/journey_gid<G>.mp4)")
    p.add_argument("--gif", action="store_true", help="also export a GIF")
    p.add_argument("--no-video", action="store_true", help="only print the journey")
    p.add_argument(
        "--plan",
        action="store_true",
        help="print the clip plan and the estimated render time, render nothing",
    )
    p.add_argument("--start", type=float, help="clip: start second on the shared timeline")
    p.add_argument("--end", type=float, help="clip: end second on the shared timeline")
    p.add_argument(
        "--cameras",
        help="clip: comma-separated camera ids or names (default: all that see the person)",
    )
    p.add_argument(
        "--speed", type=float, default=1.0, help="clip: playback speed multiplier (default 1)"
    )
    p.add_argument("--max-seconds", type=float, help="clip: cap on the clip length in seconds")
    p.add_argument("--blur-others", action="store_true", help="clip: blur every other person")
    p.add_argument(
        "--no-other-boxes", action="store_true", help="clip: box only the followed person"
    )
    p.add_argument(
        "--thumbs",
        type=int,
        default=4,
        help="clip: side thumbnails of other cameras (0 = none, default 4)",
    )
    p.add_argument(
        "--width", type=int, help="clip: output width in pixels (default render.journey_width)"
    )
    p.add_argument(
        "--fps", type=float, help="clip: output frame rate (default: source fps, max 30)"
    )


cmd_journey.add_arguments = _journey_args  # type: ignore[attr-defined]
