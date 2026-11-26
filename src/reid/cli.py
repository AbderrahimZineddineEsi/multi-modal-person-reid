"""Command line interface: ``reid <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from reid.cli_benchmark import cmd_benchmark
from reid.config import Config, load_config
from reid.storage import db, files

log = logging.getLogger(__name__)


def _make_engine(cfg: Config, cameras: Any, conn: Any) -> Any:
    """``IdentityEngine`` if :mod:`reid.identity` is importable, else None (extraction only)."""
    try:
        from reid.identity import IdentityEngine
    except ImportError as exc:
        log.warning("reid.identity unavailable (%s): running extraction only, gids stay NULL", exc)
        return None
    from reid.storage.vectors import VectorStore

    return IdentityEngine(cfg, conn, VectorStore(files.vectors_dir(cfg.data_dir)), cameras)


def cmd_process(args: argparse.Namespace) -> int:
    """``reid process --cameras X.yaml [--preset lite]``: extraction + identity association."""
    from tqdm import tqdm

    from reid.config import load_cameras
    from reid.pipeline import run

    cfg = _build_config(args)
    logging.basicConfig(level=cfg.runtime.log_level, format="%(levelname)s %(name)s: %(message)s")
    from reid.storage.portable import apply_env_models

    # an environment's vectors all come from one body/face model: a new run must keep it
    cfg, note = apply_env_models(cfg, cfg.data_dir)
    if note:
        sys.stderr.write(f"note: processing keeps this environment's embedding models ({note})\n")
    cameras = load_cameras(args.cameras)
    files.ensure_dirs(cfg.data_dir)
    conn = db.connect(files.db_path(cfg.data_dir))
    bars: dict[str, Any] = {}

    def progress(cam: str, done: int, total: int) -> None:
        bar = bars.get(cam) or bars.setdefault(
            cam, tqdm(total=total or None, desc=cam, disable=None)
        )
        bar.n = done
        bar.refresh()

    try:
        engine = None if args.no_identity else _make_engine(cfg, cameras, conn)
        res = run(cfg, cameras, engine, progress, conn=conn)
        out = res.to_dict()
        if engine is not None:
            from reid import analytics

            out["run_summary"] = analytics.run_summary(conn, res.run_id)
    finally:
        for b in bars.values():
            b.close()
        conn.close()
    # same file the app writes, so the Metrics tab shows timings and GPU memory of CLI runs too
    stats_path = Path(cfg.data_dir) / "cache" / f"run_{res.run_id}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(res.to_dict(), indent=2, default=str))
    sys.stdout.write(json.dumps(out, indent=2, default=str) + "\n")
    from reid.models.gpu_memory import format_peak

    sys.stderr.write(format_peak(res.gpu_memory) + "\n")
    return 0


def _process_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cameras", required=True, help="camera set YAML (configs/cameras/X.yaml)")
    p.add_argument("--no-identity", action="store_true", help="extraction only (gids stay NULL)")


cmd_process.add_arguments = _process_args  # type: ignore[attr-defined]


def cmd_render(args: argparse.Namespace) -> int:
    """``reid render --camera ID|all [--gid G] [--start S --end E] [--mosaic] [--gif]``."""

    from reid.viz import render as R

    cfg = _build_config(args)
    logging.basicConfig(level=cfg.runtime.log_level, format="%(levelname)s %(name)s: %(message)s")
    if not args.camera and not args.mosaic:
        sys.stderr.write("reid render: give --camera ID|all and/or --mosaic\n")
        return 1
    out_dir = Path(args.out) if args.out else files.render_path(cfg.data_dir, "")
    conn = db.connect(files.db_path(cfg.data_dir))
    written: list[Path] = []
    try:
        cam_ids = [c.id for c in db.list_cameras(conn)]
        todo = [] if not args.camera else cam_ids if args.camera == "all" else [args.camera]
        suffix = f"_gid{args.gid}" if args.gid is not None else ""
        only = [args.gid] if args.only and args.gid is not None else None
        for cid in todo:
            out = out_dir / f"{cid}{suffix}.mp4"
            R.render_camera(
                conn, cfg, cid, out, gid_filter=only, start_s=args.start, end_s=args.end,
                scale=args.scale, highlight_gid=None if only else args.gid,
            )  # fmt: skip
            written.append(out)
        if args.mosaic:
            out = out_dir / f"mosaic{suffix}.mp4"
            R.render_mosaic(
                conn, cfg, None if args.camera in (None, "all") else todo, out,
                highlight_gid=args.gid, start_s=args.start, end_s=args.end,
            )  # fmt: skip
            written.append(out)
        if args.gif:
            for mp4 in list(written):
                written.append(
                    R.export_gif(
                        mp4, mp4.with_suffix(".gif"), cfg.render.gif_fps, cfg.render.gif_width
                    )
                )
    finally:
        conn.close()
    for p in written:
        sys.stdout.write(f"{p}\n")
    return 0


def _render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", help="camera id, or 'all'")
    p.add_argument("--gid", type=int, help="highlight this identity (others are dimmed)")
    p.add_argument("--only", action="store_true", help="with --gid: draw only that identity")
    p.add_argument("--start", type=float, help="start second (video time; mosaic: timeline offset)")
    p.add_argument("--end", type=float, help="end second")
    p.add_argument("--scale", type=float, help="output scale factor (default render.scale)")
    p.add_argument("--mosaic", action="store_true", help="also render the synchronized mosaic")
    p.add_argument("--gif", action="store_true", help="also export a GIF of every output")
    p.add_argument("--out", help="output directory (default data_dir/renders)")


cmd_render.add_arguments = _render_args  # type: ignore[attr-defined]


def cmd_search(args: argparse.Namespace) -> int:
    """``reid search --image X [--mode body|face|both]``: print the SearchHit as JSON."""
    import dataclasses

    import cv2

    from reid import search
    from reid.storage.portable import apply_env_models
    from reid.storage.vectors import VectorStore

    cfg = _build_config(args)
    cfg, note = apply_env_models(cfg)
    if note:
        sys.stderr.write(f"reid search: {note}\n")
    img = cv2.imread(args.image)
    if img is None:
        sys.stderr.write(f"reid search: cannot read image {args.image!r}\n")
        return 1
    conn = db.connect(files.db_path(cfg.data_dir))
    vectors = VectorStore(files.vectors_dir(cfg.data_dir))
    try:
        fn = {"body": search.search_body, "face": search.search_face, "both": search.search_both}
        kw = {"is_crop": True} if args.crop and args.mode != "body" else {}
        hit = fn[args.mode](img, conn, vectors, cfg, **kw)
    finally:
        conn.close()
    out = dataclasses.asdict(hit)
    out["status"] = hit.status.value
    sys.stdout.write(json.dumps(out, indent=2, default=str) + "\n")
    return 0 if hit.status.value in ("MATCH", "AMBIGUOUS") else 3


def _search_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", required=True, help="query image (BGR file readable by OpenCV)")
    p.add_argument("--mode", choices=("body", "face", "both"), default="both")
    p.add_argument("--crop", action="store_true", help="the image is a tight face crop")


cmd_search.add_arguments = _search_args  # type: ignore[attr-defined]


def cmd_journey(args: argparse.Namespace) -> int:
    """``reid journey --gid G [--out F] [--gif] [--no-video]``: print the journey, render a clip."""

    from reid import analytics

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
        if args.no_video:
            return 0
        from reid.viz.journey import follow_person
        from reid.viz.render import export_gif

        out = (
            Path(args.out)
            if args.out
            else files.render_path(cfg.data_dir, f"journey_gid{args.gid}.mp4")
        )
        try:
            follow_person(conn, cfg, args.gid, out, start_s=args.start, end_s=args.end)
        except FileNotFoundError as exc:
            sys.stderr.write(f"reid journey: {exc}\n")
            return 1
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
    p.add_argument("--start", type=float, help="clip: start second on the shared timeline")
    p.add_argument("--end", type=float, help="clip: end second on the shared timeline")


cmd_journey.add_arguments = _journey_args  # type: ignore[attr-defined]


def cmd_evaluate(args: argparse.Namespace) -> int:
    """``reid evaluate --data-dir D --gt gt.csv``: metrics against ground truth (``reid.eval``)."""
    from reid.eval import evaluate, save_result, to_markdown

    cfg = _build_config(args)
    try:
        result = evaluate(
            cfg.data_dir,
            args.gt,
            iou=args.iou,
            run_id=args.run_id,
            min_gt_height=args.min_gt_height,
            retrieval=not args.no_retrieval,
            queries_per_id=args.queries_per_id,
            preset=args.preset,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"reid evaluate: {exc}\n")
        return 1
    path = save_result(cfg.data_dir, result)
    sys.stdout.write(to_markdown(result) + f"\nsaved: {path}\n")
    return 0


def _evaluate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gt", required=True, help="uniform GT CSV (written by `reid dataset fetch`)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a box match")
    p.add_argument("--run-id", type=int, help="run to evaluate (default: latest)")
    p.add_argument("--min-gt-height", type=float, default=40.0, help="ignore GT boxes shorter (px)")
    p.add_argument("--queries-per-id", type=int, default=5, help="retrieval queries per person")
    p.add_argument("--no-retrieval", action="store_true", help="skip the held-out retrieval part")


cmd_evaluate.add_arguments = _evaluate_args  # type: ignore[attr-defined]


def cmd_dataset(args: argparse.Namespace) -> int:
    """``reid dataset fetch aicity24 --scene 075 --cameras 4 --minutes 2 --out DIR``."""
    from reid.datasets.aicity24 import fetch_aicity24, result_dict

    every = args.camera_ids == "all" or args.cameras == 0
    res = fetch_aicity24(
        args.scene,
        cameras=0 if every else args.cameras,
        camera_ids=None if every else _csv(args.camera_ids, int),
        minutes=args.minutes if args.minutes > 0 else None,
        out=args.out,
        height=args.height or None,
        yaml_path=args.yaml,
        stream=args.stream,
        keep_raw=args.keep_raw,
        raw_dir=args.raw_dir,
    )
    sys.stdout.write(json.dumps(result_dict(res), indent=2) + "\n")
    return 0


def _dataset_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="dataset_cmd", required=True)
    f = sub.add_parser("fetch", help="download and convert a labelled multi-camera subset")
    f.add_argument("name", choices=("aicity24",), help="dataset (AI City 2024 MTMC, CC-BY-4.0)")
    f.add_argument("--scene", default="075", help="scene id (071-080 are retail), default 075")
    f.add_argument(
        "--cameras", type=int, default=4, help="number of cameras (auto-selected; 0 = all)"
    )
    f.add_argument(
        "--camera-ids", help="comma list of camera numbers, e.g. 700,701, or 'all' (overrides)"
    )
    f.add_argument("--minutes", type=float, default=2.0, help="keep the first M minutes (0 = all)")
    f.add_argument(
        "--height", type=int, default=720,
        help="downscale to this height (0 = keep; with --minutes 0 too: no re-encode)",
    )  # fmt: skip
    f.add_argument("--out", required=True, help="output directory (videos, gt.csv, meta.json)")
    f.add_argument("--yaml", help="camera YAML (default configs/cameras/aicity24_<scene>.yaml)")
    f.add_argument("--stream", action="store_true", help="let ffmpeg read only the first minutes")
    f.add_argument("--keep-raw", action="store_true", help="keep the downloaded full videos")
    f.add_argument("--raw-dir", help="where downloads go (default OUT/raw), removed afterwards")


cmd_dataset.add_arguments = _dataset_args  # type: ignore[attr-defined]


def _csv(v: str | None, cast: Callable[[str], Any]) -> list[Any] | None:
    return [cast(x) for x in v.split(",") if x.strip() != ""] if v else None


def cmd_virtual_cams(args: argparse.Namespace) -> int:
    """``reid virtual-cams VIDEO``: derive N virtual cameras plus a camera YAML."""
    from reid.virtual_cams import make_virtual_cameras

    yp = make_virtual_cameras(
        args.video,
        args.out_dir,
        n=args.n,
        overlap=args.overlap,
        mirror=_csv(args.mirror, int),
        delays=_csv(args.delays, float),
        out_width=args.width,
        start_s=args.start,
        end_s=args.end,
        face_cams=_csv(args.face_cams, int),
        yaml_path=args.yaml,
    )
    sys.stdout.write(f"{yp}\n")
    return 0


def _vc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", help="source video")
    p.add_argument("--out-dir", required=True, help="where to write the camera videos")
    p.add_argument("--n", type=int, default=2, help="number of virtual cameras")
    p.add_argument("--overlap", type=float, default=0.15, help="overlap of adjacent crops")
    p.add_argument("--mirror", help="comma list of camera indices to mirror, e.g. 1")
    p.add_argument("--delays", help="comma list of per-camera start_ts offsets (s)")
    p.add_argument("--width", type=int, help="downscale each crop to this width")
    p.add_argument("--start", type=float, default=0.0, help="start second in the source")
    p.add_argument("--end", type=float, help="end second in the source")
    p.add_argument("--face-cams", help="comma list of camera indices flagged is_face_cam")
    p.add_argument("--yaml", help="camera YAML output path (default <out-dir>/cameras.yaml)")


cmd_virtual_cams.add_arguments = _vc_args  # type: ignore[attr-defined]


def cmd_app(args: argparse.Namespace) -> int:
    """``reid app [--workspace DIR --env NAME] [--share --auth user:pass] [--port N]``."""
    try:
        from reid.app import launch
    except ImportError as exc:
        sys.stderr.write(f"reid app: needs the app extra (pip install 'reid[app]'): {exc}\n")
        return 1
    auth = None
    if args.auth:
        user, sep, password = args.auth.partition(":")
        if not (sep and user and password):
            sys.stderr.write("reid app: --auth expects user:password\n")
            return 1
        auth = (user, password)
    cfg = _build_config(args)
    workspace, env = _app_workspace(args, cfg)
    logging.basicConfig(level=cfg.runtime.log_level, format="%(levelname)s %(name)s: %(message)s")
    demo, creds = launch(
        cfg, share=args.share, auth=auth, workspace=workspace, env=env,
        read_only=args.read_only, port=args.port, host=args.host, block=False,
    )  # fmt: skip
    if creds is not None and auth is None:
        sys.stdout.write(f"login for the shared link: {creds[0]} / {creds[1]}\n")
    demo.block_thread()
    return 0


def _app_workspace(args: argparse.Namespace, cfg: Config) -> tuple[Path, str]:
    """``(workspace, initial environment)`` from ``--workspace/--env`` or ``--data-dir``."""
    if args.workspace:
        return Path(args.workspace).expanduser(), args.env or "default"
    data_dir = cfg.data_dir.resolve()
    if args.data_dir:  # one data_dir: its parent is the workspace, its folder the environment
        return data_dir.parent, args.env or data_dir.name
    return data_dir, args.env or "default"


def _app_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", metavar="DIR", help="folder whose subfolders are environments")
    p.add_argument("--env", metavar="NAME", help="environment selected at start (default: default)")
    p.add_argument("--share", action="store_true", help="public link (a login is generated)")
    p.add_argument("--auth", metavar="USER:PASS", help="require this login")
    p.add_argument("--read-only", action="store_true", help="hide the delete and wipe tools")
    p.add_argument("--admin", action="store_true", help=argparse.SUPPRESS)  # old no-op: always on
    p.add_argument("--port", type=int, help="port (default: first free from 7860)")
    p.add_argument("--host", help="interface to bind (default 127.0.0.1)")


cmd_app.add_arguments = _app_args  # type: ignore[attr-defined]


def _build_config(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    if args.data_dir:
        overrides["paths.data_dir"] = args.data_dir
    preset = getattr(args, "preset", None)
    if preset is None and getattr(args, "command", "") in ("process", "app"):
        preset = "auto"  # [planner] pick lite/balanced/accurate from the hardware
    return load_config(args.config, overrides, preset=preset)


def collect_info(cfg: Config) -> dict[str, Any]:
    """Data-dir statistics (file level plus DB row counts if a DB exists)."""
    stats: dict[str, Any] = dict(files.data_stats(cfg.data_dir))
    dbp = files.db_path(cfg.data_dir)
    if dbp.exists():
        conn = db.connect(dbp)
        try:
            stats["db"] = db.table_counts(conn)
            stats["schema_version"] = db.schema_version(conn)
        finally:
            conn.close()
    return stats


def cmd_env(args: argparse.Namespace) -> int:
    """``reid env list|export|import|relink``: move environments between machines."""
    from reid.storage import portable as pt

    try:
        if args.env_cmd == "list":
            rows = pt.list_envs(Path(args.workspace))
            if not rows:
                sys.stdout.write(f"no environment in {args.workspace}\n")
            for r in rows:
                sys.stdout.write(
                    f"{r.name:<24} {r.cameras} cams  {r.people} people  {r.runs} runs  "
                    f"{pt.human_size(r.size_bytes):>9}  "
                    f"videos {r.videos_present}/{r.cameras}  preset {r.preset or '?'}\n"
                )
        elif args.env_cmd == "export":
            res = pt.export_env(
                args.data_dir_env, args.out,
                with_videos=args.with_videos, with_renders=args.with_renders,
            )  # fmt: skip
            if res.missing_videos:
                sys.stderr.write(f"warning: no video found for {', '.join(res.missing_videos)}\n")
            sys.stdout.write(
                f"{res.out}  ({pt.human_size(res.size_bytes)}, {len(res.videos)} videos)\n"
            )
        elif args.env_cmd == "import":
            r = pt.import_env(args.archive, args.workspace, name=args.name, force=args.force)
            sys.stdout.write(
                f"imported '{r.name}' -> {r.path}\n"
                f"{r.cameras} cameras, schema v{r.schema_version}, "
                f"{r.videos_relinked} bundled videos\n"
            )
            if r.videos_missing:
                sys.stdout.write(
                    f"no local video for: {', '.join(r.videos_missing)} (clips and heatmaps need "
                    "them; use `reid env relink`)\n"
                )
        else:
            r = pt.relink_env(args.data_dir_env, args.videos)
            for cid, path in r.relinked.items():
                sys.stdout.write(f"relinked  {cid} -> {path}\n")
            for cid in r.already_ok:
                sys.stdout.write(f"ok        {cid}\n")
            for cid, hits in r.ambiguous.items():
                sys.stdout.write(f"ambiguous {cid}: {', '.join(hits)}\n")
            for cid in r.missing:
                sys.stdout.write(f"missing   {cid}\n")
            return 0 if not (r.missing or r.ambiguous) else 3
    except (ValueError, FileExistsError, RuntimeError) as exc:
        sys.stderr.write(f"reid env {args.env_cmd}: {exc}\n")
        return 1
    return 0


def _env_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="env_cmd", required=True)
    q = sub.add_parser("list", help="list the environments of a workspace")
    q.add_argument("--workspace", required=True, metavar="DIR")
    q = sub.add_parser("export", help="zip an environment (for another machine)")
    q.add_argument("--data-dir", dest="data_dir_env", required=True, metavar="ENV_DIR")
    q.add_argument("--out", required=True, metavar="FILE.zip")
    q.add_argument("--with-videos", action="store_true", help="bundle the camera videos (big)")
    q.add_argument("--with-renders", action="store_true", help="include renders/ (default: skip)")
    q = sub.add_parser("import", help="unzip an exported environment into a workspace")
    q.add_argument("archive", metavar="FILE.zip")
    q.add_argument("--workspace", required=True, metavar="DIR")
    q.add_argument("--name", help="environment name (default: the one in the zip)")
    q.add_argument("--force", action="store_true", help="replace an existing environment")
    q = sub.add_parser("relink", help="point cameras at local copies of their videos")
    q.add_argument("--data-dir", dest="data_dir_env", required=True, metavar="ENV_DIR")
    q.add_argument("--videos", required=True, metavar="DIR")


cmd_env.add_arguments = _env_args  # type: ignore[attr-defined]


def cmd_sync(args: argparse.Namespace) -> int:
    """``reid sync push|pull --workspace DIR --remote TARGET``: keep two workspaces identical."""
    from reid import sync as sy

    try:
        local_root = Path(args.workspace).expanduser()
        local, remote = sy.DirTransfer(local_root), sy.make_transfer(args.remote)
        rep = sy.sync(
            local, remote, local_root, args.sync_cmd, envs=args.env, dry_run=args.dry_run,
            force=args.force, delete=args.delete, with_renders=args.with_renders,
        )  # fmt: skip
    except (sy.SyncError, OSError, ValueError) as exc:
        sys.stderr.write(f"reid sync {args.sync_cmd}: {exc}\n")
        return 1
    tag = "[dry run] " if args.dry_run else ""
    if not rep.envs:
        sys.stdout.write("no environment to sync\n")
    for e in rep.envs:
        extra = f"  {e.copied_files} files, {portable_size(e.copied_bytes)}" if e.done else ""
        extra += f"  (old destination kept as {e.backup})" if e.backup else ""
        sys.stdout.write(f"{tag}{e.env:<24} {e.decision.action:<10} {e.decision.reason}{extra}\n")
    if rep.refused:
        sys.stderr.write("some environments were refused: no data was overwritten there\n")
        return 3
    return 0


def portable_size(n: float) -> str:
    from reid.storage.portable import human_size

    return human_size(n)


def _sync_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="sync_cmd", required=True)
    for name, text in (
        ("push", "copy local environments to the remote folder / rclone remote"),
        ("pull", "copy the remote's environments into the local workspace"),
    ):
        q = sub.add_parser(name, help=text, description=text)
        q.add_argument("--workspace", required=True, metavar="LOCAL_DIR")
        q.add_argument(
            "--remote",
            required=True,
            metavar="TARGET",
            help="a folder, or an rclone path like gdrive:reid_data/environments",
        )
        q.add_argument("--env", action="append", metavar="NAME", help="only this one (repeatable)")
        q.add_argument("--dry-run", action="store_true", help="show what would happen")
        q.add_argument(
            "--force",
            action="store_true",
            help="overwrite a changed destination (kept as .bak-<time>)",
        )
        q.add_argument(
            "--delete",
            action="store_true",
            help="move environments missing at the source aside (.bak-<time>)",
        )
        q.add_argument("--with-renders", action="store_true", help="also copy renders/")


cmd_sync.add_arguments = _sync_args  # type: ignore[attr-defined]


def _hardware_report(cfg: Config) -> str:  # [planner]
    """Hardware, what ``--preset auto`` picks here, and where this config's models would run."""
    from reid.models.planner import choose_preset, detect_hardware, plan_devices

    hw = detect_hardware()
    rt = cfg.runtime
    name, why = choose_preset(hw, rt.vram_budget_gb, rt.device_plan)
    plan = plan_devices(cfg, hw)
    lines = ["\n# hardware", hw.describe(), f"preset auto -> {name} ({why})", "# device plan"]
    lines += [f"  {x}" for x in plan.summary_lines()]
    lines += [f"  {a}" for a in plan.adjustments]
    return "\n".join(lines) + "\n"


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    sys.stdout.write("# config\n")
    sys.stdout.write(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    sys.stdout.write(_hardware_report(cfg))  # [planner]
    sys.stdout.write(f"\n# data_dir: {cfg.data_dir.resolve()}\n")
    sys.stdout.write(json.dumps(collect_info(cfg), indent=2) + "\n")
    return 0


COMMANDS: dict[str, tuple[Callable[[argparse.Namespace], int], str]] = {
    "process": (cmd_process, "process camera videos into tracklets and identities"),
    "render": (cmd_render, "render annotated videos from stored track points"),
    "search": (cmd_search, "search identities by body/face image"),
    "journey": (cmd_journey, "show or render the journey of one identity"),
    "evaluate": (cmd_evaluate, "evaluate against a labelled dataset"),
    "dataset": (cmd_dataset, "fetch labelled multi-camera footage (AI City 2024)"),
    "virtual-cams": (cmd_virtual_cams, "derive virtual cameras from one video"),
    "app": (cmd_app, "launch the Gradio app"),
    "env": (cmd_env, "list, export, import or relink environments (Colab to PC)"),
    "benchmark": (cmd_benchmark, "official AI City 2024 benchmark: run scenes, score with HOTA"),
    "sync": (cmd_sync, "sync environments with a Drive folder / rclone remote (PC <-> Colab)"),
    "info": (cmd_info, "print the effective config and data_dir statistics"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reid", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="YAML config merged over the defaults")
    common.add_argument(
        "--preset",
        help="model preset: auto|lite|balanced|accurate|max (or a YAML path); "
        "`process` and `app` default to auto (chosen from the GPU)",
    )
    common.add_argument("--data-dir", help="override paths.data_dir")
    common.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (func, help_) in COMMANDS.items():
        p = sub.add_parser(name, parents=[common], help=help_, description=help_)
        p.set_defaults(func=func)
        if hasattr(func, "add_arguments"):  # per-command options, see cmd_search
            func.add_arguments(p)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``reid`` console script."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NotImplementedError as exc:
        sys.stderr.write(f"reid {args.command}: not implemented yet ({exc})\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
