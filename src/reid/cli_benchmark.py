"""``reid benchmark aicity24 run|score``: the official AI City 2024 Track 1 benchmark.

``run`` fetches, processes, evaluates and exports whole 16-camera scenes (see
:mod:`reid.eval.aicity_bench`); ``score`` runs NVIDIA's official HOTA evaluator on what ``run``
produced (:mod:`reid.eval.aicity_score`). Both can be repeated: finished work is skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _scenes(value: str) -> list[str]:
    from reid.datasets.aicity24 import parse_scene_id

    return [parse_scene_id(s) for s in value.split(",") if s.strip()]


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Dispatch ``reid benchmark aicity24 run|score``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if args.bench_cmd == "run":
        return _run(args)
    return _score(args)


def _run(args: argparse.Namespace) -> int:
    from reid.eval.aicity_bench import BenchOptions, run_scenes

    overrides = {}
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    sub = "smoke" if args.smoke else ""
    ws = Path(args.workspace) / sub
    opts = BenchOptions(
        workspace=ws,
        raw_dir=(Path(args.raw_dir) / sub) if args.raw_dir else ws / ".raw",
        preset=args.preset,
        frame_stride=args.frame_stride,
        overrides=overrides,
        config_path=args.config,
        keep_videos=args.keep_videos,
        eval_2d=not args.no_eval_2d,
        pack=not args.no_pack,
        smoke=args.smoke,
        job=args.job,
    )
    statuses = run_scenes(args.scenes[:1] if args.smoke else args.scenes, opts)
    if args.smoke:
        from reid.eval.aicity_projection import finish_smoke

        finish_smoke(ws, args.preset, args.scenes, args.project_gpus, args.num_cores)
    summary = ", ".join(f"{s} {st}" for s, st in statuses.items())
    sys.stdout.write(f"benchmark run finished ({summary}) in {ws}\n")
    # a failed scene is recorded in benchmark.json; only a run where every scene failed is an error
    return 0 if any(st in ("ok", "skipped") for st in statuses.values()) else 1


def _score(args: argparse.Namespace) -> int:
    from reid.eval.aicity_score import score_workspace

    try:
        res = score_workspace(
            args.workspace, args.preset, args.num_cores, per_scene=not args.no_per_scene
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"reid benchmark aicity24 score: {exc}\n")
        return 1
    o = res["overall"]
    sys.stdout.write(
        f"overall HOTA {o['HOTA']:.2f}  DetA {o['DetA']:.2f}  AssA {o['AssA']:.2f}  "
        f"LocA {o['LocA']:.2f}\nwrote {Path(args.workspace) / f'RESULTS_{args.preset}.md'}\n"
    )
    return 0


def _benchmark_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="dataset", required=True)
    ds = sub.add_parser("aicity24", help="AI City 2024 MTMC (NVIDIA PhysicalAI-SmartSpaces)")
    cmds = ds.add_subparsers(dest="bench_cmd", required=True)
    r = cmds.add_parser("run", help="fetch, process, evaluate and export whole scenes")
    r.add_argument("--scenes", required=True, type=_scenes, help="comma list, e.g. 075,078,082")
    r.add_argument("--workspace", required=True, metavar="DIR", help="environments go here")
    r.add_argument("--preset", default="balanced", help="model preset (default balanced)")
    r.add_argument("-c", "--config", help="YAML config merged over the preset")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value")
    r.add_argument("--raw-dir", metavar="DIR", help="downloaded scenes (default WORKSPACE/.raw)")
    r.add_argument("--keep-videos", choices=("720", "none"), default="720")
    r.add_argument("--frame-stride", type=int, default=2, help="process every n-th frame")
    r.add_argument("--no-eval-2d", action="store_true", help="skip the 2D metrics and retrieval")
    r.add_argument("--no-pack", action="store_true", help="keep environment folders (no zip)")
    r.add_argument("--job", default="run", help="job name (progress_<job>.json)")
    r.add_argument(
        "--smoke", action="store_true",
        help="first scene only, 2 cameras, 2 minutes, into WORKSPACE/smoke; then score it and "
        "print a projection of the full run",
    )  # fmt: skip
    r.add_argument("--project-gpus", type=int, default=1, help="GPUs of the run to project")
    r.add_argument("--num-cores", type=int, default=4, help="cores for the smoke scoring")
    s = cmds.add_parser("score", help="official HOTA evaluation of the processed scenes")
    s.add_argument("--workspace", required=True, metavar="DIR")
    s.add_argument("--preset", default="balanced")
    s.add_argument("--num-cores", type=int, default=4, help="cores for the official evaluator")
    s.add_argument("--no-per-scene", action="store_true", help="only the overall run")


cmd_benchmark.add_arguments = _benchmark_args  # type: ignore[attr-defined]
