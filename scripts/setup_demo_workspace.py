"""Build ``data/demo_workspace``: existing environments, annotated with the attribute head.

One command to get a workspace the Find tab can be tried on, from data that is already on this
machine (nothing is re-processed and the original folders are never modified):

1. imports the Kaggle AI City environment zips (``reid env import``: videos come with the zip);
2. copies the warehouse showcase environment (its videos stay where they are);
3. installs the attribute head (``reid attributes use-head``);
4. annotates every environment (``reid attributes backfill``; seconds each, no GPU);
5. prints the command that starts the app.

Idempotent: environments already in the workspace are kept (``--force`` re-imports them), the
head is only copied when it changed and tracklets already annotated are skipped. Example::

    python scripts/setup_demo_workspace.py
    python scripts/setup_demo_workspace.py --workspace data/demo_workspace --head path/to/head.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from reid import cli
from reid.attributes.annotate import backfill_workspace, load_configured_head
from reid.config import load_config
from reid.storage import portable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEAD = ROOT / "data/attributes/kaggle_default/attributes/head"
DEFAULT_ZIPS = "data/kaggle_runs/*/aicity24_bench/envs/aicity24_*_balanced.zip"
DEFAULT_WAREHOUSE = ROOT / "data/showcase/public/warehouseA"
_COPY_SKIP = shutil.ignore_patterns("renders", "*.tmp", "reid.db-wal", "reid.db-shm")
log = logging.getLogger("setup_demo_workspace")


def import_zips(zips: list[Path], workspace: Path, force: bool) -> list[str]:
    """Import every zip not yet in the workspace; returns the names present afterwards."""
    names = []
    for z in zips:
        name = z.stem
        if (workspace / name).is_dir() and not force:
            log.info("%s: already imported", name)
        else:
            t0 = time.perf_counter()
            r = portable.import_env(z, workspace, name=name, force=force)
            log.info(
                "%s: imported in %.0fs (%d cameras, %d videos relinked, %d missing)",
                name, time.perf_counter() - t0, r.cameras, r.videos_relinked, len(r.videos_missing),
            )  # fmt: skip
        names.append(name)
    return names


def copy_showcase(src: Path, workspace: Path, force: bool) -> str | None:
    """Copy a plain data_dir (no zip) into the workspace as an environment (renders skipped)."""
    if not (src / "reid.db").is_file():
        log.warning("%s has no reid.db: skipped", src)
        return None
    dest = workspace / src.name
    if dest.is_dir() and not force:
        log.info("%s: already copied", src.name)
        return src.name
    if dest.is_dir():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=_COPY_SKIP)
    marker = dest / "env.json"
    if not marker.exists():
        marker.write_text(json.dumps({"name": src.name, "description": "copied by setup script"}))
    log.info("%s: copied from %s", src.name, src)
    return src.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workspace", type=Path, default=ROOT / "data/demo_workspace")
    ap.add_argument("--head", type=Path, default=DEFAULT_HEAD, help="head.npz or its folder")
    ap.add_argument("--zips", default=DEFAULT_ZIPS, help="glob of environment zips (from ROOT)")
    ap.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE, help="plain env to copy")
    ap.add_argument("--no-warehouse", action="store_true")
    ap.add_argument("--weights-dir", default="", help="weights cache (default ~/.cache/reid)")
    ap.add_argument("--force", action="store_true", help="re-import / re-copy / re-annotate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    zips = sorted(Path(p) for p in glob.glob(str(ROOT / args.zips)))
    if not zips:
        log.warning("no zip matches %s", args.zips)
    t0 = time.perf_counter()
    import_zips(zips, workspace, args.force)
    if not args.no_warehouse:
        copy_showcase(args.warehouse, workspace, args.force)

    wd = ["--set", f"paths.weights_dir={args.weights_dir}"] if args.weights_dir else []
    if cli.main(["attributes", *wd, "use-head", str(args.head)]) != 0:
        return 1
    head = load_configured_head(load_config(None, {"paths.weights_dir": args.weights_dir}))
    if head is None:
        log.error("the installed head cannot be loaded")
        return 1
    t_ann = time.perf_counter()
    results = backfill_workspace(workspace, head, args.force)
    ann_s = time.perf_counter() - t_ann
    sys.stdout.write("\nenvironment                  state      annotated  tracklets  time\n")
    for r in results:
        sys.stdout.write(
            f"{r.env:<28} {r.state:<10} {r.total:>9}  {r.n_tracklets:>9}  {r.seconds:>5.1f}s"
            f"  {r.reason}\n"
        )
    sys.stdout.write(
        f"\nannotation {ann_s:.1f}s, whole setup {time.perf_counter() - t0:.0f}s\n"
        f"\nstart the app with:\n    reid app --workspace {workspace}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
