"""``reid attributes fetch|embed|train|report``: the pedestrian attribute head.

``fetch`` downloads PA-100K and PETA; ``embed`` embeds every image once with the frozen body
ReID model (resumable shards); ``train`` fits the MLP head on the stored embeddings; ``report``
evaluates it on the test splits (see :mod:`reid.attributes`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def cmd_attributes(args: argparse.Namespace) -> int:
    """Dispatch ``reid attributes fetch|embed|train|report``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return {"fetch": _fetch, "embed": _embed, "train": _train, "report": _report}[args.attr_cmd](
        args
    )


def _fetch(args: argparse.Namespace) -> int:
    from reid.datasets.par_fetch import fetch

    out = fetch(args.out, args.from_dir)
    sys.stdout.write(f"attribute datasets ready in {out}\n")
    return 0


def _embed(args: argparse.Namespace) -> int:
    from reid.attributes.datasets import load_pa100k, load_peta
    from reid.attributes.embed import build_embedder, embed_datasets

    data = Path(args.data)
    loaders = {"pa100k": load_pa100k, "peta": load_peta}
    datasets = {}
    for name, loader in loaders.items():
        root = data / name if (data / name).is_dir() else data
        try:
            datasets[name] = loader(root)
        except FileNotFoundError as exc:
            logging.getLogger(__name__).warning("%s skipped: %s", name, exc)
    if not datasets:
        sys.stderr.write(f"reid attributes embed: no PA-100K or PETA found under {data}\n")
        return 1
    try:
        i, n = (int(v) for v in args.part.split("/"))
        if not 0 <= i < n:
            raise ValueError
    except ValueError:
        sys.stderr.write(
            f"reid attributes embed: --part must be I/N with 0 <= I < N, got {args.part}\n"
        )
        return 2
    emb = build_embedder(args.preset, args.batch)
    stats = embed_datasets(
        emb, datasets, args.out, batch=args.batch, workers=args.workers,
        shard_size=args.shard_size, part=(i, n),
    )  # fmt: skip
    sys.stdout.write(f"embedded {int(stats['items'])} items ({stats['img_per_s']:.1f} img/s)\n")
    return 0


def _train(args: argparse.Namespace) -> int:
    from reid.attributes.train import TrainConfig, run_train

    cfg = TrainConfig(
        hidden=args.hidden, dropout=args.dropout, lr=args.lr, wd=args.wd, batch=args.batch,
        epochs=args.epochs, patience=args.patience, peta_weight=args.peta_weight, seed=args.seed,
    )  # fmt: skip
    path = run_train(args.emb, args.out, cfg)
    sys.stdout.write(f"wrote {path}\n")
    return 0


def _report(args: argparse.Namespace) -> int:
    from reid.attributes.report import run_report

    try:
        path = run_report(args.emb, args.head, args.out, args.tag)
    except FileNotFoundError as exc:
        sys.stderr.write(f"reid attributes report: {exc}\n")
        return 1
    sys.stdout.write(f"wrote {path}\n")
    return 0


def _attributes_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="attr_cmd", required=True)
    f = sub.add_parser("fetch", help="download PA-100K and PETA (no i-LID)")
    f.add_argument("--out", required=True, metavar="DIR")
    f.add_argument("--from-dir", metavar="DIR", help="reuse zips/folders already present")
    e = sub.add_parser("embed", help="embed every image once (resumable)")
    e.add_argument("--data", required=True, metavar="DIR", help="the `fetch --out` folder")
    e.add_argument("--out", required=True, metavar="DIR", help="embeddings folder")
    e.add_argument("--preset", default="balanced", help="model preset (default balanced)")
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--workers", type=int, default=None, help="data workers (default min(4, cpus))")
    e.add_argument("--shard-size", type=int, default=4096, help="items per shard file")
    e.add_argument(
        "--part", default="0/1", metavar="I/N",
        help="embed only shards k with k %% N == I (one process per GPU); finish with 0/1",
    )  # fmt: skip
    t = sub.add_parser("train", help="train the MLP head on stored embeddings")
    t.add_argument("--emb", required=True, metavar="DIR")
    t.add_argument("--out", required=True, metavar="DIR", help="writes head.npz here")
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--peta-weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report", help="test-set metrics, per-label table, voting curve")
    r.add_argument("--emb", required=True, metavar="DIR")
    r.add_argument("--head", required=True, metavar="PATH", help="head.npz or its folder")
    r.add_argument("--out", required=True, metavar="DIR")
    r.add_argument("--tag", default="default")


cmd_attributes.add_arguments = _attributes_args  # type: ignore[attr-defined]
