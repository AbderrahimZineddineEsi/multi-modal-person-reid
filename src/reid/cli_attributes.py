"""``reid attributes fetch|embed|train|report|hpo|hpo-final|eval-external``: attribute head.

``fetch`` downloads PA-100K and PETA; ``embed`` embeds every image once with the frozen body
ReID model (resumable shards); ``train`` fits the MLP head on the stored embeddings; ``report``
evaluates it on the test splits; ``hpo`` searches the head's hyperparameters with Optuna and
``hpo-final`` compares default and best on several seeds (see :mod:`reid.attributes`).
``use-head`` installs a trained head where the pipeline finds it and ``backfill`` annotates the
tracklets of existing environments with it (:mod:`reid.attributes.annotate`). ``eval-external``
scores the head zero-shot on a dataset it never saw (:mod:`reid.attributes.external_run`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def cmd_attributes(args: argparse.Namespace) -> int:
    """Dispatch ``reid attributes <fetch|embed|train|report|hpo|...|eval-external>``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    cmds = {
        "fetch": _fetch, "embed": _embed, "train": _train, "report": _report,
        "hpo": _hpo, "hpo-final": _hpo_final, "use-head": _use_head, "backfill": _backfill,
        "eval-external": _eval_external,
    }  # fmt: skip
    return cmds[args.attr_cmd](args)


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
        weighting=args.weighting, use_flips=args.use_flips, train_view=args.train_view,
    )  # fmt: skip
    path = run_train(args.emb, args.out, cfg)
    sys.stdout.write(f"wrote {path}\n")
    return 0


def _hpo(args: argparse.Namespace) -> int:
    from reid.attributes.hpo import run_hpo

    try:
        study = run_hpo(
            args.emb, args.out, name=args.study, trials=args.trials,
            timeout_min=args.timeout_min, seed=args.seed, worker=args.worker,
        )  # fmt: skip
    except (ImportError, FileNotFoundError) as exc:
        sys.stderr.write(f"reid attributes hpo: {exc}\n")
        return 1
    sys.stdout.write(f"study {args.study}: {len(study.trials)} trials in {args.out}\n")
    return 0


def _hpo_final(args: argparse.Namespace) -> int:
    from reid.attributes.hpo_report import run_hpo_final

    try:
        path = run_hpo_final(args.emb, args.out, args.study, args.seeds)
    except (ImportError, FileNotFoundError, RuntimeError, KeyError) as exc:
        sys.stderr.write(f"reid attributes hpo-final: {exc}\n")
        return 1
    sys.stdout.write(f"wrote {path}\n")
    return 0


def _use_head(args: argparse.Namespace) -> int:
    import json
    import shutil

    from reid.attributes.annotate import default_head_path
    from reid.attributes.head import head_file, load_head
    from reid.attributes.ui_labels import RELIABLE
    from reid.cli import _build_config
    from reid.storage.attributes_db import head_sha256

    src = head_file(args.path)
    try:
        head = load_head(src)
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"reid attributes use-head: cannot read {src}: {exc}\n")
        return 1
    dst = default_head_path(_build_config(args))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    trained = [n for n, t in zip(head.labels, head.trained_labels, strict=True) if t]
    summary = {
        "installed": str(dst),
        "sha256": head_sha256(head)[:16],
        "body_model": head.body_model,
        "neck_feat": head.neck_feat,
        "labels": len(head.labels),
        "trained_labels": len(trained),
        "reliable_in_ui": len([n for n in trained if n in RELIABLE]),
        "best_val_mA": head.hparams.get("best_val_ma"),
    }
    metrics = sorted(src.parent.parent.glob("report/metrics_*.json"))
    if metrics:
        for name, res in json.loads(metrics[0].read_text()).get("datasets", {}).items():
            summary[f"test_mA_{name}"] = round(float(res["ma"]), 4)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(
        "next: reid attributes backfill --workspace <WS>   (annotates existing data)\n"
    )
    return 0


def _backfill(args: argparse.Namespace) -> int:
    from reid.attributes.annotate import backfill_env, backfill_workspace, load_configured_head
    from reid.cli import _build_config

    cfg = _build_config(args)
    head = load_configured_head(cfg)
    if head is None:
        sys.stderr.write("reid attributes backfill: no head installed (reid attributes use-head)\n")
        return 1
    if args.workspace:
        results = backfill_workspace(Path(args.workspace).expanduser(), head, args.force)
    else:
        results = [backfill_env(cfg.data_dir, head, args.force)]
    for r in results:
        note = f"  ({r.reason})" if r.reason else ""
        sys.stdout.write(
            f"{r.env:<28} {r.state:<10} +{r.annotated} annotated, {r.total}/{r.n_tracklets} "
            f"tracklets, {r.seconds:.1f}s{note}\n"
        )
    sys.stdout.write(f"{sum(r.seconds for r in results):.1f}s total\n")
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


def _eval_external(args: argparse.Namespace) -> int:
    from reid.attributes.annotate import configured_head_path
    from reid.attributes.external_run import run_eval_external
    from reid.cli import _build_config

    head = args.head or configured_head_path(_build_config(args))
    try:
        path = run_eval_external(
            args.dataset, args.root, head, args.out, emb_dir=args.emb,
            in_domain_path=args.in_domain, attribute_file=args.attribute_file,
            repeats=args.repeats, min_pos=args.min_pos, seed=args.seed, preset=args.preset,
            batch=args.batch, workers=args.workers,
        )  # fmt: skip
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        sys.stderr.write(f"reid attributes eval-external: {exc}\n")
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
    t.add_argument("--weighting", choices=("ratio", "none"), default="ratio",
                   help="ratio = exp(1-r)/exp(r) label weights; none = all 1")  # fmt: skip
    t.add_argument("--use-flips", action=argparse.BooleanOptionalAction, default=True,
                   help="train on the flipped copies too (default yes)")  # fmt: skip
    t.add_argument("--train-view", action=argparse.BooleanOptionalAction, default=True,
                   help="train the view_* labels (default yes)")  # fmt: skip
    h = sub.add_parser("hpo", help="Optuna search of the head's hyperparameters (needs [train])")
    h.add_argument("--emb", required=True, metavar="DIR")
    h.add_argument("--out", required=True, metavar="DIR", help="journal, study and results here")
    h.add_argument("--study", default="attr_hpo")
    h.add_argument("--trials", type=int, default=100, help="trials of THIS process")
    h.add_argument("--timeout-min", type=float, default=None, help="minutes, this process")
    h.add_argument("--seed", type=int, default=0, help="sampler seed (+ --worker)")
    h.add_argument("--worker", type=int, default=0, help="process index (one per GPU)")
    hf = sub.add_parser("hpo-final", help="default vs best on several seeds, test read here")
    hf.add_argument("--emb", required=True, metavar="DIR")
    hf.add_argument("--out", required=True, metavar="DIR", help="the `hpo --out` folder")
    hf.add_argument("--study", default="attr_hpo")
    hf.add_argument("--seeds", type=int, default=3)
    u = sub.add_parser("use-head", help="install a trained head where the pipeline finds it")
    u.add_argument("path", metavar="PATH", help="head.npz or its folder")
    b = sub.add_parser("backfill", help="annotate the tracklets of existing environments")
    b.add_argument("--workspace", metavar="DIR", help="every environment of this workspace")
    b.add_argument("--data-dir", default=argparse.SUPPRESS, metavar="DIR", help="one environment")
    b.add_argument("--force", action="store_true", help="recompute tracklets already annotated")
    x = sub.add_parser("eval-external", help="zero-shot test of the head on another dataset")
    x.add_argument("--dataset", required=True, choices=("market1501",))
    x.add_argument("--root", required=True, metavar="DIR", help="unzipped dataset + attribute file")
    x.add_argument(
        "--attribute-file", metavar="FILE", help="market_attribute.mat (if not in --root)"
    )
    x.add_argument(
        "--head", metavar="PATH", help="head.npz or folder (default: the installed head)"
    )
    x.add_argument("--out", default="data/attributes/external", metavar="DIR")
    x.add_argument("--emb", metavar="DIR", help="embeddings folder (default <out>/emb)")
    x.add_argument("--in-domain", metavar="JSON", help="metrics_<tag>.json of the head's report")
    x.add_argument("--repeats", type=int, default=20, help="random draws of the identity vote")
    x.add_argument("--min-pos", type=int, default=30, help="min positive AND negative identities")
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--preset", default="balanced", help="model preset (default balanced)")
    x.add_argument("--batch", type=int, default=64)
    x.add_argument("--workers", type=int, default=None, help="data workers (default min(4, cpus))")
    r = sub.add_parser("report", help="test-set metrics, per-label table, voting curve")
    r.add_argument("--emb", required=True, metavar="DIR")
    r.add_argument("--head", required=True, metavar="PATH", help="head.npz or its folder")
    r.add_argument("--out", required=True, metavar="DIR")
    r.add_argument("--tag", default="default")


cmd_attributes.add_arguments = _attributes_args  # type: ignore[attr-defined]
