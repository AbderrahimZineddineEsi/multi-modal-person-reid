"""``reid attributes eval-external``: embed an external dataset once, then score the head on it.

Zero-shot, evaluation only: the head is never trained or tuned on the external dataset. The
embeddings use the same frozen backbone and the same code as the training sets
(:func:`reid.attributes.embed.embed_split`, no flipped copies for a test split) and are stored in
the same layout (``<emb>/<dataset>/test``), resumable, so a crash or a second run re-embeds
nothing. Writes ``metrics_<dataset>.json`` and ``RESULTS_<dataset>.md``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reid.attributes import store
from reid.attributes.datasets import ParSplit
from reid.attributes.datasets_market import load_market1501
from reid.attributes.embed import EtaLog, embed_split, pending_items
from reid.attributes.external import MIN_POS, REPEATS, evaluate_external, in_domain
from reid.attributes.external_md import render
from reid.attributes.head import HeadParams, head_file, load_head, predict

log = logging.getLogger(__name__)

LOADERS: dict[str, Callable[..., ParSplit]] = {"market1501": load_market1501}


def find_in_domain_metrics(head_path: str | Path) -> Path | None:
    """``report/metrics_*.json`` next to a ``<run>/head/head.npz`` (the ``train`` layout)."""
    run = head_file(head_path).resolve().parent.parent
    found = sorted(run.glob("report/metrics_*.json"))
    return found[0] if found else None


def ensure_embedded(
    sp: ParSplit,
    head: HeadParams,
    emb_dir: Path,
    *,
    embedder: Any = None,
    preset: str = "balanced",
    batch: int = 64,
    workers: int | None = None,
    shard_size: int = store.DEFAULT_SHARD,
) -> None:
    """Embed ``sp`` unless a finished, matching embedding is already in ``emb_dir``."""
    sdir = store.split_dir(emb_dir, sp.dataset, sp.split)
    old = store.read_index(sdir)
    if (
        old
        and old.get("complete")
        and old.get("images") == sp.images
        and old.get("model") == head.body_model
        and old.get("neck_feat") == head.neck_feat
        and pending_items(sp, emb_dir, old["shard_size"]) == 0
    ):
        log.info("%s/%s: embeddings already complete, skipping the model", sp.dataset, sp.split)
        return
    if embedder is None:
        from reid.attributes.embed import build_embedder

        embedder = build_embedder(preset, batch)
    if (embedder.name, embedder.neck_feat) != (head.body_model, head.neck_feat):
        raise ValueError(
            f"head was trained on {head.body_model}/{head.neck_feat}, "
            f"the embedder is {embedder.name}/{embedder.neck_feat}"
        )
    eta = EtaLog(pending_items(sp, emb_dir, shard_size))
    embed_split(embedder, sp, emb_dir, batch=batch, workers=workers, shard_size=shard_size, eta=eta)


def run_eval_external(
    dataset: str,
    root: str | Path,
    head_path: str | Path,
    out: str | Path,
    *,
    emb_dir: str | Path | None = None,
    in_domain_path: str | Path | None = None,
    attribute_file: str | Path | None = None,
    repeats: int = REPEATS,
    min_pos: int = MIN_POS,
    seed: int = 0,
    embedder: Any = None,
    **embed_kwargs: Any,
) -> Path:
    """Embed (if needed) and evaluate; returns the path of ``RESULTS_<dataset>.md``."""
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(LOADERS)}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(emb_dir) if emb_dir else out / "emb"
    head = load_head(head_path)
    sp = LOADERS[dataset](root, attribute_file)
    ensure_embedded(sp, head, emb_dir, embedder=embedder, **embed_kwargs)
    se = store.load_split(emb_dir, dataset, "test", include_flips=False)
    if se.x.shape[1] != head.in_dim:
        raise ValueError(f"embedding dim {se.x.shape[1]} != head input {head.in_dim}")
    res = evaluate_external(
        head,
        predict(head, se.x),
        se.targets,
        se.persons.tolist(),
        repeats=repeats,
        min_pos=min_pos,
        seed=seed,
    )
    ref_path = Path(in_domain_path) if in_domain_path else find_in_domain_metrics(head_path)
    ref = in_domain(json.loads(ref_path.read_text()), res["mapped_labels"]) if ref_path else {}
    from reid.storage.attributes_db import head_sha256

    meta = {
        "dataset": dataset, "head": str(head_file(head_path)), "head_sha256": head_sha256(head),
        "body_model": head.body_model, "neck_feat": head.neck_feat,
        "in_domain_source": str(ref_path) if ref_path else None,
        "unreadable_images": se.index.get("failed", 0), "seed": seed,
    }  # fmt: skip
    (out / f"metrics_{dataset}.json").write_text(
        json.dumps({"meta": meta, "result": res, "in_domain": ref}, indent=1)
    )
    path = out / f"RESULTS_{dataset}.md"
    path.write_text(render(res, ref, meta))
    return path
