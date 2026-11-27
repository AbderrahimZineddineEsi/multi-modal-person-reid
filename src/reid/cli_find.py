"""``reid find``: attribute filter search of an environment, JSON on stdout (like ``reid search``).

    reid find --data-dir ENV --attr female --any upper_red,upper_pink --attr backpack=no --strict

``--attr LABEL`` requires a label, ``--attr LABEL=no`` forbids it, ``--any A,B`` needs at least one
of the labels. Results are people (best appearance each, with all their appearances) or single
appearances (``--group appearance``), best match first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Any

from reid import find
from reid.storage import db, files


def _parse_attrs(items: list[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for item in items:
        label, sep, val = item.partition("=")
        if sep and val.lower() not in ("no", "false", "0", "not"):
            if val.lower() not in ("yes", "true", "1"):
                raise ValueError(f"--attr expects LABEL or LABEL=no, got {item!r}")
        out[label.strip()] = not (sep and val.lower() in ("no", "false", "0", "not"))
    return out


def _hit_json(h: find.Hit, labels: list[str], shown: list[str]) -> dict[str, Any]:
    idx = {n: i for i, n in enumerate(labels)}
    return {
        "tracklet_id": h.tracklet_id,
        "gid": h.gid,
        "camera": h.camera_id,
        "start_ts": round(h.start_ts, 2),
        "end_ts": round(h.end_ts, 2),
        "match_pct": round(h.match_pct, 1),
        "passes": h.passes,
        "n_samples": h.n_samples,
        "crop": h.crop_path,
        "probs": {n: round(float(h.probs[idx[n]]), 3) for n in shown},
    }


def cmd_find(args: argparse.Namespace) -> int:
    """``reid find``: rank annotated tracklets by attributes and print the top results."""
    from reid.cli import _build_config

    cfg = _build_config(args)
    query = find.FindQuery(
        cameras=args.camera or None,
        t_from=args.t_from,
        t_to=args.t_to,
        require=_parse_attrs(args.attr or []),
        any_of=[[x.strip() for x in g.split(",") if x.strip()] for g in args.any or []],
        strict=args.strict,
    )
    conn = db.connect(files.db_path(cfg.data_dir))
    try:
        res = find.find(conn, query)
    except ValueError as exc:
        sys.stderr.write(f"reid find: {exc}\n")
        return 2
    finally:
        conn.close()
    if res.head is None:
        sys.stderr.write(
            "reid find: no attributes in this environment (reid attributes backfill)\n"
        )
        return 1
    labels = res.head.labels
    shown = list(dict.fromkeys([*query.require, *(x for g in query.any_of for x in g)]))
    if args.group == "person":
        people = find.group_by_person(res.hits)[: args.top]
        results = [
            {
                "gid": p.gid,
                "match_pct": round(p.best.match_pct, 1),
                "cameras": p.cameras,
                "n_appearances": len(p.hits),
                "n_matching_appearances": sum(h.passes for h in p.hits),
                "best": _hit_json(p.best, labels, shown),
                "appearances": [_hit_json(h, labels, shown) for h in p.hits if h.passes][:20],
            }
            for p in people
        ]
        n_people = len(find.group_by_person([h for h in res.hits if h.passes]))
    else:
        results = [_hit_json(h, labels, shown) for h in res.hits[: args.top]]
        n_people = None
    out = {
        "head": dataclasses.asdict(res.head) | {"thresholds": None, "labels": len(labels)},
        "query": dataclasses.asdict(query),
        "total_tracklets": res.total_tracklets,
        "n_in_scope": res.n_in_scope,
        "n_matching": res.n_matching,
        "n_people_matching": n_people,
        "group": args.group,
        "results": results,
    }
    sys.stdout.write(json.dumps(out, indent=2, default=str) + "\n")
    return 0


def _find_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", action="append", metavar="ID", help="only this camera (repeatable)")
    p.add_argument("--from", dest="t_from", type=float, metavar="S", help="from (epoch seconds)")
    p.add_argument("--to", dest="t_to", type=float, metavar="S", help="to (epoch seconds)")
    p.add_argument("--attr", action="append", metavar="LABEL[=no]", help="require / forbid a label")
    p.add_argument("--any", action="append", metavar="A,B", help="at least one of these labels")
    p.add_argument("--strict", action="store_true", help="only people passing every threshold")
    p.add_argument("--group", choices=("person", "appearance"), default="person")
    p.add_argument("--top", type=int, default=10, help="results to print (default 10)")


cmd_find.add_arguments = _find_args  # type: ignore[attr-defined]
