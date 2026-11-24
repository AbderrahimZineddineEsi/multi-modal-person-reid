"""Database -> display models: the glue between ``reid.analytics`` and ``components``."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from reid import analytics
from reid.app import components as C
from reid.app.media import thumb_uri
from reid.app.state import AppContext
from reid.storage import db, files


def crop_rel(ctx: AppContext, t) -> str:
    """Stored best-crop path of a tracklet (relative to data_dir), with the canonical fallback."""
    return t.best_crop_path or crop_rel_id(ctx, t.id)


def crop_rel_id(ctx: AppContext, tracklet_id: int) -> str:
    return files.crop_path(ctx.data_dir, tracklet_id).relative_to(ctx.data_dir).as_posix()


def best_crop_path(ctx: AppContext, tracklets: list) -> Path | None:
    """Crop file of the person's highest-quality tracklet that has one on disk."""
    for t in sorted(tracklets, key=lambda x: -x.mean_quality):
        p = ctx.data_dir / crop_rel(ctx, t)
        if p.exists():
            return p
    return None


def _by_gid(conn: sqlite3.Connection) -> dict[int, list]:
    by_gid: dict[int, list] = {}
    for t in db.get_tracklets(conn):
        if t.gid is not None:
            by_gid.setdefault(t.gid, []).append(t)
    return by_gid


def gallery_items(ctx: AppContext, conn: sqlite3.Connection) -> tuple[list, list[int]]:
    """(``gr.Gallery`` items, gids): best crop with a caption for every person that has one."""
    by_gid = _by_gid(conn)
    names = person_names(conn)
    items, gids = [], []
    for gid in sorted(by_gid):
        p = best_crop_path(ctx, by_gid[gid])
        if p is not None:
            items.append((str(p), names.get(gid, f"Person {gid}")))
            gids.append(gid)
    return items, gids


def people_cards(ctx: AppContext, conn: sqlite3.Connection) -> list[C.PersonCard]:
    """One card per identity, ordered by gid."""
    by_gid = _by_gid(conn)
    cards = []
    for ident in db.list_identities(conn):
        ts = by_gid.get(ident.gid, [])
        cards.append(
            C.PersonCard(
                gid=ident.gid,
                label=ident.label,
                n_tracklets=len(ts),
                n_cameras=len({t.camera_id for t in ts}),
                duration_s=sum(t.end_ts - t.start_ts for t in ts),
                has_face=ident.has_face,
                thumb=thumb_uri(best_crop_path(ctx, ts), ctx.data_dir),
            )
        )
    return cards


def person_choices(cards: list[C.PersonCard]) -> list[tuple[str, int]]:
    return [(f"{c.name} (#{c.gid})" if c.label else f"Person {c.gid}", c.gid) for c in cards]


def person_names(conn: sqlite3.Connection) -> dict[int, str]:
    return {i.gid: i.label or f"Person {i.gid}" for i in db.list_identities(conn)}


def journey_steps(ctx: AppContext, conn: sqlite3.Connection, gid: int) -> list[C.StepView]:
    return [
        C.StepView(
            s.camera_name, s.start_ts, s.end_ts, s.gap_s, s.assign_source,
            thumb_uri(s.best_crop_path or crop_rel_id(ctx, s.tracklet_id), ctx.data_dir, 120),
        )
        for s in analytics.journey(conn, gid)
    ]  # fmt: skip


def camera_rows(conn: sqlite3.Connection) -> list[C.CameraRow]:
    seen: dict[str, set[int]] = {}
    for t in db.get_tracklets(conn):
        if t.gid is not None:
            seen.setdefault(t.camera_id, set()).add(t.gid)
    return [
        C.CameraRow(
            c.name, f"{c.width}x{c.height} @ {c.fps:g} fps", len(seen.get(c.id, ())), c.is_face_cam
        )
        for c in db.list_cameras(conn)
    ]


def vram_now() -> tuple[float, float] | None:
    """(used, total) GPU memory in GiB via torch, or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**30, total / 2**30
    except ImportError:
        return None


def summary_tiles(summary: dict[str, Any], stats: dict[str, Any] | None = None) -> str:
    """KPI tiles for a ``run_summary`` (plus fps and VRAM when run statistics exist)."""
    n_people = int(summary["n_people"])
    items = [
        C.metric_tile("People", str(n_people), f"{summary['n_tracklets']} sightings in total"),
        C.metric_tile(
            "Identified", f"{summary['identified_pct']:.0f}%", "sightings with a person",
            summary["identified_pct"],
        ),
        C.metric_tile(
            "Face confirmed", f"{summary['face_confirmed_pct']:.0f}%", "people with a face seen",
            summary["face_confirmed_pct"],
        ),
        C.metric_tile(
            "Sightings per person", f"{summary['mean_tracklets_per_identity']:.1f}",
            "lower means less fragmentation",
        ),
    ]  # fmt: skip
    if stats:
        need = (stats.get("gpu_memory") or {}).get("required_mb")
        peak = need or (stats.get("vram_peak_mb") or {}).get("total_peak")
        items.append(
            C.metric_tile(
                "Speed",
                f"{stats.get('fps', 0):.1f} fps",
                "frames per second, all stages",
                key=True,
            )
        )
        if peak:
            items.append(
                C.metric_tile(
                    "GPU memory",
                    f"{peak / 1024:.2f} GB",
                    "needed: whole-process peak" if need else "PyTorch peak only (older run)",
                    key=True,
                )
            )
    return C.tiles(items)
