"""Data side of the Find tab: the cached attribute table, the query and the people to show.

No Gradio and no HTML here. The attribute table of the selected environment is loaded once and
kept in ``ctx.find_cache``; its key holds the identity fingerprint and the number of annotation
rows, so an identity edit (merge, split, delete) or a new annotation rebuilds it by itself and an
environment switch simply misses the cache.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from reid.app.state import AppContext
from reid.attributes import ui_labels
from reid.attributes.annotate import EnvStatus, backfill_env, env_status, load_configured_head
from reid.find import (
    AttributeTable,
    FindQuery,
    FindResult,
    PersonHit,
    find,
    group_by_person,
    load_table,
)

PAGE = 24  # people (or appearances) shown before "Show more"
VIEWS = ("People", "Appearances")


@dataclass(frozen=True)
class Selection:
    """Everything the filter panel says, in plain values."""

    cameras: tuple[str, ...] = ()  # empty = all cameras
    t_from: float | None = None
    t_to: float | None = None
    chips: tuple[str, ...] = ()  # UI ids (``ui_labels``)
    strict: bool = False
    view: str = "People"

    @property
    def has_terms(self) -> bool:
        return bool(self.chips)


@dataclass
class Outcome:
    """What one query produced; ``result`` is None unless the environment is searchable."""

    status: EnvStatus
    result: FindResult | None = None
    people: list[PersonHit] = field(default_factory=list)
    table: AttributeTable | None = None

    @property
    def ready(self) -> bool:
        return self.result is not None and self.result.head is not None


def get_table(ctx: AppContext, conn: sqlite3.Connection) -> AttributeTable | None:
    """The environment's attribute table, rebuilt only when its annotations or gids changed."""
    n_rows = int(conn.execute("SELECT COUNT(*) FROM tracklet_attributes").fetchone()[0])
    key = (str(ctx.data_dir), ctx.fingerprint(conn), n_rows)
    cached = ctx.find_cache.get("table")
    if cached is not None and cached[0] == key:
        return cached[1]
    table = load_table(conn)
    ctx.find_cache["table"] = (key, table)
    return table


def status_of(ctx: AppContext, conn: sqlite3.Connection) -> EnvStatus:
    """Whether the environment can be searched by attributes (and if not, why)."""
    head = load_configured_head(ctx.cfg, warn=False)
    return env_status(conn, ctx.data_dir, head)


def annotate_now(ctx: AppContext) -> str:
    """Compute the attributes of the whole environment; returns a one-line summary."""
    head = load_configured_head(ctx.cfg, warn=False)
    if head is None:
        return "No attribute head is installed (reid attributes use-head PATH)."
    res = backfill_env(ctx.data_dir, head)
    if res.state == "annotated" and res.total == 0:
        return "Nothing to annotate: this environment has no stored embeddings."
    if res.state == "annotated":
        return f"Annotated {res.annotated:,} appearances in {res.seconds:.1f} s."
    return res.reason or "Nothing to annotate."


def time_bounds(table: AttributeTable | None) -> tuple[float, float]:
    """``(first start, last end)`` in seconds, for the time sliders."""
    if table is None or len(table) == 0:
        return 0.0, 1.0
    return float(table.start_ts.min()), float(table.end_ts.max())


def build_query(sel: Selection, table: AttributeTable | None) -> FindQuery:
    """Selection -> :class:`FindQuery`; a time bound at the end of the footage is no bound."""
    require, any_of = ui_labels.selection_terms(sel.chips)
    lo, hi = time_bounds(table)
    t_from = sel.t_from if sel.t_from is not None and sel.t_from > lo else None
    t_to = sel.t_to if sel.t_to is not None and sel.t_to < hi else None
    return FindQuery(
        cameras=list(sel.cameras) or None, t_from=t_from, t_to=t_to, require=require,
        any_of=any_of, strict=sel.strict,
    )  # fmt: skip


def run_find(ctx: AppContext, sel: Selection) -> Outcome:
    """Run the query on the cached table (a few milliseconds on 2,000 appearances)."""
    with ctx.db() as conn:
        status = status_of(ctx, conn)
        if status.state != "ready":
            return Outcome(status)
        table = get_table(ctx, conn)
        if table is None:
            return Outcome(EnvStatus("not_annotated", status.n_tracklets))
        result = find(conn, build_query(sel, table), table)
    return Outcome(status, result, group_by_person(result.hits), table)


def person_key(p: PersonHit) -> str:
    """Stable id of a card: ``g<gid>`` for a person, ``t<tracklet>`` for an unidentified one."""
    return f"g{p.gid}" if p.gid is not None else f"t{p.best.tracklet_id}"


def find_person(outcome: Outcome, key: str) -> PersonHit | None:
    """The person with ``key`` in the outcome (None when the filters moved on since the click)."""
    return next((p for p in outcome.people if person_key(p) == key), None)


def section_state(sel: Selection, ids: Any) -> int:
    """How many of the chips in ``ids`` are selected (for the accordion titles)."""
    return sum(1 for i in sel.chips if i in set(ids))
