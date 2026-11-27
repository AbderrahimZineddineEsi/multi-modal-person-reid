"""HTML of the Find tab: result header, person and appearance cards, detail panel, empty states.

Pure builders over :class:`reid.app.find_data.Outcome`. Cards are real ``<button>`` elements that
call ``rdPick(key)`` (defined in ``theme.launch_style``), which hands the key to a hidden textbox
so a click reaches Python. The identity colour (``--pc``) rails every card, as in the other tabs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from reid.app import components as C
from reid.app import find_states
from reid.app.find_data import PAGE, Outcome, Selection, person_key
from reid.app.media import thumb_uri
from reid.attributes import ui_labels
from reid.find import Hit, PersonHit
from reid.storage import files
from reid.viz.colors import gid_color, hex_color


def clock(ts: float) -> str:
    """``m:ss`` (or ``h:mm:ss``) position in the footage."""
    s = int(ts)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def pct_text(pct: float) -> str:
    return "<1%" if 0 < pct < 1 else f"{pct:.0f}%"


def _crop(data_dir, h: Hit, height: int = 240) -> str | None:
    rel = h.crop_path or files.crop_path(data_dir, h.tracklet_id).relative_to(data_dir).as_posix()
    return thumb_uri(rel, data_dir, height)


def _img(uri: str | None, alt: str) -> str:
    return f'<img src="{uri}" alt="{escape(alt)}" loading="lazy">' if uri else ""


def _color(gid: int | None) -> str:
    return hex_color(gid_color(gid))


def person_name(p: PersonHit, names: Mapping[int, str]) -> str:
    return names.get(p.gid, f"Person {p.gid}") if p.gid is not None else "Unidentified"


def selection_summary(sel: Selection) -> str:
    """The active filters as small read-only pills ("Female", "Upper red or orange")."""
    if not sel.chips:
        return ""
    require, any_of = ui_labels.selection_terms(sel.chips)
    by_label = {i.label: i for i in ui_labels.UI_LABELS.values()}
    pills = []
    for label, want in require.items():
        item = next(i for i in ui_labels.UI_LABELS.values() if i.label == label and i.value == want)
        pills.append(item.name)
    for group in any_of:
        pills.append(" or ".join(by_label[g].name for g in group))
    inner = "".join(f"<span>{escape(t)}</span>" for t in pills)
    return f'<div class="rd-fpills">{inner}</div>'


# --------------------------------------------------------------------------- header


def stats_html(out: Outcome, sel: Selection) -> str:
    """ "12 people, 37 appearances, of 1,787" with the mode and the active filters."""
    res = out.result
    if res is None:
        return ""
    n_people = sum(1 for p in out.people if p.best.passes)
    in_scope = res.n_in_scope
    if not sel.has_terms:
        big, unit = f"{len(out.people):,}", "people" if len(out.people) != 1 else "person"
        detail = f"{in_scope:,} appearances"
    else:
        big, unit = f"{n_people:,}", "people" if n_people != 1 else "person"
        detail = f"{res.n_matching:,} appearances"
    scope = f"of {in_scope:,}"
    if in_scope != res.total_tracklets:
        scope += f" in scope ({res.total_tracklets:,} in all)"
    mode = ""
    if sel.has_terms:
        text = "Strict: only predicted matches" if sel.strict else "Ranked: closest match first"
        mode = f'<span class="rd-fmode">{text}</span>'
    return (
        f'<div class="rd-fstats"><div class="rd-fbig"><b>{big}</b><span>{unit}</span></div>'
        f'<div class="rd-fsub">{detail} <i>{scope}</i></div>{mode}'
        f"{selection_summary(sel)}</div>"
    )


# --------------------------------------------------------------------------- cards


def _match(h: Hit, show: bool) -> tuple[str, str]:
    """(percentage badge, bar) of a card; nothing without filters."""
    if not show:
        return "", ""
    pct = max(0.0, min(h.match_pct, 100.0))
    return (
        f'<span class="rd-fpct">{pct_text(pct)}</span>',
        f'<div class="rd-fbar"><i style="width:{pct:.0f}%"></i></div>',
    )


def _card(key: str, color: str, thumb: str, name: str, gid_tag: str, meta: str, extra: str,
          match: tuple[str, str], passes: bool, cls: str = "") -> str:  # fmt: skip
    badge, bar = match
    miss = "" if passes else " rd-miss"
    return (
        f'<button type="button" class="rd-fcard{miss}{cls}" style="--pc:{color}" '
        f'onclick="rdPick(\'{key}\')" aria-label="Open {escape(name)}">'
        f'<div class="rd-fthumb">{thumb}{badge}</div>{bar}'
        f'<div class="rd-fbody"><div class="rd-fname">{escape(name)}{gid_tag}</div>'
        f'<div class="rd-fmeta">{meta}</div>{extra}</div></button>'
    )


def people_grid(
    data_dir, out: Outcome, sel: Selection, names: Mapping[int, str],
    cams: Mapping[str, str], shown: int = PAGE,
) -> tuple[str, int]:  # fmt: skip
    """Cards of the first ``shown`` people and how many more there are."""
    cards = []
    for p in out.people[:shown]:
        n_ok = sum(1 for h in p.hits if h.passes)
        n_all = len(p.hits)
        what = C.plural(n_all, "appearance")
        if sel.has_terms and n_ok != n_all:
            what = f"{n_ok} matching of {n_all} appearances"
        cam_names = [cams.get(c, c) for c in p.cameras]
        extra_cams = f" +{len(cam_names) - 3}" if len(cam_names) > 3 else ""
        cam_txt = ", ".join(cam_names[:3]) + extra_cams
        tag = f'<span class="rd-gid">#{p.gid}</span>' if p.gid is not None else ""
        meta = f"{escape(what)}"
        extra = f'<div class="rd-fcams">{escape(cam_txt)}</div>'
        cards.append(
            _card(
                person_key(p), _color(p.gid), _img(_crop(data_dir, p.best), person_name(p, names)),
                person_name(p, names), tag, meta, extra, _match(p.best, sel.has_terms),
                p.best.passes,
            )
        )  # fmt: skip
    return _grid(cards), max(0, len(out.people) - shown)


def appearances_grid(
    data_dir, out: Outcome, sel: Selection, names: Mapping[int, str],
    cams: Mapping[str, str], shown: int = PAGE,
) -> tuple[str, int]:  # fmt: skip
    """One card per appearance (tracklet), best match first."""
    assert out.result is not None
    cards = []
    for h in out.result.hits[:shown]:
        who = names.get(h.gid, f"Person {h.gid}") if h.gid is not None else "Unidentified"
        key = f"g{h.gid}" if h.gid is not None else f"t{h.tracklet_id}"
        cam = cams.get(h.camera_id, h.camera_id)
        meta = f"{clock(h.start_ts)} to {clock(h.end_ts)}"
        extra = f'<div class="rd-fcams">{escape(cam)}</div>'
        tag = f'<span class="rd-gid">#{h.gid}</span>' if h.gid is not None else ""
        cards.append(
            _card(key, _color(h.gid), _img(_crop(data_dir, h), who), who, tag, meta, extra,
                  _match(h, sel.has_terms), h.passes)
        )  # fmt: skip
    return _grid(cards), max(0, len(out.result.hits) - shown)


def _grid(cards: Sequence[str]) -> str:
    return f'<div class="rd-fgrid">{"".join(cards)}</div>' if cards else ""


def results_html(
    data_dir, out: Outcome, sel: Selection, names: Mapping[int, str],
    cams: Mapping[str, str], shown: int = PAGE,
) -> tuple[str, int]:  # fmt: skip
    """The grid (or the helpful empty state) and the number of cards not shown yet."""
    assert out.result is not None
    if not out.result.hits:
        return find_states.empty_results(out, sel), 0
    build = people_grid if sel.view == "People" else appearances_grid
    html, more = build(data_dir, out, sel, names, cams, shown)
    dashed = any(not p.best.passes for p in out.people[:shown]) and sel.has_terms
    legend = (
        '<p class="rd-flegend"><span class="rd-fkey"></span>Dashed cards are the closest '
        "candidates: they miss at least one filter.</p>"
        if dashed
        else ""
    )
    return legend + html, more


# --------------------------------------------------------------------------- detail


def _chips(chips: Sequence[ui_labels.AttrChip], on: set[str]) -> str:
    out = []
    for c in chips:
        cls = "rd-attr" + (" rd-on" if c.id in on else "") + ("" if c.reliable else " rd-soft")
        out.append(
            f'<span class="{cls}" style="--p:{c.prob * 100:.0f}%">{escape(c.text)}'
            f"<b>{c.prob * 100:.0f}%</b></span>"
        )
    return "".join(out)


def detail_html(
    data_dir, person: PersonHit, chips: Sequence[ui_labels.AttrChip], sel: Selection,
    names: Mapping[int, str], cams: Mapping[str, str], face: bool, has_terms: bool,
) -> tuple[str, str]:  # fmt: skip
    """Drawer content as ``(header, body)``: who it is, then how they look and every appearance."""
    color = _color(person.gid)
    name = person_name(person, names)
    tag = f'<span class="rd-gid">#{person.gid}</span>' if person.gid is not None else ""
    badges = [C.badge("face confirmed", "ok") if face else C.badge("body only")]
    badges.append(C.badge(C.plural(len(person.cameras), "camera")))
    if has_terms:
        badges.insert(0, C.badge(f"best match {pct_text(person.best.match_pct)}", "accent"))
    head = (
        f'<div class="rd-dhead" style="--pc:{color}"><div class="rd-avatar">'
        f"{_img(_crop(data_dir, person.best, 320), name)}</div>"
        f'<div class="rd-dtitle"><div class="rd-person-name">{escape(name)}{tag}</div>'
        f'<div class="rd-row">{"".join(badges)}</div></div></div>'
    )
    on = set(sel.chips)
    looks = (
        f'<h4 class="rd-dh">Looks like</h4><div class="rd-attrs">{_chips(chips, on)}</div>'
        '<p class="rd-hint">Gender and age are what the model sees, not a fact. Dashed chips '
        "are less reliable.</p>"
        if chips
        else ""
    )
    tiles = []
    for h in person.hits[:36]:
        pct = f"<b>{pct_text(h.match_pct)}</b>" if has_terms else ""
        miss = "" if h.passes or not has_terms else " rd-miss"
        cam = escape(cams.get(h.camera_id, h.camera_id))
        thumb = _img(_crop(data_dir, h, 160), "")
        tiles.append(
            f'<div class="rd-atile{miss}"><div class="rd-athumb">{thumb}</div>'
            f"<div><span>{cam}</span><em>{clock(h.start_ts)} to {clock(h.end_ts)}</em>{pct}</div>"
            "</div>"
        )
    more = (
        f'<p class="rd-hint">and {len(person.hits) - 36} more appearances</p>'
        if len(person.hits) > 36
        else ""
    )
    n_ok = sum(1 for h in person.hits if h.passes)
    title = C.plural(len(person.hits), "appearance")
    if has_terms and n_ok != len(person.hits):
        title = f"{n_ok} matching of {len(person.hits)} appearances"
    return (
        head,
        f'<div class="rd-detail">{looks}<h4 class="rd-dh">{escape(title)}</h4>'
        f'<div class="rd-atiles">{"".join(tiles)}</div>{more}</div>',
    )
