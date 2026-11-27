"""The curated filter catalogue of the Find tab: names, sections, swatches, reliability.

The head predicts 67 labels; the UI offers a readable subset grouped like a video-analytics filter
panel. ``reliable`` comes from the shipped head's test metrics: a label is reliable when its mean
accuracy (mA) is at least 70 % on every test dataset that has at least 30 positives for it
(``ui_candidates`` of ``data/attributes/tuned_view/report/metrics_tuned_view.json``, the
``tuned_view`` head = best Optuna configuration retrained with viewpoint labels, written by
``reid attributes report`` on 2026-11-27; 41 labels). The list is hard-coded because
the package must not read ``data/``. Age group child and the shoulder bag are shown
even though they are not reliable, flagged; other unreliable labels only appear with the
"experimental filters" switch. Gender and age are always shown as *apparent*.

Pure data plus two helpers: :func:`selection_terms` turns the chips the user selected into the
``require`` / ``any_of`` terms of :class:`reid.find.FindQuery`; :func:`describe` turns a
probability dict into the chips of a person card.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: ``ui_candidates`` of metrics_tuned_view.json (2026-11-27): mA >= 70 % where >= 30 test positives.
RELIABLE: frozenset[str] = frozenset(
    """adult backpack female glasses hair_black hair_brown hair_grey hair_white handbag hat
    jeans leather_shoes long_coat long_hair long_sleeve lower_black lower_blue lower_brown
    lower_grey lower_red lower_white senior short_sleeve shorts trousers tshirt upper_black
    upper_blue upper_brown upper_green upper_grey upper_logo upper_orange upper_red upper_splice
    upper_stripes upper_white upper_yellow view_back view_front view_side""".split()
)

#: Swatch colours (hex) by colour name; hair has its own palette entry for blonde.
COLOUR_HEX: dict[str, str] = {
    "black": "#1d1d20",
    "blue": "#2f64d4",
    "brown": "#7a4b2b",
    "green": "#2f8f57",
    "grey": "#8b8f94",
    "orange": "#f28a1f",
    "pink": "#f39ac0",
    "purple": "#7f58c4",
    "red": "#d7322d",
    "white": "#f5f5f2",
    "yellow": "#f3d02a",
    "blonde": "#e4c373",
}
COLOURS = ("black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink")
COLOURS = (*COLOURS, "brown")


@dataclass(frozen=True)
class UiLabel:
    """One selectable chip or swatch."""

    id: str  # unique in the UI ("male" is the head label "female" with value False)
    label: str  # head label name
    name: str  # display name
    value: bool = True  # required value of ``label``
    hex: str | None = None  # swatch colour
    reliable: bool = True
    always_show: bool = False  # shown even when not reliable (flagged)


@dataclass(frozen=True)
class UiSection:
    """A collapsible group of filters."""

    id: str
    title: str
    kind: str  # "chips" | "swatches"
    items: tuple[UiLabel, ...]
    exclusive: bool = False  # at most one meaningful choice (several selected = "any of")
    note: str = ""  # hint shown under the title


def _chip(label: str, name: str, **kw: object) -> UiLabel:
    rel = kw.pop("reliable", label in RELIABLE)
    return UiLabel(label, label, name, reliable=bool(rel), **kw)  # type: ignore[arg-type]


def _swatches(prefix: str, names: Iterable[str]) -> tuple[UiLabel, ...]:
    return tuple(
        UiLabel(f"{prefix}{c}", f"{prefix}{c}", c.capitalize(), hex=COLOUR_HEX[c],
                reliable=f"{prefix}{c}" in RELIABLE)
        for c in names
    )  # fmt: skip


SECTIONS: tuple[UiSection, ...] = (
    UiSection(
        "person", "Person", "chips",
        (
            UiLabel("female", "female", "Female (apparent)", reliable=True),
            UiLabel("male", "female", "Male (apparent)", value=False, reliable=True),
            _chip("child", "Child (apparent)", always_show=True),
            _chip("adult", "Adult (apparent)", always_show=True),
            _chip("senior", "Senior (apparent)", always_show=True),
        ),
        note="Apparent gender and age group, judged from how the person looks. Age is less "
        "reliable than gender.",
    ),
    UiSection(
        "upper", "Upper body", "chips",
        (
            _chip("short_sleeve", "Short sleeves"),
            _chip("long_sleeve", "Long sleeves"),
            _chip("tshirt", "T-shirt"),
            _chip("long_coat", "Long coat"),
            _chip("jacket", "Jacket"),
            _chip("sweater", "Sweater"),
            _chip("formal_upper", "Formal"),
            _chip("upper_logo", "Logo"),
            _chip("upper_stripes", "Stripes"),
            _chip("upper_splice", "Two-tone"),
        ),
    ),
    UiSection("upper_colour", "Upper colour", "swatches", _swatches("upper_", COLOURS)),
    UiSection(
        "lower", "Lower body", "chips",
        (
            _chip("trousers", "Trousers"),
            _chip("jeans", "Jeans"),
            _chip("shorts", "Shorts"),
            _chip("skirt_dress", "Skirt / dress"),
            _chip("formal_lower", "Formal"),
        ),
    ),
    UiSection("lower_colour", "Lower colour", "swatches", _swatches("lower_", COLOURS)),
    UiSection(
        "head", "Head", "chips",
        (_chip("hat", "Hat"), _chip("glasses", "Glasses"), _chip("long_hair", "Long hair")),
    ),
    UiSection(
        "hair_colour", "Hair colour", "swatches",
        _swatches("hair_", ("black", "brown", "grey", "white", "blonde")),
    ),
    UiSection(
        "carrying", "Carrying", "chips",
        (
            _chip("backpack", "Backpack"),
            _chip("handbag", "Handbag"),
            _chip("shoulder_bag", "Shoulder bag", always_show=True),
        ),
    ),
)  # fmt: skip
_EXCLUSIVE_IDS = (("female", "male"), ("child", "adult", "senior"), ("short_sleeve", "long_sleeve"))

UI_LABELS: dict[str, UiLabel] = {i.id: i for s in SECTIONS for i in s.items}
_SECTION_OF: dict[str, UiSection] = {i.id: s for s in SECTIONS for i in s.items}
#: Chips that exclude each other: gender, age group, sleeve length (ids, not head labels).
EXCLUSIVE_GROUPS: tuple[tuple[str, ...], ...] = _EXCLUSIVE_IDS


def visible_items(section: UiSection, show_experimental: bool = False) -> tuple[UiLabel, ...]:
    """Items of ``section`` to show: reliable ones, the always-shown ones, or all if asked."""
    return tuple(i for i in section.items if show_experimental or i.reliable or i.always_show)


def selection_terms(selected: Iterable[str]) -> tuple[dict[str, bool], list[list[str]]]:
    """UI ids -> ``(require, any_of)`` for :class:`reid.find.FindQuery`.

    A chip becomes a required label (``male`` -> ``female`` must be False). Several swatches of
    one colour section are an any-of group (red OR orange). Several chips of an exclusive group
    (child + adult) are an any-of group too; both genders cancel out.
    """
    ids = [i for i in dict.fromkeys(selected)]
    unknown = [i for i in ids if i not in UI_LABELS]
    if unknown:
        raise ValueError(f"unknown filter ids: {unknown}")
    require: dict[str, bool] = {}
    any_of: list[list[str]] = []
    for sec in SECTIONS:
        picked = [UI_LABELS[i] for i in ids if _SECTION_OF[i] is sec]
        if sec.kind == "swatches":
            if picked:
                any_of.append([p.label for p in picked])
            continue
        for group in _EXCLUSIVE_IDS:
            sel = [p for p in picked if p.id in group]
            if len(sel) > 1:
                picked = [p for p in picked if p not in sel]
                if all(p.value for p in sel):
                    any_of.append([p.label for p in sel])
        for p in picked:
            require[p.label] = p.value
    return require, any_of


@dataclass(frozen=True)
class AttrChip:
    """One line of a person card: ``text`` with its probability."""

    id: str
    text: str
    prob: float
    reliable: bool


def describe(
    probs: Mapping[str, float], min_prob: float = 0.5, max_items: int = 14
) -> list[AttrChip]:
    """Readable attributes of a probability dict (label -> p), most certain first.

    Gender and age are phrased "Apparent ..."; of the exclusive groups (gender, age, sleeves) and
    of each colour section only the most probable member is shown; the rest are chips with
    ``p >= min_prob``.
    """
    out: list[AttrChip] = []
    p = probs.get
    if "female" in probs:
        f = float(p("female", 0.0))
        pick = UI_LABELS["female"] if f >= 0.5 else UI_LABELS["male"]
        out.append(AttrChip(pick.id, pick.name, max(f, 1.0 - f), True))
    for group in _EXCLUSIVE_IDS[1:]:
        best = max(group, key=lambda i: p(UI_LABELS[i].label, 0.0))
        if p(best, 0.0) >= min_prob:
            ui = UI_LABELS[best]
            out.append(AttrChip(ui.id, ui.name, float(p(best, 0.0)), ui.reliable))
    for sec in SECTIONS:
        items = [i for i in sec.items if i.label in probs]
        if sec.kind == "swatches" and items:
            best = max(items, key=lambda i: probs[i.label])
            if probs[best.label] >= min_prob:
                out.append(
                    AttrChip(best.id, f"{sec.title}: {best.name}", probs[best.label], best.reliable)
                )
        elif sec.kind == "chips" and sec.id != "person":
            seen = {i for g in _EXCLUSIVE_IDS for i in g}
            for i in items:
                if i.id not in seen and probs[i.label] >= min_prob:
                    out.append(AttrChip(i.id, i.name, float(probs[i.label]), i.reliable))
    out.sort(key=lambda c: -c.prob)
    return out[:max_items]
