"""Control-room theme: graphite surfaces, ivory chrome, colour reserved for people.

The interface itself stays neutral; the only saturated colours on screen are the identity
colours (``reid.viz.colors``), so a person looks the same in a card, the journey rail, the charts
and the rendered videos. The stylesheet is shipped as package data.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import gradio as gr

STATIC = Path(__file__).parent / "static"

# surface and text tokens shared by the theme and static/*.css (kept in sync by hand)
BG, PANEL, RAISED, LINE = "#0d1012", "#14181b", "#1b2125", "#2a3338"
TEXT, MUTED, ACCENT = "#ebe8df", "#929a9e", "#ebe8df"
DANGER = "#e5655c"

_ACCENT = gr.themes.Color(
    c50="#faf9f5", c100="#f3f1e9", c200="#ebe8df", c300="#d9d5c8", c400="#c4bfae",
    c500=ACCENT, c600="#a8a392", c700="#7d7a6d", c800="#55534a", c900="#34332e", c950="#1f1e1b",
    name="ivory",
)  # fmt: skip
_NEUTRAL = gr.themes.Color(
    c50="#f1f2f1", c100="#e3e6e5", c200="#c7cdcc", c300="#a9b1b1", c400="#929a9e",
    c500="#6c767a", c600="#4b565b", c700="#343d42", c800="#2a3338", c900="#1b2125",
    c950="#14181b", name="graphite",
)  # fmt: skip


class ReidTheme(gr.themes.Base):
    """Graphite surfaces, ivory accent, Instrument Sans with IBM Plex Mono for ids and time.
    Light and dark values are identical, so the app stays dark whatever the browser prefers."""

    def __init__(self) -> None:
        super().__init__(
            primary_hue=_ACCENT,
            secondary_hue=_ACCENT,
            neutral_hue=_NEUTRAL,
            radius_size=gr.themes.sizes.radius_sm,
            text_size=gr.themes.sizes.text_md,
            font=[
                gr.themes.GoogleFont("Instrument Sans"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
            font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Menlo", "monospace"],
        )
        tokens: dict[str, Any] = {
            "body_background_fill": BG,
            "body_text_color": TEXT,
            "body_text_color_subdued": MUTED,
            "background_fill_primary": PANEL,
            "background_fill_secondary": BG,
            "block_background_fill": PANEL,
            "block_border_color": LINE,
            "block_border_width": "1px",
            "block_label_background_fill": "transparent",
            "block_label_text_color": MUTED,
            "block_title_text_color": MUTED,
            "block_shadow": "none",
            "border_color_primary": LINE,
            "border_color_accent": ACCENT,
            "panel_background_fill": PANEL,
            "input_background_fill": RAISED,
            "input_border_color": LINE,
            "input_border_color_focus": ACCENT,
            "input_shadow": "none",
            "input_shadow_focus": f"0 0 0 3px {ACCENT}22",
            "table_even_background_fill": PANEL,
            "table_odd_background_fill": RAISED,
            "table_border_color": LINE,
            "button_primary_background_fill": ACCENT,
            "button_primary_background_fill_hover": "#ffffff",
            "button_primary_text_color": "#101314",
            "button_primary_border_color": ACCENT,
            "button_secondary_background_fill": RAISED,
            "button_secondary_background_fill_hover": "#232b30",
            "button_secondary_text_color": TEXT,
            "button_secondary_border_color": LINE,
            "button_cancel_background_fill": "#2a1513",
            "button_cancel_background_fill_hover": DANGER,
            "button_cancel_text_color": "#ff9f98",
            "button_cancel_text_color_hover": "#190807",
            "button_cancel_border_color": "#7a2e29",
            "button_cancel_border_color_hover": DANGER,
            "button_large_radius": "8px",
            "button_medium_radius": "8px",
            "slider_color": ACCENT,
            "checkbox_background_color": RAISED,
            "checkbox_background_color_selected": ACCENT,
            "checkbox_border_color": LINE,
            "link_text_color": ACCENT,
            "link_text_color_hover": "#ffffff",
        }
        known = inspect.signature(gr.themes.Base.set).parameters
        both = {**tokens, **{f"{k}_dark": v for k, v in tokens.items() if f"{k}_dark" in known}}
        super().set(**both)


# runs once at page load: dark mode, the card click hook of the Find tab, Esc closes its drawer
JS = """() => {
  document.body.classList.add('dark');
  window.rdPick = (key) => {
    const box = document.querySelector('#find-pick textarea, #find-pick input');
    if (!box) return;
    box.value = key + '|' + Date.now();
    box.dispatchEvent(new Event('input', {bubbles: true}));
  };
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') document.querySelector('.rd-drawer-close')?.click();
  });
}"""

CSS_FILES = ("base.css", "components.css", "find.css", "chrome.css")


def css() -> str:
    """The stylesheet: the files of ``static/`` in :data:`CSS_FILES` order, then the rules
    generated from the Find filter catalogue."""
    from reid.app.find_css import catalogue_css

    parts = [(STATIC / name).read_text(encoding="utf-8") for name in CSS_FILES]
    return "\n".join([*parts, catalogue_css()])


def launch_style() -> dict[str, Any]:
    """Keyword arguments for ``Blocks.launch``: Gradio 6 takes theme and CSS at launch time."""
    return {"theme": ReidTheme(), "css": css(), "js": JS}
