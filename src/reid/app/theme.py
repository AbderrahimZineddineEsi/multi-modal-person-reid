"""Dark theme with a single periwinkle accent; the stylesheet is shipped as package data."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import gradio as gr

STATIC = Path(__file__).parent / "static"

# surface and text tokens shared by the theme and style.css (kept in sync by hand, 8 values)
BG, PANEL, RAISED, LINE = "#0c0e14", "#12151d", "#181c26", "#252a3a"
TEXT, MUTED, ACCENT = "#e7e9f0", "#8a91a6", "#8b8bff"

_ACCENT = gr.themes.Color(
    c50="#f0f0ff", c100="#e2e2ff", c200="#c9c9ff", c300="#adadff", c400="#9b9bff",
    c500=ACCENT, c600="#6f6fe6", c700="#5a5ac4", c800="#47479b", c900="#35356f", c950="#232349",
    name="periwinkle",
)  # fmt: skip
_NEUTRAL = gr.themes.Color(
    c50="#f3f4f8", c100="#e7e9f0", c200="#c9cddb", c300="#aab0c3", c400="#8a91a6",
    c500="#6b7288", c600="#4d5368", c700="#363b4d", c800="#252a3a", c900="#181c26",
    c950="#12151d", name="ink",
)  # fmt: skip


class ReidTheme(gr.themes.Base):
    """Calm dark surfaces, one accent, Geist type. Light and dark values are identical, so the
    app stays dark whatever the browser prefers."""

    def __init__(self) -> None:
        super().__init__(
            primary_hue=_ACCENT,
            secondary_hue=_ACCENT,
            neutral_hue=_NEUTRAL,
            radius_size=gr.themes.sizes.radius_md,
            text_size=gr.themes.sizes.text_md,
            font=[gr.themes.GoogleFont("Geist"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
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
            "input_shadow_focus": f"0 0 0 3px {ACCENT}33",
            "table_even_background_fill": PANEL,
            "table_odd_background_fill": RAISED,
            "table_border_color": LINE,
            "button_primary_background_fill": ACCENT,
            "button_primary_background_fill_hover": "#a0a0ff",
            "button_primary_text_color": "#10101c",
            "button_primary_border_color": ACCENT,
            "button_secondary_background_fill": RAISED,
            "button_secondary_background_fill_hover": "#202533",
            "button_secondary_text_color": TEXT,
            "button_secondary_border_color": LINE,
            "button_large_radius": "10px",
            "button_medium_radius": "10px",
            "slider_color": ACCENT,
            "checkbox_background_color": RAISED,
            "checkbox_background_color_selected": ACCENT,
            "checkbox_border_color": LINE,
            "link_text_color": ACCENT,
            "link_text_color_hover": "#b0b0ff",
        }
        known = inspect.signature(gr.themes.Base.set).parameters
        both = {**tokens, **{f"{k}_dark": v for k, v in tokens.items() if f"{k}_dark" in known}}
        super().set(**both)


def css() -> str:
    """Contents of ``static/style.css``."""
    return (STATIC / "style.css").read_text(encoding="utf-8")


def launch_style() -> dict[str, Any]:
    """Keyword arguments for ``Blocks.launch``: Gradio 6 takes theme and CSS at launch time."""
    return {"theme": ReidTheme(), "css": css(), "js": "() => document.body.classList.add('dark')"}
