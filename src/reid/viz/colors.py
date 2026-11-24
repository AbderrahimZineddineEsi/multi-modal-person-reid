"""Stable, distinct, colourblind-aware colours per identity (deterministic from the gid)."""

from __future__ import annotations

import colorsys

RGB = tuple[int, int, int]

# Okabe-Ito (colourblind safe) followed by Paul Tol "bright"/"vibrant" extras, ordered so that
# consecutive gids differ strongly in hue and lightness. Black and plain grey are excluded:
# grey is reserved for unidentified tracklets.
PALETTE: tuple[RGB, ...] = (
    (230, 159, 0),  # orange
    (86, 180, 233),  # sky blue
    (0, 158, 115),  # bluish green
    (204, 121, 167),  # reddish purple
    (240, 228, 66),  # yellow
    (0, 114, 178),  # blue
    (213, 94, 0),  # vermillion
    (170, 51, 119),  # wine
    (34, 136, 51),  # green
    (102, 204, 238),  # cyan
    (238, 102, 119),  # coral
    (136, 106, 224),  # violet
    (187, 187, 80),  # olive
    (0, 153, 136),  # teal
)

UNIDENTIFIED: RGB = (128, 128, 128)


def gid_color(gid: int | None, unidentified: RGB = UNIDENTIFIED) -> RGB:
    """RGB colour of ``gid`` (grey for ``None``). Pure function of the gid.

    The first ``len(PALETTE)`` gids use the curated palette; further gids use golden-angle
    hues with alternating lightness so that neighbours stay distinguishable.
    """
    if gid is None:
        return unidentified
    i = int(gid) - 1
    if 0 <= i < len(PALETTE):
        return PALETTE[i]
    h = (i * 0.61803398875) % 1.0
    s, v = (0.72, 0.95) if i % 2 else (0.55, 0.80)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def gid_color_bgr(gid: int | None, unidentified: RGB = UNIDENTIFIED) -> tuple[int, int, int]:
    """Same as :func:`gid_color` but in OpenCV BGR order."""
    r, g, b = gid_color(gid, unidentified)
    return b, g, r


def text_on(rgb: RGB) -> RGB:
    """Black or white, whichever is more legible on ``rgb`` (WCAG relative luminance)."""

    def lin(c: int) -> float:
        x = c / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    return (18, 18, 22) if lum > 0.32 else (255, 255, 255)


def hex_color(rgb: RGB) -> str:
    """``#rrggbb`` string (matplotlib / plotly)."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)
