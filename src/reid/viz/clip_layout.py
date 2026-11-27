"""Pixel geometry of a follow clip frame: main view, thumbnail strip, person card, timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_TILE_H = 40  # smallest thumbnail height worth drawing
MIN_WIDTH, MAX_WIDTH = 320, 3840


@dataclass(frozen=True)
class Layout:
    """Pixel geometry of the output frame; never produces a tile smaller than ``MIN_TILE_H``."""

    width: int
    height: int
    margin: int
    main_w: int
    main_h: int
    strip_w: int  # 0 = no side strip
    card_h: int
    tile_w: int
    tile_h: int
    n_thumbs: int  # thumbnails that actually fit
    tl_h: int  # timeline panel height


def make_layout(
    width: int, aspect: float, n_rows: int, n_thumbs: int, strip: bool = True
) -> Layout:
    """Frame geometry for ``width`` (px), a camera ``aspect`` = height / width, ``n_rows``
    timeline rows and up to ``n_thumbs`` thumbnails (fewer when they would get too small)."""
    W = int(np.clip(width, MIN_WIDTH, MAX_WIDTH)) // 2 * 2
    m = max(6, round(W / 107))
    aspect = float(aspect) if aspect and aspect > 0 else 9 / 16
    has_strip = bool(strip)
    strip_w = int(W * 0.26) // 2 * 2 if has_strip else 0
    main_w = (W - strip_w - 3 * m if has_strip else W - 2 * m) // 2 * 2
    main_h = int(main_w * aspect) // 2 * 2
    tl_h = int(min(34 + 20 * max(n_rows, 1), 380)) if n_rows > 0 else 90
    tl_h = max(tl_h, 90)
    card_h = int(np.clip(W * 0.075, 64, 110)) if has_strip else 0
    n, tile_h = 0, 0
    for n in range(max(n_thumbs, 0) if has_strip else 0, 0, -1):
        fit = (main_h - card_h) / n - m
        tile_h = int(min(strip_w * aspect, fit)) // 2 * 2
        if tile_h >= MIN_TILE_H:
            break
    else:
        n, tile_h = 0, 0
    tile_w = int(tile_h / aspect) // 2 * 2 if n else 0
    H = (main_h + tl_h + 3 * m) // 2 * 2
    return Layout(W, H, m, main_w, main_h, strip_w, card_h, tile_w, tile_h, n, tl_h)
