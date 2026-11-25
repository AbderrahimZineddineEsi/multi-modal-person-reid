"""Plane homographies: scaling to a processed resolution, inversion and point mapping.

A homography here is a 3x3 matrix ``H`` acting on homogeneous column vectors, ``q ~ H @ p``.
``CameraSpec.homography`` is image -> floor in the pixels of the SOURCE video;
:func:`scale_homography` converts it to the (possibly downscaled) processed resolution that
every stored box uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Matrix = Sequence[Sequence[float]] | np.ndarray


def scale_homography(
    h_src: Matrix, src_size: tuple[int, int], proc_size: tuple[int, int]
) -> list[list[float]]:
    """Image -> floor homography for processed-resolution pixels.

    ``h_src`` maps SOURCE pixels to the floor. A processed pixel ``p`` corresponds to the source
    pixel ``diag(src_w/proc_w, src_h/proc_h, 1) @ p``, so ``H_proc = H_src @ that scale``. Equal
    sizes return the matrix unchanged. Sizes are ``(width, height)``.
    """
    h = np.asarray(h_src, dtype=np.float64).reshape(3, 3)
    if tuple(src_size) != tuple(proc_size):
        h = h @ np.diag([src_size[0] / proc_size[0], src_size[1] / proc_size[1], 1.0])
    return h.tolist()


def invert_homography(h: Matrix) -> np.ndarray:
    """Inverse of ``h`` normalised so that ``[2][2] == 1`` (when that entry is not ~0)."""
    inv = np.linalg.inv(np.asarray(h, dtype=np.float64).reshape(3, 3))
    return inv / inv[2, 2] if abs(inv[2, 2]) > 1e-12 else inv


def apply_homography(h: Matrix, pts: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` points through ``h`` (perspective divide included); returns ``(N, 2)``."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ np.asarray(h, np.float64).T
    return q[:, :2] / q[:, 2:3]
