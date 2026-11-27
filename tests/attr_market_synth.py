"""A tiny fake Market-1501 folder and ``market_attribute.mat`` (no real data, no network)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat

COLS = ["age", "backpack", "bag", "handbag", "clothes", "down", "up", "hair", "hat", "gender"]
COLS += [f"up{c}" for c in ("black", "white", "red", "purple", "yellow", "gray", "blue", "green")]
COLS += [f"down{c}" for c in ("black", "white", "pink", "purple", "yellow", "gray", "blue")]
COLS += ["downgreen", "downbrown"]


def make_attrs(n: int) -> dict[str, np.ndarray]:
    """Per-identity columns with values 1 / 2; identity ``i`` has a varied but fixed pattern."""
    i = np.arange(n)
    a = {c: np.ones(n, np.uint8) for c in COLS}
    a["gender"] = (1 + i % 2).astype(np.uint8)  # 1 male / 2 female
    a["hair"] = (1 + (i // 2) % 2).astype(np.uint8)
    a["up"] = (1 + (i // 3) % 2).astype(np.uint8)
    a["clothes"] = (2 - (i % 5 == 0)).astype(np.uint8)  # every 5th identity wears a dress
    a["down"] = (1 + (i // 2) % 2).astype(np.uint8)
    a["backpack"] = (1 + (i % 3 == 0)).astype(np.uint8)
    a["age"] = np.full(n, 2, np.uint8)
    a["upblack"] = (1 + (i % 4 < 2)).astype(np.uint8)  # ids with i % 4 >= 2: no upper colour yes
    a["downblue"] = (1 + (i % 3 == 1)).astype(np.uint8)
    a["downpink"] = (1 + (i % 7 == 6)).astype(np.uint8)
    return a


def make_market(root: Path, n_ids: int = 12, per_id: int = 4, mat_dir: Path | None = None):
    """Write ``bounding_box_test`` (+ junk, a distractor, an unannotated id, ``query``) and the
    attribute file; returns ``(ids, attrs)`` where ``ids[k]`` is the id of attribute row ``k``."""
    rng = np.random.default_rng(0)
    base = root / "Market-1501-v15.09.15"
    for sub in ("bounding_box_test", "query", "bounding_box_train"):
        (base / sub).mkdir(parents=True)
    ids = [f"{2 * k + 1:04d}" for k in range(n_ids)]  # 0001, 0003, ...

    def write(folder: str, name: str) -> None:
        img = rng.integers(0, 255, (24, 12, 3)).astype(np.uint8)
        cv2.imwrite(str(base / folder / name), img)

    for pid in ids:
        for n in range(per_id):
            write("bounding_box_test", f"{pid}_c{n % 6 + 1}s1_{n:06d}_00.jpg")
    write("bounding_box_test", "-1_c1s1_000401_03.jpg")  # junk
    write("bounding_box_test", "0000_c2s1_000002_01.jpg")  # distractor
    write("bounding_box_test", "1500_c2s1_000003_01.jpg")  # id without attributes
    (base / "bounding_box_test" / "Thumbs.db").write_bytes(b"x")
    write("query", f"{ids[0]}_c1s1_000999_00.jpg")
    attrs = make_attrs(n_ids)
    test = {c: v.reshape(1, -1) for c, v in attrs.items()}
    test["image_index"] = np.array(ids, dtype=object).reshape(1, -1)
    train = {c: v[:2].reshape(1, -1) for c, v in attrs.items()}
    train["image_index"] = np.array(["0002", "0007"], dtype=object).reshape(1, -1)
    mat_dir = mat_dir or root
    mat_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        str(mat_dir / "market_attribute.mat"), {"market_attribute": {"test": test, "train": train}}
    )
    return ids, attrs
