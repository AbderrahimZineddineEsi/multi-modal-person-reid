"""Market-1501 test split with the identity-level attribute annotations (external evaluation only).

Market-1501 (Zheng et al. 2015) is never used to train or tune the attribute head; it is a
zero-shot test of how the head transfers. The 27 attributes of Lin et al. 2019 are annotated per
identity, so every image of an identity gets the identity's labels (see
:func:`reid.attributes.schema.market_targets`). Only ``bounding_box_test`` is read: junk images
(identity ``-1``) and distractors (identity ``0000``) have no annotation and are dropped, which
leaves the images of the 750 test identities. ``query`` is not included: its images are further
crops of the same identities and the paper's attribute evaluation uses ``bounding_box_test``
only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from reid.attributes.datasets import IMAGE_EXTS, ParSplit, _subdirs, find_file
from reid.attributes.schema import check_market_columns, market_targets

log = logging.getLogger(__name__)

ATTRIBUTE_FILE = "market_attribute.mat"
TEST_DIR = "bounding_box_test"
JUNK_IDS = ("-1", "0000")


def _scalar(x: object) -> str:
    return str(np.asarray(x).ravel()[0])


def read_market_attributes(mat_path: str | Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Test-identity ids and their attribute columns from ``market_attribute.mat``."""
    from scipy.io import loadmat

    test = loadmat(str(mat_path))["market_attribute"][0, 0]["test"][0, 0]
    names = [n for n in test.dtype.names if n != "image_index"]
    check_market_columns(names)
    ids = [_scalar(x) for x in test["image_index"].ravel()]
    cols = {n: np.asarray(test[n]).ravel() for n in names}
    if any(len(v) != len(ids) for v in cols.values()):
        raise ValueError(f"{mat_path}: attribute columns and image_index differ in length")
    return ids, cols


def find_market_root(root: str | Path) -> Path:
    """Folder that contains ``bounding_box_test`` (``root`` itself or up to 3 levels below)."""
    for d in _subdirs(Path(root), 3):
        if (d / TEST_DIR).is_dir():
            return d
    raise FileNotFoundError(f"{TEST_DIR} not found under {root}")


def load_market1501(root: str | Path, attribute_file: str | Path | None = None) -> ParSplit:
    """The test identities of Market-1501 as a :class:`ParSplit` (``dataset='market1501'``).

    ``root`` holds the unzipped dataset (``bounding_box_test`` directly or a few folders down) and,
    anywhere below it, ``market_attribute.mat`` (or pass ``attribute_file``). Images are sorted
    by name; ``persons`` holds the 4-digit identity string of each image.
    """
    root = Path(root)
    base = find_market_root(root)
    mat = Path(attribute_file) if attribute_file else find_file(root, ATTRIBUTE_FILE)
    if mat is None or not Path(mat).is_file():
        raise FileNotFoundError(f"{ATTRIBUTE_FILE} not found under {root}")
    ids, cols = read_market_attributes(mat)
    row = {pid: i for i, pid in enumerate(ids)}
    rel: list[str] = []
    persons: list[str] = []
    skipped = 0
    for img in sorted((base / TEST_DIR).iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        pid = img.name.split("_")[0]
        if pid in JUNK_IDS:
            continue
        if pid not in row:
            skipped += 1
            continue
        rel.append(f"{TEST_DIR}/{img.name}")
        persons.append(pid)
    if skipped:
        log.warning("Market-1501: %d images of identities without attributes skipped", skipped)
    id_targets = market_targets(cols)
    targets = id_targets[[row[p] for p in persons]]
    log.info("Market-1501 test: %d images of %d identities", len(rel), len(set(persons)))
    return ParSplit("market1501", "test", base, rel, targets, persons)
