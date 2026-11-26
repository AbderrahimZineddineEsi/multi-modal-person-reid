"""PA-100K and PETA loaders on tiny fake data."""

import cv2
import numpy as np
import pytest
from scipy.io import savemat

from reid.attributes import datasets as d
from reid.attributes.schema import INDEX, N_LABELS, PA100K_NAMES


def _fake_pa100k(root, n=(6, 2, 2), images=True):
    ann = root / "ann"
    ann.mkdir(parents=True)
    rng = np.random.default_rng(0)
    attrs = np.empty((26, 1), object)
    for i, name in enumerate(PA100K_NAMES):
        attrs[i, 0] = np.array([name])
    mat = {"attributes": attrs}
    k = 0
    for split, cnt in zip(d.SPLITS, n, strict=True):
        names = np.empty((cnt, 1), object)
        for i in range(cnt):
            k += 1
            names[i, 0] = np.array([f"{k:06d}.jpg"])
            if images:
                (root / "data").mkdir(exist_ok=True)
                cv2.imwrite(str(root / "data" / f"{k:06d}.jpg"), np.full((8, 4, 3), k, np.uint8))
        mat[f"{split}_images_name"] = names
        mat[f"{split}_label"] = rng.integers(0, 2, (cnt, 26)).astype(np.uint8)
    savemat(str(ann / "annotation.mat"), mat)
    return mat


def test_load_pa100k(tmp_path):
    mat = _fake_pa100k(tmp_path)
    sp = d.load_pa100k(tmp_path)
    assert [len(sp[s]) for s in d.SPLITS] == [6, 2, 2]
    assert sp["train"].targets.shape == (6, N_LABELS)
    assert sp["train"].paths()[0].is_file() and sp["train"].persons == []
    assert sp["val"].images[0].endswith("000007.jpg")
    assert (sp["test"].targets[:, INDEX["female"]] == mat["test_label"][:, 0]).all()


def test_load_pa100k_annotation_only(tmp_path):
    _fake_pa100k(tmp_path, images=False)
    with pytest.raises(FileNotFoundError):
        d.load_pa100k(tmp_path)
    sp = d.load_pa100k(tmp_path, require_images=False)
    assert len(sp["train"]) == 6


def _fake_peta(root, n_persons=30, with_ilid=True):
    base = root / "PETA dataset"
    layout = {"SubA": "prefix", "SubB": "prefix", "CUHK": "filename", "i-LID": "prefix"}
    k = 0
    for sub, kind in layout.items():
        if sub == "i-LID" and not with_ilid:
            continue
        arch = base / sub / "archive"
        arch.mkdir(parents=True)
        lines = []
        for pid in range(n_persons):
            tokens = ["personalMale", "upperBodyBlack"] if pid % 2 else ["personalFemale"]
            key = f"{pid:04d}.png" if kind == "filename" else str(pid)
            lines.append(" ".join([key, *tokens]))
            for j in range(2 + pid % 2):
                k += 1
                name = f"{pid:04d}.png" if kind == "filename" else f"{pid}_{j}_FRAME.bmp"
                if kind == "filename" and j:
                    continue  # CUHK: one image per ID, the file name is the ID
                cv2.imwrite(str(arch / name), np.full((8, 4, 3), k % 255, np.uint8))
        (arch / "Label.txt").write_text("\n".join(lines) + "\n")
    (base.parent / "ReadMe.txt").write_text("readme")
    return base


def test_load_peta_skips_ilid_and_splits_by_person(tmp_path):
    _fake_peta(tmp_path)
    sp = d.load_peta(tmp_path)
    all_persons = [p for s in d.SPLITS for p in sp[s].persons]
    assert not any(p.startswith("i-LID") for p in all_persons)
    assert not any("i-LID" in r for s in d.SPLITS for r in sp[s].images)
    sets = [set(sp[s].persons) for s in d.SPLITS]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(x) for x in sets) == 90  # 3 subsets x 30 persons
    assert len(sets[0]) == 45 and len(sets[1]) == 9 and len(sets[2]) == 36  # 50 / 10 / 40 %
    for s in d.SPLITS:
        assert len(sp[s]) == len(sp[s].persons) == len(sp[s].targets)
        assert (sp[s].root / sp[s].images[0]).is_file()
    cuhk = [p for p in all_persons if p.startswith("CUHK/")]
    assert cuhk and all(p.endswith(".png") for p in cuhk)  # file name used as the ID
    t = sp["train"].targets
    assert set(np.unique(t[:, INDEX["female"]])) <= {0, 1}


def test_load_peta_split_is_seeded(tmp_path):
    _fake_peta(tmp_path)
    a, b = d.load_peta(tmp_path, seed=0), d.load_peta(tmp_path, seed=0)
    c = d.load_peta(tmp_path, seed=1)
    assert a["train"].images == b["train"].images
    assert a["train"].images != c["train"].images


def test_find_peta_root_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        d.find_peta_root(tmp_path)
    base = _fake_peta(tmp_path, with_ilid=False)
    assert d.find_peta_root(tmp_path) == base


def test_positive_counts_and_groups(tmp_path):
    _fake_peta(tmp_path)
    sp = d.load_peta(tmp_path)["train"]
    counts = d.positive_counts(sp)
    pos, neg = counts["female"]
    assert pos + neg == len(sp)
    groups = d.group_by_person(sp.persons)
    assert sum(len(v) for v in groups.values()) == len(sp)
