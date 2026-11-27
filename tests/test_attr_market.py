"""Market-1501 -> unified schema mapping and the test-split loader (synthetic folder)."""

import numpy as np
import pytest

from reid.attributes import schema as s
from reid.attributes.datasets import group_by_person
from reid.attributes.datasets_market import load_market1501
from tests.attr_market_synth import COLS, make_attrs, make_market


def idx(name: str) -> int:
    return s.INDEX[name]


def one(**kw) -> np.ndarray:
    """Targets of one identity: all columns 1 (no / long / short...) except those given."""
    a = {c: np.array([1], np.uint8) for c in COLS}
    a.update({k: np.array([v], np.uint8) for k, v in kw.items()})
    return s.market_targets(a)[0]


def test_columns_cover_the_27_attributes():
    assert len(COLS) == len(s.MARKET_COLUMNS) == 27
    s.check_market_columns(COLS)
    with pytest.raises(ValueError, match="hat"):
        s.check_market_columns([c for c in COLS if c != "hat"])


def test_mapped_labels_exist_and_unmapped_stay_unknown():
    assert len(s.MARKET_LABELS) == 26
    assert all(lab in s.INDEX for lab in s.MARKET_LABELS)
    t = one(gender=2, hat=2)
    for lab in s.LABELS:
        if lab not in s.MARKET_LABELS:
            assert t[idx(lab)] == -1, lab
    for name in ("child", "adult", "senior", "shoulder_bag", "lower_pink", "upper_pink"):
        assert t[idx(name)] == -1
    assert set(s.MARKET_UNMAPPED) == {"age", "bag", "downpink"}


def test_one_two_encoding_not_zero_one():
    assert one(gender=2)[idx("female")] == 1 and one(gender=1)[idx("female")] == 0
    assert one(hair=2)[idx("long_hair")] == 1 and one(hair=1)[idx("long_hair")] == 0
    assert one(hat=2)[idx("hat")] == 1 and one(hat=1)[idx("hat")] == 0
    assert one(backpack=2)[idx("backpack")] == 1 and one(handbag=2)[idx("handbag")] == 1
    assert one(gender=0)[idx("female")] == -1  # anything but 1 / 2 is unknown


def test_sleeves_are_the_two_values_of_up():
    r = one(up=2)
    assert (r[idx("short_sleeve")], r[idx("long_sleeve")]) == (1, 0)
    r = one(up=1)
    assert (r[idx("short_sleeve")], r[idx("long_sleeve")]) == (0, 1)


def test_lower_clothing_rules():
    def lower(clothes, down):
        r = one(clothes=clothes, down=down)
        return [int(r[idx(n)]) for n in ("trousers", "shorts", "skirt_dress")]

    assert lower(2, 1) == [1, 0, 0]  # pants, long
    assert lower(2, 2) == [0, 1, 0]  # pants, short = shorts
    assert lower(1, 1) == [0, 0, 1]  # dress, long
    assert lower(1, 2) == [0, 0, 1]  # dress, short
    assert lower(0, 1) == [-1, -1, -1]


def test_colour_group_without_a_yes_is_all_zero_not_unknown():
    none = one()  # every colour column is 1 = no
    for lab in s.MARKET_LABELS:
        if lab.startswith(("upper_", "lower_")):
            assert none[idx(lab)] == 0, lab
    r = one(upgray=2, downblue=2, downpink=2)
    assert r[idx("upper_grey")] == 1 and r[idx("upper_black")] == 0
    assert r[idx("lower_blue")] == 1 and r[idx("lower_black")] == 0
    assert r[idx("lower_pink")] == -1  # pink is not mapped; its identities have 0 elsewhere


def test_loader_keeps_test_identities_only(tmp_path):
    ids, attrs = make_market(tmp_path / "market", n_ids=12, per_id=4, mat_dir=tmp_path / "x" / "y")
    sp = load_market1501(tmp_path)  # attribute file and folder found anywhere below root
    assert (sp.dataset, sp.split, len(sp)) == ("market1501", "test", 12 * 4)
    assert sp.images == sorted(sp.images) and all(
        i.startswith("bounding_box_test/") for i in sp.images
    )
    assert not any(i.split("/")[-1].startswith(("-1", "0000", "1500")) for i in sp.images)
    assert all(p.is_file() for p in sp.paths())
    assert sp.persons[0] == "0001" and sorted(set(sp.persons)) == ids
    assert sp.targets.shape == (48, s.N_LABELS) and sp.targets.dtype == np.int8
    # labels are copied from the identity row to each of its images
    want = s.market_targets(attrs)
    for k, pid in enumerate(ids):
        rows = [i for i, p in enumerate(sp.persons) if p == pid]
        assert len(rows) == 4 and (sp.targets[rows] == want[k]).all()
    assert sorted(len(v) for v in group_by_person(sp.persons).values()) == [4] * 12


def test_loader_errors_are_clear(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="bounding_box_test"):
        load_market1501(tmp_path / "empty")
    make_market(tmp_path / "m", mat_dir=tmp_path / "m2")
    with pytest.raises(FileNotFoundError, match="market_attribute.mat"):
        load_market1501(tmp_path / "m")  # the attribute file is outside root


def test_make_attrs_is_consistent():
    a = make_attrs(12)
    assert set(a) == set(COLS) and all(len(v) == 12 for v in a.values())
