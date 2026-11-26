"""Unified attribute schema: label list and the PA-100K / PETA mappings."""

import numpy as np

from reid.attributes import schema as s


def idx(name: str) -> int:
    return s.INDEX[name]


def test_67_unique_labels_and_groups():
    assert s.N_LABELS == 67 == len(set(s.LABELS)) == len(s.GROUPS)
    sizes = {g: s.GROUPS.count(g) for g in dict.fromkeys(s.GROUPS)}
    assert sizes == {
        "person": 4, "view": 3, "head": 4, "hair_colour": 5, "upper": 11, "upper_colour": 11,
        "lower": 7, "lower_colour": 11, "shoes": 4, "carrying": 7,
    }  # fmt: skip
    assert s.LABELS[:4] == ("female", "child", "adult", "senior")  # table order


def test_pa100k_26_map_one_to_one():
    assert len(s.PA100K_MAP) == 26
    assert len(set(s.PA100K_COLUMNS)) == 26  # distinct unified labels
    lab = np.zeros((2, 26), np.uint8)
    lab[0, 5] = 1  # Side
    t = s.pa100k_targets(lab)
    assert t.shape == (2, 67) and t.dtype == np.int8
    assert t[0, idx("view_side")] == 1 and t[1, idx("view_side")] == 0
    for _, unified in s.PA100K_MAP:
        assert t[0, idx(unified)] in (0, 1)
    assert (t[:, idx("scarf")] == -1).all() and (t[:, idx("upper_black")] == -1).all()
    assert (t != -1).sum(1).tolist() == [26, 26]
    # every column of the matrix lands in the label that carries its name
    for col, (_, unified) in enumerate(s.PA100K_MAP):
        one = np.zeros((1, 26), np.uint8)
        one[0, col] = 1
        assert np.flatnonzero(s.pa100k_targets(one)[0] == 1).tolist() == [idx(unified)]


def row(*tokens: str) -> np.ndarray:
    return s.peta_targets([list(tokens)])[0]


def test_peta_sex_and_age_bins():
    r = row("personalMale", "personalLess15")
    assert r[idx("female")] == 0 and r[idx("child")] == 1 and r[idx("adult")] == 0
    assert r[idx("senior")] == 0
    r = row("personalFemale", "personalLess45")
    assert r[idx("female")] == 1 and r[idx("child")] == 0 and r[idx("adult")] == 1
    r = row("personalLarger60")
    assert r[idx("female")] == -1  # neither sex token
    assert (r[idx("senior")], r[idx("child")], r[idx("adult")]) == (1, 0, 0)
    r = row("personalLess30")  # the 16-30 bin overlaps 15-17 and 18: unknown
    assert (r[idx("child")], r[idx("adult")], r[idx("senior")]) == (-1, -1, 0)
    r = row("hairBlack")  # no age token: all age labels unknown
    assert (r[[idx("child"), idx("adult"), idx("senior")]] == -1).all()


def test_peta_colour_group_rule():
    r = row("upperBodyBlue", "upperBodyWhite", "hairYellow")
    assert r[idx("upper_blue")] == 1 and r[idx("upper_white")] == 1 and r[idx("upper_red")] == 0
    assert r[idx("hair_blonde")] == 1 and r[idx("hair_black")] == 0
    assert (r[[idx("lower_black"), idx("lower_blue")]] == -1).all()  # no lower colour token
    assert (row("personalMale")[[idx("hair_black"), idx("upper_red")]] == -1).all()


def test_peta_or_merges_and_spellings():
    assert row("lowerBodyJeans")[[idx("trousers"), idx("jeans")]].tolist() == [1, 1]
    assert row("lowerBodyTrousers")[[idx("trousers"), idx("jeans")]].tolist() == [1, 0]
    assert row("lowerBodyHotPants")[idx("shorts")] == 1
    assert row("lowerBodyLongSkirt")[idx("skirt_dress")] == 1
    assert row("upperBodySuit")[idx("formal_upper")] == 1
    assert row("lowerBodySuits")[idx("formal_lower")] == 1
    assert row("upperBodyThickStripes")[idx("upper_stripes")] == 1
    assert row("carryingSuitcase")[idx("suitcase")] == row("carryingLuggageCase")[idx("suitcase")]
    assert row("carryingMessengerBag")[idx("shoulder_bag")] == 1
    assert row("carryingBabyBuggy")[idx("stroller")] == 1


def test_peta_sneaker_spelling_and_glasses():
    assert row("footwearSneakers")[idx("sneakers")] == 1
    assert row("footwearSneaker")[idx("sneakers")] == 1
    assert row("personalMale")[idx("sneakers")] == 0
    assert row("accessorySunglasses")[idx("glasses")] == 1
    assert row("personalMale")[idx("glasses")] == -1  # no sunglasses is not "no glasses"


def test_peta_no_source_labels_are_unknown():
    r = row("personalMale", "accessoryHat")
    for name in ("view_front", "view_side", "view_back", "long_coat", "upper_splice", "handbag",
                 "lower_pattern", "holding_front"):  # fmt: skip
        assert r[idx(name)] == -1
    assert r[idx("hat")] == 1 and r[idx("scarf")] == 0
