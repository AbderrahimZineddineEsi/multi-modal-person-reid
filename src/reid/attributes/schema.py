"""Unified pedestrian-attribute label schema and the PA-100K / PETA -> schema mappings.

Every image gets, per label, 1 (present), 0 (absent) or -1 (unknown: masked out of the loss and
of the metrics). A label with no source in a dataset is -1 for that dataset. The order of
:data:`LABELS` is the output order of the head. The 26 PA-100K attributes map one to one onto
distinct labels (:data:`PA100K_MAP`), so PA-100K test metrics on those 26 are comparable with
published numbers. Mapping rules for PETA are documented next to each table.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

UNKNOWN = -1
COLOURS = ("black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white")
COLOURS = (*COLOURS, "yellow")
_HAIR_COLOURS = (
    ("black", "Black"),
    ("brown", "Brown"),
    ("grey", "Grey"),
    ("white", "White"),
    ("blonde", "Yellow"),
)


def _build_labels() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def add(group: str, names: Sequence[str]) -> None:
        out.extend((n, group) for n in names)

    add("person", ["female", "child", "adult", "senior"])
    add("view", ["view_front", "view_side", "view_back"])
    add("head", ["hat", "glasses", "scarf", "long_hair"])
    add("hair_colour", [f"hair_{n}" for n, _ in _HAIR_COLOURS])
    add(
        "upper",
        ["short_sleeve", "long_sleeve", "long_coat", "jacket", "tshirt", "sweater"]
        + ["formal_upper", "upper_logo", "upper_plaid", "upper_stripes", "upper_splice"],
    )
    add("upper_colour", [f"upper_{c}" for c in COLOURS])
    add(
        "lower",
        ["trousers", "jeans", "shorts", "skirt_dress", "formal_lower", "lower_stripes"]
        + ["lower_pattern"],
    )
    add("lower_colour", [f"lower_{c}" for c in COLOURS])
    add("shoes", ["boots", "sneakers", "leather_shoes", "sandals"])
    add(
        "carrying",
        ["backpack", "shoulder_bag", "handbag", "plastic_bag", "suitcase", "holding_front"]
        + ["stroller"],
    )
    return out


_LABELS = _build_labels()
LABELS: tuple[str, ...] = tuple(n for n, _ in _LABELS)
GROUPS: tuple[str, ...] = tuple(g for _, g in _LABELS)
N_LABELS = len(LABELS)
INDEX: dict[str, int] = {n: i for i, n in enumerate(LABELS)}

#: PA-100K attribute order in ``annotation.mat`` -> unified label name.
PA100K_MAP: tuple[tuple[str, str], ...] = (
    ("Female", "female"),
    ("AgeOver60", "senior"),
    ("Age18-60", "adult"),
    ("AgeLess18", "child"),
    ("Front", "view_front"),
    ("Side", "view_side"),
    ("Back", "view_back"),
    ("Hat", "hat"),
    ("Glasses", "glasses"),
    ("HandBag", "handbag"),
    ("ShoulderBag", "shoulder_bag"),
    ("Backpack", "backpack"),
    ("HoldObjectsInFront", "holding_front"),
    ("ShortSleeve", "short_sleeve"),
    ("LongSleeve", "long_sleeve"),
    ("UpperStride", "upper_stripes"),
    ("UpperLogo", "upper_logo"),
    ("UpperPlaid", "upper_plaid"),
    ("UpperSplice", "upper_splice"),
    ("LowerStripe", "lower_stripes"),
    ("LowerPattern", "lower_pattern"),
    ("LongCoat", "long_coat"),
    ("Trousers", "trousers"),
    ("Shorts", "shorts"),
    ("Skirt&Dress", "skirt_dress"),
    ("boots", "boots"),
)
PA100K_NAMES: tuple[str, ...] = tuple(a for a, _ in PA100K_MAP)
PA100K_COLUMNS: tuple[int, ...] = tuple(INDEX[u] for _, u in PA100K_MAP)

#: PETA labels that are the OR of binary tokens (token absent = 0). ``footwearSneaker`` is the
#: ReadMe spelling, ``footwearSneakers`` the one in Label.txt.
PETA_OR: dict[str, tuple[str, ...]] = {
    "hat": ("accessoryHat",),
    "scarf": ("accessoryMuffler",),
    "long_hair": ("hairLong",),
    "short_sleeve": ("upperBodyShortSleeve",),
    "long_sleeve": ("upperBodyLongSleeve",),
    "jacket": ("upperBodyJacket",),
    "tshirt": ("upperBodyTshirt",),
    "sweater": ("upperBodySweater",),
    "formal_upper": ("upperBodyFormal", "upperBodySuit"),
    "upper_logo": ("upperBodyLogo",),
    "upper_plaid": ("upperBodyPlaid",),
    "upper_stripes": ("upperBodyThinStripes", "upperBodyThickStripes"),
    "trousers": ("lowerBodyTrousers", "lowerBodyJeans"),
    "jeans": ("lowerBodyJeans",),
    "shorts": ("lowerBodyShorts", "lowerBodyHotPants"),
    "skirt_dress": ("lowerBodyShortSkirt", "lowerBodyLongSkirt"),
    "formal_lower": ("lowerBodyFormal", "lowerBodySuits"),
    "lower_stripes": ("lowerBodyThinStripes",),
    "boots": ("footwearBoots",),
    "sneakers": ("footwearSneakers", "footwearSneaker"),
    "leather_shoes": ("footwearLeatherShoes",),
    "sandals": ("footwearSandals",),
    "backpack": ("carryingBackpack",),
    "shoulder_bag": ("carryingMessengerBag",),
    "plastic_bag": ("carryingPlasticBags",),
    "suitcase": ("carryingLuggageCase", "carryingSuitcase"),
    "stroller": ("carryingBabyBuggy",),
}
#: Colour groups: (label prefix, token prefix, [(label suffix, token suffix)]).
PETA_COLOUR_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("hair_", "hair", _HAIR_COLOURS),
    ("upper_", "upperBody", tuple((c, c.capitalize()) for c in COLOURS)),
    ("lower_", "lowerBody", tuple((c, c.capitalize()) for c in COLOURS)),
)
_AGE_TOKENS = (
    "personalLess15",
    "personalLess30",
    "personalLess45",
    "personalLess60",
    "personalLarger60",
)


def check_pa100k_names(names: Sequence[str]) -> None:
    """Raise if the attribute names of an ``annotation.mat`` differ from :data:`PA100K_NAMES`."""
    if tuple(names) != PA100K_NAMES:
        raise ValueError(f"unexpected PA-100K attributes: {list(names)}")


def pa100k_targets(labels26: np.ndarray) -> np.ndarray:
    """PA-100K ``(N, 26)`` 0/1 labels -> ``int8 (N, 67)`` unified targets (-1 where no source)."""
    arr = np.asarray(labels26)
    if arr.ndim != 2 or arr.shape[1] != len(PA100K_MAP):
        raise ValueError(f"expected (N, 26) labels, got {arr.shape}")
    out = np.full((arr.shape[0], N_LABELS), UNKNOWN, np.int8)
    out[:, list(PA100K_COLUMNS)] = (arr > 0).astype(np.int8)
    return out


def _peta_one(tokens: Collection[str], row: np.ndarray) -> None:
    """Fill one ``(67,)`` row of PETA targets from the positive tokens of one image."""
    t = tokens
    male, female = "personalMale" in t, "personalFemale" in t
    if male != female:
        row[INDEX["female"]] = int(female)
    ages = {a for a in _AGE_TOKENS if a in t}
    if ages:  # no age token at all: the three age labels stay unknown
        row[INDEX["senior"]] = int("personalLarger60" in ages)
        if "personalLess15" in ages:
            row[INDEX["child"]], row[INDEX["adult"]] = 1, 0
        elif ages & {"personalLess45", "personalLess60"}:
            row[INDEX["child"]], row[INDEX["adult"]] = 0, 1
        elif "personalLarger60" in ages:
            row[INDEX["child"]], row[INDEX["adult"]] = 0, 0
        # only Less30: bins overlap 15-17, child and adult stay unknown
    if "accessorySunglasses" in t:
        row[INDEX["glasses"]] = 1
    for label, tokens_or in PETA_OR.items():
        row[INDEX[label]] = int(any(x in t for x in tokens_or))
    for lab_prefix, tok_prefix, pairs in PETA_COLOUR_GROUPS:
        if any(tok_prefix + suffix in t for _, suffix in pairs):
            for name, suffix in pairs:
                row[INDEX[lab_prefix + name]] = int(tok_prefix + suffix in t)


def peta_targets(tokens_per_image: Sequence[Collection[str]]) -> np.ndarray:
    """PETA positive tokens of each image -> ``int8 (N, 67)`` unified targets.

    An absent binary token means 0, except where the rules say unknown: a colour group without
    any colour token, ``glasses`` without sunglasses, sex without either token, the age labels
    without any age token, ``child``/``adult`` for the overlapping 16-30 bin.
    """
    out = np.full((len(tokens_per_image), N_LABELS), UNKNOWN, np.int8)
    for i, tokens in enumerate(tokens_per_image):
        _peta_one(set(tokens), out[i])
    return out
