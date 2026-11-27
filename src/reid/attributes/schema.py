"""Unified pedestrian-attribute label schema and the PA-100K / PETA / Market-1501 mappings.

Every image gets, per label, 1 (present), 0 (absent) or -1 (unknown: masked out of the loss and
of the metrics). A label with no source in a dataset is -1 for that dataset. The order of
:data:`LABELS` is the output order of the head. The 26 PA-100K attributes map one to one onto
distinct labels (:data:`PA100K_MAP`), so PA-100K test metrics on those 26 are comparable with
published numbers. Mapping rules for PETA are documented next to each table.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

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


# ----------------------------------------------------------------------------- Market-1501
#: Market-1501 attribute columns (Lin et al. 2019, ``market_attribute.mat``, identity level).
#: Every value is 1 or 2 (not 0 / 1): ``gender`` 1 male / 2 female, ``hair`` 1 short / 2 long,
#: ``up`` 1 long / 2 short sleeve, ``down`` 1 long / 2 short lower clothing, ``clothes`` 1 dress /
#: 2 pants, the rest 1 no / 2 yes. Only one colour of the upper (lower) group is 2, none when the
#: colour is outside the list. Encodings checked on images of the test identities.
MARKET_COLUMNS: frozenset[str] = frozenset(
    {"age", "backpack", "bag", "handbag", "clothes", "down", "up", "hair", "hat", "gender"}
    | {f"up{c}" for c in ("black", "white", "red", "purple", "yellow", "gray", "blue", "green")}
    | {f"down{c}" for c in ("black", "white", "pink", "purple", "yellow", "gray", "blue")}
    | {"downgreen", "downbrown"}
)
#: (Market column, unified label): value 2 -> 1, value 1 -> 0.
MARKET_BINARY: tuple[tuple[str, str], ...] = (
    ("gender", "female"),
    ("hair", "long_hair"),
    ("up", "short_sleeve"),
    ("hat", "hat"),
    ("backpack", "backpack"),
    ("handbag", "handbag"),
)
_MARKET_UP = (("black", "black"), ("white", "white"), ("red", "red"), ("purple", "purple"))
_MARKET_UP += (("yellow", "yellow"), ("gray", "grey"), ("blue", "blue"), ("green", "green"))
_MARKET_DOWN = (("black", "black"), ("white", "white"), ("purple", "purple"), ("yellow", "yellow"))
_MARKET_DOWN += (("gray", "grey"), ("blue", "blue"), ("green", "green"), ("brown", "brown"))
#: Colours: (Market column, unified label). A group where no colour is 2 means a colour outside
#: Market's list, so every mapped colour of the group is 0 (not unknown).
MARKET_COLOURS: tuple[tuple[str, str], ...] = tuple(
    [(f"up{m}", f"upper_{u}") for m, u in _MARKET_UP]
    + [(f"down{m}", f"lower_{u}") for m, u in _MARKET_DOWN]
)
#: Derived labels (rules in :func:`market_targets`): ``long_sleeve`` = not ``up`` short,
#: ``trousers`` = pants and long, ``shorts`` = pants and short, ``skirt_dress`` = dress.
MARKET_DERIVED: tuple[str, ...] = ("long_sleeve", "trousers", "shorts", "skirt_dress")
#: Market columns that are NOT mapped, with the reason (shown in the report).
MARKET_UNMAPPED: dict[str, str] = {
    "age": "classes young / teenager / adult / old have no stated age ranges (paper and README "
    "even name them differently); 640 of 750 identities are 'teenager', so child / adult / senior "
    "boundaries cannot be justified",
    "bag": "'other types of bag' (paper): not the same as shoulder_bag, whose boundary with "
    "handbag differs between datasets",
    "downpink": "its positives mix pink and red clothes (the paper lists 'red'), no schema label",
}
MARKET_LABELS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [u for _, u in MARKET_BINARY] + list(MARKET_DERIVED) + [u for _, u in MARKET_COLOURS]
    )
)


def _market_yes(v: np.ndarray) -> np.ndarray:
    """Market 2 -> 1, 1 -> 0, anything else -> -1 (``int8``)."""
    a = np.asarray(v).ravel()
    return np.where(a == 2, 1, np.where(a == 1, 0, UNKNOWN)).astype(np.int8)


def market_targets(attrs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Market-1501 per-identity columns ``{name: (n,)}`` -> ``int8 (n, 67)`` unified targets.

    ``short_sleeve``/``long_sleeve`` are the two values of ``up``; ``skirt_dress`` is
    ``clothes == dress``; ``trousers`` / ``shorts`` are ``clothes == pants`` with ``down`` long /
    short (any unknown input makes the derived label unknown). Everything unmapped stays -1.
    """
    n = len(np.asarray(attrs["gender"]).ravel())
    out = np.full((n, N_LABELS), UNKNOWN, np.int8)
    for col, label in (*MARKET_BINARY, *MARKET_COLOURS):
        out[:, INDEX[label]] = _market_yes(attrs[col])
    short = _market_yes(attrs["up"])
    out[:, INDEX["long_sleeve"]] = np.where(short < 0, UNKNOWN, 1 - short)
    pants, short_low = _market_yes(attrs["clothes"]), _market_yes(attrs["down"])  # 2 = pants
    out[:, INDEX["skirt_dress"]] = np.where(pants < 0, UNKNOWN, 1 - pants)
    both = (pants >= 0) & (short_low >= 0)
    out[:, INDEX["trousers"]] = np.where(both, (pants == 1) & (short_low == 0), UNKNOWN)
    out[:, INDEX["shorts"]] = np.where(both, (pants == 1) & (short_low == 1), UNKNOWN)
    return out


def check_market_columns(names: Collection[str]) -> None:
    """Raise if the attribute columns of a ``market_attribute.mat`` are not the expected 27."""
    missing = MARKET_COLUMNS - set(names)
    if missing:
        raise ValueError(f"Market-1501 attribute file lacks columns: {sorted(missing)}")
