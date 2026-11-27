"""Find: ranking, strict mode, any-of, must-not, camera/time filters, grouping, UI catalogue."""

import numpy as np
import pytest

from reid import find
from reid.attributes import ui_labels as ui
from reid.attributes.schema import INDEX, LABELS
from tests.attr_synth import annotated_env

ROWS = [
    # camera, t0, t1, gid, probs
    ("c1", 0, 10, 1, {"female": 0.95, "upper_red": 0.9, "backpack": 0.1}),
    ("c2", 20, 30, 1, {"female": 0.9, "upper_red": 0.6, "backpack": 0.8}),  # same person, again
    ("c1", 40, 50, 2, {"female": 0.2, "upper_red": 0.92, "backpack": 0.9}),  # male, red, backpack
    ("c2", 60, 70, None, {"female": 0.97, "upper_pink": 0.85}),  # unidentified, pink
    ("c1", 80, 90, 3, {"female": 0.55, "upper_blue": 0.9}),
]


@pytest.fixture
def env(conn):
    rec, ids = annotated_env(conn, ROWS)
    return conn, rec, ids


def _tids(res):
    return [h.tracklet_id for h in res.hits]


def test_ranked_order_and_match_pct(env):
    conn, _, ids = env
    res = find.find(conn, find.FindQuery(require={"female": True, "upper_red": True}))
    assert res.total_tracklets == res.n_in_scope == len(ROWS) and len(res.hits) == len(ROWS)
    assert _tids(res)[:3] == [ids[0], ids[1], ids[2]]  # woman in red first, then the red man
    top = res.hits[0]
    assert top.score == pytest.approx((np.log(0.95) + np.log(0.9)) / 2)
    assert top.match_pct == pytest.approx(100 * np.exp(top.score))
    assert [h.score for h in res.hits] == sorted((h.score for h in res.hits), reverse=True)
    assert res.n_matching == 2 and {h.tracklet_id for h in res.hits if h.passes} == set(ids[:2])


def test_strict_keeps_only_passing(env):
    conn, _, ids = env
    q = find.FindQuery(require={"female": True, "upper_red": True}, strict=True)
    res = find.find(conn, q)
    assert set(_tids(res)) == {ids[0], ids[1]} and res.n_matching == 2
    assert res.total_tracklets == len(ROWS)


def test_must_not_and_any_of(env):
    conn, _, ids = env
    res = find.find(conn, find.FindQuery(require={"backpack": False}, strict=True))
    assert set(_tids(res)) == {ids[0], ids[3], ids[4]}
    q = find.FindQuery(require={"female": True}, any_of=[["upper_red", "upper_pink"]], strict=True)
    assert set(_tids(find.find(conn, q))) == {ids[0], ids[1], ids[3]}
    top = find.find(conn, find.FindQuery(any_of=[["upper_red", "upper_pink"]])).hits[0]
    assert top.score == pytest.approx(np.log(0.92))  # log of the best member


def test_probability_clipping(conn):
    _, ids = annotated_env(conn, [("c1", 0, 5, 1, {"female": 0.0})])
    h = find.find(conn, find.FindQuery(require={"female": True})).hits[0]
    assert h.score == pytest.approx(np.log(1e-4)) and h.match_pct > 0


def test_camera_and_time_filters(env):
    conn, _, ids = env
    res = find.find(conn, find.FindQuery(cameras=["c2"]))
    assert set(_tids(res)) == {ids[1], ids[3]} and res.n_in_scope == 2
    res = find.find(conn, find.FindQuery(t_from=45, t_to=65))  # overlap, not containment
    assert set(_tids(res)) == {ids[2], ids[3]}
    assert find.find(conn, find.FindQuery(cameras=["nope"])).hits == []


def test_no_terms_orders_by_time(env):
    conn, _, ids = env
    res = find.find(conn, find.FindQuery())
    assert _tids(res) == ids and all(h.match_pct == 100.0 and h.passes for h in res.hits)


def test_unknown_label_and_empty_group(env):
    conn = env[0]
    with pytest.raises(ValueError, match="unknown attribute"):
        find.find(conn, find.FindQuery(require={"flying": True}))
    with pytest.raises(ValueError):
        find.find(conn, find.FindQuery(any_of=[[]]))


def test_group_by_person_and_limit(env):
    conn, _, ids = env
    res = find.find(conn, find.FindQuery(require={"female": True, "upper_red": True}))
    people = find.group_by_person(res.hits)
    assert len(people) == 4  # gids 1, 2, 3 and the unidentified tracklet
    first = people[0]
    assert first.gid == 1 and first.best.tracklet_id == ids[0] and first.score == first.best.score
    assert [h.tracklet_id for h in first.hits] == [ids[0], ids[1]] and first.cameras == ["c1", "c2"]
    assert sum(p.gid is None for p in people) == 1
    assert [p.score for p in people] == sorted((p.score for p in people), reverse=True)
    lim = find.find(conn, find.FindQuery(require={"female": True}, limit_people=2))
    assert {h.gid for h in lim.hits} == {1, None} and len(lim.hits) == 3


def test_person_attributes_weighted_mean(env):
    conn, rec, ids = env
    pa = find.person_attributes(conn, 1)
    assert pa["female"] == pytest.approx((0.95 + 0.9) / 2)  # equal n_samples
    assert pa["backpack"] == pytest.approx((0.1 + 0.8) / 2)
    table = find.load_table(conn)
    assert find.person_attributes(conn, 1, table)["upper_red"] == pytest.approx(0.75)
    assert find.person_attributes(conn, 999) == {}
    conn.execute("UPDATE tracklet_attributes SET n_samples=3 WHERE tracklet_id=?", (ids[1],))
    conn.execute("UPDATE tracklet_attributes SET n_samples=1 WHERE tracklet_id=?", (ids[0],))
    assert find.person_attributes(conn, 1)["female"] == pytest.approx((0.95 + 3 * 0.9) / 4)


def test_table_reuse_and_no_annotations(env, tmp_path):
    conn = env[0]
    table = find.load_table(conn)
    assert len(table) == len(ROWS) and table.total_tracklets == len(ROWS)
    a = find.find(conn, find.FindQuery(require={"backpack": True}), table)
    assert a.hits[0].tracklet_id == env[2][2]
    from reid.storage import db

    empty = db.connect(tmp_path / "e.db")
    res = find.find(empty, find.FindQuery(require={"female": True}))
    assert res.head is None and res.hits == [] and find.load_table(empty) is None


def test_catalogue_matches_schema():
    assert len(ui.RELIABLE) == 41 and ui.RELIABLE <= set(LABELS)
    for item in ui.UI_LABELS.values():
        assert item.label in INDEX
        assert item.reliable == (item.label in ui.RELIABLE)
    assert all(i.hex for s in ui.SECTIONS if s.kind == "swatches" for i in s.items)
    always = [i.id for i in ui.UI_LABELS.values() if i.always_show]
    assert {"child", "adult", "shoulder_bag"} <= set(always)
    sec = next(s for s in ui.SECTIONS if s.id == "carrying")
    assert [i.id for i in ui.visible_items(sec)] == ["backpack", "handbag", "shoulder_bag"]
    # an unreliable, not-always-shown label is hidden until "experimental filters" is on
    sec, item = next(
        (s, i) for s in ui.SECTIONS for i in s.items if not i.reliable and not i.always_show
    )
    assert item.id not in [i.id for i in ui.visible_items(sec)]
    assert item.id in [i.id for i in ui.visible_items(sec, show_experimental=True)]


def test_selection_terms():
    req, any_of = ui.selection_terms(["male", "backpack", "upper_red", "upper_pink", "lower_blue"])
    assert req == {"female": False, "backpack": True}
    assert any_of == [["upper_red", "upper_pink"], ["lower_blue"]]
    req, any_of = ui.selection_terms(["child", "adult", "female", "male"])
    assert req == {} and any_of == [["child", "adult"]]
    with pytest.raises(ValueError):
        ui.selection_terms(["bogus"])


def test_describe():
    probs = {"female": 0.2, "adult": 0.1, "child": 0.8, "senior": 0.1, "upper_red": 0.9,
             "upper_blue": 0.2, "backpack": 0.7, "handbag": 0.3, "long_sleeve": 0.6,
             "short_sleeve": 0.4}  # fmt: skip
    chips = {c.id: c for c in ui.describe(probs)}
    assert chips["male"].text == "Male (apparent)" and chips["male"].prob == pytest.approx(0.8)
    assert chips["child"].text == "Child (apparent)" and not chips["child"].reliable
    assert chips["upper_red"].text == "Upper colour: Red" and "upper_blue" not in chips
    assert "backpack" in chips and "handbag" not in chips and "short_sleeve" not in chips
