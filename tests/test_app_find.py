"""Find tab: query handlers, cached table, HTML builders and wiring (synthetic, CPU, no models)."""

import gradio as gr
import pytest

from reid import app
from reid.app import find_css, find_data, find_views, theme
from reid.app.find_data import Selection
from reid.app.state import AppContext
from reid.app.tabs import find as find_tab
from reid.config import load_config
from reid.storage import db
from tests.attr_synth import annotated_env

ROWS = [  # camera, t0, t1, gid, probabilities
    ("a", 0, 10, 1, {"female": 0.95, "upper_red": 0.9, "backpack": 0.9}),
    ("b", 20, 30, 1, {"female": 0.9, "upper_red": 0.8}),
    ("a", 5, 15, 2, {"female": 0.1, "upper_blue": 0.9}),
    ("b", 40, 50, None, {"female": 0.92, "upper_red": 0.7}),
    ("a", 60, 70, 3, {"female": 0.6, "upper_pink": 0.6}),
]


@pytest.fixture
def ctx(tmp_path):
    cfg = load_config(None, {"paths.data_dir": str(tmp_path)})
    conn = db.connect(tmp_path / "reid.db")
    for g in (1, 2, 3):
        db.create_identity(conn, g)
    annotated_env(conn, ROWS)
    db.refresh_identity_stats(conn)
    conn.close()
    return AppContext(cfg)


def test_selection_and_query(ctx):
    sel = find_tab.to_selection(
        ["a"], 0.0, 1e9, True, "Appearances", ["female"], ["upper_red"], None
    )
    assert sel.chips == ("female", "upper_red") and sel.strict and sel.view == "Appearances"
    assert find_tab.to_selection([], 0, 0, False, "nonsense").view == "People"
    with ctx.db() as conn:
        table = find_data.get_table(ctx, conn)
    q = find_data.build_query(sel, table)
    assert q.cameras == ["a"] and q.t_from is None and q.t_to is None  # bounds = no bound
    assert q.require == {"female": True} and q.any_of == [["upper_red"]]
    q = find_data.build_query(Selection(t_from=30.0, t_to=40.0), table)
    assert (q.t_from, q.t_to) == (30.0, 40.0)
    assert find_data.time_bounds(table) == (0.0, 70.0)
    with pytest.raises(ValueError):
        find_data.build_query(Selection(chips=("nope",)), table)


def test_table_is_cached_and_rebuilt_on_change(ctx):
    with ctx.db() as conn:
        first = find_data.get_table(ctx, conn)
        assert find_data.get_table(ctx, conn) is first
        conn.execute("UPDATE tracklets SET gid=2 WHERE gid=3")  # an identity edit
        conn.commit()
        second = find_data.get_table(ctx, conn)
    assert second is not first and 3 not in second.gids


def test_run_find_ranks_people(ctx):
    out = find_data.run_find(ctx, Selection(chips=("female", "upper_red")))
    assert out.ready and out.result.total_tracklets == 5
    assert [p.gid for p in out.people][:2] == [1, None]  # unidentified stays its own entry
    assert out.result.n_matching == 3
    strict = find_data.run_find(ctx, Selection(chips=("male", "upper_red"), strict=True))
    assert strict.ready and strict.people == []
    assert find_data.find_person(out, "g1").gid == 1
    assert find_data.find_person(out, "g99") is None
    assert find_data.person_key(out.people[1]).startswith("t")


def test_render_people_appearances_and_states(ctx):
    state, annotate, main, stats, grid, more, *labels = find_tab.render(
        ctx, Selection(chips=("female", "upper_red"))
    )
    assert state == "" and main["visible"] and annotate["visible"] is False
    assert "<b>2</b>" in stats and "appearances" in stats and "Ranked" in stats
    assert grid.count("rd-fcard") == 4 and "rdPick('g1')" in grid  # every person, best first
    assert "rd-miss" in grid and "Dashed cards" in grid  # a close candidate that misses a filter
    assert labels[0]["label"] == "Person (1)" and labels[1]["label"] == "Upper body (1)"
    _, _, _, _, apps, *_ = find_tab.render(ctx, Selection(chips=("female",), view="Appearances"))
    assert apps.count("rd-fcard") == 5  # ranked mode lists every appearance
    _, _, _, stats_none, grid_none, *_ = find_tab.render(ctx, Selection())
    assert "matching" not in grid_none and "rd-fpct" not in grid_none and "<b>4</b>" in stats_none
    _, _, _, _, empty, *_ = find_tab.render(
        ctx, Selection(chips=("male", "upper_red"), strict=True)
    )
    assert "No one passes every filter" in empty


def test_pagination(ctx):
    _, _, _, _, grid, more, *_ = find_tab.render(ctx, Selection(view="Appearances"), shown=2)
    assert grid.count("rd-fcard") == 2 and more["visible"] and "3 left" in more["value"]


def test_not_annotated_and_empty_environments(tmp_path):
    from reid.attributes.head import save_head
    from tests.attr_synth import add_camera, add_tracklet, make_head

    head = tmp_path / "head.npz"
    save_head(head, make_head(in_dim=1280))
    cfg = load_config(None, {"paths.data_dir": str(tmp_path / "e"), "attributes.head": str(head)})
    conn = db.connect(tmp_path / "e" / "reid.db")
    add_camera(conn, "a")
    add_tracklet(conn, db.create_run(conn, "{}"), "a", 0, 5)
    conn.close()
    env = AppContext(cfg)
    state, annotate, main, *_ = find_tab.render(env, Selection())
    assert main["visible"] is False and annotate["visible"] and "not computed" in state
    # no head installed: the explanation replaces the tool and there is nothing to click
    bare = load_config(None, {"paths.data_dir": str(tmp_path / "e"), "attributes.head": "none.npz"})
    env.cfg.attributes.head = bare.attributes.head
    state, annotate, *_ = find_tab.render(env, Selection())
    assert "No attribute head" in state and annotate["visible"] is False
    assert "No attribute head" in find_data.annotate_now(env)
    nothing = AppContext(load_config(None, {"paths.data_dir": str(tmp_path / "new")}))
    assert "Nothing to search yet" in find_tab.render(nothing, Selection())[0]


def test_detail_drawer(ctx):
    sel = Selection(chips=("female", "upper_red"))
    drawer, head, body, gid, to_explore, clip_btn, clip, note = find_tab.open_detail(ctx, "g1", sel)
    assert drawer["visible"] and gid == 1 and to_explore["visible"] and clip_btn["visible"]
    assert "Person 1" in head and "best match" in head and "Apparent" not in head
    assert "Female (apparent)" in body and "Looks like" in body
    assert "2 appearances" in body and clip["visible"] is False and note == ""
    _, head, _, gid, to_explore, *_ = find_tab.open_detail(ctx, "t4", sel)  # unidentified
    assert gid is None and to_explore["visible"] is False and "Unidentified" in head
    assert find_tab.open_detail(ctx, "g99", sel)[0] == gr.update()  # stale card: nothing opens


def test_views_escape_and_format():
    assert find_views.clock(65) == "1:05" and find_views.clock(3725) == "1:02:05"
    assert find_views.pct_text(0.2) == "<1%" and find_views.pct_text(87.4) == "87%"
    pills = find_views.selection_summary(Selection(chips=("male", "upper_red", "upper_orange")))
    assert "Male (apparent)" in pills and "Red or Orange" in pills
    assert find_views.selection_summary(Selection()) == ""


def test_catalogue_css_and_js_ship_with_the_app():
    css = theme.css()
    assert 'input[name="upper_red"]' in css and "--sw: #d7322d" in css
    assert "border-style: dashed" in find_css.catalogue_css()  # unreliable chips are flagged
    assert "rdPick" in theme.launch_style()["js"] and ".rd-drawer" in css


def test_find_tab_is_built(tmp_path):
    demo = app.build_app(load_config(None, {"paths.data_dir": str(tmp_path)}))
    labels = [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert labels.index("Find") == labels.index("Search") + 1


def test_scope_updates_follow_the_data(ctx):
    cams, lo, hi, limits = find_tab.scope_updates(ctx)
    assert [c for _, c in cams["choices"]] == ["a", "b"] and cams["value"] == []
    assert (lo["minimum"], hi["maximum"], limits) == (0.0, 70.0, (0.0, 70.0))
    # a moved bound and a camera choice survive a refresh; a bound still at the end follows it
    cams, lo, hi, _ = find_tab.scope_updates(
        ctx, ["b", "gone"], 10.0, 70.0, (0.0, 70.0), reset=False
    )
    assert cams["value"] == ["b"] and lo["value"] == 10.0 and hi["value"] == 70.0
    _, lo, hi, _ = find_tab.scope_updates(ctx, [], 0.0, 1.0, (0.0, 1.0), reset=False)
    assert hi["value"] == 70.0 and lo["value"] == 0.0  # sliders at the old ends follow
