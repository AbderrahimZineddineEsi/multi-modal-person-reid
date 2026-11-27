"""App construction and tab handlers against a synthetic data_dir (CPU, no models)."""

import gradio as gr
import numpy as np
import pytest

from reid import app
from reid.app import environments as envs
from reid.app import theme, views
from reid.app.state import AppContext
from reid.app.tabs import admin, cameras, explore, metrics, overview, search
from reid.config import load_config
from reid.identity.associate import IdentityEngine
from reid.storage import db
from tests.synth import centers, jitter, make_env, make_tracklet
from tests.test_viz import FPS, H, W, add_tracklet, make_video


@pytest.fixture
def ctx(tmp_path):
    """Two cameras with real tiny videos, two identities, one unidentified tracklet."""
    cfg = load_config(None, {"paths.data_dir": str(tmp_path)})
    conn = db.connect(tmp_path / "reid.db")
    run = db.create_run(conn, cfg.to_json())
    for cid in ("a", "b"):
        make_video(tmp_path / f"{cid}.mp4")
        from reid.types import CameraInfo

        db.upsert_camera(
            conn, CameraInfo(cid, f"cam_{cid}", str(tmp_path / f"{cid}.mp4"), FPS, W, H, 0.0)
        )
    for g in (1, 2):
        db.create_identity(conn, g)
    add_tracklet(conn, run, "a", 0, 20, 1, face=True)
    add_tracklet(conn, run, "a", 5, 30, 2)
    add_tracklet(conn, run, "b", 28, 40, 1)
    add_tracklet(conn, run, "b", 2, 10, None)
    db.refresh_identity_stats(conn)
    conn.close()
    return AppContext(cfg)


class Progress:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))


def tab_labels(demo: gr.Blocks) -> list[str]:
    return [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]


def test_build_app_is_lazy_and_has_expected_tabs(tmp_path):
    cfg = load_config(None, {"paths.data_dir": str(tmp_path)})
    demo = app.build_app(cfg)
    assert demo.ctx.models_loaded() is False
    assert tab_labels(demo) == [
        "Overview",
        "Process",
        "Explore",
        "Search",
        "Find",
        "Cameras",
        "Metrics",
        "Admin",
    ]


def test_theme_and_css_ship_with_package():
    style = theme.launch_style()
    assert isinstance(style["theme"], theme.ReidTheme)
    assert ".rd-tile" in style["css"]
    for name in theme.CSS_FILES:  # keep the stylesheet split by concern, not one monolith
        assert len((theme.STATIC / name).read_text().splitlines()) < 200


def test_launch_requires_auth_when_sharing(tmp_path, monkeypatch):
    seen = {}

    def fake_launch(self, **kw):
        seen.update(kw)

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    cfg = load_config(None, {"paths.data_dir": str(tmp_path)})
    _, creds = app.launch(cfg, share=True, block=False)
    assert creds is not None and creds[0] == "reid" and len(creds[1]) >= 8
    assert seen["auth"] == creds and seen["share"] is True
    _, creds = app.launch(cfg, share=False, block=False)
    assert creds is None and seen["auth"] is None
    _, creds = app.launch(cfg, share=True, auth=("u", "p"), block=False)
    assert creds == ("u", "p")
    assert str(tmp_path.parent.resolve()) in seen["allowed_paths"]  # the workspace


def test_views_and_overview(ctx):
    with ctx.db() as conn:
        cards = views.people_cards(ctx, conn)
        assert [c.gid for c in cards] == [1, 2] and cards[0].n_cameras == 2
        assert views.person_choices(cards) == [("Person 1", 1), ("Person 2", 2)]
        assert [r.name for r in views.camera_rows(conn)] == ["cam_a", "cam_b"]
    tiles, grid, side = overview.render(ctx)
    assert "People" in tiles and "Identified" in tiles and grid.count("rd-person") >= 2
    assert "cam_a" in side


def test_overview_empty(tmp_path):
    empty = AppContext(load_config(None, {"paths.data_dir": str(tmp_path / "new")}))
    assert "Nothing processed yet" in overview.render(empty)[0]


def test_render_cache_hits_and_invalidates(ctx):
    calls = []

    def make(p):
        calls.append(p)
        p.write_bytes(b"x")

    with ctx.db() as conn:
        a = ctx.cached(conn, "k", {"n": 1}, ".bin", make)
        b = ctx.cached(conn, "k", {"n": 1}, ".bin", make)
        assert a == b and len(calls) == 1
        ctx.cached(conn, "k", {"n": 2}, ".bin", make)
        assert len(calls) == 2
        conn.execute("UPDATE tracklets SET gid=2 WHERE gid=1")  # identities changed
        c = ctx.cached(conn, "k", {"n": 1}, ".bin", make)
        assert c != a and len(calls) == 3


def test_explore_show_and_clip(ctx):
    items, gids, drop = explore.refresh_people(ctx, None)
    assert drop["value"] == 1 and set(gids) <= {1, 2}
    profile, journey, timeline, clip = explore.show(ctx, 1)
    assert "Person 1" in profile and "cam_a" in journey and clip is None
    assert "no longer exists" in str(pytest.raises(gr.Error, explore.show, ctx, 99).value)
    prog = Progress()
    path = explore.make_clip(ctx, 1, prog)
    assert path.endswith(".mp4") and prog.calls and "follow_gid1_" in path
    assert explore.make_clip(ctx, 1, prog) == path  # cached


def test_cameras_handlers(ctx):
    cam, people = cameras.choices(ctx)
    assert cam["value"] == "a" and people["choices"][0] == ("Everyone", 0)
    assert "rd-matrix" in cameras.handoffs(ctx)
    prog = Progress()
    assert "camera_a_" in cameras.video(ctx, "a", 0, 1.0, prog)
    assert "camera_a_gid2_" in cameras.video(ctx, "a", 2, 1.0, prog)
    assert cameras.mosaic(ctx, 1, 1.0, prog).endswith(".mp4")
    with pytest.raises(gr.Error):
        cameras.video(ctx, None, 0, 1.0, prog)


def test_metrics_view_and_eval(ctx):
    run_update = metrics.run_choices(ctx)
    assert run_update["value"] == 1
    kpis, table, mem, frag = metrics.run_view(ctx, 1)
    assert (
        "People" in kpis
        and table["visible"] is False
        and "No timings" in mem
        and "Person 1" in frag
    )
    ctx.save_run_stats(1, {"fps": 12.5, "ms_per_frame": {"detect": 10.0}, "vram_peak_mb": {}})
    kpis, table, *_ = metrics.run_view(ctx, 1)
    assert "12.5 fps" in kpis and table["visible"] is True
    assert "No evaluation yet" in metrics.eval_results(ctx)[0]
    (ctx.data_dir / "eval").mkdir(exist_ok=True)
    (ctx.data_dir / "eval" / "a.json").write_text(
        '{"mtmc": {"idf1": 0.5, "mota": 0.4, "id_switches": 3},'
        ' "retrieval": {"rank1": 0.9, "mAP": 0.8}}'
    )
    tiles, data = metrics.eval_results(ctx)
    assert "50.0%" in tiles and "90.0%" in tiles and data["mtmc"]["idf1"] == 0.5


def test_admin_actions(ctx):
    msg, *_ = admin.do_rename(ctx, 1, "  Ana ")
    assert "Renamed" in msg
    with ctx.db() as conn:
        assert views.person_names(conn)[1] == "Ana"
    with pytest.raises(gr.Error):
        admin.do_merge(ctx, 1, None)
    with pytest.raises(gr.Error):
        admin.do_wipe(ctx, "not the name")
    assert (ctx.data_dir / "reid.db").exists()
    msg, *_ = admin.do_wipe(ctx, ctx.env)
    assert "emptied" in msg
    with ctx.db() as conn:
        assert db.list_identities(conn) == []


def test_admin_merge_and_split(tmp_path):
    rng = np.random.default_rng(0)
    c = centers(2, seed=5, shared=0.0)
    env = make_env(tmp_path)
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    ts = [make_tracklet(env, "c1", 10 * i, 10 * i + 5, c[i % 2], rng) for i in range(4)]
    for t in ts:
        eng.on_tracklet(t)
    eng.end_run(env.run_id)
    ctx = AppContext(env.cfg)
    with ctx.db() as conn:
        gids = [i.gid for i in db.list_identities(conn)]
        tl = db.get_tracklets(conn, gid=gids[0])
    assert len(gids) >= 2
    msg, *_ = admin.do_merge(ctx, gids[0], gids[1])
    assert "Merged" in msg
    msg, *_ = admin.do_split(ctx, tl[0].id)
    assert "Created" in msg


class _FakeBody:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, crops):
        return np.stack([self.vec for _ in crops])


class _FakeModels:
    face = None

    def __init__(self, vec):
        self.body = _FakeBody(vec)


def test_search_handler_with_fake_models(tmp_path):
    rng = np.random.default_rng(1)
    c = centers(3, seed=21, shared=0.0)
    env = make_env(tmp_path, **{"search.body_min_cos": 0.5})
    eng = IdentityEngine(env.cfg, env.conn, env.vectors)
    for p in range(3):
        for k, cam in enumerate(("c1", "c2")):
            t = make_tracklet(env, cam, 100 * p + 40 * k, 100 * p + 40 * k + 10, c[p], rng)
            eng.on_tracklet(t)
    eng.end_run(env.run_id)
    ctx = AppContext(env.cfg)
    ctx.models = lambda cfg=None: _FakeModels(jitter(c[1], 1, 0.1, rng)[0].astype(np.float32))
    img = np.zeros((40, 20, 3), np.uint8)
    for mode in ("Body", "Both"):
        html, gid, btn = search.run_search(ctx, img, "", mode, False, Progress())
        assert "Match" in html and gid is not None and btn["visible"] is True
        assert "Open in Explore" not in html and "rd-query" in html
    with pytest.raises(gr.Error):
        search.run_search(ctx, None, "", "Body", False, Progress())
    with pytest.raises(gr.Error):
        search.run_search(ctx, None, "ftp://x/y.jpg", "Body", False, Progress())


def test_read_only_hides_delete_tools(tmp_path):
    cfg = load_config(None, {"paths.data_dir": str(tmp_path / "e")})
    full = app.build_app(cfg)
    ro = app.build_app(cfg, read_only=True)

    def buttons(demo):
        return {b.value for b in demo.blocks.values() if isinstance(b, gr.Button)}

    assert "Delete person" in buttons(full) and "Delete person" not in buttons(ro)


def test_environments_switch_and_refresh(tmp_path, ctx):
    """Environments next to the fixture's data_dir; switching re-points the context."""
    ws = ctx.workspace
    assert ctx.env == ctx.data_dir.name and ctx.cfg.data_dir == ctx.data_dir
    envs.create_environment(ws, "other", "second")
    names = [e.name for e in ctx.environments()]
    assert ctx.env in names and "other" in names
    assert "2 cameras" in next(e for e in ctx.environments() if e.name == ctx.env).stats
    first = ctx.env
    ctx.select_env("other")
    assert ctx.data_dir == ws / "other" and ctx.cfg.data_dir == ws / "other"
    assert "Nothing processed yet" in overview.render(ctx)[0]
    assert explore.refresh_people(ctx)[2]["value"] is None
    ctx.select_env(first)
    assert overview.render(ctx)[1].count("rd-person") >= 2
    with pytest.raises(ValueError):
        ctx.select_env("missing")


def test_admin_delete_actions(ctx):
    ws = ctx.workspace
    with pytest.raises(gr.Error):
        admin.do_delete_person(ctx, 1, False, False)  # not confirmed
    msg, *_ = admin.do_delete_person(ctx, 1, False, True)
    assert "Deleted person 1" in msg
    with ctx.db() as conn:
        assert [i.gid for i in db.list_identities(conn)] == [2]
    msg, *_ = admin.do_delete_run(ctx, 1, True)
    assert "Deleted run 1" in msg
    with ctx.db() as conn:
        assert db.list_runs(conn) == [] and db.get_tracklets(conn) == []
    envs.create_environment(ws, "throwaway")
    ctx.select_env("throwaway")
    with pytest.raises(gr.Error):
        admin.do_delete_env(ctx, "nope")
    assert "deleted" in admin.do_delete_env(ctx, "throwaway")
    assert ctx.env != "throwaway" and not (ws / "throwaway").exists()
