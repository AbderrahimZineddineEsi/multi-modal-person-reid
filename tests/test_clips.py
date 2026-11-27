"""Clip planning, layout, cost model and rendering on the tiny synthetic environment (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from reid.config import load_config
from reid.storage import db
from reid.types import CameraInfo
from reid.viz import clip_cost, journey
from reid.viz.clip_layout import MIN_TILE_H, make_layout
from reid.viz.clip_plan import ClipOptions, plan_clip
from reid.viz.draw import CameraTracks, Style, draw_view
from tests.test_viz import FPS, H, W, add_tracklet, env, make_video, probe  # noqa: F401


def add_camera(conn, tmp, cid, start=0.0):
    make_video(tmp / f"{cid}.mp4")
    db.upsert_camera(conn, CameraInfo(cid, f"cam_{cid}", str(tmp / f"{cid}.mp4"), FPS, W, H, start))


def run_id(conn) -> int:
    return db.list_runs(conn)[0]["id"]


def test_camera_filter_limits_segments_and_thumbnails(env):  # noqa: F811
    cfg, conn, tmp = env
    add_camera(conn, tmp, "c")
    add_tracklet(conn, run_id(conn), "c", 0, 30, 1)  # gid 1 also in c, overlapping a
    every = plan_clip(conn, 1, ClipOptions())
    assert set(every.camera_ids) == {"a", "b", "c"}
    only = plan_clip(conn, 1, ClipOptions(cameras=("a", "b")))
    assert set(only.camera_ids) <= {"a", "b"}
    assert all(set(s.thumbs) <= {"a", "b"} - {s.camera_id} for s in only.segments)
    assert [c for c, _ in only.cameras] == ["a", "b"]
    assert plan_clip(conn, 1, ClipOptions(cameras=())).empty
    assert plan_clip(conn, 1, ClipOptions(cameras=("zzz",))).empty


def test_window_clips_segments_and_drops_the_first_gap(env):  # noqa: F811
    cfg, conn, _ = env
    full = plan_clip(conn, 1, ClipOptions())
    assert len(full.segments) == 2 and full.segments[1].gap_s is not None
    win = plan_clip(conn, 1, ClipOptions(start_s=3.0, end_s=3.5))
    assert [s.camera_id for s in win.segments] == ["b"]
    assert win.segments[0].t0 == pytest.approx(3.0) and win.segments[0].t1 == pytest.approx(3.5)
    assert win.segments[0].gap_s is None
    assert plan_clip(conn, 1, ClipOptions(start_s=100.0)).empty


def test_speed_scales_frame_count(env):  # noqa: F811
    cfg, conn, _ = env
    one = plan_clip(conn, 1, ClipOptions(speed=1.0, fps=10))
    two = plan_clip(conn, 1, ClipOptions(speed=2.0, fps=10))
    four = plan_clip(conn, 1, ClipOptions(speed=4.0, fps=10))
    assert one.out_frames > two.out_frames > four.out_frames
    # every segment is at most one frame off half / a quarter of the speed-1 length
    for a, b in zip(one.segments, two.segments, strict=True):
        assert abs(b.n_frames - a.n_frames / 2) <= 1
    assert two.source_seconds == pytest.approx(one.source_seconds)
    assert two.out_seconds == pytest.approx(two.out_frames / 10)


def test_max_length_truncates_and_says_so(env):  # noqa: F811
    cfg, conn, _ = env
    full = plan_clip(conn, 1, ClipOptions(fps=10))
    capped = plan_clip(conn, 1, ClipOptions(fps=10, max_seconds=1.5))
    assert capped.truncated and not full.truncated
    assert capped.out_frames == 15 and capped.out_seconds == pytest.approx(1.5)
    assert capped.full_out_seconds == pytest.approx(full.out_seconds)
    assert capped.segments[-1].t1 < full.segments[-1].t1 or len(capped.segments) < 2
    assert not plan_clip(conn, 1, ClipOptions(fps=10, max_seconds=999)).truncated


def test_unidentified_tracklets_and_unknown_people_are_ignored(env):  # noqa: F811
    cfg, conn, _ = env  # the fixture has an unassigned tracklet in camera b (frames 2..10)
    plan = plan_clip(conn, 1, ClipOptions())
    null_ids = {t.id for t in db.get_tracklets(conn) if t.gid is None}
    assert null_ids and not null_ids & {s.tracklet_id for s in plan.segments}
    assert plan_clip(conn, 99, ClipOptions()).empty


@pytest.mark.parametrize("width", [320, 640, 1280, 1920, 3840])
def test_layout_never_produces_tiny_or_negative_tiles(width):
    for n_cams in range(1, 31):
        for aspect in (9 / 16, 3 / 4, 1.0):
            lay = make_layout(width, aspect, n_cams, n_cams - 1)
            assert lay.main_w > 0 and lay.main_h > 0 and lay.height % 2 == 0
            if lay.n_thumbs:
                assert lay.tile_h >= MIN_TILE_H and 0 < lay.tile_w <= lay.strip_w
                used = lay.n_thumbs * (lay.tile_h + lay.margin)
                assert used <= lay.main_h - lay.card_h + 1
            else:
                assert lay.tile_h == 0 and lay.tile_w == 0
    assert make_layout(1280, 9 / 16, 3, 3, strip=False).strip_w == 0


def test_estimate_grows_with_frames_and_thumbnails(env):  # noqa: F811
    cfg, conn, tmp = env
    for cid in ("c", "d"):
        add_camera(conn, tmp, cid)
        add_tracklet(conn, run_id(conn), cid, 0, 40, 1)
    est = lambda **kw: clip_cost.estimate(plan_clip(conn, 1, ClipOptions(fps=10, **kw)))  # noqa: E731
    assert est(speed=1) > est(speed=2) > est(speed=4) > 0
    assert est(max_thumbs=0) < est(max_thumbs=1) < est(max_thumbs=3)
    assert est(width=1280) > est(width=640)
    assert est(blur_others=True) > est(blur_others=False)
    assert est(max_seconds=1) < est()
    assert clip_cost.estimate(plan_clip(conn, 99, ClipOptions())) == 0.0


def test_calibration_scales_later_estimates(env, tmp_path):  # noqa: F811
    cfg, conn, _ = env
    plan = plan_clip(conn, 1, ClipOptions(fps=10))
    assert clip_cost.load_model(tmp_path).factor == 1.0
    modelled = clip_cost.modelled_seconds(plan)
    entry = clip_cost.record(tmp_path, plan, seconds=3 * modelled, frames=plan.out_frames)
    assert entry["factor"] == pytest.approx(3.0, rel=1e-3)
    assert (tmp_path / "cache" / "clip_speed.json").exists()
    model = clip_cost.load_model(tmp_path)
    assert clip_cost.estimate(plan, model) == pytest.approx(3 * modelled, rel=1e-3)
    clip_cost.record(tmp_path, plan, seconds=modelled, frames=plan.out_frames)  # blends
    assert 1.0 < clip_cost.load_model(tmp_path).factor < 3.0
    assert clip_cost.record(tmp_path, plan, 0.0, 0) == {}
    assert clip_cost.fmt_seconds(42) == "42 s" and clip_cost.fmt_seconds(185) == "3m 05s"


def test_options_do_not_mutate_the_shared_config():
    cfg = load_config(None, {})
    opts = ClipOptions(blur_others=True, width=800, fps=12.0)
    copy = opts.apply_cfg(cfg)
    assert copy.render.blur_others and copy.render.journey_width == 800
    assert not cfg.render.blur_others and cfg.render.journey_width != 800
    assert opts.key() != ClipOptions().key() and ClipOptions().key() == ClipOptions().key()
    assert ClipOptions(speed=1000, width=10).clean().speed == 64.0
    assert ClipOptions(width=10).clean().width == 320


def test_render_matches_plan(env):  # noqa: F811
    cfg, conn, tmp = env
    cfg = load_config(None, {"paths.data_dir": str(tmp)})
    for kw in (
        {"width": 480},
        {"width": 480, "speed": 2.0},
        {"width": 480, "thumbs": False, "blur_others": True, "show_others": False},
        {"width": 480, "cameras": ("a",), "max_seconds": 1.0},
    ):
        opts = ClipOptions(**kw)
        plan = plan_clip(conn, 1, opts)
        out = tmp / "e2e.mp4"
        res = journey.render_clip(conn, cfg, 1, out, opts, plan=plan)
        n, w, h = probe(out)
        assert n == plan.out_frames == res.frames
        assert (w, h) == (plan.layout.width, plan.layout.height)
        assert res.seconds > 0 and set(res.timings) == {"main", "thumbs", "encode"}
    assert not cfg.render.blur_others  # render_clip never touches the caller's config


def test_render_errors(env):  # noqa: F811
    cfg, conn, tmp = env
    with pytest.raises(ValueError):
        journey.render_clip(conn, cfg, 99, tmp / "x.mp4")
    (tmp / "b.mp4").unlink()
    with pytest.raises(FileNotFoundError):  # camera b is a main view of gid 1
        journey.render_clip(conn, cfg, 1, tmp / "x.mp4", ClipOptions(width=480))
    # a missing video that is only a thumbnail does not stop the clip
    out = journey.render_clip(
        conn, cfg, 1, tmp / "y.mp4", ClipOptions(width=480, end_s=1.5, cameras=("a", "b"))
    )
    assert out.frames > 0


def test_other_boxes_off_draws_only_the_highlighted_person(env, monkeypatch):  # noqa: F811
    from reid.viz import draw

    cfg, conn, _ = env
    trk = CameraTracks(conn, "a", 0, 40, 30)
    frame = np.full((H, W, 3), 90, np.uint8)
    seen: list[list[int | None]] = []
    real = draw.draw_dets
    monkeypatch.setattr(
        draw,
        "draw_dets",
        lambda f, dets, *a, **k: (seen.append([d.gid for d in dets]), real(f, dets, *a, **k)),
    )
    st = Style.from_cfg(cfg, highlight_gid=2)
    draw_view(frame.copy(), trk, 12, (W, H), st)
    st.draw_others = False
    draw_view(frame.copy(), trk, 12, (W, H), st)
    assert sorted(seen[0]) == [1, 2] and seen[1] == [2]


def test_blur_still_hides_people_whose_boxes_are_not_drawn(env):  # noqa: F811
    cfg, conn, _ = env
    trk = CameraTracks(conn, "b", 0, 40, 30)  # frame 5: only an unidentified person is in view
    noisy = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    plain = Style.from_cfg(cfg, highlight_gid=1)
    plain.draw_others = False
    blurred = Style.from_cfg(load_config(None, {"render.blur_others": True}), highlight_gid=1)
    blurred.draw_others = False
    a = draw_view(noisy.copy(), trk, 5, (W, H), plain, hud=False)
    b = draw_view(noisy.copy(), trk, 5, (W, H), blurred, hud=False)
    box = (slice(40, 80), slice(22, 38))  # inside the person's box (x 20..40, y 30..90)
    assert b[box].std() < 0.5 * a[box].std()
