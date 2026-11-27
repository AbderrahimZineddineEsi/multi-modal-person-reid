"""Clips tab logic (data, preview, cached render) and ``reid journey --plan``, CPU only."""

from __future__ import annotations

import json

import pytest

from reid.app import clips_data as D
from reid.app import clips_view as V
from reid.app.tabs import clips as clips_tab
from reid.viz.clip_plan import ClipOptions
from tests.test_app_handlers import ctx  # noqa: F401  (the fixture)


def options(**kw) -> ClipOptions:
    base = dict(width=480, fps=10.0)
    return ClipOptions(**{**base, **kw})


def test_scope_choices_and_prefill(ctx):  # noqa: F811
    with ctx.db() as conn:
        sc = D.person_scope(conn, 1)
        assert [i for _, i in sc.cameras] == ["a", "b"] and sc.default == ["a", "b"]
        assert 0 <= sc.lo < sc.hi
        assert D.person_scope(conn, 2).default == ["a"]
        assert D.person_scope(conn, None).default == []  # nobody picked: footage range
        assert [g for _, g in D.people_choices(conn)] == [1, 2]
    p = D.make_prefill(1, ["a"], 1.0, None)
    assert p and p["gid"] == 1 and p["cameras"] == ["a"] and p["end"] is None
    assert D.make_prefill(1)["cameras"] is None  # empty selection = the person's cameras
    assert D.make_prefill(None) is None
    assert D.make_prefill(1)["nonce"] != 0


def test_scope_updates_keep_and_clamp(ctx):  # noqa: F811
    cams, t_from, t_to, limits = clips_tab.scope_updates(ctx, 1, ["b", "zzz"], -50, 1e9)
    assert cams["value"] == ["b"]  # unknown cameras dropped
    assert t_from["value"] == limits[0] and t_to["value"] == limits[1]  # clamped
    cams, t_from, t_to, _ = clips_tab.scope_updates(ctx, 1, ["zzz"], 3.0, 2.0)
    assert cams["value"] == ["a", "b"] and t_from["value"] < t_to["value"]  # fell back
    assert clips_tab.people_update(ctx, 2)["value"] == 2
    assert clips_tab.people_update(ctx, 77)["value"] == 1  # vanished person: first one
    assert "value" not in clips_tab.people_update(ctx, None, keep_value=True)


def test_to_options_zero_means_off():
    o = D.to_options(["a"], 1.0, 2.0, 4, 0, True, False, True, 3, 640, 0)
    assert o.cameras == ("a",) and o.max_seconds is None and o.fps is None
    assert o.speed == 4.0 and o.blur_others and not o.show_others and o.max_thumbs == 3
    assert D.to_options([], None, None, 1, 60, False, True, False, 0, 640, 12).cameras == ()


def test_preview_and_notices(ctx):  # noqa: F811
    plan, est = D.preview_plan(ctx, 1, options())
    assert est > 0 and not plan.empty
    tiles = V.tiles_html(plan, est, False)
    assert "Clip length" in tiles and "Render time" in tiles and f"{plan.layout.width}x" in tiles
    assert "plotly" in type(V.timetable_figure(plan)).__module__
    assert V.plan_table_html(plan).count("<tr>") == len(plan.segments) + 1
    assert V.notices_html(ctx, plan, est) == ""  # videos exist, nothing capped
    capped, est2 = D.preview_plan(ctx, 1, options(max_seconds=1.0))
    assert "capped" in V.notices_html(ctx, capped, est2)
    empty, _ = D.preview_plan(ctx, 1, options(cameras=()))
    assert "Nothing to render" in V.notices_html(ctx, empty, 0.0)
    assert V.tiles_html(empty, 0.0, False) == "" and V.timetable_figure(empty) is None
    assert "Reused" in V.result_note(plan, est, None) and "Rendered in" in V.result_note(
        plan, est, 5
    )


def test_missing_video_is_a_notice_not_a_crash(ctx):  # noqa: F811
    plan, est = D.preview_plan(ctx, 2, options())  # gid 2 only appears in camera a
    (ctx.data_dir / "b.mp4").unlink()
    assert V.notices_html(ctx, plan, est) == ""  # b is not needed for this clip
    plan1, est1 = D.preview_plan(ctx, 1, options())
    assert "Source video not available" in V.notices_html(ctx, plan1, est1)
    with pytest.raises(FileNotFoundError):
        D.render_cached(ctx, 1, options(), lambda *a: None)


def test_render_cached_keys_every_option_and_calibrates(ctx):  # noqa: F811
    calls: list[tuple[int, int]] = []
    path, plan, secs = D.render_cached(ctx, 1, options(), lambda d, t: calls.append((d, t)))
    assert path.exists() and secs and secs > 0 and "clip_gid1_" in path.name
    cal = json.loads((ctx.data_dir / "cache" / "clip_speed.json").read_text())
    assert cal["last_frames"] == plan.out_frames and cal["seconds_per_frame"] > 0
    assert ctx.clip_model().factor == pytest.approx(cal["factor"])
    again, _, secs2 = D.render_cached(ctx, 1, options(), lambda *a: None)
    assert again == path and secs2 is None  # cache hit
    keys = [options(speed=2.0), options(blur_others=True), options(show_others=False),
            options(thumbs=False), options(width=640), options(cameras=("a",)),
            options(max_seconds=1.0), options(start_s=3.0)]  # fmt: skip
    paths = {D.render_cached(ctx, 1, o, lambda *a: None)[0] for o in keys}
    assert len(paths) == len(keys) and path not in paths
    assert not ctx.cfg.render.blur_others  # the shared config was never touched
    with pytest.raises(ValueError):
        D.render_cached(ctx, 1, options(cameras=()), lambda *a: None)


def test_cli_journey_plan(ctx, capsys):  # noqa: F811
    from reid import cli

    base = ["journey", "--data-dir", str(ctx.data_dir), "--gid", "1", "--plan"]
    assert cli.main([*base, "--speed", "2", "--max-seconds", "0.5", "--cameras", "cam_a"]) == 0
    out = capsys.readouterr().out
    assert "estimated render time" in out and "capped" in out and "cam_a" in out
    assert "cam_b" not in out.split("plan:")[1]
    assert not list(ctx.data_dir.glob("renders/journey_gid1*"))  # nothing rendered
    assert cli.main([*base, "--cameras", "nope"]) == 1
    full = ["journey", "--data-dir", str(ctx.data_dir), "--gid", "1", "--thumbs", "0"]
    assert cli.main([*full, "--width", "480", "--speed", "4", "--blur-others",
                     "--no-other-boxes"]) == 0  # fmt: skip
    assert (ctx.data_dir / "renders" / "journey_gid1.mp4").exists()
    assert (ctx.data_dir / "cache" / "clip_speed.json").exists()
