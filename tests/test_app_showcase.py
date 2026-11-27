"""Showcase mode: the allow-list policy, blurred thumbnails, render config and cache keys."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from reid.app import showcase as sc
from reid.app import views
from reid.app.media import thumb_uri
from reid.app.showcase_gt import gt_keep_ids
from reid.app.state import AppContext
from reid.app.tabs import cameras
from reid.config import load_config
from reid.storage import attributes_db as adb
from reid.storage import db, files
from reid.types import CameraInfo, Tracklet, TrackPoint
from reid.viz.privacy import strong_blur
from tests.attr_synth import LABELS, make_head
from tests.test_viz import FPS, H, W, add_tracklet, make_video

FEMALE = LABELS.index("female")


class Progress:
    def __call__(self, *a, **k):
        pass


def policy(env: str, allow, max_female: float = 0.6) -> sc.Showcase:
    return sc.Showcase(max_female, {env: None if allow == "all" else frozenset(allow)})


def build_env(tmp_path, female: dict[int, float | None], showcase=None, allow=(1,)):
    """cams a, b; tracklets 1 (gid 1, a), 2 (gid 2, a), 3 (gid 1, b), 4 (no gid, b), all with a
    textured crop on disk; ``female`` gives P(female) per tracklet id (None: no attribute row)."""
    cfg = load_config(None, {"paths.data_dir": str(tmp_path)})
    conn = db.connect(tmp_path / "reid.db")
    run = db.create_run(conn, cfg.to_json())
    for cid in ("a", "b"):
        make_video(tmp_path / f"{cid}.mp4")
        db.upsert_camera(
            conn, CameraInfo(cid, f"cam_{cid}", str(tmp_path / f"{cid}.mp4"), FPS, W, H, 0.0)
        )
    for g in (1, 2):
        db.create_identity(conn, g)
    add_tracklet(conn, run, "a", 0, 20, 1)
    add_tracklet(conn, run, "a", 5, 30, 2)
    add_tracklet(conn, run, "b", 28, 40, 1)
    add_tracklet(conn, run, "b", 2, 10, None)
    db.refresh_identity_stats(conn)
    rng = np.random.default_rng(0)
    for tid in (1, 2, 3, 4):
        p = files.crop_path(tmp_path, tid)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (120, 60, 3), dtype=np.uint8))
        conn.execute(
            "UPDATE tracklets SET best_crop_path=? WHERE id=?",
            (files.rel_to_data(tmp_path, p), tid),
        )
    rec = adb.upsert_head(conn, make_head())
    rows = []
    for tid, f in female.items():
        if f is not None:
            probs = np.full(len(LABELS), 0.02, np.float32)
            probs[FEMALE] = f
            rows.append((tid, probs, 4))
    adb.upsert_tracklet_attributes(conn, rec.id, rows)
    conn.commit()
    conn.close()
    pol = showcase if showcase is not None else policy(tmp_path.name, allow)
    return AppContext(cfg, showcase=pol)


MEN = {1: 0.1, 2: 0.1, 3: 0.2}  # no row for tracklet 4


def test_policy_decisions():
    p = policy("env", [3, 4])
    assert p.permits("env", 3, 0.1)
    assert not p.permits("env", 5, 0.1)  # gid not on the list
    assert not p.permits("env", None, 0.1)  # unidentified under a list
    assert not p.permits("other", 3, 0.1)  # environment not listed: everybody blurred
    assert not p.permits("env", 3, 0.6)  # female guard: strictly below max_female
    assert not p.permits("env", 3, float("nan"))
    assert p.permits("env", 3, None)  # no attribute row: the gid decides
    everyone = policy("env", "all", max_female=0.5)
    assert everyone.permits("env", 99, 0.2) and everyone.permits("env", None, None)
    assert not everyone.permits("env", 99, 0.7)
    assert policy("a", [1]).key() != policy("a", [2]).key()


def test_policy_file(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(
        json.dumps(
            {"max_female": 0.5, "envs": {"x": {"allow_gids": [1, 2]}, "y": {"allow_gids": "all"}}}
        )
    )
    p = sc.Showcase.load(f)
    assert p.max_female == 0.5 and p.envs == {"x": frozenset({1, 2}), "y": None}
    f.write_text(json.dumps({"envs": {"x": {"allow_gids": "some"}}}))
    with pytest.raises(ValueError):
        sc.Showcase.load(f)
    with pytest.raises(ValueError):
        sc.Showcase.load(tmp_path / "missing.json")


def test_strong_blur_hides_detail_and_keeps_size():
    img = np.random.default_rng(1).integers(0, 255, (120, 60, 3), dtype=np.uint8)
    out = strong_blur(img)
    assert out.shape == img.shape
    assert out.astype(float).std() < 0.2 * img.astype(float).std()


def test_view_blurs_everything_not_proven_allowed(tmp_path):
    ctx = build_env(tmp_path, {1: 0.1, 2: 0.1, 3: 0.9})
    with ctx.db() as conn:
        view = ctx.show_view(conn)
    assert view.is_sharp(views.crop_rel_id(ctx, 1))  # allowed gid, male
    assert not view.is_sharp(views.crop_rel_id(ctx, 2))  # other gid
    assert not view.is_sharp(views.crop_rel_id(ctx, 3))  # allowed gid but female guard
    assert not view.is_sharp(views.crop_rel_id(ctx, 4))  # unidentified
    assert not view.is_sharp("crops/00000/unknown.jpg") and not view.is_sharp(None)
    assert view.is_sharp(tmp_path / views.crop_rel_id(ctx, 1))  # absolute paths are mapped
    assert view.gids == frozenset()  # a gid with one failing tracklet is hidden in videos


def test_unlisted_environment_is_all_blurred(tmp_path):
    ctx = build_env(tmp_path, MEN, showcase=policy("elsewhere", "all"))
    with ctx.db() as conn:
        view = ctx.show_view(conn)
        assert view.sharp == frozenset() and view.gids == frozenset()
        assert all(
            c.thumb != thumb_uri(views.crop_rel(ctx, t), tmp_path)
            for c in views.people_cards(ctx, conn)
            for t in db.get_tracklets(conn, gid=c.gid)
        )


def test_thumbs_differ_between_blurred_and_sharp_and_cache_separately(tmp_path):
    ctx = build_env(tmp_path, MEN)
    rel1, rel2 = views.crop_rel_id(ctx, 1), views.crop_rel_id(ctx, 2)
    sharp = thumb_uri(rel1, tmp_path)
    blurred = thumb_uri(rel1, tmp_path, blur=True)
    assert sharp != blurred and thumb_uri(rel1, tmp_path) == sharp  # cache key has the decision
    with ctx.db() as conn:
        view = ctx.show_view(conn)
        assert sc.thumb(view, rel1, tmp_path) == sharp
        assert sc.thumb(view, rel2, tmp_path) == thumb_uri(rel2, tmp_path, blur=True)
        cards = {c.gid: c for c in views.people_cards(ctx, conn)}
        steps = views.journey_steps(ctx, conn, 2)
    assert cards[1].thumb == sharp
    assert cards[2].thumb == thumb_uri(rel2, tmp_path, blur=True) != thumb_uri(rel2, tmp_path)
    assert steps[0].thumb == thumb_uri(rel2, tmp_path, 120, blur=True)


def test_gallery_items_use_blurred_copies(tmp_path):
    ctx = build_env(tmp_path, MEN)
    with ctx.db() as conn:
        items, _ = views.gallery_items(ctx, conn)
    paths = dict(zip([n for _, n in items], [p for p, _ in items], strict=True))
    assert paths["Person 1"] == str(files.crop_path(tmp_path, 1))
    copy = paths["Person 2"]
    assert "renders/app/showcase" in copy and copy != str(files.crop_path(tmp_path, 2))
    assert cv2.imread(copy).shape == cv2.imread(str(files.crop_path(tmp_path, 2))).shape


def test_off_mode_is_unchanged(tmp_path):
    ctx = build_env(tmp_path, MEN, showcase=policy("x", []))
    ctx.showcase = None
    with ctx.db() as conn:
        assert ctx.show_view(conn) is None
        assert ctx.render_cfg(conn, 1) is ctx.cfg
        cards = views.people_cards(ctx, conn)
        items, _ = views.gallery_items(ctx, conn)
        key = json.dumps(["k", {"a": 1}, ctx.fingerprint(conn)], sort_keys=True, default=str)
        out = ctx.cached(conn, "k", {"a": 1}, ".txt", lambda p: p.write_text("x"))
    assert cards[0].thumb == thumb_uri(views.crop_rel_id(ctx, 1), tmp_path)
    assert items[0][0] == str(files.crop_path(tmp_path, 1))
    assert out.name == f"k_{hashlib.sha1(key.encode()).hexdigest()[:8]}.txt"  # same key as before


def test_render_cfg_copy_and_highlight(tmp_path):
    ctx = build_env(tmp_path, MEN, allow=(1, 2))
    before = ctx.cfg.render.model_dump()
    with ctx.db() as conn:
        cfg = ctx.render_cfg(conn, 1)
        other = ctx.render_cfg(conn, 99)
        none = ctx.render_cfg(conn)
    r = cfg.render
    assert r.blur_others and r.blur_strength == 1.0 and r.blur_keep_gids == [1, 2]
    assert not r.blur_highlight and other.render.blur_highlight and not none.render.blur_highlight
    assert r.privacy_gt == "" and ctx.cfg.render.model_dump() == before and cfg is not ctx.cfg


def test_render_cfg_with_gt_keeps_only_allowed_ids(tmp_path):
    ctx = build_env(tmp_path, MEN, allow=(1,))
    (tmp_path / "gt.csv").write_text(
        "camera,frame,gid,x1,y1,x2,y2\n"
        + "".join(f"a,{f},7,{10 + 2 * f},30,{30 + 2 * f},90\n" for f in range(0, 20, 4))
    )
    with ctx.db() as conn:
        # tracklets 1 (gid 1) and 2 (gid 2) have identical boxes in camera a on frames 5..19:
        # make the GT person unambiguous by removing tracklet 2's points there
        conn.execute("DELETE FROM track_points WHERE tracklet_id=2")
        conn.commit()
        cfg = ctx.render_cfg(conn, 1)
    assert cfg.render.privacy_gt == str(tmp_path / "gt.csv")
    assert cfg.render.privacy_keep_gt == [7]


def test_gt_keep_ids_majority_vote(tmp_path):
    conn = db.connect(tmp_path / "reid.db")
    run = db.create_run(conn, "{}")
    db.upsert_camera(conn, CameraInfo("a", "a", "a.mp4", 10.0, 640, 480))
    for gid, x0 in ((1, 0.0), (2, 200.0), (3, 400.0)):
        t = Tracklet("a", gid, 0, 39, 0.0, 3.9, gid=gid)
        t.points = [TrackPoint(f, f / 10, x0, 10, x0 + 50, 150, 0.9) for f in range(40)]
        db.insert_tracklets(conn, run, [t])
    rows = ["camera,frame,gid,x1,y1,x2,y2"]
    for f in range(0, 40, 4):
        rows += [f"a,{f},10,0,10,50,150", f"a,{f},20,200,10,250,150", f"a,{f},30,900,10,950,150"]
    (tmp_path / "gt.csv").write_text("\n".join(rows) + "\n")
    # GT 10 -> gid 1 (allowed), GT 20 -> gid 2 (not), GT 30 matches nothing (blurred)
    assert gt_keep_ids(conn, tmp_path / "gt.csv", frozenset({1})) == [10]
    assert gt_keep_ids(conn, tmp_path / "gt.csv", frozenset({1, 2, 3})) == [10, 20]


def test_cache_key_includes_showcase(tmp_path):
    ctx = build_env(tmp_path, MEN)
    on = ctx.showcase
    made = []

    def make(p):
        made.append(p)
        p.write_text("x")

    with ctx.db() as conn:
        a = ctx.cached(conn, "k", {"a": 1}, ".txt", make)
        ctx.showcase = None
        off = ctx.cached(conn, "k", {"a": 1}, ".txt", make)
        ctx.showcase = policy(tmp_path.name, [2])
        other = ctx.cached(conn, "k", {"a": 1}, ".txt", make)
        ctx.showcase = on
        again = ctx.cached(conn, "k", {"a": 1}, ".txt", make)
    assert len({a, off, other}) == 3 and again == a and len(made) == 3


def test_camera_render_in_showcase_uses_blurring_copy(tmp_path, monkeypatch):
    ctx = build_env(tmp_path, MEN, allow=(1,))
    seen = []

    def fake_render(conn, cfg, camera, path, **kw):
        seen.append((cfg, kw))
        path.write_bytes(b"v")

    monkeypatch.setattr(cameras, "render_camera", fake_render)
    out = cameras.video(ctx, "a", 2, 5, Progress())
    cfg, kw = seen[0]
    assert cfg is not ctx.cfg and cfg.render.blur_others and cfg.render.blur_highlight  # gid 2
    assert cfg.render.blur_keep_gids == [1] and kw["highlight_gid"] == 2
    assert not ctx.cfg.render.blur_others
    ctx.showcase = None
    off = cameras.video(ctx, "a", 2, 5, Progress())
    assert off != out and seen[1][0] is ctx.cfg


def test_real_camera_render_runs_with_blur(tmp_path):
    ctx = build_env(tmp_path, MEN, allow=(1,))
    out = cameras.video(ctx, "a", None, 1, Progress())
    assert cv2.VideoCapture(out).read()[0]


def _frame(path, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, img = cap.read()
    cap.release()
    assert ok
    return img.astype(float)


def test_camera_pixels_are_blurred_in_showcase(tmp_path):
    ctx = build_env(tmp_path, MEN, allow=())
    blurred = cameras.video(ctx, "a", None, 2, Progress())
    ctx.showcase = None
    plain = cameras.video(ctx, "a", None, 2, Progress())
    x0, x1 = 10 + 2 * 12, 30 + 2 * 12  # a tracked person at frame 12
    box = (slice(40, 80), slice(x0, x1))
    assert np.abs(_frame(blurred, 12)[box] - _frame(plain, 12)[box]).mean() > 3.0


def test_clip_in_showcase_renders_with_blurred_person_card(tmp_path):
    from reid.app import clips_data as D
    from reid.viz.clip_plan import ClipOptions

    ctx = build_env(tmp_path, MEN, allow=(1,))
    opts = ClipOptions(width=480, fps=10.0)
    path, plan, _ = D.render_cached(ctx, 2, opts, lambda *a: None)  # gid 2 is not allowed
    assert path.exists() and plan.options.blur_others  # forced on although opts said no
    assert not ctx.cfg.render.blur_others and not ctx.cfg.render.blur_highlight
    ctx.showcase = None
    plain, _, _ = D.render_cached(ctx, 2, opts, lambda *a: None)
    assert plain != path


def test_highlighted_person_is_blurred_when_not_allowed(tmp_path):
    from reid.viz.render import render_camera

    ctx = build_env(tmp_path, MEN, allow=())
    with ctx.db() as conn:
        shown = ctx.render_cfg(conn, 1)
        shown.render.blur_highlight = False  # what the highlight exemption would have done
        hidden = ctx.render_cfg(conn, 1)
        assert hidden.render.blur_highlight
        render_camera(conn, shown, "a", tmp_path / "shown.mp4", end_s=2, highlight_gid=1)
        render_camera(conn, hidden, "a", tmp_path / "hidden.mp4", end_s=2, highlight_gid=1)
    box = (slice(25, 95), slice(8, 50))  # around tracklet 1 (gid 1) at frame 3, edges included
    diff = np.abs(_frame(tmp_path / "shown.mp4", 3)[box] - _frame(tmp_path / "hidden.mp4", 3)[box])
    assert diff.mean() > 1.0 and diff.max() > 30
