"""Visualization tests on a tiny synthetic video + database (CPU only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reid.config import load_config
from reid.storage import db
from reid.types import CameraInfo, Tracklet, TrackPoint
from reid.viz import colors, heatmap, journey, render, timeline

W, H, FPS, N = 160, 120, 10.0, 40


def make_video(path: Path, n: int = N, fps: float = FPS) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        f = np.full((H, W, 3), 90, np.uint8)
        cv2.rectangle(f, (10 + 2 * i, 30), (30 + 2 * i, 90), (200, 80, 60), -1)
        vw.write(f)
    vw.release()


def add_tracklet(conn, run, cam, f0, f1, gid, fps=FPS, start_ts=0.0, face=False):
    pts = [
        TrackPoint(f, start_ts + f / fps, 10 + 2 * f, 30, 30 + 2 * f, 90, 0.9)
        for f in range(f0, f1)
    ]
    t = Tracklet(cam, 1, f0, f1 - 1, start_ts + f0 / fps, start_ts + (f1 - 1) / fps)
    t.gid, t.points = gid, pts
    t.assign_source = "body" if gid else ""
    t.face_quality = 0.8 if face else None
    db.insert_tracklets(conn, run, [t])
    return t


@pytest.fixture
def env(tmp_path):
    cfg = load_config(None, {"paths.data_dir": str(tmp_path)})
    conn = db.connect(tmp_path / "reid.db")
    run = db.create_run(conn, cfg.to_json())
    for cid, start in (("a", 0.0), ("b", 0.0)):
        make_video(tmp_path / f"{cid}.mp4")
        db.upsert_camera(
            conn, CameraInfo(cid, f"cam_{cid}", str(tmp_path / f"{cid}.mp4"), FPS, W, H, start)
        )
    for g in (1, 2):
        db.create_identity(conn, g)
    add_tracklet(conn, run, "a", 0, 20, 1, face=True)
    add_tracklet(conn, run, "a", 5, 30, 2)
    add_tracklet(conn, run, "b", 28, 40, 1)
    add_tracklet(conn, run, "b", 2, 10, None)
    db.refresh_identity_stats(conn)
    yield cfg, conn, tmp_path
    conn.close()


def probe(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    out = (n, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return out


def test_palette_deterministic_and_distinct():
    assert [colors.gid_color(g) for g in range(1, 60)] == [
        colors.gid_color(g) for g in range(1, 60)
    ]
    first = [colors.gid_color(g) for g in range(1, len(colors.PALETTE) + 1)]
    assert len(set(first)) == len(first)
    assert colors.gid_color(None) == colors.UNIDENTIFIED
    assert colors.gid_color(500) not in (colors.UNIDENTIFIED,)
    assert colors.gid_color_bgr(1) == colors.gid_color(1)[::-1]


def test_render_camera_size_frames_and_diff(env):
    cfg, conn, tmp = env
    out = render.render_camera(conn, cfg, "a", tmp / "out" / "a.mp4")
    n, w, h = probe(out)
    assert (n, w, h) == (N, W, H)
    src = cv2.VideoCapture(str(tmp / "a.mp4"))
    dst = cv2.VideoCapture(str(out))
    _, f_src = src.read()
    _, f_dst = dst.read()
    assert np.abs(f_src.astype(int) - f_dst.astype(int)).mean() > 1.0  # boxes / HUD drawn


def test_render_camera_range_scale_highlight(env):
    cfg, conn, tmp = env
    out = render.render_camera(
        conn, cfg, "a", tmp / "r.mp4", start_s=1.0, end_s=2.0, scale=0.5, highlight_gid=1
    )
    assert probe(out) == (10, W // 2, H // 2)
    out2 = render.render_camera(conn, cfg, "a", tmp / "f.mp4", gid_filter=[2], end_s=1.0)
    assert probe(out2)[0] == 10


def test_ffmpeg_h264(env):
    import shutil
    import subprocess

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe missing")
    cfg, conn, tmp = env
    out = render.render_camera(conn, cfg, "a", tmp / "h.mp4", end_s=1.0)
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv", out],
        capture_output=True,
        text=True,
    ).stdout
    assert "h264" in info and "yuv420p" in info


def test_export_gif(env):
    cfg, conn, tmp = env
    mp4 = render.render_camera(conn, cfg, "a", tmp / "g.mp4", end_s=1.0)
    gif = render.export_gif(mp4, tmp / "g.gif", fps=5, width=80)
    assert gif.exists() and gif.read_bytes()[:3] == b"GIF"


def test_frame_index_alignment():
    # cam started 2.0 s later, 20 fps, 100 frames
    assert render.frame_index_at(1.9, 2.0, 20.0, 100) is None
    assert render.frame_index_at(2.0, 2.0, 20.0, 100) == 0
    assert render.frame_index_at(3.0, 2.0, 20.0, 100) == 20
    assert render.frame_index_at(7.0, 2.0, 20.0, 100) is None
    assert render.grid_shape(2) == (2, 1) and render.grid_shape(4) == (2, 2)
    assert render.grid_shape(6) == (3, 2)


def test_mosaic_timeline_and_no_signal(env):
    cfg, conn, tmp = env
    # camera b starts 2 s after a on the shared timeline and runs at another fps
    make_video(tmp / "b.mp4", n=20, fps=5.0)
    db.upsert_camera(conn, CameraInfo("b", "cam_b", str(tmp / "b.mp4"), 5.0, W, H, 2.0))
    out = render.render_mosaic(conn, cfg, ["a", "b"], tmp / "m.mp4", tile_width=160)
    n, w, h = probe(out)
    # timeline: a = [0, 4) s, b = [2, 6) s -> 6 s at out_fps = max fps (10) -> 60 frames
    assert n == 60
    cap = cv2.VideoCapture(str(out))
    frames = [cap.read()[1] for _ in range(n)]
    tw = (w - 3 * 4) // 2
    tile_b = lambda f: f[4 : 4 + (h - 8), 8 + tw : 8 + 2 * tw]  # noqa: E731
    tile_a = lambda f: f[4 : 4 + (h - 8), 4 : 4 + tw]  # noqa: E731
    body = lambda t: t[-35:, :]  # below the centred text  # noqa: E731
    # at t=1 s camera b has no signal (flat dark tile), camera a shows the video
    assert body(tile_b(frames[10])).mean() < 40
    assert body(tile_a(frames[10])).mean() > 60
    # at t=5 s camera a is over, camera b shows video
    assert body(tile_a(frames[50])).mean() < 40
    assert body(tile_b(frames[50])).mean() > 60


def test_plan_segments_matches_journey(env):
    from reid import analytics

    cfg, conn, _ = env
    # gid 1: a [0, 1.9] then b [2.8, 3.9]: no overlap -> exactly the journey order
    segs = journey.plan_segments(conn, 1)
    steps = analytics.journey(conn, 1)
    assert [s.tracklet_id for s in segs] == [s.tracklet_id for s in steps]
    assert [s.camera_id for s in segs] == ["a", "b"]
    assert segs[0].gap_s is None and segs[1].gap_s == pytest.approx(segs[1].t0 - segs[0].t1)
    assert segs[0].t1 <= segs[1].t0


def test_plan_segments_overlap_handoff(env):
    cfg, conn, tmp = env
    run = db.list_runs(conn)[0]["id"]
    db.create_identity(conn, 3)
    add_tracklet(conn, run, "a", 0, 20, 3)  # [0, 1.9]
    add_tracklet(conn, run, "b", 10, 30, 3)  # [1.0, 2.9] overlaps -> hand-off at 2.0
    segs = journey.plan_segments(conn, 3)
    assert [s.camera_id for s in segs] == ["a", "b"]
    assert segs[1].t0 == pytest.approx(segs[0].t1)
    assert segs[1].gap_s == pytest.approx(0.0)
    assert journey.plan_segments(conn, 99) == []


def test_follow_person_clip(env):
    cfg, conn, tmp = env
    cfg = load_config(None, {"paths.data_dir": str(tmp), "render.journey_width": 640})
    out = journey.follow_person(conn, cfg, 1, tmp / "j.mp4")
    segs = journey.plan_segments(conn, 1)
    n, w, h = probe(out)
    assert w == 640 and h % 2 == 0
    assert n == sum(max(1, int((s.t1 - s.t0) * FPS)) for s in segs)


def test_timeline_plots(env, tmp_path):
    cfg, conn, tmp = env
    f1 = timeline.plot_timeline(conn, gid=1, out_path=tmp / "t1.png")
    f2 = timeline.plot_timeline(conn, out_path=tmp / "t2.png")
    assert (tmp / "t1.png").stat().st_size > 1000 and (tmp / "t2.png").stat().st_size > 1000
    assert len(f1.axes[0].get_yticks()) == 2  # two cameras
    assert [t.get_text() for t in f2.axes[0].get_yticklabels()] == ["ID 1", "ID 2", "unassigned"]
    if timeline.HAS_PLOTLY:
        assert timeline.timeline_figure(conn, 1).data
    else:
        with pytest.raises(ImportError):
            timeline.timeline_figure(conn, 1)


def test_heatmap_shapes_background_and_transitions(env, tmp_path):
    cfg, conn, tmp = env
    hm = heatmap.foot_heatmap(conn, cfg, "a")
    assert hm.shape == (round(H * cfg.render.heat_scale), round(W * cfg.render.heat_scale))
    assert hm.max() == pytest.approx(1.0)
    pts = heatmap.foot_points(conn, "a")
    assert len(pts) == 20 + 25  # every point, not one per sighting
    assert len(heatmap.foot_points(conn, "a", gid=2)) == 25
    bg = heatmap.camera_background(conn, cfg, "a")
    assert bg.shape == (H, W, 3) and (tmp / "cache" / "background_a.jpg").exists()
    out = heatmap.render_heatmap(conn, cfg, "a", tmp / "h.png", gid=1)
    assert cv2.imread(str(out)).shape == (H, W, 3)
    fig = heatmap.plot_transitions(conn, tmp / "tr.png")
    assert (tmp / "tr.png").exists() and fig is not None


def test_background_path_extension_mismatch(env):
    cfg, conn, tmp = env
    cv2.imwrite(str(tmp / "bg.jpg"), np.full((H, W, 3), 33, np.uint8))
    db.upsert_camera(
        conn, CameraInfo("a", "cam_a", str(tmp / "a.mp4"), FPS, W, H, 0.0, background_path="bg.png")
    )  # stored as .png but the file is a .jpg (v1 B17)
    assert heatmap.camera_background(conn, cfg, "a").mean() == pytest.approx(33, abs=2)


def test_floor_plan(env):
    cfg, conn, tmp = env
    with pytest.raises(ValueError):
        heatmap.plot_floor_heatmap(conn, cfg)
    db.upsert_camera(
        conn,
        CameraInfo("a", "cam_a", str(tmp / "a.mp4"), FPS, W, H, 0.0,
                   homography=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]),
    )  # fmt: skip
    fp = heatmap.floor_points(conn)
    assert set(fp) == {"a"} and fp["a"].shape == (45, 2)
    heatmap.plot_floor_heatmap(conn, cfg, tmp / "fh.png")
    heatmap.plot_floor_trajectories(conn, tmp / "ft.png")
    assert (tmp / "fh.png").exists() and (tmp / "ft.png").exists()


def test_cli_render_and_journey(env, capsys):
    from reid import cli

    cfg, conn, tmp = env
    conn.close()
    base = ["--data-dir", str(tmp)]
    assert cli.main(["render", *base, "--camera", "a", "--end", "1", "--gif"]) == 0
    assert (tmp / "renders" / "a.mp4").exists() and (tmp / "renders" / "a.gif").exists()
    assert cli.main(["render", *base, "--mosaic", "--gid", "1", "--end", "1"]) == 0
    assert (tmp / "renders" / "mosaic_gid1.mp4").exists()
    assert cli.main(["journey", *base, "--gid", "1", "--set", "render.journey_width=480"]) == 0
    assert (tmp / "renders" / "journey_gid1.mp4").exists()
    assert cli.main(["journey", *base, "--gid", "42", "--no-video"]) == 1
    assert cli.main(["render", *base]) == 1


def test_spotlight_brightness_zero_hides_everything_else():
    """With brightness 0 only the highlighted box area keeps its pixels (hides bystanders)."""
    import numpy as np

    from reid.viz.draw import _spotlight

    frame = np.full((200, 300, 3), 200, np.uint8)
    _spotlight(frame, [(100.0, 50.0, 140.0, 150.0)], 0.0)
    assert frame[5, 5].max() == 0 and frame[195, 290].max() == 0  # far from the box: black
    assert frame[100, 120].min() > 150  # inside the box: kept
    dimmed = np.full((50, 50, 3), 200, np.uint8)
    _spotlight(dimmed, [], 0.62)
    assert int(dimmed[0, 0, 0]) == 124  # default behaviour unchanged


def test_spotlight_margin_zero_keeps_only_the_box():
    import numpy as np

    from reid.viz.draw import _spotlight

    frame = np.full((200, 300, 3), 200, np.uint8)
    _spotlight(frame, [(100.0, 50.0, 140.0, 150.0)], 0.0, 0.0)
    assert frame[100, 120].min() == 200  # inside the box: untouched
    assert frame[100, 150].max() == 0 and frame[45, 120].max() == 0  # just outside: black


def test_privacy_blur_blurs_others_and_keeps_the_highlighted(tmp_path):
    import numpy as np

    from reid.viz.privacy import blur_boxes, load_privacy_boxes

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    before = img.copy()
    blur_boxes(img, [(10.0, 10.0, 50.0, 90.0)], 0.35)
    assert np.abs(img[20:80, 15:45].astype(int) - before[20:80, 15:45]).mean() > 10  # blurred
    assert (img[:, 120:] == before[:, 120:]).all()  # elsewhere untouched
    csv = tmp_path / "gt.csv"
    csv.write_text("camera,frame,gid,x1,y1,x2,y2\ncam_a,5,1,0,0,10,20\ncam_a,5,2,30,0,40,20\n")
    idx = load_privacy_boxes(csv, keep=[2])
    assert len(idx) == 1 and len(idx.get(("cam_a", 5))) == 1  # id 2 kept unblurred
    assert idx.get(("cam_a", 6)) is None and idx.get(("cam_b", 5)) is None


def test_dets_at_holds_boxes_between_strided_frames():
    """frame_stride 2 stores points on even frames only; odd frames must still show the box."""
    import numpy as np

    from reid.viz.draw import CameraTracks

    dt = [
        ("tracklet_id", "i8"),
        ("frame", "i8"),
        ("x1", "f4"),
        ("y1", "f4"),
        ("x2", "f4"),
        ("y2", "f4"),
    ]
    pts = np.array([(1, f, 0, 0, 10, 20) for f in range(0, 20, 2)], dtype=dt)
    trk = object.__new__(CameraTracks)
    trk.pts, trk._frames, trk.gid_of, trk.face_gids = pts, pts["frame"], {1: 5}, set()
    trk.trail_len, trk.hold = 30, 3
    trk._tr = {1: (pts["frame"], np.zeros((len(pts), 2)))}
    assert [d.gid for d in trk.dets_at(4)] == [5]
    assert [d.gid for d in trk.dets_at(5)] == [5]  # odd frame: held from frame 4
    assert trk.dets_at(30) == []  # long after the tracklet ended: nothing


def test_blur_keep_gids_blurs_everyone_else_and_keeps_own_box_sharp():
    """No highlight: blur_keep_gids stay sharp; overlap outside the kept box does not blur it."""
    import numpy as np

    from reid.viz.draw import Det, Style, _privacy_blur

    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (120, 300, 3), dtype=np.uint8)
    keep, other, near = (
        (20.0, 20.0, 60.0, 110.0),
        (200.0, 20.0, 240.0, 110.0),
        (64.0, 20.0, 100.0, 110.0),
    )

    def det(gid, box, tid):
        return Det(tid, gid, box, np.zeros((0, 2)))

    st = Style(blur_others=True, blur_strength=0.35, blur_keep_gids=frozenset({6}))
    img = base.copy()
    _privacy_blur(
        img, "c", 0, [det(6, keep, 1), det(7, other, 2), det(8, near, 3)], st, 1.0, (0, 0)
    )

    def diff(b):
        return np.abs(
            img[b[1] : b[3], b[0] : b[2]].astype(int) - base[b[1] : b[3], b[0] : b[2]]
        ).mean()

    assert (
        diff((20, 20, 60, 110)) == 0
    )  # kept person: sharp, though a blur halo of "near" reaches it
    assert diff((205, 30, 235, 100)) > 10  # other person: blurred
    assert diff((70, 30, 95, 100)) > 10  # neighbour: blurred
