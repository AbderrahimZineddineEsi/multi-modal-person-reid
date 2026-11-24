"""HTML builders of the UI (pure functions, no Gradio needed)."""

import numpy as np

from reid.app import components as C
from reid.viz.colors import gid_color, hex_color


def test_formatting():
    assert C.fmt_dur(4.3) == "4.3 s"
    assert C.fmt_dur(125) == "2m 05s"
    assert C.fmt_clock(65.34) == "1:05.3"
    assert C.plural(1, "camera") == "1 camera" and C.plural(2, "camera") == "2 cameras"


def test_metric_tile_escapes_and_meter():
    html = C.metric_tile("<b>People</b>", "12", "hint & more", pct=250)
    assert "&lt;b&gt;People" in html and "<b>People" not in html
    assert "hint &amp; more" in html
    assert 'style="width:100%"' in html  # clamped
    assert "rd-off" in C.metric_tile("x", "1")  # no meter requested: invisible placeholder


def test_status_badges():
    assert "rd-ok" in C.status_badge("MATCH") and "Match" in C.status_badge("MATCH")
    assert "rd-warn" in C.status_badge("AMBIGUOUS")
    assert "rd-dim" in C.status_badge("UNKNOWN") and "Unknown person" in C.status_badge("UNKNOWN")
    assert "No face found" in C.status_badge("NO_FACE")


def test_identity_card_uses_person_colour_and_escapes_label():
    p = C.PersonCard(
        3, "<script>x</script>", 2, 2, 7.5, has_face=True, thumb="data:image/jpeg;base64,AA"
    )
    html = C.identity_card(p)
    assert hex_color(gid_color(3)) in html
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "face confirmed" in html and 'src="data:image/jpeg' in html
    assert "2 sightings in 2 cameras" in html
    anon = C.identity_card(C.PersonCard(5, None, 1, 1, 1.0))
    assert "Person 5" in anon and "body only" in anon and "rd-initial" in anon


def test_person_grid_empty_and_filled():
    assert "No people yet" in C.person_grid([])
    html = C.person_grid([C.PersonCard(1), C.PersonCard(2)])
    assert html.count("rd-person") >= 2 and "rd-grid" in html


def test_journey_badges():
    steps = [
        C.StepView("Door", 0.0, 4.0, None, "face"),
        C.StepView("Aisle", 3.0, 9.0, -1.0, "body"),
        C.StepView("Exit", 20.0, 25.0, 11.0, "consolidate"),
    ]
    html = C.journey_html(2, steps)
    assert "first seen" in html and "overlaps by 1.0 s" in html and "after 11.0 s" in html
    assert html.index("Door") < html.index("Aisle") < html.index("Exit")
    assert "No sightings" in C.journey_html(2, [])


def test_candidates_and_transitions():
    html = C.candidates_html([(1, 0.9), (2, 0.5)], {1: "Ana", 2: "Bo"}, best=1)
    assert "rd-best" in html and "Ana" in html and "0.90" in html and "width:50%" in html
    assert C.candidates_html([], {}, None) == ""
    mat = np.array([[0, 3], [1, 0]])
    t = C.transitions_html(["A", "B"], mat)
    assert ">3<" in t and "--a:1.00" in t and "--a:0.33" in t
    assert "No transitions yet" in C.transitions_html(["A"], np.zeros((1, 1)))


def test_cameras_and_header():
    html = C.cameras_html(
        [C.CameraRow("Door", "640x480 @ 25 fps", 1, True), C.CameraRow("B", "x", 3)]
    )
    assert "face cam" in html and "1 person" in html and "3 people" in html
    head = C.page_header("reid", "sub", [C.badge("admin mode", "warn")])
    assert "<h1>reid</h1>" in head and "admin mode" in head
