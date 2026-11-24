"""Pose-to-box matching and face quality (pure numpy)."""

import numpy as np
import pytest

from reid.models.face import face_quality, frontalness, size_factor
from reid.models.pose import box_iou, match_pose_to_box


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    assert box_iou(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_pose_matches_target_not_first_or_neighbour():
    # crop with the target in the middle and a bigger neighbour listed first (v1 took index 0)
    target = np.array([40, 10, 80, 110], np.float32)
    poses = np.array([[0, 0, 45, 120], [38, 12, 82, 108], [90, 0, 130, 120]], np.float32)
    assert match_pose_to_box(poses, target) == 1


def test_pose_requires_target_centre_inside_pose_box():
    target = np.array([40, 10, 80, 110], np.float32)  # centre (60, 60)
    poses = np.array([[0, 0, 55, 120], [90, 0, 130, 120]], np.float32)  # neither contains it
    assert match_pose_to_box(poses, target) == -1
    assert match_pose_to_box(np.zeros((0, 4)), target) == -1


def test_pose_prefers_overlap_among_containing_boxes():
    target = np.array([40, 10, 80, 110], np.float32)
    poses = np.array([[0, 0, 200, 200], [42, 12, 78, 108]], np.float32)  # big one also contains
    assert match_pose_to_box(poses, target) == 1


def test_size_factor():
    box = lambda s: np.array([0, 0, s, s * 1.2], np.float32)  # noqa: E731
    assert size_factor(box(20), 32, 112) == 0.0
    assert size_factor(box(112), 32, 112) == 1.0
    assert size_factor(box(200), 32, 112) == 1.0
    assert size_factor(box(72), 32, 112) == pytest.approx(0.5)


def _kps(nose_dx=0.0, nose_y=0.11):
    return np.array(
        [[-25, -10], [25, -10], [nose_dx, nose_y * 100], [-20, 25], [20, 25]], np.float32
    )


def test_frontalness_orders_poses():
    frontal, three_q, profile = frontalness(_kps()), frontalness(_kps(15)), frontalness(_kps(30))
    assert frontal > three_q > profile >= 0.0
    assert frontal == pytest.approx(1.0, abs=0.05)
    assert frontalness(np.zeros((5, 2))) == 0.0


def test_face_quality_combines_factors():
    box = np.array([0, 0, 112, 112], np.float32)
    q = face_quality(0.9, box, _kps(), 32, 112)
    assert q == pytest.approx(0.9, abs=0.05)
    assert face_quality(0.9, box * 0.2, _kps(), 32, 112) == 0.0  # below min size
    assert face_quality(0.9, box, _kps(30), 32, 112) < q
