"""Plain dataclasses shared across the package (see docs/dev/PLAN.md section 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in source-frame pixels


@dataclass(slots=True)
class Detection:
    """A person detection in one frame."""

    box: Box
    conf: float
    cls: int = 0


@dataclass(slots=True)
class TrackPoint:
    """One tracked box of a tracklet at one frame (every tracked frame is persisted)."""

    frame: int
    ts: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    sampled: bool = False

    @property
    def box(self) -> Box:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box (floor contact point)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(slots=True)
class Sample:
    """A quality-gated body observation with its embedding (in-memory)."""

    frame: int
    quality: float
    embedding: np.ndarray | None = None
    crop: np.ndarray | None = None


@dataclass(slots=True)
class FaceObs:
    """A face observation paired to a tracklet (in-memory)."""

    frame: int
    box: Box
    det_score: float
    quality: float
    embedding: np.ndarray | None = None
    image: np.ndarray | None = None


@dataclass(slots=True)
class CameraInfo:
    """Persisted camera record (``cameras`` table)."""

    id: str
    name: str
    video_path: str
    fps: float
    width: int = 0
    height: int = 0
    start_ts: float = 0.0
    is_face_cam: bool = False
    homography: list[list[float]] | None = None
    background_path: str | None = None


@dataclass
class Tracklet:
    """A track of one person in one camera, between appearance and loss.

    DB columns come first; the trailing fields are in-memory only (never stored in
    ``tracklets``): ``points`` go to ``track_points``, embeddings go to the vector store.
    """

    camera_id: str
    local_track_id: int
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    id: int | None = None
    run_id: int | None = None
    n_points: int = 0
    n_samples: int = 0
    mean_quality: float = 0.0
    gid: int | None = None
    assign_score: float | None = None
    assign_source: str = ""  # 'body' | 'face' | 'consolidate' | 'manual' | ''
    best_crop_path: str | None = None
    best_face_path: str | None = None
    face_quality: float | None = None
    # in-memory
    points: list[TrackPoint] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    faces: list[FaceObs] = field(default_factory=list)
    best_crop: np.ndarray | None = None
    body_embedding: np.ndarray | None = None  # quality-weighted mean, L2-normalized
    face_embedding: np.ndarray | None = None

    @property
    def embeddings(self) -> np.ndarray:
        """Stacked sample embeddings, shape (n, D); empty (0, 0) if none."""
        embs = [s.embedding for s in self.samples if s.embedding is not None]
        return np.stack(embs) if embs else np.zeros((0, 0), dtype=np.float32)

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


@dataclass(slots=True)
class Identity:
    """A global person identity (``identities`` table)."""

    gid: int
    created_at: float = 0.0
    label: str | None = None
    first_ts: float | None = None
    last_ts: float | None = None
    n_tracklets: int = 0
    has_face: bool = False


class SearchStatus(str, Enum):
    """Outcome of a search."""

    MATCH = "MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    NO_FACE = "NO_FACE"


@dataclass(slots=True)
class SearchHit:
    """Result of a body or face search for one query."""

    status: SearchStatus
    gid: int | None = None  # best guess; set for MATCH and AMBIGUOUS
    score: float = 0.0
    modality: str = "body"  # 'body' | 'face'
    second_gid: int | None = None
    second_score: float | None = None
    ratio: float | None = None  # top-2 softmax ratio (body only)
    candidates: list[tuple[int, float]] = field(default_factory=list)  # (gid, score), desc
    journey: list[Any] = field(default_factory=list)  # filled by analytics
