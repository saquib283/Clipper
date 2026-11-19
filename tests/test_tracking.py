from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import ClipError, TrackingPlan  # noqa: E402
from lib.tracking import (  # noqa: E402
    analyze_video_tracking,
    compute_vertical_crop_width,
    detect_scene_cuts,
    interpolate_x_positions,
    smooth_crop_x0,
)


def test_compute_vertical_crop_width_is_even_and_clamped() -> None:
    assert compute_vertical_crop_width(1920, 1080) == 608
    assert compute_vertical_crop_width(1280, 720) == 404
    assert compute_vertical_crop_width(500, 1000) == 500


def test_smooth_crop_applies_deadzone_ema_and_snaps_on_scene_cut() -> None:
    # 1920 wide, crop_w=608 -> valid x0 in [0, 1312], center x0 = 656
    centers: list[float | None] = [500.0, 505.0, 510.0, 1500.0, 1500.0]
    cuts = {3}
    x0s = smooth_crop_x0(centers, cuts, src_w=1920, crop_w=608, alpha=0.25, deadzone_px=15.0)
    # Frame 0 snaps to target 500 - 304 = 196; frames 1-2 stay in deadzone
    assert x0s[0] == 196
    assert x0s[1] == 196
    assert x0s[2] == 196
    # Scene cut at index 3 snaps immediately to 1500 - 304 = 1196 instead of lagging via EMA
    assert x0s[3] == 1196
    assert x0s[4] == 1196


def test_detect_scene_cuts_and_interpolation() -> None:
    h_a = np.zeros((16, 16), dtype=np.float32)
    h_a[0, 0] = 1.0
    h_b = np.zeros((16, 16), dtype=np.float32)
    h_b[15, 15] = 1.0
    cuts = detect_scene_cuts([h_a, h_a, h_b, h_b], threshold=0.45)
    assert cuts == {2}

    interp = interpolate_x_positions([100, 200], [0, 4], total_frames=6)
    assert interp == [100, 125, 150, 175, 200, 200]

    cut_interp = interpolate_x_positions([100, 500], [0, 4], total_frames=6, scene_cuts={1})
    assert cut_interp == [100, 100, 100, 100, 500, 500]


def test_smooth_and_interpolate_edge_cases() -> None:
    assert smooth_crop_x0([], set(), src_w=1920, crop_w=608) == []
    assert interpolate_x_positions([], [], total_frames=0) == []
    assert interpolate_x_positions([], [], total_frames=3) == [0, 0, 0]

    # Clamping to [0, src_w - crop_w] and EMA step outside deadzone
    x0s = smooth_crop_x0([0.0, 2000.0], set(), src_w=1920, crop_w=608, alpha=0.5, deadzone_px=10.0)
    assert x0s[0] == 0
    # cur_center moves from 0 to 1000 -> x0 = 1000 - 304 = 696
    assert x0s[1] == 696


def test_analyze_video_tracking_reads_video_and_handles_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(ClipError, match="Could not open video") as exc:
        analyze_video_tracking(missing)
    assert exc.value.code == "MEDIA_ERROR"

    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 180),
    )
    assert writer.isOpened()
    try:
        frame_a = np.full((180, 320, 3), 40, dtype=np.uint8)
        frame_b = np.full((180, 320, 3), 210, dtype=np.uint8)
        for _ in range(5):
            writer.write(frame_a)
        for _ in range(5):
            writer.write(frame_b)
    finally:
        writer.release()

    plan = analyze_video_tracking(video_path, sample_fps=5.0)
    assert isinstance(plan, TrackingPlan)
    assert plan.src_width == 320
    assert plan.src_height == 180
    assert plan.crop_width == 100
    assert plan.total_frames == 10
    assert len(plan.x0_per_frame) == 10
    assert plan.face_ratio == 0.0
    # No faces -> defaults to centered crop: 160 - 50 = 110
    assert all(x == 110 for x in plan.x0_per_frame)


def test_analyze_video_tracking_scales_detected_faces_and_falls_back_to_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lib.tracking as tr

    video_path = tmp_path / "hd.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (1280, 720),
    )
    assert writer.isOpened()
    try:
        frame = np.full((720, 1280, 3), 80, dtype=np.uint8)
        for _ in range(4):
            writer.write(frame)
    finally:
        writer.release()

    class StubCascade:
        def __init__(self, path: str) -> None:
            self.is_profile = "profile" in path
            self.calls = 0

        def detectMultiScale(self, gray, scaleFactor: float, minNeighbors: int, minSize: tuple[int, int]):
            self.calls += 1
            if not self.is_profile and self.calls == 1:
                # Two detections on scaled width=640 (scale=0.5): largest has center x=500 -> 1000 in 1280
                return np.array([[10, 10, 40, 40], [450, 100, 100, 100]])
            if self.is_profile:
                # Fallback profile detection at scaled x center=500 -> 1000 in 1280
                return np.array([[460, 100, 80, 80]])
            return ()

    monkeypatch.setattr(tr.cv2, "CascadeClassifier", StubCascade)
    plan = analyze_video_tracking(video_path, sample_fps=5.0)
    assert plan.src_width == 1280
    assert plan.src_height == 720
    assert plan.crop_width == 404
    assert plan.total_frames == 4
    assert plan.face_ratio == 1.0
    # center=1000, half crop=202 -> x0 = 798
    assert plan.x0_per_frame == [798, 798, 798, 798]


def test_analyze_video_tracking_raises_when_cascade_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import lib.tracking as tr

    video_path = tmp_path / "empty_cascade.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 180),
    )
    assert writer.isOpened()
    try:
        frame = np.full((180, 320, 3), 50, dtype=np.uint8)
        writer.write(frame)
    finally:
        writer.release()

    class EmptyCascade:
        def __init__(self, path: str) -> None:
            self.path = path

        def empty(self) -> bool:
            return True

    monkeypatch.setattr(tr.cv2, "CascadeClassifier", EmptyCascade)
    with pytest.raises(ClipError) as exc:
        analyze_video_tracking(video_path)
    assert exc.value.code == "MEDIA_ERROR"
    assert "cascade" in exc.value.message.lower()
