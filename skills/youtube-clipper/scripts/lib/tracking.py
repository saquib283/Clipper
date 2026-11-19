from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lib.models import ClipError, TrackingPlan


def compute_vertical_crop_width(src_w: int, src_h: int) -> int:
    """Compute an even 9:16 crop width clamped to the source frame width."""
    target = int(round(src_h * 9.0 / 16.0))
    clamped = min(src_w, max(2, target))
    return clamped if clamped % 2 == 0 else max(2, clamped - 1)


def detect_scene_cuts(histograms: list[np.ndarray], threshold: float = 0.45) -> set[int]:
    """Identify sample indices where Bhattacharyya histogram distance indicates a scene cut."""
    cuts: set[int] = set()
    for i in range(1, len(histograms)):
        dist = float(cv2.compareHist(histograms[i - 1], histograms[i], cv2.HISTCMP_BHATTACHARYYA))
        if dist >= threshold:
            cuts.add(i)
    return cuts


def smooth_crop_x0(
    face_centers: list[float | None],
    scene_cuts: set[int],
    src_w: int,
    crop_w: int,
    alpha: float = 0.22,
    deadzone_px: float = 18.0,
) -> list[int]:
    """Smooth face-centered crop x0 positions with deadzone filtering, EMA, and scene-cut snapping."""
    max_x0 = max(0, src_w - crop_w)
    default_center = src_w / 2.0
    if not face_centers:
        return []

    cur_center = face_centers[0] if face_centers[0] is not None else default_center
    x0s: list[int] = []

    for idx, det in enumerate(face_centers):
        if idx in scene_cuts:
            cur_center = det if det is not None else default_center
        elif det is not None and abs(det - cur_center) > deadzone_px:
            cur_center = (1.0 - alpha) * cur_center + alpha * det

        x0 = int(round(cur_center - crop_w / 2.0))
        x0s.append(max(0, min(max_x0, x0)))

    return x0s


def interpolate_x_positions(
    sample_x0: list[int],
    sample_indices: list[int],
    total_frames: int,
    scene_cuts: set[int] | None = None,
) -> list[int]:
    """Expand sampled x0 positions to every frame, holding position before scene cuts."""
    if total_frames <= 0:
        return []
    if not sample_x0:
        return [0] * total_frames

    cuts = scene_cuts or set()
    out = [sample_x0[0]] * total_frames
    for i in range(len(sample_indices) - 1):
        f0, f1 = sample_indices[i], sample_indices[i + 1]
        x0, x1 = sample_x0[i], sample_x0[i + 1]
        span = max(1, f1 - f0)
        is_cut = (i + 1) in cuts
        for f in range(f0, min(f1, total_frames)):
            if is_cut:
                out[f] = x0
            else:
                t = (f - f0) / span
                out[f] = int(round(x0 + t * (x1 - x0)))
    for f in range(sample_indices[-1], total_frames):
        out[f] = sample_x0[-1]
    return out


def analyze_video_tracking(video_path: Path, sample_fps: float = 5.0) -> TrackingPlan:
    """Analyze a video at sampled frames to produce a scene-aware face-tracking crop plan."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ClipError("MEDIA_ERROR", f"Could not open video for tracking: {video_path}")

    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        crop_w = compute_vertical_crop_width(src_w, src_h)

        cascades_dir = Path(cv2.data.haarcascades)
        frontal = cv2.CascadeClassifier(str(cascades_dir / "haarcascade_frontalface_default.xml"))
        profile = cv2.CascadeClassifier(str(cascades_dir / "haarcascade_profileface.xml"))
        if getattr(frontal, "empty", lambda: False)() or getattr(profile, "empty", lambda: False)():
            raise ClipError("MEDIA_ERROR", "Failed to load OpenCV Haar face cascade XML.")

        step = max(1, int(round(fps / max(0.5, sample_fps))))
        sample_indices: list[int] = []
        face_centers: list[float | None] = []
        histograms: list[np.ndarray] = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                scale = min(1.0, 640.0 / max(1, src_w))
                small = (
                    cv2.resize(frame, (max(1, int(src_w * scale)), max(1, int(src_h * scale))))
                    if scale < 1.0
                    else frame
                )
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                histograms.append(hist)

                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))
                if len(faces) == 0:
                    faces = profile.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))

                if len(faces) > 0:
                    largest = max(faces, key=lambda box: int(box[2]) * int(box[3]))
                    cx_small = float(largest[0] + largest[2] / 2.0)
                    face_centers.append(cx_small / scale)
                else:
                    face_centers.append(None)
                sample_indices.append(frame_idx)
            frame_idx += 1
    except cv2.error as exc:
        raise ClipError("MEDIA_ERROR", f"Video tracking failed for {video_path}: {exc}") from exc
    finally:
        cap.release()

    actual_total = max(total_frames, frame_idx)
    detected_count = sum(1 for c in face_centers if c is not None)
    face_ratio = round(detected_count / max(1, len(face_centers)), 3)

    cuts = detect_scene_cuts(histograms)
    sample_x0 = smooth_crop_x0(face_centers, cuts, src_w=src_w, crop_w=crop_w)
    x0_per_frame = interpolate_x_positions(sample_x0, sample_indices, actual_total, scene_cuts=cuts)

    return TrackingPlan(
        src_width=src_w,
        src_height=src_h,
        crop_width=crop_w,
        fps=fps,
        total_frames=actual_total,
        face_ratio=face_ratio,
        x0_per_frame=x0_per_frame,
    )
