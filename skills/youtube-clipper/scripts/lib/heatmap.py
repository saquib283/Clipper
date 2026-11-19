from __future__ import annotations

from typing import Any

from lib.models import HeatmapPeak


def extract_heatmap_peaks(
    heatmap: list[dict[str, Any]] | None,
    threshold: float = 0.65,
    merge_gap_sec: float = 20.0,
    pad_sec: float = 10.0,
    max_peaks: int = 15,
    duration_sec: float | None = None,
) -> list[HeatmapPeak]:
    """Extract, merge, pad, and rank Most Replayed heatmap peaks by score."""
    if not heatmap:
        return []

    hot = [
        (float(b["start_time"]), float(b["end_time"]), float(b["value"]))
        for b in heatmap
        if float(b.get("value", 0.0)) >= threshold
    ]
    if not hot:
        return []

    hot.sort(key=lambda item: item[0])
    merged: list[list[float]] = []
    for start, end, val in hot:
        mid = (start + end) / 2.0
        if not merged or start - merged[-1][1] > merge_gap_sec:
            merged.append([start, end, mid, val])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            if val > merged[-1][3]:
                merged[-1][2] = mid
                merged[-1][3] = val

    peaks: list[HeatmapPeak] = []
    for start, end, peak_time, score in merged:
        padded_start = max(0.0, start - pad_sec)
        padded_end = end + pad_sec
        if duration_sec is not None:
            padded_end = min(duration_sec, padded_end)
        peaks.append(
            HeatmapPeak(
                start=round(padded_start, 2),
                end=round(padded_end, 2),
                peak_time=round(peak_time, 2),
                score=round(score, 4),
            )
        )

    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks[:max_peaks]


def score_time_range(peaks: list[HeatmapPeak], start: float, end: float) -> float:
    """Return the highest heatmap peak score overlapping [start, end], or 0.0."""
    best = 0.0
    for peak in peaks:
        if peak.end >= start and peak.start <= end:
            best = max(best, peak.score)
    return best
