from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.heatmap import extract_heatmap_peaks, score_time_range  # noqa: E402


def test_extract_heatmap_peaks_merges_adjacent_and_sorts_by_score() -> None:
    heatmap = [
        {"start_time": 0.0, "end_time": 10.0, "value": 0.2},
        {"start_time": 100.0, "end_time": 110.0, "value": 0.72},
        {"start_time": 110.0, "end_time": 120.0, "value": 0.85},
        {"start_time": 125.0, "end_time": 135.0, "value": 0.70},
        {"start_time": 400.0, "end_time": 410.0, "value": 0.96},
    ]
    peaks = extract_heatmap_peaks(heatmap, threshold=0.65, merge_gap_sec=20.0, pad_sec=10.0, duration_sec=600.0)
    assert len(peaks) == 2
    assert peaks[0].score == 0.96
    assert peaks[0].start == 390.0
    assert peaks[0].end == 420.0
    assert peaks[1].score == 0.85
    assert peaks[1].start == 90.0
    assert peaks[1].end == 145.0


def test_extract_heatmap_peaks_handles_empty_and_clamps_bounds() -> None:
    assert extract_heatmap_peaks(None) == []
    assert extract_heatmap_peaks([]) == []
    peaks = extract_heatmap_peaks(
        [{"start_time": 2.0, "end_time": 8.0, "value": 0.9}],
        pad_sec=10.0,
        duration_sec=12.0,
    )
    assert peaks[0].start == 0.0
    assert peaks[0].end == 12.0


def test_score_time_range_returns_highest_overlapping_peak() -> None:
    peaks = extract_heatmap_peaks(
        [
            {"start_time": 50.0, "end_time": 60.0, "value": 0.75},
            {"start_time": 200.0, "end_time": 210.0, "value": 0.92},
        ],
        pad_sec=5.0,
    )
    assert score_time_range(peaks, 48.0, 55.0) == 0.75
    assert score_time_range(peaks, 10.0, 20.0) == 0.0
