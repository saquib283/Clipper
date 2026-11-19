from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import CaptionSegment, HeatmapPeak  # noqa: E402
from lib.subtitles import (  # noqa: E402
    build_transcript_blocks,
    parse_json3,
    parse_vtt,
    segments_to_proportional_words,
    slice_words_for_clip,
)


def test_parse_json3_extracts_segments_and_word_offsets() -> None:
    payload = {
        "events": [
            {
                "tStartMs": 10000,
                "dDurationMs": 2000,
                "segs": [
                    {"utf8": "Nobody"},
                    {"utf8": " talks", "tOffsetMs": 400},
                    {"utf8": " about", "tOffsetMs": 800},
                    {"utf8": " this", "tOffsetMs": 1300},
                ],
            },
            {"tStartMs": 12000, "dDurationMs": 500, "segs": [{"utf8": "\n"}]},
        ]
    }
    segments, words = parse_json3(json.dumps(payload))
    assert len(segments) == 1
    assert segments[0].text == "Nobody talks about this"
    assert segments[0].start == 10.0
    assert [w.word for w in words] == ["Nobody", "talks", "about", "this"]
    assert words[1].start == 10.4
    assert words[0].end == 10.4


def test_parse_vtt_deduplicates_rolling_auto_captions() -> None:
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.000 --> 00:00:05.000
Hello world
this is a test

00:00:05.000 --> 00:00:07.000
<c>this</c> is a test
final line
"""
    segments = parse_vtt(vtt)
    assert [s.text for s in segments] == ["Hello world", "this is a test", "final line"]
    assert segments[0].start == 1.0
    assert segments[2].end == 7.0


def test_slice_words_and_proportional_fallback_rezero_timestamps() -> None:
    payload = {
        "events": [
            {
                "tStartMs": 10000,
                "dDurationMs": 2000,
                "segs": [{"utf8": "First"}, {"utf8": " second", "tOffsetMs": 1000}],
            }
        ]
    }
    _, words = parse_json3(json.dumps(payload))
    sliced = slice_words_for_clip(words, start=10.5, end=12.5)
    assert len(sliced) == 1
    assert sliced[0].word == "second"
    assert sliced[0].start == 0.5

    segs = [CaptionSegment(start=10.0, end=12.0, text="one two")]
    prop = segments_to_proportional_words(segs, start=10.0, end=12.0)
    assert [(w.word, w.start, w.end) for w in prop] == [("one", 0.0, 1.0), ("two", 1.0, 2.0)]


def test_build_transcript_blocks_windows_long_vods_by_heatmap() -> None:
    segs = [
        CaptionSegment(start=10.0, end=15.0, text="intro talk"),
        CaptionSegment(start=3600.0, end=3605.0, text="huge moment"),
        CaptionSegment(start=5000.0, end=5005.0, text="quiet part"),
    ]
    peaks = [HeatmapPeak(start=3590.0, end=3640.0, peak_time=3602.0, score=0.95)]
    blocks, truncated = build_transcript_blocks(
        segs,
        peaks,
        duration_sec=6000.0,
        max_windows=1,
        full_transcript=False,
    )
    assert truncated is True
    assert len(blocks) == 1
    assert "huge moment" in blocks[0].text
    assert blocks[0].heatmap_score == 0.95
