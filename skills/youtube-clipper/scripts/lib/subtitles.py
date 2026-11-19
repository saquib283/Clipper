from __future__ import annotations

import html
import json
import re

from lib.heatmap import score_time_range
from lib.models import CaptionSegment, HeatmapPeak, TranscriptBlock, WordTiming

VTT_TS_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _ts_to_sec(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h) if h else 0
    return round(hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0, 3)


def parse_json3(content: str) -> tuple[list[CaptionSegment], list[WordTiming]]:
    data = json.loads(content)
    events = data.get("events", [])
    segments: list[CaptionSegment] = []
    words: list[WordTiming] = []

    for event in events:
        segs = event.get("segs")
        if not segs or "tStartMs" not in event:
            continue
        start_ms = float(event["tStartMs"])
        dur_ms = float(event.get("dDurationMs", 0.0))
        end_ms = start_ms + max(dur_ms, 200.0)

        raw_pieces: list[tuple[str, float]] = []
        for piece in segs:
            token = _clean_text(str(piece.get("utf8", "")))
            if not token:
                continue
            offset_ms = float(piece.get("tOffsetMs", 0.0))
            raw_pieces.append((token, (start_ms + offset_ms) / 1000.0))

        if not raw_pieces:
            continue

        line_text = " ".join(p[0] for p in raw_pieces).strip()
        if not line_text:
            continue

        segments.append(
            CaptionSegment(
                start=round(start_ms / 1000.0, 3),
                end=round(end_ms / 1000.0, 3),
                text=line_text,
            )
        )

        # Build word timings: use exact tOffsetMs when present, or distribute within piece span
        for p_idx, (token, p_start) in enumerate(raw_pieces):
            p_end = (
                raw_pieces[p_idx + 1][1]
                if p_idx + 1 < len(raw_pieces) and raw_pieces[p_idx + 1][1] > p_start
                else max(p_start + 0.15, end_ms / 1000.0)
            )
            sub_tokens = token.split()
            span = max(0.1, p_end - p_start)
            step = span / max(1, len(sub_tokens))
            for s_idx, sub in enumerate(sub_tokens):
                w_start = p_start + s_idx * step
                w_end = p_start + (s_idx + 1) * step
                words.append(WordTiming(word=sub, start=round(w_start, 3), end=round(w_end, 3)))

    # Fix any overlapping word end times across consecutive events
    for i in range(len(words) - 1):
        if words[i].end > words[i + 1].start and words[i + 1].start > words[i].start:
            words[i].end = words[i + 1].start

    return segments, words


def parse_vtt(content: str) -> list[CaptionSegment]:
    lines = content.replace("\r\n", "\n").split("\n")
    segments: list[CaptionSegment] = []
    last_emitted = ""
    i = 0
    while i < len(lines):
        match = VTT_TS_RE.search(lines[i])
        if not match:
            i += 1
            continue
        start = _ts_to_sec(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _ts_to_sec(match.group(5), match.group(6), match.group(7), match.group(8))
        i += 1
        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != "":
            cleaned = _clean_text(lines[i])
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        # Deduplicate rolling captions where first line repeats previous cue
        filtered = [ln for ln in cue_lines if ln != last_emitted]
        text = " ".join(filtered).strip()
        if text:
            segments.append(CaptionSegment(start=start, end=end, text=text))
            last_emitted = filtered[-1]
    return segments


def slice_words_for_clip(words: list[WordTiming], start: float, end: float) -> list[WordTiming]:
    result: list[WordTiming] = []
    clip_dur = max(0.0, end - start)
    for w in words:
        if w.start < start or w.start >= end:
            continue
        rel_start = max(0.0, round(w.start - start, 3))
        rel_end = min(clip_dur, round(w.end - start, 3))
        if rel_end > rel_start:
            result.append(WordTiming(word=w.word, start=rel_start, end=rel_end))
    return result


def segments_to_proportional_words(segments: list[CaptionSegment], start: float, end: float) -> list[WordTiming]:
    words: list[WordTiming] = []
    clip_dur = max(0.0, end - start)
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        tokens = seg.text.split()
        if not tokens:
            continue
        seg_start = max(start, seg.start)
        seg_end = min(end, seg.end)
        span = max(0.1, seg_end - seg_start)
        step = span / len(tokens)
        for idx, tok in enumerate(tokens):
            w_start = round((seg_start - start) + idx * step, 3)
            w_end = min(clip_dur, round((seg_start - start) + (idx + 1) * step, 3))
            if w_end > w_start:
                words.append(WordTiming(word=tok, start=w_start, end=w_end))
    return words


def build_transcript_blocks(
    segments: list[CaptionSegment],
    heatmap_peaks: list[HeatmapPeak],
    block_sec: float = 18.0,
    duration_sec: float = 0.0,
    max_windows: int = 15,
    full_transcript: bool = False,
) -> tuple[list[TranscriptBlock], bool]:
    if not segments:
        return [], False

    truncated = False
    selected_segments = segments
    if duration_sec > 5400.0 and not full_transcript and heatmap_peaks:
        windows = sorted(heatmap_peaks[:max_windows], key=lambda p: p.start)
        filtered = [
            seg
            for seg in segments
            if any(seg.end >= (w.start - 30.0) and seg.start <= (w.end + 30.0) for w in windows)
        ]
        if filtered:
            selected_segments = filtered
            truncated = True

    blocks: list[TranscriptBlock] = []
    cur_start = selected_segments[0].start
    cur_end = selected_segments[0].end
    cur_texts: list[str] = [selected_segments[0].text]

    for seg in selected_segments[1:]:
        if (seg.end - cur_start) <= block_sec and (seg.start - cur_end) <= 4.0:
            cur_end = max(cur_end, seg.end)
            cur_texts.append(seg.text)
        else:
            blocks.append(
                TranscriptBlock(
                    start=round(cur_start, 2),
                    end=round(cur_end, 2),
                    text=" ".join(cur_texts),
                    heatmap_score=score_time_range(heatmap_peaks, cur_start, cur_end),
                )
            )
            cur_start = seg.start
            cur_end = seg.end
            cur_texts = [seg.text]

    blocks.append(
        TranscriptBlock(
            start=round(cur_start, 2),
            end=round(cur_end, 2),
            text=" ".join(cur_texts),
            heatmap_score=score_time_range(heatmap_peaks, cur_start, cur_end),
        )
    )
    return blocks, truncated
