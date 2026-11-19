from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from lib.models import CaptionSegment, ClipError, HeatmapPeak, WordTiming

PTS_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
RMS_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+(?:\.\d+)?|-inf)")


def parse_ffmpeg_rms_levels(stderr_text: str, window_sec: float = 5.0) -> list[tuple[float, float]]:
    """Parse FFmpeg astats RMS_level lines into mean dB buckets of size window_sec."""
    buckets: dict[int, list[float]] = {}
    current_t = 0.0
    for line in stderr_text.splitlines():
        pts_m = PTS_RE.search(line)
        if pts_m:
            current_t = float(pts_m.group(1))
            continue
        rms_m = RMS_RE.search(line)
        if rms_m:
            raw = rms_m.group(1)
            db = -80.0 if raw == "-inf" else max(-80.0, float(raw))
            bucket = int(current_t // window_sec)
            buckets.setdefault(bucket, []).append(db)

    return [
        (round(b * window_sec, 2), sum(vals) / len(vals))
        for b, vals in sorted(buckets.items())
    ]


def peaks_from_rms_windows(
    windows: list[tuple[float, float]],
    duration_sec: float,
    window_sec: float = 5.0,
    max_peaks: int = 15,
) -> list[HeatmapPeak]:
    """Normalize RMS windows, merge nearby hot segments, and return top HeatmapPeaks."""
    if not windows:
        return []
    db_vals = [w[1] for w in windows]
    lo, hi = min(db_vals), max(db_vals)
    span = max(hi - lo, 1.0)

    normalized = [(t, (db - lo) / span) for t, db in windows]
    hot = [(t, score) for t, score in normalized if score >= 0.7]
    if not hot:
        return []

    merged: list[list[float]] = []
    for t, score in hot:
        end = t + window_sec
        mid = t + window_sec / 2.0
        if not merged or t - merged[-1][1] > 20.0:
            merged.append([t, end, mid, score])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            if score > merged[-1][3]:
                merged[-1][2] = mid
                merged[-1][3] = score

    peaks = [
        HeatmapPeak(
            start=round(max(0.0, s - 10.0), 2),
            end=round(min(duration_sec, e + 10.0), 2),
            peak_time=round(mid, 2),
            score=round(sc, 4),
        )
        for s, e, mid, sc in merged
    ]
    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks[:max_peaks]


def detect_audio_peaks(
    audio_path: Path,
    duration_sec: float,
    window_sec: float = 5.0,
    max_peaks: int = 15,
) -> list[HeatmapPeak]:
    """Scan an audio/video file with FFmpeg astats and return top energy peaks."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        "asetnsamples=n=44100,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ClipError("FFMPEG_FAILED", f"Audio peak scan failed: {(proc.stderr or '').strip()[:400]}")
    levels = parse_ffmpeg_rms_levels(proc.stderr or "", window_sec=window_sec)
    return peaks_from_rms_windows(levels, duration_sec=duration_sec, window_sec=window_sec, max_peaks=max_peaks)


@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str):
    """Load and cache a CPU int8 faster-whisper WhisperModel instance."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _decode_audio_16k_mono(audio_path: Path) -> np.ndarray:
    """Decode any FFmpeg-supported audio/video file to 16 kHz mono float32 PCM."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()[:400]
        raise ClipError("FFMPEG_FAILED", f"Audio decode failed for transcription: {err}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe_audio(
    audio_path: Path,
    model_size: str = "base",
    language: str | None = None,
    clip_offset: float = 0.0,
) -> tuple[list[CaptionSegment], list[WordTiming], str]:
    """Transcribe audio locally with faster-whisper and offset timestamps by clip_offset."""
    pcm = _decode_audio_16k_mono(audio_path)
    model = _load_whisper_model(model_size)
    raw_segments, info = model.transcribe(
        pcm,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    segments: list[CaptionSegment] = []
    words: list[WordTiming] = []
    for seg in raw_segments:
        text = str(seg.text).strip()
        if not text:
            continue
        segments.append(
            CaptionSegment(
                start=round(float(seg.start) + clip_offset, 3),
                end=round(float(seg.end) + clip_offset, 3),
                text=text,
            )
        )
        for w in getattr(seg, "words", None) or []:
            tok = str(w.word).strip()
            if tok:
                words.append(
                    WordTiming(
                        word=tok,
                        start=round(float(w.start) + clip_offset, 3),
                        end=round(float(w.end) + clip_offset, 3),
                    )
                )
    detected_lang = str(getattr(info, "language", None) or language or "en")
    return segments, words, detected_lang
