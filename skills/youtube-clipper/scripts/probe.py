from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.audio_peaks import detect_audio_peaks, transcribe_audio
from lib.heatmap import extract_heatmap_peaks
from lib.live_dvr import prepare_live_probe, words_to_json3
from lib.models import CaptionSegment, ClipError, HeatmapPeak, WordTiming
from lib.subtitles import build_transcript_blocks, parse_json3, parse_vtt
from lib.youtube import download_audio_only, download_subtitles, fetch_video_info


def _transcribe_fallback(
    audio_path: Path,
    duration_sec: float,
    peaks: list[HeatmapPeak],
    max_windows: int,
    work_dir: Path,
) -> tuple[list[CaptionSegment], list[WordTiming], str]:
    """Transcribe downloaded audio directly or across peak-centered windows for long VODs."""
    if duration_sec <= 1200.0 or not peaks:
        segments, words, lang = transcribe_audio(audio_path, model_size="base")
        return segments, words, lang

    all_segments: list[CaptionSegment] = []
    all_words: list[WordTiming] = []
    detected_lang = "en"
    tmp_dir = work_dir / "cache" / "whisper_windows"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, peak in enumerate(peaks[:max_windows]):
        w_start = max(0.0, peak.peak_time - 45.0)
        w_dur = min(90.0, max(10.0, duration_sec - w_start))
        slice_wav = tmp_dir / f"win_{idx:02d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{w_start:.2f}",
                "-t",
                f"{w_dur:.2f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(slice_wav),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        segs, words, detected_lang = transcribe_audio(slice_wav, model_size="base", clip_offset=w_start)
        all_segments.extend(segs)
        all_words.extend(words)
    all_segments.sort(key=lambda s: s.start)
    all_words.sort(key=lambda w: w.start)
    return all_segments, all_words, detected_lang


def run_probe(
    url: str,
    out_dir: Path | None = None,
    lang: str = "en",
    max_windows: int = 15,
    full_transcript: bool = False,
    cookies_from_browser: str | None = None,
    live_window: float = 300.0,
) -> dict[str, Any]:
    """Probe a YouTube URL (VOD or active livestream DVR tail), extract peaks/captions, and write analysis.json."""
    info = fetch_video_info(url, cookies_from_browser=cookies_from_browser)
    is_live = bool(info.get("is_live"))
    if is_live and live_window <= 0:
        raise ClipError(
            "LIVE_STREAM_ACTIVE",
            "This YouTube broadcast is currently live and --live-window was set to 0.",
            hint="Re-run with --live-window 300 (default) to clip from the recent live DVR window, or wait for the VOD.",
        )

    video_id = str(info.get("id") or "video")
    duration_sec = float(info.get("duration") or 0.0)
    work_dir = out_dir or (Path.cwd() / "clips" / video_id / "work")
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    heatmap_peaks = extract_heatmap_peaks(
        info.get("heatmap"),
        max_peaks=max_windows,
        duration_sec=duration_sec if duration_sec > 0 else None,
    )
    chapters = [
        {
            "title": str(ch.get("title", "")),
            "start": round(float(ch.get("start_time", 0.0)), 2),
            "end": round(float(ch.get("end_time", 0.0)), 2),
        }
        for ch in (info.get("chapters") or [])
    ]

    segments: list[CaptionSegment] = []
    words: list[WordTiming] = []
    has_word_timestamps = False
    sub_path: Path | None = None
    audio_path: Path | None = None
    live_manifest_path: Path | None = None
    audio_peaks: list[HeatmapPeak] = []
    caption_source = "none"
    detected_lang = lang

    if is_live:
        audio_path, live_manifest_path, duration_sec = prepare_live_probe(
            info, cache_dir, window_sec=float(live_window)
        )
        audio_peaks = detect_audio_peaks(audio_path, duration_sec=duration_sec, max_peaks=max_windows)
        segments, words, detected_lang = _transcribe_fallback(
            audio_path, duration_sec, audio_peaks, max_windows, work_dir
        )
        caption_source = "faster_whisper"
        if segments:
            sub_path = cache_dir / "whisper_words.json3"
            sub_path.write_text(words_to_json3(segments, words), encoding="utf-8")
            has_word_timestamps = len(words) > 0
    else:
        sub_path, caption_source, detected_lang = download_subtitles(
            url, cache_dir, lang=lang, cookies_from_browser=cookies_from_browser
        )
        if sub_path and caption_source == "youtube_json3":
            segments, words = parse_json3(sub_path.read_text(encoding="utf-8"))
            has_word_timestamps = len(words) > 0
        elif sub_path and caption_source == "youtube_vtt":
            segments = parse_vtt(sub_path.read_text(encoding="utf-8"))
        else:
            audio_path = download_audio_only(url, cache_dir, cookies_from_browser=cookies_from_browser)
            audio_peaks = detect_audio_peaks(audio_path, duration_sec=duration_sec, max_peaks=max_windows)
            candidate_peaks = heatmap_peaks or audio_peaks
            segments, words, detected_lang = _transcribe_fallback(
                audio_path, duration_sec, candidate_peaks, max_windows, work_dir
            )
            caption_source = "faster_whisper"
            if segments and words:
                sub_path = cache_dir / "whisper_words.json3"
                sub_path.write_text(words_to_json3(segments, words), encoding="utf-8")
                has_word_timestamps = True

    signal_peaks = heatmap_peaks or audio_peaks
    blocks, truncated = build_transcript_blocks(
        segments=segments,
        heatmap_peaks=signal_peaks,
        duration_sec=duration_sec,
        max_windows=max_windows,
        full_transcript=full_transcript,
    )

    analysis = {
        "video": {
            "id": video_id,
            "url": url,
            "title": str(info.get("title", "")),
            "channel": str(info.get("channel") or info.get("uploader") or ""),
            "duration_sec": duration_sec,
            "upload_date": str(info.get("upload_date", "")),
            "is_live": is_live,
            "was_live": bool(info.get("was_live", False)),
        },
        "signals": {
            "caption_source": caption_source,
            "language": detected_lang,
            "has_heatmap": len(heatmap_peaks) > 0,
            "has_word_timestamps": has_word_timestamps,
            "truncated_to_windows": truncated,
            "chapters": chapters,
            "heatmap_peaks": [p.to_dict() for p in heatmap_peaks],
            "audio_peaks": [p.to_dict() for p in audio_peaks],
        },
        "transcript_blocks": [b.to_dict() for b in blocks],
        "cache_paths": {
            "subtitles": str(sub_path.resolve()) if sub_path else None,
            "audio": str(audio_path.resolve()) if audio_path else None,
            "live_manifest": str(live_manifest_path.resolve()) if live_manifest_path else None,
        },
    }

    analysis_path = work_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "ok": True,
        "video_id": video_id,
        "title": analysis["video"]["title"],
        "is_live": is_live,
        "duration_sec": duration_sec,
        "caption_source": caption_source,
        "has_heatmap": len(heatmap_peaks) > 0,
        "heatmap_peaks_count": len(heatmap_peaks),
        "audio_peaks_count": len(audio_peaks),
        "transcript_blocks_count": len(blocks),
        "analysis_path": str(analysis_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 1 probing."""
    parser = argparse.ArgumentParser(description="Probe a YouTube URL for metadata, heatmap peaks, and captions.")
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-windows", type=int, default=15)
    parser.add_argument("--full-transcript", action="store_true")
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--live-window", type=float, default=300.0, help="Seconds of live DVR history to analyze (0 to reject active livestreams).")
    args = parser.parse_args(argv)

    try:
        summary = run_probe(
            url=args.url,
            out_dir=args.out,
            lang=args.lang,
            max_windows=args.max_windows,
            full_transcript=args.full_transcript,
            cookies_from_browser=args.cookies_from_browser,
            live_window=args.live_window,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except ClipError as err:
        print(json.dumps(err.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
