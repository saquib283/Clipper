from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.audio_peaks import detect_audio_peaks, transcribe_audio
from lib.heatmap import extract_heatmap_peaks
from lib.models import CaptionSegment, ClipError, HeatmapPeak
from lib.subtitles import build_transcript_blocks, parse_json3, parse_vtt
from lib.youtube import download_audio_only, download_subtitles, fetch_video_info


def _transcribe_fallback(
    audio_path: Path,
    duration_sec: float,
    peaks: list[HeatmapPeak],
    max_windows: int,
    work_dir: Path,
) -> tuple[list[CaptionSegment], str]:
    """Transcribe downloaded audio directly or across peak-centered windows for long VODs."""
    if duration_sec <= 1200.0 or not peaks:
        segments, _, lang = transcribe_audio(audio_path, model_size="base")
        return segments, lang

    all_segments: list[CaptionSegment] = []
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
        segs, _, detected_lang = transcribe_audio(slice_wav, model_size="base", clip_offset=w_start)
        all_segments.extend(segs)
    all_segments.sort(key=lambda s: s.start)
    return all_segments, detected_lang


def run_probe(
    url: str,
    out_dir: Path | None = None,
    lang: str = "en",
    max_windows: int = 15,
    full_transcript: bool = False,
    cookies_from_browser: str | None = None,
) -> dict[str, Any]:
    """Probe a YouTube URL, extract peaks and captions, and write analysis.json."""
    info = fetch_video_info(url, cookies_from_browser=cookies_from_browser)
    if info.get("is_live"):
        raise ClipError(
            "LIVE_STREAM_ACTIVE",
            "This YouTube broadcast is currently live. youtube-clipper supports finished videos and livestream VODs.",
            hint="Wait until the stream finishes or clip the VOD once YouTube publishes it.",
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

    sub_path, caption_source, detected_lang = download_subtitles(
        url, cache_dir, lang=lang, cookies_from_browser=cookies_from_browser
    )
    segments: list[CaptionSegment] = []
    has_word_timestamps = False
    audio_path: Path | None = None
    audio_peaks: list[HeatmapPeak] = []

    if sub_path and caption_source == "youtube_json3":
        segments, words = parse_json3(sub_path.read_text(encoding="utf-8"))
        has_word_timestamps = len(words) > 0
    elif sub_path and caption_source == "youtube_vtt":
        segments = parse_vtt(sub_path.read_text(encoding="utf-8"))
    else:
        audio_path = download_audio_only(url, cache_dir, cookies_from_browser=cookies_from_browser)
        audio_peaks = detect_audio_peaks(audio_path, duration_sec=duration_sec, max_peaks=max_windows)
        candidate_peaks = heatmap_peaks or audio_peaks
        segments, detected_lang = _transcribe_fallback(
            audio_path, duration_sec, candidate_peaks, max_windows, work_dir
        )
        caption_source = "faster_whisper"

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
            "is_live": False,
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
        },
    }

    analysis_path = work_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "ok": True,
        "video_id": video_id,
        "title": analysis["video"]["title"],
        "duration_sec": duration_sec,
        "caption_source": caption_source,
        "has_heatmap": len(heatmap_peaks) > 0,
        "heatmap_peaks_count": len(heatmap_peaks),
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
    args = parser.parse_args(argv)

    try:
        summary = run_probe(
            url=args.url,
            out_dir=args.out,
            lang=args.lang,
            max_windows=args.max_windows,
            full_transcript=args.full_transcript,
            cookies_from_browser=args.cookies_from_browser,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except ClipError as err:
        print(json.dumps(err.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
