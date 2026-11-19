from __future__ import annotations

import json
import math
import subprocess
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from lib.ffmpeg_ops import probe_media
from lib.models import CaptionSegment, ClipError, WordTiming

FFMPEG_HLS_WHITELIST = "file,http,https,tcp,tls,crypto"


def _fetch_text(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise ClipError("LIVE_DVR_FAILED", f"Failed to fetch HLS playlist: {exc}") from exc


def select_hls_formats(info: dict[str, Any], max_height: int = 1080) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select best <=1080p HLS video and audio formats from yt-dlp info JSON."""
    formats = [
        f
        for f in (info.get("formats") or [])
        if f.get("url") and str(f.get("protocol", "")).startswith("m3u8")
    ]
    if not formats:
        return None, None

    audio_only = [f for f in formats if f.get("vcodec") == "none"]
    audio_only.sort(key=lambda f: (float(f.get("tbr") or f.get("abr") or 0.0), str(f.get("format_id", ""))))
    best_audio = audio_only[-1] if audio_only else None

    video_candidates = [
        f
        for f in formats
        if f.get("vcodec") != "none" and int(f.get("height") or 0) <= max_height
    ]
    if not video_candidates:
        video_candidates = [f for f in formats if f.get("vcodec") != "none"]

    def _video_rank(f: dict[str, Any]) -> tuple[int, int, float]:
        vcodec = str(f.get("vcodec") or "").lower()
        is_avc = 1 if vcodec.startswith("avc1") else 0
        height = int(f.get("height") or 0)
        tbr = float(f.get("tbr") or 0.0)
        return (is_avc, height, tbr)

    video_candidates.sort(key=_video_rank)
    best_video = video_candidates[-1] if video_candidates else None
    return best_video, best_audio


def parse_m3u8_segments(m3u8_text: str, base_url: str = "") -> list[dict[str, Any]]:
    """Parse an HLS media playlist into ordered segment entries with sequence numbers."""
    media_seq = 0
    pending_dur: float | None = None
    segments: list[dict[str, Any]] = []

    for raw_line in m3u8_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                media_seq = int(line.split(":", 1)[1].strip())
            except ValueError:
                media_seq = 0
        elif line.startswith("#EXTINF:"):
            val = line.split(":", 1)[1].split(",", 1)[0].strip()
            try:
                pending_dur = max(0.1, float(val))
            except ValueError:
                pending_dur = 2.0
        elif not line.startswith("#") and pending_dur is not None:
            seg_url = urljoin(base_url, line) if base_url else line
            segments.append(
                {
                    "seq": media_seq + len(segments),
                    "duration": pending_dur,
                    "url": seg_url,
                }
            )
            pending_dur = None
    return segments


def build_dvr_window_manifest(
    video_segments: list[dict[str, Any]],
    audio_segments: list[dict[str, Any]] | None,
    window_sec: float,
) -> dict[str, Any]:
    """Align video/audio HLS segments by sequence number and slice the trailing window_sec."""
    v_by_seq = {int(s["seq"]): s for s in video_segments}
    if audio_segments is not None:
        a_by_seq = {int(s["seq"]): s for s in audio_segments}
        common_seqs = sorted(set(v_by_seq) & set(a_by_seq))
    else:
        a_by_seq = {}
        common_seqs = sorted(v_by_seq)

    selected_seqs: list[int] = []
    acc = 0.0
    for seq in reversed(common_seqs):
        selected_seqs.append(seq)
        acc += float(v_by_seq[seq]["duration"])
        if acc >= window_sec:
            break
    selected_seqs.reverse()

    entries: list[dict[str, Any]] = []
    cursor = 0.0
    max_dur = 2.0
    for seq in selected_seqs:
        v = v_by_seq[seq]
        dur = float(v["duration"])
        max_dur = max(max_dur, dur)
        entries.append(
            {
                "seq": seq,
                "start": round(cursor, 3),
                "end": round(cursor + dur, 3),
                "duration": round(dur, 3),
                "video_url": v["url"],
                "audio_url": a_by_seq[seq]["url"] if seq in a_by_seq else None,
            }
        )
        cursor += dur

    return {
        "mode": "dvr",
        "duration_sec": round(cursor, 3),
        "target_duration": max(1, int(math.ceil(max_dur))),
        "segments": entries,
    }


def render_m3u8_text(urls_and_durations: list[tuple[str, float]], target_duration: int = 4) -> str:
    """Build a finite VOD HLS playlist ending with #EXT-X-ENDLIST."""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{max(1, int(target_duration))}",
        "#EXT-X-MEDIA-SEQUENCE:0",
    ]
    for url, dur in urls_and_durations:
        lines.append(f"#EXTINF:{dur:.3f},")
        lines.append(url)
    lines.append("#EXT-X-ENDLIST")
    return "\n".join(lines) + "\n"


def slice_manifest_segments(
    manifest: dict[str, Any],
    start: float,
    end: float,
    pad_sec: float = 2.0,
) -> tuple[list[dict[str, Any]], float]:
    """Select manifest segments overlapping [max(0, start - pad_sec), end + pad_sec]."""
    win_start = max(0.0, start - pad_sec)
    win_end = end + pad_sec
    segs = [
        s
        for s in (manifest.get("segments") or [])
        if float(s["end"]) > win_start and float(s["start"]) < win_end
    ]
    if not segs:
        raise ClipError("MEDIA_ERROR", f"No live DVR segments overlap requested window {start:.2f}s..{end:.2f}s.")
    return segs, float(segs[0]["start"])


def words_to_json3(segments: list[CaptionSegment], words: list[WordTiming]) -> str:
    """Serialize Whisper segments and word timings to YouTube json3 format for render reuse."""
    events: list[dict[str, Any]] = []
    if words:
        for seg in segments:
            seg_words = [w for w in words if w.end > seg.start and w.start < seg.end]
            if not seg_words:
                continue
            base_ms = int(round(seg_words[0].start * 1000))
            end_ms = int(round(seg_words[-1].end * 1000))
            segs_payload = []
            for idx, w in enumerate(seg_words):
                entry: dict[str, Any] = {"utf8": w.word if idx == 0 else f" {w.word}"}
                offset = max(0, int(round(w.start * 1000)) - base_ms)
                if offset > 0:
                    entry["tOffsetMs"] = offset
                segs_payload.append(entry)
            events.append(
                {
                    "tStartMs": base_ms,
                    "dDurationMs": max(200, end_ms - base_ms),
                    "segs": segs_payload,
                }
            )
    else:
        for seg in segments:
            base_ms = int(round(seg.start * 1000))
            dur_ms = max(200, int(round((seg.end - seg.start) * 1000)))
            events.append({"tStartMs": base_ms, "dDurationMs": dur_ms, "segs": [{"utf8": seg.text}]})
    return json.dumps({"events": events}, ensure_ascii=False)


def prepare_live_probe(
    info: dict[str, Any],
    cache_dir: Path,
    window_sec: float = 300.0,
) -> tuple[Path, Path, float]:
    """Capture bounded live audio (instant DVR tail or live-edge fallback) and save live_manifest.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    v_fmt, a_fmt = select_hls_formats(info)
    if v_fmt is None:
        raise ClipError("LIVE_DVR_FAILED", "No playable HLS stream formats were found for this live broadcast.")

    v_url = str(v_fmt["url"])
    v_segs = parse_m3u8_segments(_fetch_text(v_url), base_url=v_url)
    a_segs: list[dict[str, Any]] | None = None
    if a_fmt is not None:
        a_url = str(a_fmt["url"])
        a_segs = parse_m3u8_segments(_fetch_text(a_url), base_url=a_url)

    manifest = build_dvr_window_manifest(v_segs, a_segs, window_sec=window_sec)
    manifest_path = cache_dir / "live_manifest.json"

    if manifest["duration_sec"] >= 15.0:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        audio_m3u8 = cache_dir / "live_audio.m3u8"
        audio_pairs = [
            (str(s["audio_url"] or s["video_url"]), float(s["duration"]))
            for s in manifest["segments"]
        ]
        audio_m3u8.write_text(
            render_m3u8_text(audio_pairs, target_duration=int(manifest["target_duration"])),
            encoding="utf-8",
        )
        audio_out = cache_dir / "live_audio.m4a"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            FFMPEG_HLS_WHITELIST,
            "-i",
            str(audio_m3u8),
            "-vn",
            "-c:a",
            "copy",
            str(audio_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not audio_out.exists():
            # Fallback to AAC transcode if stream copy fails on raw ADTS/TS framing
            cmd[-3:-1] = ["-c:a", "aac", "-b:a", "128k"]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if proc.returncode != 0 or not audio_out.exists():
                raise ClipError("FFMPEG_FAILED", f"Live DVR audio download failed: {proc.stderr.strip()[:400]}")
        return audio_out, manifest_path, float(manifest["duration_sec"])

    # No-DVR fallback: capture bounded real-time window from the live edge
    cap_sec = min(float(window_sec), 180.0)
    capture_mp4 = cache_dir / "live_capture.mp4"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", v_url]
    if a_fmt is not None:
        cmd.extend(["-i", str(a_fmt["url"]), "-map", "0:v:0", "-map", "1:a:0"])
    cmd.extend(["-t", f"{cap_sec:.2f}", "-c", "copy", "-movflags", "+faststart", str(capture_mp4)])
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not capture_mp4.exists():
        raise ClipError("FFMPEG_FAILED", f"Live-edge capture failed: {proc.stderr.strip()[:400]}")

    actual_dur = probe_media(capture_mp4).duration or cap_sec
    local_manifest = {
        "mode": "local_capture",
        "duration_sec": round(actual_dur, 3),
        "capture_path": str(capture_mp4.resolve()),
    }
    manifest_path.write_text(json.dumps(local_manifest, indent=2), encoding="utf-8")
    return capture_mp4, manifest_path, float(actual_dur)


def materialize_live_clip(
    manifest_path: Path,
    start: float,
    end: float,
    clip_temp: Path,
    pad_sec: float = 2.0,
) -> tuple[Path, float]:
    """Download/remux only the buffered DVR segments needed for [start, end], returning (path, trim_offset)."""
    if not manifest_path.is_file():
        raise ClipError("MEDIA_ERROR", f"Live manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ClipError("MEDIA_ERROR", f"Failed to parse live manifest {manifest_path}: {exc}") from exc

    mode = manifest.get("mode")
    if mode == "local_capture":
        cap_path = Path(str(manifest.get("capture_path", "")))
        if not cap_path.is_file():
            raise ClipError("MEDIA_ERROR", f"Live capture file missing: {cap_path}")
        return cap_path.resolve(), start

    if mode != "dvr":
        raise ClipError("MEDIA_ERROR", f"Unsupported live manifest mode: {mode}")

    segs, slice_start = slice_manifest_segments(manifest, start=start, end=end, pad_sec=pad_sec)
    target_dur = int(manifest.get("target_duration", 4))
    has_separate_audio = any(s.get("audio_url") for s in segs)

    video_m3u8 = clip_temp / "clip_video.m3u8"
    video_m3u8.write_text(
        render_m3u8_text([(str(s["video_url"]), float(s["duration"])) for s in segs], target_duration=target_dur),
        encoding="utf-8",
    )

    out_mp4 = clip_temp / "section.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        FFMPEG_HLS_WHITELIST,
        "-i",
        str(video_m3u8),
    ]
    if has_separate_audio:
        audio_m3u8 = clip_temp / "clip_audio.m3u8"
        audio_m3u8.write_text(
            render_m3u8_text(
                [(str(s["audio_url"] or s["video_url"]), float(s["duration"])) for s in segs],
                target_duration=target_dur,
            ),
            encoding="utf-8",
        )
        cmd.extend(["-protocol_whitelist", FFMPEG_HLS_WHITELIST, "-i", str(audio_m3u8), "-map", "0:v:0", "-map", "1:a:0"])
    cmd.extend(["-c", "copy", "-movflags", "+faststart", str(out_mp4)])

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not out_mp4.exists():
        raise ClipError("FFMPEG_FAILED", f"Live DVR clip materialization failed: {proc.stderr.strip()[:400]}")

    return out_mp4, max(0.0, round(start - slice_start, 3))

