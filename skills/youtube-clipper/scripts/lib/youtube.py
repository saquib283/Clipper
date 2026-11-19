from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.models import ClipError


def _ytdlp_base_cmd(cookies_from_browser: str | None = None) -> list[str]:
    """Build the base yt-dlp invocation with optional browser cookies."""
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings"]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    return cmd


def _classify_ytdlp_error(stderr: str, url: str) -> ClipError:
    """Map yt-dlp stderr messages to structured ClipError codes."""
    low = stderr.lower()
    if "sign in to confirm you" in low or "bot" in low:
        return ClipError(
            "BOT_CHECK",
            f"YouTube requested bot verification for {url}.",
            hint="Retry with --cookies-from-browser chrome (or edge/firefox).",
        )
    if "sign in to confirm your age" in low or "members-only" in low or "private video" in low:
        return ClipError(
            "AUTH_REQUIRED",
            f"Video requires authentication or membership: {url}",
            hint="Retry with --cookies-from-browser chrome (or edge/firefox).",
        )
    return ClipError("UNAVAILABLE", f"yt-dlp failed for {url}: {stderr.strip()[:400]}")


def fetch_video_info(url: str, cookies_from_browser: str | None = None) -> dict[str, Any]:
    """Fetch video metadata, chapters, and heatmap JSON without downloading media."""
    cmd = _ytdlp_base_cmd(cookies_from_browser) + ["--dump-single-json", "--skip-download", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise _classify_ytdlp_error(proc.stderr, url)
    return json.loads(proc.stdout)


def download_subtitles(
    url: str,
    out_dir: Path,
    lang: str = "en",
    cookies_from_browser: str | None = None,
) -> tuple[Path | None, str, str]:
    """Download manual or auto-generated subtitles preferring json3 over vtt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt, source_kind in (("json3", "youtube_json3"), ("vtt", "youtube_vtt")):
        template = str(out_dir / "subtitles.%(ext)s")
        cmd = _ytdlp_base_cmd(cookies_from_browser) + [
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            f"{lang}.*,{lang}",
            "--sub-format",
            fmt,
            "-o",
            template,
            url,
        ]
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        matches = sorted(out_dir.glob(f"subtitles*.{fmt}"))
        if matches:
            chosen = matches[0]
            detected_lang = chosen.stem.split(".")[-1] if "." in chosen.stem else lang
            return chosen, source_kind, detected_lang
    return None, "none", lang


def download_audio_only(
    url: str,
    out_dir: Path,
    cookies_from_browser: str | None = None,
) -> Path:
    """Download the best available audio-only stream for fallback transcription."""
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "source_audio.%(ext)s")
    cmd = _ytdlp_base_cmd(cookies_from_browser) + [
        "-f",
        "bestaudio[ext=m4a]/bestaudio/best",
        "-o",
        template,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise _classify_ytdlp_error(proc.stderr, url)
    matches = sorted(out_dir.glob("source_audio.*"))
    if not matches:
        raise ClipError("DOWNLOAD_FAILED", "Audio download completed but no output file was found.")
    return matches[0]


def download_video_section(
    url: str,
    start: float,
    end: float,
    out_path: Path,
    cookies_from_browser: str | None = None,
) -> Path:
    """Download only the requested [start, end] section of a YouTube video as mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start:.2f}-{end:.2f}"
    formats = [
        "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "best[height<=1080]/best",
    ]
    last_stderr = ""
    for fmt in formats:
        if out_path.exists():
            out_path.unlink()
        cmd = _ytdlp_base_cmd(cookies_from_browser) + [
            "-f",
            fmt,
            "--download-sections",
            section,
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out_path),
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode == 0 and out_path.exists():
            return out_path
        last_stderr = proc.stderr
    raise _classify_ytdlp_error(last_stderr, url)
