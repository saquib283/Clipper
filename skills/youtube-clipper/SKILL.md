---
name: youtube-clipper
description: >-
  Use when the user asks to clip, cut, extract highlights, or create YouTube
  Shorts, Instagram Reels, or TikToks from a YouTube video, podcast, livestream
  VOD, or active YouTube live broadcast URL.
---

# YouTube Clipper

Turn YouTube videos, podcasts, livestream VODs, and active YouTube live broadcasts into ready-to-post short clips with zero external API keys. Let `<SKILL_DIR>` be the folder containing this `SKILL.md`.

**NEVER create a `.venv`, run `pip install`, or write ad-hoc Python/FFmpeg/ASS scripts.** Invoke [run.py](./scripts/run.py) with any system `python`—it automatically routes to the pre-built isolated `.venv` (`yt_dlp`, `cv2`, `faster_whisper`) and runs FFmpeg with Windows-safe relative subtitle paths.

## Workflow

1. **Verify setup (once per session if unsure):**
   ```powershell
   python "<SKILL_DIR>/scripts/run.py" doctor --json
   ```
   If `"ok": false`, report the failing entry in `checks` and stop.

2. **Probe without downloading the video:**
   ```powershell
   python "<SKILL_DIR>/scripts/run.py" probe "<YOUTUBE_URL>"
   ```
   Flags: `--out <work_dir>`, `--lang <code>`, `--max-windows <n>`, `--full-transcript`, `--live-window <sec>` (default `300`, up to `900` for active live DVR tail; `0` rejects live), `--cookies-from-browser chrome|edge|firefox`.
   `probe` fetches metadata, Most Replayed heatmap, and YouTube `json3`/`vtt` subtitles first; for active live streams it grabs bounded DVR audio + `live_manifest.json` and transcribes with `faster-whisper`.
   - Exit `2` with `LIVE_STREAM_ACTIVE`: only occurs when `--live-window 0` was passed.
   - `BOT_CHECK` or `AUTH_REQUIRED`: ask whether to retry with `--cookies-from-browser`.

3. **Select clips (Agent Brain):**
   Read `analysis.json` (`analysis_path` from `probe`; its folder is `<WORK_DIR>`) and [clip-rubric.md](./references/clip-rubric.md).
   - Score hooks, payoffs, `signals.heatmap_peaks`, and `signals.audio_peaks`; snap `start`/`end` to sentence boundaries in `transcript_blocks` (default `3` non-overlapping clips, `25–60s` or requested length).
   - Choose `layout` (`face-track` default, `auto`, `blurred-fit`, `split`, `landscape`), `caption_style` (`bold-highlight`, `clean`, `none`), and optional `effects` (`0–3` tasteful cues from `signals.available_sfx` per `clip-rubric.md`, `>=6s` apart—never spam).
   - If the user asked to preview candidates first, share the shortlist with timestamps and wait; otherwise write `<WORK_DIR>/clips_plan.json` (`cache_live_manifest` copied when live; `title`, `caption_text`, `hashtags`, `why_selected` populated) and proceed.

4. **Render targeted clips:**
   ```powershell
   python "<SKILL_DIR>/scripts/run.py" render --plan "<WORK_DIR>/clips_plan.json"
   ```
   Flags: `--out <output_dir>`, `--local-source <video_path>`, `--keep-temp`.
   `render` downloads only winning sections (`--download-sections` or DVR segments from `live_manifest.json`), runs face-tracking/layout crops, normalizes audio, burns escaped ASS captions via relative `ass=captions.ass`, and outputs `render_report.json` + `post_kit.md`.

5. **Deliver results:**
   Read `render_report.json` and `post_kit.md`, then return clickable `file://` links to each `.mp4` and `post_kit.md`, summarizing titles, timestamps, `layout_used`, and any `warnings`.

## Red Flags — Never Do These

| Mistake | Do This Instead |
|---|---|
| Creating `.venv`, running `pip install`, or ad-hoc scripts | Invoke `scripts/run.py`; it auto-routes to the pre-built `.venv`. |
| Downloading full multi-hour MP4 via raw `yt-dlp` | Run `run.py probe` first; `run.py render` fetches only winning sections. |
| Extracting full-VOD WAV or running CPU Whisper on hours | Let `run.py probe` use YouTube `json3`/heatmap first and windowed fallback. |
| Fixed sliding windows or cutting raw peaks mid-sentence | Score hooks/payoffs and snap `start`/`end` to sentences per `clip-rubric.md`. |
| Raw `ffmpeg` center crop or `-vf ass=C:\...` on Windows | Use `run.py render` for layouts and Windows-safe relative `ass=captions.ass`. |
| Unescaped ASS scripts, external LLM APIs, or skipping `post_kit.md` | Populate plan copy fields from `analysis.json`; let `run.py render` generate ASS + `post_kit.md`. |
