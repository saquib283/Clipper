# Clipper (`youtube-clipper` Antigravity Skill)

Zero-API-key AI video clipper skill for Antigravity that turns **YouTube videos, podcasts, livestream VODs, and active YouTube live broadcasts** into ready-to-post Shorts, Reels, and TikToks—plus the complete **TeeniTinyClips** social media brand package.

## Features
- **Zero-Download VOD Probing (`probe`):** Pulls metadata, chapters, YouTube **Most Replayed** heatmap peaks, and word-timed `json3`/`vtt` subtitles via `yt-dlp` without downloading multi-hour videos. Falls back to FFmpeg RMS audio peaks + local CPU `faster-whisper` (`int8`) when captions are missing.
- **Instant Live DVR Clipping (`--live-window`):** Slices the most recent 5–15 minutes of active YouTube Live HLS DVR buffers at full download speed (no waiting in real time) and remuxes only the winning clip segments.
- **5 Social-Ready Layouts (`render`):**
  - `face-track` — 9:16 (`1080×1920`) OpenCV face tracking with HSV scene-cut snapping + deadzone/EMA smoothing
  - `auto` — switches between `face-track` and `blurred-fit` based on detected face presence
  - `blurred-fit` — full uncropped 16:9 frame over a dark blurred 9:16 canvas (ideal for gaming streams with mid-side facecams, tutorials, and wide shots)
  - `split` — top gameplay/content + bottom zoomed webcam corner (`br`, `bl`, `tr`, `tl`)
  - `landscape` — original 16:9 with clean burned subtitles
- **Karaoke Captions & Social Audio:** Word-highlighted `.ass` subtitles in the Shorts/Reels/TikTok safe zone (Windows-safe relative `ass=captions.ass`), `-14 LUFS` `loudnorm`, H.264 `yuv420p` (`+faststart`), AAC 192k, and auto-generated `post_kit.md`.
- **Trend-Tuned Meme SFX & Visual Punches:** Built-in royalty-free starter soundboard (`assets/sfx/`) + automatic drop-in discovery in `custom_sfx/`, subtle visual punches (`zoom-punch`, `bw-flash`, `red-flash`, `shake`), automatic limiting/gain staging, and hard anti-bombardment guardrails (`<= 3` cues/min, `>= 6s` spacing).
- **TeeniTinyClips Brand Package:** Logos, safe-zone verified `2560×1440` YouTube banner, `1640×624` Facebook cover, transparent `150×150` YouTube watermark, 4 Instagram Story Highlight covers, and bios in `brand/TeeniTinyClips/`.

## Quick Setup (Windows)

Prerequisites: **FFmpeg / FFprobe** on `PATH` and **Python 3.12–3.14** (prebuilt binary wheels for `faster-whisper` and `opencv-python-headless`).

```powershell
python scripts/install_skill.py
```

This creates an isolated `.venv`, installs dependencies, generates the built-in starter SFX pack, and syncs `skills/youtube-clipper/` globally to `~/.gemini/config/skills/youtube-clipper/` so the skill works from any workspace.

Verify installation:
```powershell
python skills/youtube-clipper/scripts/run.py doctor --json
```

Run tests:
```powershell
.\.venv\Scripts\pytest.exe -v
```

## Repository Structure
- `skills/youtube-clipper/` — Antigravity skill (`SKILL.md`, `references/clip-rubric.md`, `scripts/`)
- `scripts/install_skill.py` — Venv bootstrapper & global skill sync script
- `custom_sfx/` — Drop-in folder for custom `.mp3`/`.wav` meme SFX (binaries gitignored; see `custom_sfx/README.txt`)
- `brand/TeeniTinyClips/` — Complete YouTube, Instagram, and Facebook brand kit
- `tests/` — 67 unit & real-FFmpeg offline integration tests
- `docs/superpowers/` — Design spec and implementation plan
