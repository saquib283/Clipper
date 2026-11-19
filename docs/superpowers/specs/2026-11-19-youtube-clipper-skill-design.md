# YouTube Clipper Antigravity Skill — Design Specification

**Date:** 2026-11-19  
**Workspace:** `c:\Users\Admin\Desktop\clipper`  
**Global Install Target:** `C:\Users\Admin\.gemini\config\skills\youtube-clipper`

---

## 1. Goal & Scope

Build a custom Antigravity AI Skill (`youtube-clipper`) backed by a modular local Python media pipeline so the user can drop any finished YouTube URL (video, podcast, or livestream VOD) into any Antigravity chat and receive ready-to-post short-form clips plus a social posting kit—with **zero external LLM API keys** and CPU-only Windows support.

### In Scope
- Finished YouTube videos, podcasts, and livestream VODs.
- Zero-download metadata/heatmap/caption probing via `yt-dlp`, with local audio-peak + `faster-whisper` (`int8` CPU) fallback when YouTube captions are missing.
- Highlight selection, boundary refinement, and social copy generation performed by the Antigravity agent itself using a structured scoring rubric.
- Targeted clip downloading (`yt-dlp --download-sections`) so multi-hour VODs process rapidly.
- Four output layouts: `face-track` (default 9:16), `blurred-fit` (9:16), `split` (9:16 top content + bottom facecam corner), `landscape` (16:9), plus `auto` (chooses `face-track` vs `blurred-fit` based on face presence).
- Word-highlighted `.ass` karaoke captions positioned in the social-media safe zone, loudness-normalized audio (`-14 LUFS`), H.264/AAC MP4 export, and a `post_kit.md` deliverable.
- Workspace repository with automated tests plus a global install/sync script so the skill works from any Antigravity workspace.

### Out of Scope
- Real-time clipping of currently live (in-progress) streams (`is_live: true` returns a structured notice instead of hanging).
- Auto-uploading/posting directly to YouTube/TikTok/Instagram APIs.
- External cloud LLM API calls.

---

## 2. Repository & Global Skill Layout

```text
c:\Users\Admin\Desktop\clipper\
├── docs\superpowers\specs\
│   └── 2026-11-19-youtube-clipper-skill-design.md
├── requirements.txt                 # yt-dlp, opencv-python-headless, faster-whisper
├── requirements-dev.txt             # pytest
├── scripts\
│   └── install_skill.py             # Creates .venv, installs deps, syncs global skill, verifies doctor
├── skills\youtube-clipper\
│   ├── SKILL.md                     # Concise agent workflow (<500 words, SDO trigger frontmatter)
│   ├── references\
│   │   └── clip-rubric.md           # Viral hook rubric, duration/boundary rules, post-kit format
│   └── scripts\
│       ├── run.py                   # Unified CLI launcher routing to the isolated .venv Python
│       ├── doctor.py                # Environment & binary diagnostics
│       ├── probe.py                 # Stage 1 CLI: metadata, heatmap, captions/Whisper fallback
│       ├── render.py                # Stage 3 CLI: download sections, crop, captions, FFmpeg export
│       └── lib\
│           ├── __init__.py
│           ├── models.py            # Shared dataclasses & JSON serialization/validation
│           ├── youtube.py           # yt-dlp metadata, subtitle, audio, and section download wrappers
│           ├── heatmap.py           # YouTube Most Replayed peak extraction & window merging
│           ├── subtitles.py         # json3/vtt parsing, word-timing extraction, transcript chunking
│           ├── audio_peaks.py       # FFmpeg loudness/RMS energy peak scan + faster-whisper wrapper
│           ├── tracking.py          # Scene-cut detection + OpenCV face tracking + EMA crop smoothing
│           ├── captions.py          # Word chunking & styled .ass karaoke subtitle generator
│           └── ffmpeg_ops.py        # Filtergraph builders & Windows-safe FFmpeg execution
└── tests\
    ├── test_heatmap.py
    ├── test_subtitles.py
    ├── test_captions.py
    ├── test_tracking.py
    ├── test_ffmpeg_ops.py
    ├── test_probe_cli.py
    └── test_render_integration.py   # Offline synthetic FFmpeg end-to-end render verification
```

### Cross-Workspace Isolation
`scripts/install_skill.py` creates a single dedicated virtual environment at `c:\Users\Admin\Desktop\clipper\.venv`, copies `skills\youtube-clipper\` into `C:\Users\Admin\.gemini\config\skills\youtube-clipper\`, and writes the absolute path to `.venv\Scripts\python.exe` into `scripts\.venv_python` in both locations. Whenever Antigravity invokes `python <skill_dir>/scripts/run.py <subcommand> ...`, `run.py` delegates to that exact `.venv` interpreter, ensuring consistent dependencies from any workspace on the machine.

---

## 3. End-to-End Architecture & Data Contracts

### Stage 1: Probe (`run.py probe`)
**CLI:**
```text
python run.py probe <youtube_url> [--out <work_dir>] [--lang en] [--max-windows 15] [--full-transcript] [--cookies-from-browser chrome|edge|firefox]
```
*(Default `--out`: `./clips/<video_id>/work` relative to the caller's current working directory.)*

**Behavior:**
1. Calls `yt-dlp --dump-single-json --skip-download` to inspect video metadata, chapters, `heatmap`, and available subtitle languages without downloading video data.
2. If `is_live` is `true`, exits cleanly with a structured JSON error explaining that active in-progress livestreams are not supported yet.
3. Extracts and merges local peaks from YouTube's `heatmap` (`score >= 0.65`, merging peaks within 20 seconds into candidate windows).
4. Downloads subtitles (`json3` preferred because auto-captions include word-level `tOffsetMs`; `vtt` fallback) into `<work_dir>/cache/`.
   - **Fallback when captions are absent:** Downloads audio-only (`m4a`/`opus`), scans RMS energy spikes via FFmpeg (`astats` over 5s windows), and transcribes with `faster-whisper` (`base`, `compute_type="int8"`, CPU). Videos `<= 20 min` are transcribed in full; longer videos transcribe up to `--max-windows` top heatmap/energy windows (90s each).
5. Groups transcript into readable ~15–20s blocks annotated with overlapping `heatmap_score`. For videos `> 90 min` without `--full-transcript`, outputs full chapter list + heatmap peaks + transcript blocks inside the top `--max-windows` high-signal windows so context remains bounded.
6. Writes `<work_dir>/analysis.json` and prints a concise JSON summary to stdout.

**`analysis.json` Schema:**
```json
{
  "video": {
    "id": "dQw4w9WgXcQ",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "title": "Video Title",
    "channel": "Channel Name",
    "duration_sec": 3600.0,
    "upload_date": "20261119",
    "is_live": false,
    "was_live": true
  },
  "signals": {
    "caption_source": "youtube_json3",
    "language": "en",
    "has_heatmap": true,
    "has_word_timestamps": true,
    "truncated_to_windows": false,
    "chapters": [{"title": "Intro", "start": 0.0, "end": 120.0}],
    "heatmap_peaks": [{"start": 410.0, "end": 465.0, "peak_time": 438.0, "score": 0.94}],
    "audio_peaks": []
  },
  "transcript_blocks": [
    {"start": 410.2, "end": 425.8, "text": "...", "heatmap_score": 0.91}
  ],
  "cache_paths": {
    "subtitles": "C:/.../cache/subtitles.json3",
    "audio": null
  }
}
```

### Stage 2: Select (Antigravity Agent Brain via `SKILL.md` + `references/clip-rubric.md`)
1. Reads `analysis.json` and `references/clip-rubric.md`.
2. Ranks candidate clips using four weighted criteria:
   - **3-Second Hook (35%)**: Immediate curiosity gap, bold claim, conflict, emotion, or punchy question.
   - **Standalone Payoff (30%)**: Complete mini-story/insight requiring zero prior video context.
   - **Signal Boost (20%)**: Corroboration from YouTube Most Replayed heatmap or audio energy peaks.
   - **Pacing & Clean Boundaries (15%)**: Target 25–60 seconds (configurable per user prompt), starting and ending on complete sentence boundaries.
3. Selects top `N` non-overlapping clips (default `3`), writes `<work_dir>/clips_plan.json`, and proceeds to Stage 3 immediately (unless the user explicitly asked to preview/choose candidates first).

**`clips_plan.json` Schema:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=...",
  "video_id": "abc123",
  "video_title": "Original Video Title",
  "cache_subtitles": "C:/.../cache/subtitles.json3",
  "cookies_from_browser": null,
  "clips": [
    {
      "id": "clip_01",
      "slug": "contrarian-hook",
      "start": 412.4,
      "end": 458.1,
      "layout": "face-track",
      "split_cam_corner": "br",
      "captions": true,
      "caption_style": "bold-highlight",
      "title": "Short Viral Title",
      "caption_text": "Ready-to-paste social caption",
      "hashtags": ["#shorts", "#podcast"],
      "why_selected": "Strong hook + 0.94 Most Replayed peak"
    }
  ]
}
```

### Stage 3: Render (`run.py render`)
**CLI:**
```text
python run.py render --plan <clips_plan.json> [--out <output_dir>] [--local-source <video_path>] [--keep-temp]
```
*(Default `--out`: `./clips/<video_id>` relative to the caller's current working directory.)*

**Per-Clip Pipeline:**
1. **Acquire clip source:** Unless `--local-source` is provided (for local files or offline tests), downloads only `[max(0, start - 2.0), end + 2.0]` via `yt-dlp --download-sections` at up to 1080p MP4/H.264, then performs a frame-accurate FFmpeg trim to `[start, end]`.
2. **Word-level caption timing:**
   - Slices word timestamps from cached `json3` subtitles if present and word-timed.
   - Otherwise runs a fast (~3–5s CPU) `faster-whisper` (`base`, `int8`, `word_timestamps=True`) pass on the short trimmed clip audio (or falls back to proportional word-splitting of VTT cue timing if `faster-whisper` is unavailable).
   - Generates `<clip_temp>/captions.ass` (1080×1920 `PlayRes`, bold sans-serif font, thick dark outline + drop shadow, safe-zone `MarginV=360` above TikTok/Reels/Shorts UI chrome, 3–5 words per line with active-word `#FFE600` highlight for `bold-highlight`, or clean bottom caption for `clean`).
3. **Layout Transform:**
   - **`face-track` (default, 1080×1920):** Samples video at 5 fps (downscaled to 640px wide) using OpenCV's bundled Haar frontal + profile face detectors (zero extra downloads). Computes frame-to-frame HSV histogram deltas to detect hard scene cuts. Within each shot, applies deadzone filtering + exponential moving average (EMA) smoothing on the horizontal crop center `x` (locking onto the largest face, defaulting to frame center if no face is found); across hard scene cuts, resets EMA so the camera snaps cleanly instead of panning mid-cut. Streams native-resolution cropped BGR slices (`crop_w × src_h`, interpolated per frame) directly into FFmpeg's stdin (`-f rawvideo -pix_fmt bgr24`), while FFmpeg performs Lanczos upscale to `1080×1920`, burns `ass=captions.ass`, and muxes the loudness-normalized audio stream.
   - **`auto`:** Runs the 5 fps face scan; uses `face-track` when a face is detected in `>= 30%` of sampled frames, otherwise switches to `blurred-fit` and records `layout_used: "blurred-fit"` in `render_report.json`.
   - **`blurred-fit` (1080×1920):** Blurred, darkened zoomed 9:16 background with full uncropped 16:9 content centered on top.
   - **`split` (1080×1920):** Top half (`1080×960`) shows scaled/cropped main content; bottom half (`1080×960`) crops and zooms the configured webcam corner (`br`, `bl`, `tr`, `tl` — bottom/top 35% height × 30% width).
   - **`landscape` (original 16:9):** Frame-accurate cut with clean bottom subtitles.
4. **Audio & Encode:** Normalizes audio with FFmpeg `loudnorm=I=-14:TP=-1.5:LRA=11`, encodes H.264 (`libx264`, `pix_fmt yuv420p`, `crf 20`, `preset medium`, `-movflags +faststart`) + AAC 192k.
5. **Deliverables:** Writes `<output_dir>/<id>_<slug>.mp4`, `<output_dir>/post_kit.md`, and `<output_dir>/render_report.json`.

### Diagnostics (`run.py doctor`)
Checks and reports JSON/human status for: Python version, `ffmpeg`, `ffprobe`, `yt-dlp`, `cv2` (+ bundled Haar cascades), `faster_whisper`, and global skill installation path.

---

## 4. Error Handling & Windows Hardening

| Failure Mode | Handling Strategy |
|---|---|
| Python 3.15/3.14 missing prebuilt ML wheels | `install_skill.py` checks wheel availability; prefers an installed Python with full binary wheels (3.15 → 3.14 → 3.12), or bootstraps standalone `uv` + Python 3.12 for `.venv` automatically—never failing on C++ compiler errors. |
| FFmpeg Windows subtitle path parsing (`ass=C:\...`) | `ffmpeg_ops.py` always runs FFmpeg with `cwd=<clip_temp_dir>` and passes relative `ass=captions.ass`, eliminating drive-letter `:` and backslash escaping bugs. |
| Video has no Most Replayed heatmap | `heatmap.py` returns `[]`; scoring uses transcript + optional audio energy peaks. |
| Video has no YouTube captions | `probe.py` downloads audio-only, scans FFmpeg RMS energy peaks, and runs local `faster-whisper` (`int8` CPU). |
| Active live broadcast (`is_live: true`) | `probe.py` aborts immediately with exit code 2 and JSON error code `LIVE_STREAM_ACTIVE`. |
| Private / age-gated / bot-checked YouTube URL | `youtube.py` maps stderr to structured error codes (`AUTH_REQUIRED`, `BOT_CHECK`, `UNAVAILABLE`) recommending `--cookies-from-browser chrome|edge`. |
| `yt-dlp --download-sections` failure on exotic streams | Retries once with safe format selector `bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best` before surfacing a structured clip-level error without crashing remaining clips in the batch. |
| Zero faces found during `face-track` | Falls back cleanly to center 9:16 crop and records a warning in `render_report.json` suggesting `blurred-fit` if content is slides/gameplay. |
| Dynamic face-track crop smoothness | Within a single shot, smooth crop coordinates are applied via a Python frame-cropper streaming native cropped BGR slices into FFmpeg (keeping upscale, audio, and ASS captions in FFmpeg) so camera motion is buttery-smooth without FFmpeg expression length limits. |

---

## 5. Skill Design (`SKILL.md`)

Follows `agy-customizations` and `writing-skills` best practices:
- **Frontmatter:** `name: youtube-clipper` and trigger-only third-person `description` starting with `Use when...` (covers YouTube links, Shorts, Reels, TikToks, highlights, podcasts, livestream VODs) **without summarizing the workflow** in the description so agents always read the full `SKILL.md`.
- **Body (<500 words):** Clear numbered 3-step runbook (`doctor` if unverified → `probe` → score via `references/clip-rubric.md` & write `clips_plan.json` → `render`), layout decision table, common mistakes / red flags (never download full video with raw `yt-dlp`, never pass raw Windows paths into FFmpeg `ass=`, never cut mid-sentence).
- **Progressive disclosure:** Scoring rubric, hook templates, and `post_kit.md` structure live in `references/clip-rubric.md`.

---

## 6. Testing & Verification Plan

1. **Offline Unit Tests (`pytest`):**
   - `test_heatmap.py`: Peak detection, threshold filtering, adjacent peak merging, empty/missing heatmap handling.
   - `test_subtitles.py`: YouTube `json3` word-offset parsing, `vtt` cue parsing, deduplication of rolling auto-caption lines, transcript chunking & long-VOD windowing.
   - `test_captions.py`: 3–5 word line wrapping, active-word ASS color tag timing, special character escaping, timestamp formatting.
   - `test_tracking.py`: Scene-cut histogram reset, EMA smoothing + deadzone clamping within frame bounds, `auto` layout decision threshold.
   - `test_ffmpeg_ops.py`: Filtergraph generation for `blurred-fit`, `split` (all 4 corners), `landscape`, loudness normalization args, relative `ass=captions.ass` invocation.
   - `test_probe_cli.py`: Mocked `yt-dlp` JSON outputs covering normal video, missing heatmap, missing captions, and `is_live: true` rejection.
2. **Offline Real-FFmpeg Integration Test (`test_render_integration.py`):**
   - Generates a 6-second synthetic 1280×720 30fps MP4 with sine audio using real `ffmpeg`, runs `render.py --local-source` across `face-track`, `blurred-fit`, `split`, and `landscape` with captions enabled, and asserts exact dimensions (`1080×1920` / `1280×720`), valid H.264/AAC streams, expected duration, and `post_kit.md` generation via `ffprobe`.
3. **Skill TDD Verification (per `writing-skills`):**
   - **RED (baseline):** Run a realistic YouTube clipping scenario with a subagent *without* `SKILL.md` to observe natural failures (e.g., downloading the entire 2-hour video, hardcoding Windows `ass=C:\...` paths, or improvising unsmoothed static crops).
   - **GREEN:** Run the same scenario *with* `SKILL.md` installed and verify compliance with the `probe` → rubric → `render` workflow.
   - **REFACTOR:** Tighten any ambiguity or loopholes discovered during GREEN testing and re-verify.
4. **Global Installation Verification:**
   - Run `scripts/install_skill.py`, verify `C:\Users\Admin\.gemini\config\skills\youtube-clipper\SKILL.md` exists, and invoke `run.py doctor` from a different working directory (e.g., `C:\Users\Admin\Desktop`) to confirm cross-workspace execution succeeds.
