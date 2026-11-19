# Highlight Selection Rubric & Post-Kit Guide

## Scoring Rubric (100 Points)

Score candidate windows from `analysis.json` (`signals.heatmap_peaks`, `signals.audio_peaks`, `signals.chapters`, and `transcript_blocks`):

| Criterion | Weight | What Scores High |
|---|---|---|
| **3-Second Hook** | 35 | Opens immediately with a bold claim, contrarian take, emotional reaction, surprising stat, or tension question—never "Hey guys" or slow throat-clearing. |
| **Standalone Payoff** | 30 | Complete micro-story or punchline understandable with zero context from earlier in the video. |
| **Signal Corroboration** | 20 | Overlaps a high YouTube Most Replayed `heatmap_score` (`>= 0.70`) or `signals.audio_peaks` energy peak. |
| **Pacing & Boundaries** | 15 | 25–60 seconds (or near the user's requested duration); starts cleanly at a sentence start and ends right after a complete payoff sentence. |

## Boundary & Overlap Rules
- Never overlap selected clips (`clip[i].end + 5s <= clip[i+1].start`).
- Never use mechanical fixed-duration sliding windows or cut blindly on peak timestamps—always read `transcript_blocks` and snap `start`/`end` to complete sentence boundaries.
- Trim filler intros ("uh, so basically...") by starting on the first high-energy clause.
- Add `0.3s` breathing room after the closing word (clamped to `video.duration_sec`) so the final syllable is never clipped.

## Layout & Caption Selection Guide
- **`face-track` (default):** Podcasts, talking heads, interviews, IRL streams.
- **`auto`:** Unknown visual style—uses `face-track` when faces appear in `>= 30%` of sampled frames, otherwise switches to `blurred-fit`.
- **`blurred-fit`:** Coding tutorials, slide decks, multi-person wide shots, or anything where cropping loses context.
- **`split`:** Gaming livestreams with a webcam overlay (`split_cam_corner`: `br`, `bl`, `tr`, or `tl`).
- **`landscape`:** User explicitly asks for 16:9 widescreen clips.
- **`caption_style`:** `"bold-highlight"` (default word-highlighted ASS captions), `"clean"`, or `"none"` (with `captions: true|false`).

## Meme SFX & Visual Punches (Strict Anti-Bombardment Rules)
- Check `signals.available_sfx` in `analysis.json` (includes built-in royalty-free starter sounds plus any custom `.mp3`/`.wav` dropped in `custom_sfx/`).
- **Never bombard clips:** Use **0–3 cues per 60s clip**, spaced at least **6 seconds apart** (hard-enforced by `render.py`). Keep `"effects": []` on serious, educational, or emotional clips unless the user asked for effects.
- **Pacing pattern (for funny / gaming / IRL clips):** At most **1 subtle hook accent** (`0–4s`, e.g., `shutter-snap`, `alert`, `soft-pop`) + **1 climax hit** (main funny/sus/chaos punchline, e.g., `vine-boom`, `sus`, `metal-pipe`) + **1 optional payoff sting** (`womp-womp`, `rizz`, `apple-pay`).
- **Gain staging:** Keep `"volume"` subtle (`0.30–0.45`, default `0.35`) so SFX season speech instead of drowning it out.
- **Visual punches (`visual`):** `"none"`, `"zoom-punch"` (0.35s 1.12x snap), `"bw-flash"` (0.6s sus/fail grayscale), `"red-flash"` (0.35s rage/alert tint), `"shake"` (0.35s micro-shake).
- Place `"at"` (clip-relative seconds) right on the exact punchline word from `cache_paths.subtitles`!

## `clips_plan.json` Schema
```json
{
  "video_url": "<video.url from analysis.json>",
  "video_id": "<video.id from analysis.json>",
  "video_title": "<video.title from analysis.json>",
  "cache_subtitles": "<cache_paths.subtitles or null>",
  "cache_live_manifest": "<cache_paths.live_manifest or null>",
  "cookies_from_browser": null,
  "clips": [
    {
      "id": "clip_01",
      "slug": "short-kebab-slug",
      "start": 412.4,
      "end": 458.1,
      "layout": "face-track",
      "split_cam_corner": "br",
      "captions": true,
      "caption_style": "bold-highlight",
      "effects": [
        {"at": 1.2, "sfx": "alert", "visual": "zoom-punch", "volume": 0.35, "duration": 0.35}
      ],
      "title": "Scroll-stopping title (<60 chars)",
      "caption_text": "1-2 sentence engaging post caption with a comment hook.",
      "hashtags": ["#shorts", "#topic"],
      "why_selected": "1-line rationale referencing hook + heatmap/energy score"
    }
  ]
}
```
If `run.py probe` used `--cookies-from-browser`, set `"cookies_from_browser"` to that same browser name.

## Post-Kit Deliverable (`post_kit.md`)
Always populate `title`, `caption_text`, `hashtags`, and `why_selected` in `clips_plan.json`. `run.py render` automatically writes `post_kit.md` and `render_report.json` alongside the rendered `.mp4` clips.
