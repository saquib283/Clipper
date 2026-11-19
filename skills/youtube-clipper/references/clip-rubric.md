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
