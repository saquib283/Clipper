from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2

from lib.audio_peaks import transcribe_audio
from lib.captions import generate_ass
from lib.ffmpeg_ops import (
    probe_media,
    render_face_tracked_clip,
    render_static_layout_clip,
    trim_exact,
)
from lib.models import CaptionSegment, ClipError, ClipSpec, ClipsPlan, WordTiming
from lib.subtitles import parse_json3, parse_vtt, segments_to_proportional_words, slice_words_for_clip
from lib.tracking import analyze_video_tracking
from lib.youtube import download_video_section


def _load_cached_subtitles(cache_path: str | None) -> tuple[list[CaptionSegment], list[WordTiming]]:
    """Load cached .json3 or .vtt subtitle artifacts from Stage 1 when present."""
    if not cache_path:
        return [], []
    p = Path(cache_path)
    if not p.exists():
        return [], []
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".json3":
        return parse_json3(text)
    if p.suffix.lower() == ".vtt":
        return parse_vtt(text), []
    return [], []


def _resolve_clip_words(
    clip: ClipSpec,
    trimmed_path: Path,
    cached_segments: list[CaptionSegment],
    cached_words: list[WordTiming],
) -> tuple[list[WordTiming], str]:
    """Resolve clip-relative word timings via json3 -> faster-whisper -> VTT proportional fallback."""
    if not clip.captions or clip.caption_style == "none":
        return [], "none"
    if cached_words:
        sliced = slice_words_for_clip(cached_words, clip.start, clip.end)
        if sliced:
            return sliced, "youtube_json3"
    try:
        _, words, _ = transcribe_audio(trimmed_path, model_size="base")
        if words:
            return words, "faster_whisper"
    except Exception:
        pass
    if cached_segments:
        prop = segments_to_proportional_words(cached_segments, clip.start, clip.end)
        if prop:
            return prop, "vtt_proportional"
    return [], "none"


def _write_post_kit(plan: ClipsPlan, clip_results: list[dict[str, Any]], out_path: Path) -> None:
    """Write a markdown summary with rendered clip metadata, titles, captions, and hashtags."""
    lines = [
        f"# Post Kit — {plan.video_title or plan.video_id}",
        "",
        f"- **Source:** {plan.video_url}",
        f"- **Clips Generated:** {sum(1 for c in clip_results if c.get('ok'))}/{len(clip_results)}",
        "",
    ]
    spec_by_id = {c.id: c for c in plan.clips}
    for res in clip_results:
        spec = spec_by_id[res["id"]]
        lines.append(f"## {spec.id}: {spec.title or spec.slug}")
        if not res.get("ok"):
            lines.append(f"- **Status:** Failed (`{res.get('error_code')}`: {res.get('message')})")
            lines.append("")
            continue
        mp4_name = Path(res["output_path"]).name
        lines.extend(
            [
                f"- **File:** `{mp4_name}`",
                f"- **Source Window:** `{spec.start:.1f}s – {spec.end:.1f}s` ({res['duration_sec']:.1f}s)",
                f"- **Layout:** `{res['layout_used']}` ({res['resolution']})",
                f"- **Captions:** `{res['caption_source']}` ({spec.caption_style})",
            ]
        )
        if spec.why_selected:
            lines.append(f"- **Why Selected:** {spec.why_selected}")
        if res.get("warnings"):
            lines.append(f"- **Warnings:** {'; '.join(res['warnings'])}")
        if spec.caption_text:
            lines.extend(["", "**Suggested Post Caption:**", "", spec.caption_text])
        if spec.hashtags:
            lines.extend(["", "**Hashtags:** " + " ".join(spec.hashtags)])
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_render(
    plan_path: Path,
    out_dir: Path | None = None,
    local_source: Path | None = None,
    keep_temp: bool = False,
) -> dict[str, Any]:
    """Download/trim, track, caption, and render all clips defined in clips_plan.json."""
    plan = ClipsPlan.from_json_file(plan_path)
    target_dir = out_dir or (Path.cwd() / "clips" / plan.video_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    cached_segments, cached_words = _load_cached_subtitles(plan.cache_subtitles)
    results: list[dict[str, Any]] = []

    for clip in plan.clips:
        clip_temp = Path(tempfile.mkdtemp(prefix=f"clip_{clip.id}_"))
        warnings: list[str] = []
        try:
            clip_dur = clip.end - clip.start
            if local_source is not None:
                source_file = local_source.resolve()
                trim_offset = clip.start
            else:
                buf_start = max(0.0, clip.start - 2.0)
                buf_end = clip.end + 2.0
                source_file = download_video_section(
                    plan.video_url,
                    start=buf_start,
                    end=buf_end,
                    out_path=clip_temp / "section.mp4",
                    cookies_from_browser=plan.cookies_from_browser,
                )
                trim_offset = clip.start - buf_start

            trimmed = trim_exact(source_file, clip_temp / "trimmed.mp4", start_offset=trim_offset, duration=clip_dur)
            media = probe_media(trimmed)

            layout_used = clip.layout
            tracking = None
            if clip.layout in ("face-track", "auto"):
                try:
                    tracking = analyze_video_tracking(trimmed, sample_fps=5.0)
                except cv2.error as exc:
                    raise ClipError("MEDIA_ERROR", f"Video tracking failed for {clip.id}: {exc}") from exc
                if clip.layout == "auto" and tracking.face_ratio < 0.30:
                    layout_used = "blurred-fit"
                    warnings.append(f"Auto layout switched to blurred-fit (face presence {tracking.face_ratio:.0%}).")
                else:
                    layout_used = "face-track"
                    if tracking.face_ratio == 0.0:
                        warnings.append(
                            "No faces detected; fell back to centered 9:16 crop. Consider blurred-fit for slides/gameplay."
                        )

            words, caption_source = _resolve_clip_words(clip, trimmed, cached_segments, cached_words)
            ass_filename: str | None = None
            if words and clip.captions and clip.caption_style != "none":
                play_x, play_y = (media.width, media.height) if layout_used == "landscape" else (1080, 1920)
                ass_content = generate_ass(words, style=clip.caption_style, play_res_x=play_x, play_res_y=play_y)
                (clip_temp / "captions.ass").write_text(ass_content, encoding="utf-8")
                ass_filename = "captions.ass"

            out_mp4 = target_dir / f"{clip.id}_{clip.slug}.mp4"
            if layout_used == "face-track":
                assert tracking is not None
                render_face_tracked_clip(trimmed, out_mp4, tracking, media, ass_filename, clip_temp)
            else:
                render_static_layout_clip(
                    trimmed,
                    out_mp4,
                    layout_used,
                    media,
                    ass_filename,
                    clip_temp,
                    split_cam_corner=clip.split_cam_corner,
                )

            out_info = probe_media(out_mp4)
            results.append(
                {
                    "id": clip.id,
                    "ok": True,
                    "output_path": str(out_mp4.resolve()),
                    "duration_sec": round(out_info.duration, 2),
                    "resolution": f"{out_info.width}x{out_info.height}",
                    "layout_used": layout_used,
                    "caption_source": caption_source,
                    "warnings": warnings,
                }
            )
        except ClipError as err:
            results.append({"id": clip.id, **err.to_dict()})
        finally:
            if not keep_temp:
                shutil.rmtree(clip_temp, ignore_errors=True)

    post_kit_path = target_dir / "post_kit.md"
    _write_post_kit(plan, results, post_kit_path)

    report = {
        "ok": all(r.get("ok") for r in results),
        "video_id": plan.video_id,
        "output_dir": str(target_dir.resolve()),
        "post_kit_path": str(post_kit_path.resolve()),
        "clips": results,
    }
    report_path = target_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Stage 3 clip rendering."""
    parser = argparse.ArgumentParser(description="Render clips from clips_plan.json.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--local-source", type=Path, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = run_render(
            plan_path=args.plan,
            out_dir=args.out,
            local_source=args.local_source,
            keep_temp=args.keep_temp,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    except ClipError as err:
        print(json.dumps(err.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
