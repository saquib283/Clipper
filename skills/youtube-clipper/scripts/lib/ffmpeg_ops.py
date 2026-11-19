from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2

from lib.models import ClipError, MediaInfo, TrackingPlan

AUDIO_LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def _parse_fps(rate_str: str) -> float:
    """Parse an ffprobe rational frame rate string (e.g. '30000/1001') into float fps."""
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        return float(num) / max(1.0, float(den))
    return float(rate_str or 30.0)


def _validate_relative_ass_filename(ass_filename: str) -> str:
    """Ensure subtitle path passed to FFmpeg's libass filter is relative and safe on Windows."""
    if not ass_filename or ":" in ass_filename or "\\" in ass_filename or Path(ass_filename).is_absolute():
        raise ClipError(
            "INVALID_PATH",
            f"Subtitle path for FFmpeg filter must be a relative filename inside cwd: {ass_filename!r}",
        )
    return ass_filename


def _audio_and_video_encode_args(
    media: MediaInfo,
    out_path: Path,
    audio_map: str = "0:a:0",
    shortest: bool = False,
) -> list[str]:
    """Build standardized H.264 + loudnorm AAC encoding arguments for final clip output."""
    args: list[str] = []
    if media.has_audio:
        args.extend(["-map", audio_map, "-af", AUDIO_LOUDNORM, "-c:a", "aac", "-b:a", "192k"])
    else:
        args.append("-an")
    args.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    )
    if shortest:
        args.append("-shortest")
    args.append(str(out_path.resolve()))
    return args


def probe_media(path: Path) -> MediaInfo:
    """Inspect video dimensions, frame rate, duration, and audio presence via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ClipError("MEDIA_ERROR", f"ffprobe failed for {path}: {proc.stderr.strip()[:400]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ClipError("MEDIA_ERROR", f"Invalid ffprobe JSON output for {path}: {exc}") from exc

    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ClipError("MEDIA_ERROR", f"No video stream found in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    raw_duration = data.get("format", {}).get("duration", 0.0)
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        duration = 0.0

    return MediaInfo(
        width=int(v_stream["width"]),
        height=int(v_stream["height"]),
        fps=round(_parse_fps(str(v_stream.get("r_frame_rate", "30/1"))), 3),
        duration=duration,
        has_audio=has_audio,
    )


def trim_exact(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path:
    """Frame-accurately trim a source clip by re-encoding to an intermediate H.264/AAC mp4."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_offset):.3f}",
        "-i",
        str(src_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ClipError("FFMPEG_FAILED", f"Exact trim failed: {proc.stderr.strip()[:400]}")
    return out_path


def build_static_filtergraph(
    layout: str,
    src_w: int,
    src_h: int,
    has_captions: bool,
    split_cam_corner: str = "br",
    ass_filename: str = "captions.ass",
) -> str:
    """Build an FFmpeg -filter_complex string for landscape, blurred-fit, or split layouts."""
    if has_captions:
        safe_ass = _validate_relative_ass_filename(ass_filename)
        ass_suffix = f",ass={safe_ass}"
    else:
        ass_suffix = ""

    if layout == "landscape":
        base = "setsar=1"
        return f"[0:v]{base}{ass_suffix}[vout]"

    if layout == "blurred-fit":
        return (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.12[bgb];"
            "[fg]scale=1080:-2:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1{ass_suffix}[vout]"
        )

    if layout == "split":
        cam_w = max(2, int(round(src_w * 0.30)) // 2 * 2)
        cam_h = max(2, int(round(src_h * 0.35)) // 2 * 2)
        x_map = {"br": src_w - cam_w, "tr": src_w - cam_w, "bl": 0, "tl": 0}
        y_map = {"br": src_h - cam_h, "bl": src_h - cam_h, "tr": 0, "tl": 0}
        if split_cam_corner not in x_map:
            raise ClipError("INVALID_LAYOUT", f"Unsupported split_cam_corner: {split_cam_corner}")
        cx, cy = x_map[split_cam_corner], y_map[split_cam_corner]
        return (
            "[0:v]split=2[top_src][bot_src];"
            "[top_src]scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[top];"
            f"[bot_src]crop={cam_w}:{cam_h}:{cx}:{cy},scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960[bot];"
            f"[top][bot]vstack=inputs=2,setsar=1{ass_suffix}[vout]"
        )

    raise ClipError("INVALID_LAYOUT", f"Unsupported static layout: {layout}")


def render_static_layout_clip(
    trimmed_path: Path,
    out_path: Path,
    layout: str,
    media: MediaInfo,
    ass_filename: str | None,
    work_dir: Path,
    split_cam_corner: str = "br",
) -> Path:
    """Render a static layout clip via FFmpeg filter_complex using work_dir as cwd for relative ASS paths."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fg = build_static_filtergraph(
        layout=layout,
        src_w=media.width,
        src_h=media.height,
        has_captions=bool(ass_filename),
        split_cam_corner=split_cam_corner,
        ass_filename=ass_filename or "captions.ass",
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(trimmed_path.resolve()),
        "-filter_complex",
        fg,
        "-map",
        "[vout]",
        *_audio_and_video_encode_args(media, out_path, audio_map="0:a:0"),
    ]
    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ClipError("FFMPEG_FAILED", f"Static render ({layout}) failed: {proc.stderr.strip()[:400]}")
    return out_path


def render_face_tracked_clip(
    trimmed_path: Path,
    out_path: Path,
    tracking: TrackingPlan,
    media: MediaInfo,
    ass_filename: str | None,
    work_dir: Path,
) -> Path:
    """Crop per-frame face-tracked columns in OpenCV and pipe raw BGR frames into FFmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf_parts = ["scale=1080:1920:flags=lanczos", "setsar=1"]
    if ass_filename:
        vf_parts.append(f"ass={_validate_relative_ass_filename(ass_filename)}")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{tracking.crop_width}x{tracking.src_height}",
        "-r",
        f"{tracking.fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(trimmed_path.resolve()),
        "-map",
        "0:v:0",
        "-vf",
        vf,
        *_audio_and_video_encode_args(media, out_path, audio_map="1:a:0", shortest=True),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(work_dir))
    cap = cv2.VideoCapture(str(trimmed_path))
    if not cap.isOpened():
        proc.kill()
        if proc.stdin:
            proc.stdin.close()
        if proc.stderr:
            proc.stderr.close()
        proc.wait()
        raise ClipError("MEDIA_ERROR", f"OpenCV failed to open {trimmed_path}")

    frame_idx = 0
    x0s = tracking.x0_per_frame
    crop_w = tracking.crop_width
    default_x0 = max(0, (tracking.src_width - crop_w) // 2)

    try:
        assert proc.stdin is not None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            x0 = x0s[frame_idx] if frame_idx < len(x0s) else (x0s[-1] if x0s else default_x0)
            max_x0 = max(0, frame.shape[1] - crop_w)
            x0 = max(0, min(max_x0, x0))
            cropped = frame[:, x0 : x0 + crop_w]
            if not cropped.flags["C_CONTIGUOUS"]:
                cropped = cropped.copy()
            proc.stdin.write(cropped.tobytes())
            frame_idx += 1
    except (BrokenPipeError, OSError):
        pass
    finally:
        cap.release()
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass

    stderr_bytes = b""
    if proc.stderr:
        try:
            stderr_bytes = proc.stderr.read()
        finally:
            proc.stderr.close()
    retcode = proc.wait()
    if retcode != 0:
        raise ClipError("FFMPEG_FAILED", f"Face-tracked render failed: {stderr_bytes.decode('utf-8', 'ignore')[:400]}")
    return out_path
