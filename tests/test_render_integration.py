from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.ffmpeg_ops import probe_media  # noqa: E402
from lib.models import ClipError, MediaInfo, TrackingPlan  # noqa: E402
import render as render_mod  # noqa: E402
from render import main as render_main, run_render  # noqa: E402


def _make_synthetic_video(out_path: Path, duration: float = 4.0, with_audio: bool = True) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=24:duration={duration}",
    ]
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"])
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if with_audio:
        cmd.extend(["-c:a", "aac"])
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)
    return out_path


def test_render_end_to_end_all_four_layouts(tmp_path: Path) -> None:
    src = _make_synthetic_video(tmp_path / "synthetic.mp4", duration=4.0)
    subs = tmp_path / "subs.json3"
    subs.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 500,
                        "dDurationMs": 2500,
                        "segs": [
                            {"utf8": "Testing"},
                            {"utf8": " viral", "tOffsetMs": 600},
                            {"utf8": " clip", "tOffsetMs": 1200},
                            {"utf8": " captions", "tOffsetMs": 1800},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan_path = tmp_path / "clips_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=synth01",
                "video_id": "synth01",
                "video_title": "Synthetic Stream",
                "cache_subtitles": str(subs),
                "clips": [
                    {"id": "c1", "slug": "face", "start": 0.5, "end": 3.0, "layout": "face-track", "title": "Face Clip", "hashtags": ["#shorts"]},
                    {"id": "c2", "slug": "blur", "start": 0.5, "end": 3.0, "layout": "blurred-fit", "title": "Blur Clip"},
                    {"id": "c3", "slug": "split", "start": 0.5, "end": 3.0, "layout": "split", "split_cam_corner": "br", "title": "Split Clip"},
                    {"id": "c4", "slug": "land", "start": 0.5, "end": 3.0, "layout": "landscape", "caption_style": "clean", "title": "Landscape Clip"},
                    {"id": "c5", "slug": "auto-clip", "start": 0.5, "end": 3.0, "layout": "auto", "title": "Auto Clip"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    report = run_render(plan_path, out_dir=out_dir, local_source=src)
    assert report["ok"] is True
    assert len(report["clips"]) == 5

    for idx in (0, 1, 2, 4):
        item = report["clips"][idx]
        mp4 = Path(item["output_path"])
        assert mp4.exists()
        info = probe_media(mp4)
        assert (info.width, info.height) == (1080, 1920)
        assert info.has_audio is True
        assert 2.0 <= info.duration <= 3.2

    assert report["clips"][4]["layout_used"] == "blurred-fit"
    land_info = probe_media(Path(report["clips"][3]["output_path"]))
    assert (land_info.width, land_info.height) == (1280, 720)
    assert (out_dir / "post_kit.md").exists()
    assert "Face Clip" in (out_dir / "post_kit.md").read_text(encoding="utf-8")
    assert (out_dir / "render_report.json").exists()


def test_render_silent_video_and_vtt_proportional_caption_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    src = _make_synthetic_video(tmp_path / "silent.mp4", duration=2.0, with_audio=False)
    vtt = tmp_path / "subs.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:00.200 --> 00:00:01.800\nSilent video with VTT captions\n",
        encoding="utf-8",
    )

    # Simulate faster-whisper failing or unavailable so VTT proportional fallback runs
    monkeypatch.setattr(
        render_mod,
        "transcribe_audio",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("whisper unavailable")),
    )

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=silent1",
                "video_id": "silent1",
                "video_title": "Silent Video",
                "cache_subtitles": str(vtt),
                "clips": [
                    {"id": "c1", "slug": "silent", "start": 0.2, "end": 1.8, "layout": "landscape", "title": "Silent"}
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_silent"
    report = run_render(plan_path, out_dir=out_dir, local_source=src)
    assert report["ok"] is True
    assert report["clips"][0]["caption_source"] == "vtt_proportional"
    out_info = probe_media(Path(report["clips"][0]["output_path"]))
    assert out_info.has_audio is False
    assert (out_info.width, out_info.height) == (1280, 720)


def test_render_auto_layout_keeps_face_track_at_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=face30",
                "video_id": "face30",
                "video_title": "Face Threshold",
                "clips": [
                    {
                        "id": "c1",
                        "slug": "auto-face",
                        "start": 10.0,
                        "end": 15.0,
                        "layout": "auto",
                        "captions": False,
                        "title": "Auto Face",
                        "why_selected": "Strong hook",
                        "caption_text": "Watch this clip!",
                        "hashtags": ["#clip", "#dev"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    downloaded: list[tuple[float, float]] = []
    trimmed_offsets: list[tuple[float, float]] = []
    face_render_called: list[bool] = []

    def fake_download(url: str, start: float, end: float, out_path: Path, cookies_from_browser=None) -> Path:
        downloaded.append((start, end))
        out_path.write_bytes(b"section")
        return out_path

    def fake_trim(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path:
        trimmed_offsets.append((start_offset, duration))
        out_path.write_bytes(b"trimmed")
        return out_path

    monkeypatch.setattr(render_mod, "download_video_section", fake_download)
    monkeypatch.setattr(render_mod, "trim_exact", fake_trim)
    monkeypatch.setattr(
        render_mod,
        "probe_media",
        lambda p: MediaInfo(1080, 1920, 30.0, 5.0, True)
        if p.name != "trimmed.mp4"
        else MediaInfo(1280, 720, 30.0, 5.0, True),
    )
    monkeypatch.setattr(
        render_mod,
        "analyze_video_tracking",
        lambda p, sample_fps=5.0: TrackingPlan(1280, 720, 404, 30.0, 150, 0.30, [400] * 150),
    )
    monkeypatch.setattr(
        render_mod,
        "render_face_tracked_clip",
        lambda trimmed, out_mp4, tracking, media, ass_filename, clip_temp: (
            face_render_called.append(True) or out_mp4.write_bytes(b"mp4") or out_mp4
        ),
    )

    monkeypatch.chdir(tmp_path)
    report = run_render(plan_path, out_dir=None, local_source=None)
    assert report["ok"] is True
    assert downloaded == [(8.0, 17.0)]
    assert trimmed_offsets == [(2.0, 5.0)]
    assert face_render_called == [True]
    assert report["clips"][0]["layout_used"] == "face-track"
    assert Path(report["output_dir"]) == (tmp_path / "clips" / "face30").resolve()
    kit_text = (tmp_path / "clips" / "face30" / "post_kit.md").read_text(encoding="utf-8")
    assert "Strong hook" in kit_text
    assert "Watch this clip!" in kit_text
    assert "#clip #dev" in kit_text


def test_render_clip_failure_and_cli_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=err01",
                "video_id": "err01",
                "video_title": "Mixed Results",
                "clips": [
                    {"id": "c1", "slug": "good", "start": 0.0, "end": 2.0, "layout": "landscape", "captions": False},
                    {"id": "c2", "slug": "bad", "start": 2.0, "end": 4.0, "layout": "landscape", "captions": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    local_src = tmp_path / "src.mp4"
    local_src.write_bytes(b"video")

    call_n = {"n": 0}

    def fake_trim(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path:
        call_n["n"] += 1
        if call_n["n"] == 2:
            raise ClipError("FFMPEG_FAILED", "simulated trim failure")
        out_path.write_bytes(b"trimmed")
        return out_path

    monkeypatch.setattr(render_mod, "trim_exact", fake_trim)
    monkeypatch.setattr(render_mod, "probe_media", lambda p: MediaInfo(1280, 720, 24.0, 2.0, True))
    monkeypatch.setattr(
        render_mod,
        "render_static_layout_clip",
        lambda trimmed, out_mp4, layout, media, ass, work_dir, split_cam_corner="br": out_mp4.write_bytes(b"mp4")
        or out_mp4,
    )

    out_dir = tmp_path / "out_err"
    rc = render_main(["--plan", str(plan_path), "--out", str(out_dir), "--local-source", str(local_src)])
    assert rc == 1
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["ok"] is False
    assert out_json["clips"][0]["ok"] is True
    assert out_json["clips"][1]["error_code"] == "FFMPEG_FAILED"
    post_kit = (out_dir / "post_kit.md").read_text(encoding="utf-8")
    assert "Failed (`FFMPEG_FAILED`" in post_kit

    # Invalid plan returns exit code 2
    bad_plan = tmp_path / "missing_plan.json"
    rc_bad = render_main(["--plan", str(bad_plan)])
    assert rc_bad == 2
    err_json = json.loads(capsys.readouterr().err)
    assert err_json["error_code"] == "INVALID_PLAN"


def test_render_whisper_caption_source_and_keep_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lib.models import CaptionSegment, WordTiming

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=wh01",
                "video_id": "wh01",
                "video_title": "Whisper Test",
                "clips": [
                    {"id": "c1", "slug": "wh", "start": 0.0, "end": 2.0, "layout": "landscape", "title": "Whisper Clip"}
                ],
            }
        ),
        encoding="utf-8",
    )
    local_src = tmp_path / "src.mp4"
    local_src.write_bytes(b"video")
    seen_temp_dirs: list[Path] = []

    def fake_trim(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path:
        seen_temp_dirs.append(out_path.parent)
        out_path.write_bytes(b"trimmed")
        return out_path

    monkeypatch.setattr(render_mod, "trim_exact", fake_trim)
    monkeypatch.setattr(render_mod, "probe_media", lambda p: MediaInfo(1280, 720, 24.0, 2.0, True))
    monkeypatch.setattr(
        render_mod,
        "transcribe_audio",
        lambda *args, **kwargs: (
            [CaptionSegment(0.1, 1.0, "Hello world")],
            [WordTiming("Hello", 0.1, 0.5), WordTiming("world", 0.5, 1.0)],
            "en",
        ),
    )
    monkeypatch.setattr(
        render_mod,
        "render_static_layout_clip",
        lambda trimmed, out_mp4, layout, media, ass, work_dir, split_cam_corner="br": out_mp4.write_bytes(b"mp4")
        or out_mp4,
    )

    out_dir = tmp_path / "out_keep"
    report = run_render(plan_path, out_dir=out_dir, local_source=local_src, keep_temp=True)
    assert report["ok"] is True
    assert report["clips"][0]["caption_source"] == "faster_whisper"
    assert len(seen_temp_dirs) == 1
    assert seen_temp_dirs[0].exists()
    assert (seen_temp_dirs[0] / "captions.ass").exists()
    import shutil
    shutil.rmtree(seen_temp_dirs[0], ignore_errors=True)


def test_render_isolates_per_clip_tracking_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=trk01",
                "video_id": "trk01",
                "video_title": "Tracking Failure Isolation",
                "clips": [
                    {"id": "c1", "slug": "bad-media", "start": 0.0, "end": 2.0, "layout": "face-track", "captions": False, "title": "Bad Media"},
                    {"id": "c2", "slug": "bad-cv2", "start": 2.0, "end": 4.0, "layout": "auto", "captions": False, "title": "Bad CV2"},
                    {"id": "c3", "slug": "good-clip", "start": 4.0, "end": 6.0, "layout": "landscape", "captions": False, "title": "Good Clip"},
                ],
            }
        ),
        encoding="utf-8",
    )
    local_src = tmp_path / "src.mp4"
    local_src.write_bytes(b"video")

    def fake_trim(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path:
        out_path.write_bytes(f"trimmed-{start_offset}".encode("utf-8"))
        return out_path

    track_calls = {"n": 0}

    def fake_track(video_path: Path, sample_fps: float = 5.0) -> TrackingPlan:
        track_calls["n"] += 1
        if track_calls["n"] == 1:
            raise ClipError("MEDIA_ERROR", "Could not open video for tracking")
        raise cv2.error("simulated OpenCV failure")

    monkeypatch.setattr(render_mod, "trim_exact", fake_trim)
    monkeypatch.setattr(render_mod, "analyze_video_tracking", fake_track)
    monkeypatch.setattr(render_mod, "probe_media", lambda p: MediaInfo(1280, 720, 24.0, 2.0, True))
    monkeypatch.setattr(
        render_mod,
        "render_static_layout_clip",
        lambda trimmed, out_mp4, layout, media, ass, work_dir, split_cam_corner="br": out_mp4.write_bytes(b"mp4")
        or out_mp4,
    )

    out_dir = tmp_path / "out_trk"
    report = run_render(plan_path, out_dir=out_dir, local_source=local_src)
    assert report["ok"] is False
    assert len(report["clips"]) == 3
    assert report["clips"][0]["ok"] is False
    assert report["clips"][0]["error_code"] == "MEDIA_ERROR"
    assert report["clips"][1]["ok"] is False
    assert report["clips"][1]["error_code"] == "MEDIA_ERROR"
    assert report["clips"][2]["ok"] is True
    assert Path(report["clips"][2]["output_path"]).exists()

    post_kit = out_dir / "post_kit.md"
    report_json = out_dir / "render_report.json"
    assert post_kit.exists()
    assert report_json.exists()
    kit_text = post_kit.read_text(encoding="utf-8")
    assert "Clips Generated:** 1/3" in kit_text
    assert "Failed (`MEDIA_ERROR`" in kit_text
    assert "Good Clip" in kit_text


def test_render_uses_cache_live_manifest_for_live_clips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = _make_synthetic_video(tmp_path / "live_capture.mp4", duration=4.0)
    manifest = tmp_path / "live_manifest.json"
    manifest.write_text(json.dumps({"mode": "local_capture", "capture_path": str(src)}), encoding="utf-8")

    plan_path = tmp_path / "live_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/live/live99",
                "video_id": "live99",
                "video_title": "Live Stream",
                "cache_live_manifest": str(manifest),
                "clips": [
                    {"id": "clip_01", "slug": "live-best", "start": 0.5, "end": 2.5, "layout": "blurred-fit", "captions": False, "title": "Live Best"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_live"
    report = run_render(plan_path, out_dir=out_dir)
    assert report["ok"] is True
    out_mp4 = Path(report["clips"][0]["output_path"])
    assert out_mp4.exists()
    info = probe_media(out_mp4)
    assert (info.width, info.height) == (1080, 1920)


