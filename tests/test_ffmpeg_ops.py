from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib import ffmpeg_ops  # noqa: E402
from lib.ffmpeg_ops import (  # noqa: E402
    AUDIO_LOUDNORM,
    build_static_filtergraph,
    probe_media,
    render_face_tracked_clip,
    render_static_layout_clip,
    trim_exact,
)
from lib.models import ClipError, MediaInfo, TrackingPlan  # noqa: E402


def test_blurred_fit_filtergraph_contains_blur_overlay_and_relative_ass() -> None:
    fg = build_static_filtergraph("blurred-fit", 1920, 1080, has_captions=True)
    assert "boxblur" in fg
    assert "overlay=(W-w)/2:(H-h)/2" in fg
    assert "ass=captions.ass" in fg
    assert "C:" not in fg


def test_split_filtergraph_handles_all_four_corners() -> None:
    for corner in ("br", "bl", "tr", "tl"):
        fg = build_static_filtergraph("split", 1920, 1080, has_captions=False, split_cam_corner=corner)
        assert "vstack=inputs=2" in fg
        assert "1080:960" in fg
        assert "ass=" not in fg


def test_landscape_filtergraph_preserves_aspect_and_adds_captions() -> None:
    fg = build_static_filtergraph("landscape", 1280, 720, has_captions=True)
    assert "ass=captions.ass" in fg


def test_build_static_filtergraph_rejects_invalid_inputs_and_absolute_ass() -> None:
    with pytest.raises(ClipError) as exc_layout:
        build_static_filtergraph("unknown", 1280, 720, has_captions=False)
    assert exc_layout.value.code == "INVALID_LAYOUT"

    with pytest.raises(ClipError) as exc_corner:
        build_static_filtergraph("split", 1280, 720, has_captions=False, split_cam_corner="center")
    assert exc_corner.value.code == "INVALID_LAYOUT"

    with pytest.raises(ClipError) as exc_ass:
        build_static_filtergraph("landscape", 1280, 720, has_captions=True, ass_filename=r"C:\tmp\captions.ass")
    assert exc_ass.value.code == "INVALID_PATH"


def test_probe_media_parses_metadata_and_handles_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen_kwargs: dict = {}

    def fake_run_ok(cmd, **kwargs):
        seen_kwargs.update(kwargs)
        payload = {
            "streams": [
                {"codec_type": "video", "width": 1280, "height": 720, "r_frame_rate": "30000/1001"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "12.345"},
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(ffmpeg_ops.subprocess, "run", fake_run_ok)
    info = probe_media(tmp_path / "demo.mp4")
    assert info == MediaInfo(width=1280, height=720, fps=29.97, duration=12.345, has_audio=True)
    assert seen_kwargs["encoding"] == "utf-8"
    assert seen_kwargs["errors"] == "replace"

    def fake_run_no_video(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"streams": [{"codec_type": "audio"}]}), stderr="")

    monkeypatch.setattr(ffmpeg_ops.subprocess, "run", fake_run_no_video)
    with pytest.raises(ClipError) as exc_no_vid:
        probe_media(tmp_path / "audio_only.m4a")
    assert exc_no_vid.value.code == "MEDIA_ERROR"

    def fake_run_fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No such file")

    monkeypatch.setattr(ffmpeg_ops.subprocess, "run", fake_run_fail)
    with pytest.raises(ClipError) as exc_fail:
        probe_media(tmp_path / "missing.mp4")
    assert exc_fail.value.code == "MEDIA_ERROR"


def test_trim_exact_uses_utf8_and_raises_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="trim failed")

    monkeypatch.setattr(ffmpeg_ops.subprocess, "run", fake_run)
    out = trim_exact(tmp_path / "src.mp4", tmp_path / "trimmed.mp4", start_offset=-1.0, duration=5.5)
    assert out == tmp_path / "trimmed.mp4"
    cmd, kw = calls[0]
    assert kw["encoding"] == "utf-8"
    assert kw["errors"] == "replace"
    assert cmd[cmd.index("-ss") + 1] == "0.000"
    assert cmd[cmd.index("-t") + 1] == "5.500"

    with pytest.raises(ClipError) as exc:
        trim_exact(tmp_path / "src.mp4", tmp_path / "trimmed.mp4", start_offset=1.0, duration=2.0)
    assert exc.value.code == "FFMPEG_FAILED"


def test_render_static_layout_clip_sets_cwd_loudnorm_and_an(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg_ops.subprocess, "run", fake_run)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    trimmed = tmp_path / "trimmed.mp4"
    out = tmp_path / "nested" / "clip.mp4"

    render_static_layout_clip(
        trimmed,
        out,
        "blurred-fit",
        MediaInfo(1280, 720, 24.0, 3.0, has_audio=True),
        "captions.ass",
        work_dir,
    )
    cmd_audio, kw_audio = calls[0]
    assert kw_audio["cwd"] == str(work_dir)
    assert kw_audio["encoding"] == "utf-8"
    assert kw_audio["errors"] == "replace"
    assert AUDIO_LOUDNORM in cmd_audio
    assert "ass=captions.ass" in cmd_audio[cmd_audio.index("-filter_complex") + 1]
    assert "C:" not in cmd_audio[cmd_audio.index("-filter_complex") + 1]

    render_static_layout_clip(
        trimmed,
        out,
        "landscape",
        MediaInfo(1280, 720, 24.0, 3.0, has_audio=False),
        None,
        work_dir,
    )
    cmd_mute, _ = calls[1]
    assert "-an" in cmd_mute
    assert AUDIO_LOUDNORM not in cmd_mute


def test_render_face_tracked_clip_pipes_frames_and_cleans_up_on_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    trimmed = tmp_path / "trimmed.mp4"
    out = tmp_path / "out.mp4"
    tracking = TrackingPlan(
        src_width=640,
        src_height=360,
        crop_width=202,
        fps=24.0,
        total_frames=2,
        face_ratio=1.0,
        x0_per_frame=[100, 110],
    )
    media = MediaInfo(640, 360, 24.0, 1.0, has_audio=True)

    # Case 1: OpenCV fails to open -> kills Popen and raises MEDIA_ERROR
    class DummyPipe:
        def __init__(self) -> None:
            self.written = bytearray()
            self.closed = False

        def write(self, data: bytes) -> int:
            self.written.extend(data)
            return len(data)

        def close(self) -> None:
            self.closed = True

    class DummyStderr:
        def __init__(self, msg: bytes = b"") -> None:
            self.msg = msg
            self.closed = False

        def read(self) -> bytes:
            return self.msg

        def close(self) -> None:
            self.closed = True

    class DummyProc:
        def __init__(self, returncode: int = 0, stderr_msg: bytes = b"") -> None:
            self.stdin = DummyPipe()
            self.stderr = DummyStderr(stderr_msg)
            self.returncode = returncode
            self.killed = False
            self.waited = False

        def kill(self) -> None:
            self.killed = True

        def wait(self) -> int:
            self.waited = True
            return self.returncode

    spawned: list[tuple[list[str], dict, DummyProc]] = []

    def make_popen(returncode: int = 0, stderr_msg: bytes = b""):
        def _popen(cmd, **kwargs):
            p = DummyProc(returncode=returncode, stderr_msg=stderr_msg)
            spawned.append((cmd, kwargs, p))
            return p

        return _popen

    class ClosedCap:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr(ffmpeg_ops.subprocess, "Popen", make_popen())
    monkeypatch.setattr(ffmpeg_ops.cv2, "VideoCapture", lambda _: ClosedCap())

    with pytest.raises(ClipError) as exc_open:
        render_face_tracked_clip(trimmed, out, tracking, media, "captions.ass", work_dir)
    assert exc_open.value.code == "MEDIA_ERROR"
    assert spawned[-1][2].killed is True
    assert spawned[-1][2].waited is True

    # Case 2: Successful frame piping sets cwd and relative ass=captions.ass
    frames = [
        np.full((360, 640, 3), 10, dtype=np.uint8),
        np.full((360, 640, 3), 20, dtype=np.uint8),
    ]

    class GoodCap:
        def __init__(self) -> None:
            self.idx = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self.idx >= len(frames):
                return False, None
            f = frames[self.idx]
            self.idx += 1
            return True, f

        def release(self) -> None:
            pass

    monkeypatch.setattr(ffmpeg_ops.subprocess, "Popen", make_popen(returncode=0))
    monkeypatch.setattr(ffmpeg_ops.cv2, "VideoCapture", lambda _: GoodCap())

    res = render_face_tracked_clip(trimmed, out, tracking, media, "captions.ass", work_dir)
    assert res == out
    cmd, kw, proc = spawned[-1]
    assert kw["cwd"] == str(work_dir)
    assert "ass=captions.ass" in cmd[cmd.index("-vf") + 1]
    assert "C:" not in cmd[cmd.index("-vf") + 1]
    assert AUDIO_LOUDNORM in cmd
    assert len(proc.stdin.written) == 2 * 360 * 202 * 3
    assert proc.stdin.closed is True

    # Case 3: FFmpeg fails with non-zero exit code -> raises FFMPEG_FAILED
    monkeypatch.setattr(ffmpeg_ops.subprocess, "Popen", make_popen(returncode=1, stderr_msg=b"encoder error"))
    with pytest.raises(ClipError) as exc_ff:
        render_face_tracked_clip(trimmed, out, tracking, media, None, work_dir)
    assert exc_ff.value.code == "FFMPEG_FAILED"
    assert "encoder error" in exc_ff.value.message
