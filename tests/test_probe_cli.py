from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import ClipError  # noqa: E402
from lib import youtube as yt_mod  # noqa: E402
import probe as probe_mod  # noqa: E402


def test_probe_writes_analysis_json_with_heatmap_and_json3(monkeypatch, tmp_path: Path) -> None:
    fake_info = {
        "id": "vid123",
        "title": "Demo Podcast",
        "channel": "Demo Channel",
        "duration": 600.0,
        "upload_date": "20261118",
        "is_live": False,
        "was_live": True,
        "chapters": [{"title": "Intro", "start_time": 0.0, "end_time": 60.0}],
        "heatmap": [{"start_time": 100.0, "end_time": 115.0, "value": 0.92}],
    }
    sub_file = tmp_path / "cache" / "subtitles.json3"
    sub_file.parent.mkdir(parents=True, exist_ok=True)
    sub_file.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 102000,
                        "dDurationMs": 3000,
                        "segs": [{"utf8": "Biggest"}, {"utf8": " takeaway", "tOffsetMs": 800}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(probe_mod, "fetch_video_info", lambda url, cookies_from_browser=None: fake_info)
    monkeypatch.setattr(
        probe_mod,
        "download_subtitles",
        lambda url, out_dir, lang="en", cookies_from_browser=None: (sub_file, "youtube_json3", "en"),
    )

    summary = probe_mod.run_probe("https://www.youtube.com/watch?v=vid123", out_dir=tmp_path)
    assert summary["ok"] is True
    assert summary["caption_source"] == "youtube_json3"
    assert summary["has_heatmap"] is True

    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["video"]["id"] == "vid123"
    assert analysis["signals"]["heatmap_peaks"][0]["score"] == 0.92
    assert analysis["transcript_blocks"][0]["text"] == "Biggest takeaway"
    assert analysis["transcript_blocks"][0]["heatmap_score"] == 0.92


def test_probe_rejects_active_livestream(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        probe_mod,
        "fetch_video_info",
        lambda url, cookies_from_browser=None: {"id": "live1", "title": "Live Now", "is_live": True, "duration": 0},
    )
    with pytest.raises(ClipError) as exc:
        probe_mod.run_probe("https://www.youtube.com/watch?v=live1", out_dir=tmp_path)
    assert exc.value.code == "LIVE_STREAM_ACTIVE"


def test_probe_falls_back_to_audio_and_whisper_when_no_subtitles(monkeypatch, tmp_path: Path) -> None:
    fake_info = {
        "id": "vid_nosub",
        "title": "No Subtitles",
        "uploader": "Fallback Channel",
        "duration": 300.0,
        "upload_date": "20261118",
        "is_live": False,
        "was_live": False,
        "chapters": None,
        "heatmap": None,
    }
    audio_file = tmp_path / "cache" / "source_audio.m4a"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"fake-audio")

    monkeypatch.setattr(probe_mod, "fetch_video_info", lambda url, cookies_from_browser=None: fake_info)
    monkeypatch.setattr(
        probe_mod,
        "download_subtitles",
        lambda url, out_dir, lang="en", cookies_from_browser=None: (None, "none", lang),
    )
    monkeypatch.setattr(
        probe_mod,
        "download_audio_only",
        lambda url, out_dir, cookies_from_browser=None: audio_file,
    )
    monkeypatch.setattr(
        probe_mod,
        "detect_audio_peaks",
        lambda ap, duration_sec, max_peaks=15: [
            probe_mod.HeatmapPeak(start=40.0, end=70.0, peak_time=55.0, score=0.88)
        ],
    )
    monkeypatch.setattr(
        probe_mod,
        "transcribe_audio",
        lambda ap, model_size="base", clip_offset=0.0: (
            [probe_mod.CaptionSegment(start=50.0, end=56.0, text="Audio peak moment")],
            [],
            "en",
        ),
    )

    summary = probe_mod.run_probe("https://www.youtube.com/watch?v=vid_nosub", out_dir=tmp_path)
    assert summary["ok"] is True
    assert summary["caption_source"] == "faster_whisper"
    assert summary["has_heatmap"] is False

    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["video"]["channel"] == "Fallback Channel"
    assert analysis["signals"]["audio_peaks"][0]["score"] == 0.88
    assert analysis["transcript_blocks"][0]["text"] == "Audio peak moment"
    assert analysis["transcript_blocks"][0]["heatmap_score"] == 0.88
    assert analysis["cache_paths"]["audio"] == str(audio_file.resolve())


def test_probe_long_vod_windowed_whisper_transcription(monkeypatch, tmp_path: Path) -> None:
    audio_file = tmp_path / "source_audio.m4a"
    audio_file.write_bytes(b"fake")
    peaks = [probe_mod.HeatmapPeak(start=100.0, end=130.0, peak_time=115.0, score=0.9)]
    ffmpeg_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        ffmpeg_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(probe_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        probe_mod,
        "transcribe_audio",
        lambda ap, model_size="base", clip_offset=0.0: (
            [probe_mod.CaptionSegment(start=clip_offset + 1.0, end=clip_offset + 4.0, text="Windowed")],
            [],
            "en",
        ),
    )

    segs, lang = probe_mod._transcribe_fallback(audio_file, duration_sec=2400.0, peaks=peaks, max_windows=5, work_dir=tmp_path)
    assert lang == "en"
    assert len(ffmpeg_calls) == 1
    assert segs[0].start == 71.0
    assert segs[0].text == "Windowed"


def test_probe_main_cli_success_and_error(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        probe_mod,
        "run_probe",
        lambda **kwargs: {"ok": True, "video_id": "abc"},
    )
    rc = probe_mod.main(["https://youtu.be/abc", "--out", str(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True

    def raise_err(**kwargs):
        raise ClipError("BOT_CHECK", "bot check", hint="use cookies")

    monkeypatch.setattr(probe_mod, "run_probe", raise_err)
    rc_err = probe_mod.main(["https://youtu.be/abc"])
    assert rc_err == 2
    err_out = json.loads(capsys.readouterr().err)
    assert err_out["error_code"] == "BOT_CHECK"


def test_youtube_wrapper_uses_utf8_and_classifies_errors(monkeypatch, tmp_path: Path) -> None:
    seen_kwargs: list[dict] = []

    def fake_run(cmd, **kwargs):
        seen_kwargs.append(kwargs)
        if "--dump-single-json" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: Sign in to confirm you're not a bot")
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(yt_mod.subprocess, "run", fake_run)
    with pytest.raises(ClipError) as exc:
        yt_mod.fetch_video_info("https://youtu.be/x", cookies_from_browser="chrome")
    assert exc.value.code == "BOT_CHECK"
    assert seen_kwargs[0]["encoding"] == "utf-8"
    assert seen_kwargs[0]["errors"] == "replace"


def test_youtube_download_subtitles_audio_and_section(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, **kwargs):
        assert kwargs.get("encoding") == "utf-8"
        if "--write-subs" in cmd and "json3" in cmd:
            (tmp_path / "subtitles.en.json3").write_text("{}", encoding="utf-8")
        elif "bestaudio[ext=m4a]/bestaudio/best" in cmd:
            (tmp_path / "source_audio.m4a").write_bytes(b"m4a")
        elif "--download-sections" in cmd:
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(yt_mod.subprocess, "run", fake_run)

    sub_path, kind, lang = yt_mod.download_subtitles("https://youtu.be/x", tmp_path, lang="en")
    assert sub_path == tmp_path / "subtitles.en.json3"
    assert kind == "youtube_json3"
    assert lang == "en"

    audio_path = yt_mod.download_audio_only("https://youtu.be/x", tmp_path)
    assert audio_path == tmp_path / "source_audio.m4a"

    clip_path = yt_mod.download_video_section("https://youtu.be/x", 10.0, 25.0, tmp_path / "clip.mp4")
    assert clip_path.exists()


def test_classify_ytdlp_error_avoids_bot_substring_false_positives() -> None:
    for msg in (
        "ERROR: Video unavailable. More information about this video",
        "ERROR: Video about robotics is unavailable",
        "ERROR: Both formats failed at the bottom of the page",
    ):
        err = yt_mod._classify_ytdlp_error(msg, "https://youtu.be/x")
        assert err.code == "UNAVAILABLE"

    bot_err = yt_mod._classify_ytdlp_error(
        "ERROR: [youtube] x: Sign in to confirm you're not a bot",
        "https://youtu.be/x",
    )
    assert bot_err.code == "BOT_CHECK"

    word_bot_err = yt_mod._classify_ytdlp_error(
        "ERROR: Blocked by bot detection",
        "https://youtu.be/x",
    )
    assert word_bot_err.code == "BOT_CHECK"
