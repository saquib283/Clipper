from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.audio_peaks import (  # noqa: E402
    _load_whisper_model,
    detect_audio_peaks,
    parse_ffmpeg_rms_levels,
    peaks_from_rms_windows,
    transcribe_audio,
)


def test_parse_ffmpeg_rms_levels_and_pick_top_peaks() -> None:
    sample_stderr = "\n".join(
        [
            "[Parsed_ametadata_1 @ 000] frame:0 pts:0 pts_time:0.0",
            "[Parsed_ametadata_1 @ 000] lavfi.astats.Overall.RMS_level=-35.0",
            "[Parsed_ametadata_1 @ 000] frame:1 pts:1 pts_time:5.0",
            "[Parsed_ametadata_1 @ 000] lavfi.astats.Overall.RMS_level=-32.0",
            "[Parsed_ametadata_1 @ 000] frame:2 pts:2 pts_time:50.0",
            "[Parsed_ametadata_1 @ 000] lavfi.astats.Overall.RMS_level=-10.0",
            "[Parsed_ametadata_1 @ 000] frame:3 pts:3 pts_time:120.0",
            "[Parsed_ametadata_1 @ 000] lavfi.astats.Overall.RMS_level=-15.0",
        ]
    )
    levels = parse_ffmpeg_rms_levels(sample_stderr, window_sec=5.0)
    assert len(levels) == 4
    peaks = peaks_from_rms_windows(levels, duration_sec=200.0, max_peaks=2)
    assert len(peaks) >= 1
    assert peaks[0].peak_time == 52.5
    assert peaks[0].score == 1.0


def test_parse_ffmpeg_rms_handles_inf_merging_and_empty() -> None:
    assert parse_ffmpeg_rms_levels("") == []
    assert peaks_from_rms_windows([], duration_sec=100.0) == []

    stderr = "\n".join(
        [
            "pts_time:0.0",
            "lavfi.astats.Overall.RMS_level=-inf",
            "pts_time:10.0",
            "lavfi.astats.Overall.RMS_level=-20.0",
            "pts_time:20.0",
            "lavfi.astats.Overall.RMS_level=-10.0",
        ]
    )
    levels = parse_ffmpeg_rms_levels(stderr, window_sec=5.0)
    assert levels[0] == (0.0, -80.0)
    peaks = peaks_from_rms_windows(levels, duration_sec=30.0, window_sec=5.0, max_peaks=5)
    assert len(peaks) == 1
    assert peaks[0].start == 0.0
    assert peaks[0].end == 30.0
    assert peaks[0].peak_time == 22.5
    assert peaks[0].score == 1.0


def test_detect_audio_peaks_invokes_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    import lib.audio_peaks as ap

    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output: bool, text: bool):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        return SimpleNamespace(
            stderr="\n".join(
                [
                    "pts_time:0.0",
                    "lavfi.astats.Overall.RMS_level=-40.0",
                    "pts_time:30.0",
                    "lavfi.astats.Overall.RMS_level=-10.0",
                ]
            )
        )

    monkeypatch.setattr(ap.subprocess, "run", fake_run)
    audio = tmp_path / "audio.wav"
    peaks = detect_audio_peaks(audio, duration_sec=60.0, window_sec=5.0, max_peaks=3)
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert str(audio) in captured["cmd"]
    assert len(peaks) == 1
    assert peaks[0].peak_time == 32.5


def test_transcribe_audio_uses_faster_whisper_and_offsets_times(monkeypatch, tmp_path: Path) -> None:
    fake_word = SimpleNamespace(word=" hello", start=0.2, end=0.7)
    fake_seg = SimpleNamespace(start=0.2, end=0.7, text=" hello", words=[fake_word])
    fake_info = SimpleNamespace(language="en")

    class FakeModel:
        def __init__(self, model_size: str, device: str, compute_type: str) -> None:
            assert device == "cpu"
            assert compute_type == "int8"

        def transcribe(self, path: str, language=None, word_timestamps: bool = True, vad_filter: bool = True):
            return [fake_seg], fake_info

    import lib.audio_peaks as ap

    monkeypatch.setattr(ap, "_load_whisper_model", lambda size: FakeModel(size, "cpu", "int8"))
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")

    segments, words, lang = transcribe_audio(audio, model_size="base", clip_offset=10.0)
    assert lang == "en"
    assert segments[0].start == 10.2
    assert words[0].word == "hello"
    assert words[0].end == 10.7


def test_load_whisper_model_forces_cpu_and_int8(monkeypatch) -> None:
    init_calls: list[tuple[str, str, str]] = []

    class StubWhisperModel:
        def __init__(self, model_size: str, device: str, compute_type: str) -> None:
            init_calls.append((model_size, device, compute_type))

    fake_pkg = ModuleType("faster_whisper")
    fake_pkg.WhisperModel = StubWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_pkg)
    _load_whisper_model.cache_clear()
    try:
        model = _load_whisper_model("tiny")
        assert isinstance(model, StubWhisperModel)
        assert init_calls == [("tiny", "cpu", "int8")]
    finally:
        _load_whisper_model.cache_clear()
