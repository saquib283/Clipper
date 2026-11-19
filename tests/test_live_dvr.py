from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.live_dvr import (  # noqa: E402
    build_dvr_window_manifest,
    materialize_live_clip,
    parse_m3u8_segments,
    render_m3u8_text,
    select_hls_formats,
    slice_manifest_segments,
    words_to_json3,
)
from lib.models import CaptionSegment, ClipError, WordTiming  # noqa: E402
from lib.subtitles import parse_json3  # noqa: E402


def test_select_hls_formats_picks_1080p_avc_and_best_audio() -> None:
    info = {
        "formats": [
            {"format_id": "233", "protocol": "m3u8_native", "vcodec": "none", "acodec": None, "tbr": 48, "url": "https://a/233.m3u8"},
            {"format_id": "234", "protocol": "m3u8_native", "vcodec": "none", "acodec": None, "tbr": 128, "url": "https://a/234.m3u8"},
            {"format_id": "311", "protocol": "m3u8_native", "vcodec": "avc1.4D4020", "height": 720, "url": "https://v/720.m3u8"},
            {"format_id": "312", "protocol": "m3u8_native", "vcodec": "avc1.4D402A", "height": 1080, "url": "https://v/1080.m3u8"},
            {"format_id": "999", "protocol": "m3u8_native", "vcodec": "avc1.640033", "height": 2160, "url": "https://v/2160.m3u8"},
        ]
    }
    v_fmt, a_fmt = select_hls_formats(info)
    assert v_fmt is not None and v_fmt["format_id"] == "312"
    assert a_fmt is not None and a_fmt["format_id"] == "234"


def test_parse_m3u8_and_build_dvr_manifest_aligns_sequences_and_slices_tail() -> None:
    v_m3u8 = "\n".join(
        [
            "#EXTM3U",
            "#EXT-X-MEDIA-SEQUENCE:100",
            "#EXTINF:2.0,",
            "https://cdn/v_100.ts",
            "#EXTINF:2.0,",
            "https://cdn/v_101.ts",
            "#EXTINF:2.0,",
            "https://cdn/v_102.ts",
            "#EXTINF:2.0,",
            "https://cdn/v_103.ts",
        ]
    )
    a_m3u8 = "\n".join(
        [
            "#EXTM3U",
            "#EXT-X-MEDIA-SEQUENCE:101",
            "#EXTINF:2.0,",
            "https://cdn/a_101.ts",
            "#EXTINF:2.0,",
            "https://cdn/a_102.ts",
            "#EXTINF:2.0,",
            "https://cdn/a_103.ts",
            "#EXTINF:2.0,",
            "https://cdn/a_104.ts",
        ]
    )
    v_segs = parse_m3u8_segments(v_m3u8)
    a_segs = parse_m3u8_segments(a_m3u8)
    assert [s["seq"] for s in v_segs] == [100, 101, 102, 103]

    # Common seqs are 101, 102, 103 (6s total); request 4s tail -> 102, 103
    manifest = build_dvr_window_manifest(v_segs, a_segs, window_sec=4.0)
    assert manifest["mode"] == "dvr"
    assert manifest["duration_sec"] == 4.0
    assert [s["seq"] for s in manifest["segments"]] == [102, 103]
    assert manifest["segments"][0]["start"] == 0.0
    assert manifest["segments"][1]["end"] == 4.0
    assert manifest["segments"][0]["video_url"] == "https://cdn/v_102.ts"
    assert manifest["segments"][0]["audio_url"] == "https://cdn/a_102.ts"


def test_slice_manifest_segments_and_render_m3u8_endlist() -> None:
    manifest = {
        "mode": "dvr",
        "duration_sec": 10.0,
        "target_duration": 2,
        "segments": [
            {"seq": i, "start": float(i * 2), "end": float((i + 1) * 2), "duration": 2.0, "video_url": f"https://v/{i}.ts", "audio_url": f"https://a/{i}.ts"}
            for i in range(5)
        ],
    }
    sliced, slice_start = slice_manifest_segments(manifest, start=4.5, end=7.5, pad_sec=2.0)
    assert [s["seq"] for s in sliced] == [1, 2, 3, 4]
    assert slice_start == 2.0

    playlist = render_m3u8_text([(s["video_url"], s["duration"]) for s in sliced], target_duration=2)
    assert playlist.startswith("#EXTM3U\n")
    assert "#EXT-X-ENDLIST" in playlist
    assert "https://v/1.ts" in playlist


def test_materialize_live_clip_supports_local_capture_and_dvr_mux(monkeypatch, tmp_path: Path) -> None:
    cap_mp4 = tmp_path / "live_capture.mp4"
    cap_mp4.write_bytes(b"mp4")
    local_manifest = tmp_path / "local_manifest.json"
    local_manifest.write_text(json.dumps({"mode": "local_capture", "capture_path": str(cap_mp4)}), encoding="utf-8")

    src, offset = materialize_live_clip(local_manifest, start=12.5, end=42.5, clip_temp=tmp_path)
    assert src == cap_mp4.resolve()
    assert offset == 12.5

    dvr_manifest = tmp_path / "dvr_manifest.json"
    dvr_manifest.write_text(
        json.dumps(
            {
                "mode": "dvr",
                "duration_sec": 10.0,
                "target_duration": 2,
                "segments": [
                    {"seq": i, "start": float(i * 2), "end": float((i + 1) * 2), "duration": 2.0, "video_url": f"https://v/{i}.ts", "audio_url": f"https://a/{i}.ts"}
                    for i in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )

    recorded_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"muxed")
        return type("Proc", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    import lib.live_dvr as ld

    monkeypatch.setattr(ld.subprocess, "run", fake_run)
    clip_tmp = tmp_path / "clip1"
    clip_tmp.mkdir()
    out_src, trim_off = materialize_live_clip(dvr_manifest, start=4.5, end=7.5, clip_temp=clip_tmp)
    assert out_src.exists()
    assert trim_off == 2.5
    assert "-protocol_whitelist" in recorded_cmds[0]
    assert "file,http,https,tcp,tls,crypto" in recorded_cmds[0]
    assert (clip_tmp / "clip_video.m3u8").exists()
    assert (clip_tmp / "clip_audio.m3u8").exists()


def test_words_to_json3_roundtrips_through_parse_json3() -> None:
    segs = [CaptionSegment(start=1.0, end=2.5, text="No way bro")]
    words = [
        WordTiming("No", 1.0, 1.4),
        WordTiming("way", 1.4, 1.9),
        WordTiming("bro", 1.9, 2.5),
    ]
    raw = words_to_json3(segs, words)
    parsed_segs, parsed_words = parse_json3(raw)
    assert len(parsed_segs) == 1
    assert [w.word for w in parsed_words] == ["No", "way", "bro"]
    assert parsed_words[1].start == 1.4


def test_materialize_live_clip_raises_on_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ClipError) as exc:
        materialize_live_clip(tmp_path / "missing.json", start=0.0, end=10.0, clip_temp=tmp_path)
    assert exc.value.code == "MEDIA_ERROR"
