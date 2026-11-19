from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import ClipError, ClipSpec, ClipsPlan  # noqa: E402
from doctor import run_doctor  # noqa: E402
import run as run_mod  # noqa: E402


def test_clips_plan_parses_and_validates(tmp_path: Path) -> None:
    plan_file = tmp_path / "clips_plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=abc123",
                "video_id": "abc123",
                "video_title": "Test Stream",
                "clips": [
                    {
                        "id": "clip_01",
                        "slug": "great-hook",
                        "start": 10.5,
                        "end": 45.0,
                        "layout": "face-track",
                        "title": "Great Hook",
                        "hashtags": ["#shorts"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = ClipsPlan.from_json_file(plan_file)
    assert plan.video_id == "abc123"
    assert len(plan.clips) == 1
    assert plan.clips[0].layout == "face-track"
    assert plan.clips[0].caption_style == "bold-highlight"


def test_clips_plan_rejects_invalid_layout_and_times(tmp_path: Path) -> None:
    bad_layout = tmp_path / "bad_layout.json"
    bad_layout.write_text(
        json.dumps(
            {
                "video_url": "https://youtu.be/abc123",
                "video_id": "abc123",
                "video_title": "Test",
                "clips": [{"id": "c1", "slug": "a", "start": 5, "end": 15, "layout": "diagonal"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ClipError) as exc:
        ClipsPlan.from_json_file(bad_layout)
    assert exc.value.code == "INVALID_PLAN"

    bad_time = tmp_path / "bad_time.json"
    bad_time.write_text(
        json.dumps(
            {
                "video_url": "https://youtu.be/abc123",
                "video_id": "abc123",
                "video_title": "Test",
                "clips": [{"id": "c1", "slug": "a", "start": 20, "end": 10}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ClipError) as exc2:
        ClipsPlan.from_json_file(bad_time)
    assert exc2.value.code == "INVALID_PLAN"


def test_clips_plan_validates_malformed_root_items_hashtags_and_times(tmp_path: Path) -> None:
    non_dict_root = tmp_path / "root_list.json"
    non_dict_root.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ClipError) as exc_root:
        ClipsPlan.from_json_file(non_dict_root)
    assert exc_root.value.code == "INVALID_PLAN"

    non_dict_clip = tmp_path / "non_dict_clip.json"
    non_dict_clip.write_text(json.dumps({"clips": ["not-an-object"]}), encoding="utf-8")
    with pytest.raises(ClipError) as exc_item:
        ClipsPlan.from_json_file(non_dict_clip)
    assert exc_item.value.code == "INVALID_PLAN"

    null_hashtags = tmp_path / "null_hashtags.json"
    null_hashtags.write_text(
        json.dumps({"clips": [{"id": "c1", "start": 0, "end": 10, "hashtags": None}]}),
        encoding="utf-8",
    )
    parsed_null = ClipsPlan.from_json_file(null_hashtags)
    assert parsed_null.clips[0].hashtags == []

    bad_hashtags = tmp_path / "bad_hashtags.json"
    bad_hashtags.write_text(
        json.dumps({"clips": [{"id": "c1", "start": 0, "end": 10, "hashtags": "#shorts"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ClipError) as exc_hash:
        ClipsPlan.from_json_file(bad_hashtags)
    assert exc_hash.value.code == "INVALID_PLAN"

    missing_end = tmp_path / "missing_end.json"
    missing_end.write_text(
        json.dumps({"clips": [{"id": "c1", "start": 0}]}),
        encoding="utf-8",
    )
    with pytest.raises(ClipError) as exc_end:
        ClipsPlan.from_json_file(missing_end)
    assert exc_end.value.code == "INVALID_PLAN"

    non_numeric_start = tmp_path / "bad_start.json"
    non_numeric_start.write_text(
        json.dumps({"clips": [{"id": "c1", "start": "ten", "end": 20}]}),
        encoding="utf-8",
    )
    with pytest.raises(ClipError) as exc_start:
        ClipsPlan.from_json_file(non_numeric_start)
    assert exc_start.value.code == "INVALID_PLAN"


def test_resolve_venv_python_handles_empty_whitespace_and_valid_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(run_mod, "SCRIPTS_DIR", tmp_path)
    marker = tmp_path / ".venv_python"

    assert run_mod.resolve_venv_python() is None

    marker.write_text("", encoding="utf-8")
    assert run_mod.resolve_venv_python() is None

    marker.write_text("   \n\t  ", encoding="utf-8")
    assert run_mod.resolve_venv_python() is None

    marker.write_text(str(tmp_path), encoding="utf-8")
    assert run_mod.resolve_venv_python() is None

    fake_python = tmp_path / "python.exe"
    fake_python.write_text("", encoding="utf-8")
    marker.write_text(f"  {fake_python} \n", encoding="utf-8")
    assert run_mod.resolve_venv_python() == fake_python


def test_doctor_reports_all_required_checks() -> None:
    report = run_doctor()
    assert report["ok"] is True
    for key in ("ffmpeg", "ffprobe", "yt_dlp", "cv2", "faster_whisper"):
        assert report["checks"][key]["ok"] is True
