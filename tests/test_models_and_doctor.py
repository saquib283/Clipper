from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import ClipError, ClipSpec, ClipsPlan  # noqa: E402
from doctor import run_doctor  # noqa: E402


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


def test_doctor_reports_all_required_checks() -> None:
    report = run_doctor()
    assert report["ok"] is True
    for key in ("ffmpeg", "ffprobe", "yt_dlp", "cv2", "faster_whisper"):
        assert report["checks"][key]["ok"] is True
