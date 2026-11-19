# YouTube Clipper Antigravity Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and globally install the `youtube-clipper` Antigravity skill backed by a modular, zero-API-key, CPU-friendly Python media pipeline for clipping finished YouTube videos, podcasts, and livestream VODs into ready-to-post short-form clips.

**Architecture:** Three-stage pipeline (`probe.py` -> Antigravity Agent Brain via `SKILL.md` + `clip-rubric.md` -> `render.py`) organized as focused single-responsibility modules under `skills/youtube-clipper/scripts/lib/`. All commands route through `scripts/run.py` to an isolated Python 3.14 `.venv` inside `c:\Users\Admin\Desktop\clipper\.venv`, while `scripts/install_skill.py` syncs the skill globally to `C:\Users\Admin\.gemini\config\skills\youtube-clipper\`.

**Tech Stack:** Python 3.14 (isolated `.venv`), `yt-dlp`, `FFmpeg`/`FFprobe`, `opencv-python-headless`, `faster-whisper` (`int8` CPU), `pytest`.

**Spec:** `docs/superpowers/specs/2026-11-19-youtube-clipper-skill-design.md`

## Global Constraints

- Workspace root: `c:\Users\Admin\Desktop\clipper`; global skill target: `C:\Users\Admin\.gemini\config\skills\youtube-clipper`.
- Zero external LLM API keys: highlight selection and post copy are done by the Antigravity agent.
- Windows CPU-only execution: `.venv` must use a Python interpreter with prebuilt binary wheels for `yt-dlp`, `opencv-python-headless`, and `faster-whisper` (verified available on Python 3.14; not available on Python 3.15).
- FFmpeg Windows path safety: every FFmpeg call that burns subtitles MUST set `cwd=<clip_temp_dir>` and pass relative `ass=captions.ass` (never `ass=C:\...`).
- Default output directories: `probe` defaults `--out` to `./clips/<video_id>/work`; `render` defaults `--out` to `./clips/<video_id>`.
- Audio encoding standard: FFmpeg `loudnorm=I=-14:TP=-1.5:LRA=11`, H.264 `libx264` (`yuv420p`, `crf=20`, `+faststart`), AAC `192k`.
- Supported layouts: `face-track` (default), `blurred-fit`, `split` (corners `br`/`bl`/`tr`/`tl`), `landscape`, and `auto` (face ratio `>= 0.30` -> `face-track`, else `blurred-fit`).
- Supported caption styles: `bold-highlight` (default, 3–5 words per chunk, yellow `#FFE600` active highlight in ASS BGR `&H0000E6FF&`, `MarginV=360` at `1080x1920`), `clean`, `none`.
- `SKILL.md` frontmatter must have `name: youtube-clipper` and a third-person `description` starting with `Use when...` describing triggering conditions ONLY (never summarizing the workflow). Body must stay `<500` words.

---

## File Map

| File | Responsibility |
|---|---|
| `requirements.txt`, `requirements-dev.txt` | Runtime & test dependency pins |
| `scripts/install_skill.py` | Selects wheel-compatible Python, creates `.venv`, installs deps, syncs global skill, writes `.venv_python`, runs `doctor` |
| `skills/youtube-clipper/scripts/run.py` | Thin cross-workspace launcher delegating `probe`, `render`, `doctor` to `.venv` Python |
| `skills/youtube-clipper/scripts/doctor.py` | Checks binaries (`ffmpeg`, `ffprobe`) and Python packages (`yt_dlp`, `cv2`, `faster_whisper`) |
| `skills/youtube-clipper/scripts/lib/models.py` | Shared dataclasses, enums/constants, JSON load/save, `ClipError` |
| `skills/youtube-clipper/scripts/lib/heatmap.py` | YouTube Most Replayed peak detection, merging, and range scoring |
| `skills/youtube-clipper/scripts/lib/subtitles.py` | `json3` word-level parser, `vtt` parser with rolling dedup, word slicing, transcript chunking & long-VOD windowing |
| `skills/youtube-clipper/scripts/lib/captions.py` | Word chunking and styled `.ass` karaoke subtitle generation |
| `skills/youtube-clipper/scripts/lib/audio_peaks.py` | FFmpeg RMS energy peak scanner + `faster-whisper` (`int8` CPU) wrapper |
| `skills/youtube-clipper/scripts/lib/youtube.py` | `yt-dlp` wrappers (metadata, subtitles, audio-only, targeted section download with retry & structured errors) |
| `skills/youtube-clipper/scripts/probe.py` | Stage 1 CLI generating `analysis.json` |
| `skills/youtube-clipper/scripts/lib/tracking.py` | HSV histogram scene-cut detection, OpenCV Haar face detection, deadzone + EMA smoothing, frame interpolation |
| `skills/youtube-clipper/scripts/lib/ffmpeg_ops.py` | `ffprobe` media inspection, exact trim, static filtergraphs (`blurred-fit`, `split`, `landscape`), BGR pipe renderer for `face-track` |
| `skills/youtube-clipper/scripts/render.py` | Stage 3 CLI executing `clips_plan.json` -> `.mp4` + `post_kit.md` + `render_report.json` |
| `skills/youtube-clipper/SKILL.md` | Concise agent runbook (<500 words) |
| `skills/youtube-clipper/references/clip-rubric.md` | Detailed highlight scoring rubric, pacing/boundary rules, and post-kit template |

---

### Task 1: Project Dependencies, Data Models, Launcher, Doctor & Environment Setup

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `skills/youtube-clipper/scripts/lib/__init__.py`
- Create: `skills/youtube-clipper/scripts/lib/models.py`
- Create: `skills/youtube-clipper/scripts/run.py`
- Create: `skills/youtube-clipper/scripts/doctor.py`
- Create: `scripts/install_skill.py`
- Test: `tests/test_models_and_doctor.py`

**Interfaces:**
- Produces in `lib/models.py`:
  - `VALID_LAYOUTS: set[str]`, `VALID_CORNERS: set[str]`, `VALID_CAPTION_STYLES: set[str]`
  - `ClipError(Exception)` with `.code: str`, `.message: str`, `.hint: str | None`, `.to_dict() -> dict`
  - `HeatmapPeak(start: float, end: float, peak_time: float, score: float)`
  - `WordTiming(word: str, start: float, end: float)`
  - `CaptionSegment(start: float, end: float, text: str)`
  - `TranscriptBlock(start: float, end: float, text: str, heatmap_score: float = 0.0)`
  - `ClipSpec(...)`, `ClipsPlan(...)` with `ClipsPlan.from_json_file(path: Path) -> ClipsPlan`
  - `TrackingPlan(src_width: int, src_height: int, crop_width: int, fps: float, total_frames: int, face_ratio: float, x0_per_frame: list[int])`
  - `MediaInfo(width: int, height: int, fps: float, duration: float, has_audio: bool)`
- Produces in `doctor.py`: `run_doctor() -> dict`

- [ ] **Step 1: Create dependency files and `install_skill.py` bootstrapper, then set up `.venv`**

Write `requirements.txt`:
```text
yt-dlp>=2026.1.0
opencv-python-headless>=4.10.0
faster-whisper>=1.1.0
```

Write `requirements-dev.txt`:
```text
pytest>=8.0.0
```

Write `scripts/install_skill.py`:
```python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_SRC = REPO_ROOT / "skills" / "youtube-clipper"
DEFAULT_GLOBAL_TARGET = Path.home() / ".gemini" / "config" / "skills" / "youtube-clipper"
VENV_DIR = REPO_ROOT / ".venv"


def find_compatible_python() -> list[str]:
    """Select an installed Python interpreter that has binary wheels for faster-whisper."""
    candidates: list[list[str]] = []
    if os.name == "nt":
        for ver in ("3.14", "3.13", "3.12"):
            probe = subprocess.run(
                ["py", f"-{ver}", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                candidates.append([probe.stdout.strip()])
    candidates.append([sys.executable])

    for cmd in candidates:
        check = subprocess.run(
            cmd + ["-m", "pip", "install", "--dry-run", "--only-binary=:all:", "faster-whisper", "opencv-python-headless", "yt-dlp"],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            return cmd
    raise RuntimeError(
        "No installed Python interpreter has prebuilt binary wheels for faster-whisper/opencv-python-headless. "
        "Install Python 3.12 or 3.14 and re-run install_skill.py."
    )


def venv_python_path(venv_dir: Path = VENV_DIR) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path = VENV_DIR, recreate: bool = False) -> Path:
    py_exe = venv_python_path(venv_dir)
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not py_exe.exists():
        base_py = find_compatible_python()
        print(f"Creating venv at {venv_dir} using {base_py[0]} ...")
        subprocess.run(base_py + ["-m", "venv", str(venv_dir)], check=True)
    subprocess.run(
        [
            str(py_exe),
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "-r",
            str(REPO_ROOT / "requirements.txt"),
            "-r",
            str(REPO_ROOT / "requirements-dev.txt"),
        ],
        check=True,
    )
    return py_exe


def sync_skill(py_exe: Path, global_target: Path = DEFAULT_GLOBAL_TARGET) -> None:
    src_marker = SKILL_SRC / "scripts" / ".venv_python"
    src_marker.parent.mkdir(parents=True, exist_ok=True)
    src_marker.write_text(str(py_exe.resolve()), encoding="utf-8")

    global_target.parent.mkdir(parents=True, exist_ok=True)
    if global_target.exists():
        shutil.rmtree(global_target)
    shutil.copytree(
        SKILL_SRC,
        global_target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    dst_marker = global_target / "scripts" / ".venv_python"
    dst_marker.write_text(str(py_exe.resolve()), encoding="utf-8")
    print(f"Synced skill to {global_target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up .venv and install youtube-clipper skill globally.")
    parser.add_argument("--target", type=Path, default=DEFAULT_GLOBAL_TARGET)
    parser.add_argument("--recreate-venv", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    py_exe = ensure_venv(VENV_DIR, recreate=args.recreate_venv)
    if not args.skip_sync and (SKILL_SRC / "SKILL.md").exists():
        sync_skill(py_exe, args.target)
    else:
        marker = SKILL_SRC / "scripts" / ".venv_python"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(py_exe.resolve()), encoding="utf-8")
        print(f"Wrote venv marker at {marker} (SKILL.md not created yet, skipped global sync)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `python scripts/install_skill.py --skip-sync`
Expected: Creates `.venv` using Python 3.14 and installs `yt-dlp`, `opencv-python-headless`, `faster-whisper`, and `pytest` cleanly.

- [ ] **Step 2: Write the failing test `tests/test_models_and_doctor.py`**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_models_and_doctor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib'`

- [ ] **Step 4: Write `lib/__init__.py`, `lib/models.py`, `doctor.py`, and `run.py`**

Create `skills/youtube-clipper/scripts/lib/__init__.py` (empty file).

Create `skills/youtube-clipper/scripts/lib/models.py`:
```python
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_LAYOUTS = {"face-track", "blurred-fit", "split", "landscape", "auto"}
VALID_CORNERS = {"br", "bl", "tr", "tl"}
VALID_CAPTION_STYLES = {"bold-highlight", "clean", "none"}


class ClipError(Exception):
    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "error_code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        return payload


@dataclass
class HeatmapPeak:
    start: float
    end: float
    peak_time: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaptionSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptBlock:
    start: float
    end: float
    text: str
    heatmap_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MediaInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


@dataclass
class TrackingPlan:
    src_width: int
    src_height: int
    crop_width: int
    fps: float
    total_frames: int
    face_ratio: float
    x0_per_frame: list[int]


def sanitize_slug(value: str, fallback: str = "clip") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return cleaned[:48] or fallback


@dataclass
class ClipSpec:
    id: str
    slug: str
    start: float
    end: float
    layout: str = "face-track"
    split_cam_corner: str = "br"
    captions: bool = True
    caption_style: str = "bold-highlight"
    title: str = ""
    caption_text: str = ""
    hashtags: list[str] = field(default_factory=list)
    why_selected: str = ""

    def validate(self) -> None:
        if self.end <= self.start:
            raise ClipError("INVALID_PLAN", f"Clip {self.id}: end ({self.end}) must be greater than start ({self.start}).")
        if self.start < 0:
            raise ClipError("INVALID_PLAN", f"Clip {self.id}: start ({self.start}) must be >= 0.")
        if self.layout not in VALID_LAYOUTS:
            raise ClipError("INVALID_PLAN", f"Clip {self.id}: unsupported layout '{self.layout}'. Valid: {sorted(VALID_LAYOUTS)}")
        if self.split_cam_corner not in VALID_CORNERS:
            raise ClipError("INVALID_PLAN", f"Clip {self.id}: unsupported split_cam_corner '{self.split_cam_corner}'.")
        if self.caption_style not in VALID_CAPTION_STYLES:
            raise ClipError("INVALID_PLAN", f"Clip {self.id}: unsupported caption_style '{self.caption_style}'.")
        self.slug = sanitize_slug(self.slug or self.id, fallback=self.id)


@dataclass
class ClipsPlan:
    video_url: str
    video_id: str
    video_title: str
    clips: list[ClipSpec]
    cache_subtitles: str | None = None
    cookies_from_browser: str | None = None

    @classmethod
    def from_json_file(cls, path: Path) -> "ClipsPlan":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ClipError("INVALID_PLAN", f"Failed to parse plan JSON at {path}: {exc}") from exc

        raw_clips = raw.get("clips")
        if not isinstance(raw_clips, list) or not raw_clips:
            raise ClipError("INVALID_PLAN", "Plan must contain a non-empty 'clips' list.")

        clips: list[ClipSpec] = []
        for idx, item in enumerate(raw_clips, start=1):
            spec = ClipSpec(
                id=str(item.get("id") or f"clip_{idx:02d}"),
                slug=str(item.get("slug") or f"clip-{idx:02d}"),
                start=float(item["start"]),
                end=float(item["end"]),
                layout=str(item.get("layout", "face-track")),
                split_cam_corner=str(item.get("split_cam_corner", "br")),
                captions=bool(item.get("captions", True)),
                caption_style=str(item.get("caption_style", "bold-highlight")),
                title=str(item.get("title", "")),
                caption_text=str(item.get("caption_text", "")),
                hashtags=[str(h) for h in item.get("hashtags", [])],
                why_selected=str(item.get("why_selected", "")),
            )
            spec.validate()
            clips.append(spec)

        return cls(
            video_url=str(raw.get("video_url", "")),
            video_id=str(raw.get("video_id", "video")),
            video_title=str(raw.get("video_title", "")),
            clips=clips,
            cache_subtitles=raw.get("cache_subtitles"),
            cookies_from_browser=raw.get("cookies_from_browser"),
        )
```

Create `skills/youtube-clipper/scripts/doctor.py`:
```python
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _check_binary(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"ok": bool(path), "path": path}


def _check_module(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "installed")
        extra: dict[str, Any] = {"ok": True, "version": str(version)}
        if name == "cv2":
            cascade = Path(mod.data.haarcascades) / "haarcascade_frontalface_default.xml"
            extra["haar_cascade_ok"] = cascade.exists()
            extra["ok"] = cascade.exists()
        return extra
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_doctor() -> dict[str, Any]:
    checks = {
        "ffmpeg": _check_binary("ffmpeg"),
        "ffprobe": _check_binary("ffprobe"),
        "yt_dlp": _check_module("yt_dlp"),
        "cv2": _check_module("cv2"),
        "faster_whisper": _check_module("faster_whisper"),
    }
    return {
        "ok": all(item["ok"] for item in checks.values()),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify youtube-clipper runtime environment.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = run_doctor()
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python: {report['python_version']} ({report['python']})")
        for name, info in report["checks"].items():
            status = "OK" if info["ok"] else "FAIL"
            detail = info.get("version") or info.get("path") or info.get("error", "")
            print(f"  [{status}] {name}: {detail}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `skills/youtube-clipper/scripts/run.py`:
```python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SUBCOMMANDS = {
    "doctor": SCRIPTS_DIR / "doctor.py",
    "probe": SCRIPTS_DIR / "probe.py",
    "render": SCRIPTS_DIR / "render.py",
}


def resolve_venv_python() -> Path | None:
    marker = SCRIPTS_DIR / ".venv_python"
    if marker.exists():
        candidate = Path(marker.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in SUBCOMMANDS:
        print(f"Usage: python run.py <{'|'.join(SUBCOMMANDS)}> [args...]", file=sys.stderr)
        return 2

    subcommand, rest = args[0], args[1:]
    target_script = SUBCOMMANDS[subcommand]
    venv_py = resolve_venv_python()
    current_py = Path(sys.executable).resolve()

    if venv_py and venv_py.resolve() != current_py and os.environ.get("YOUTUBE_CLIPPER_REEXEC") != "1":
        env = os.environ.copy()
        env["YOUTUBE_CLIPPER_REEXEC"] = "1"
        completed = subprocess.run([str(venv_py), str(target_script), *rest], env=env)
        return completed.returncode

    completed = subprocess.run([sys.executable, str(target_script), *rest])
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests and doctor via `run.py`**

Run: `.\.venv\Scripts\pytest.exe tests/test_models_and_doctor.py -v`
Run: `python skills/youtube-clipper/scripts/run.py doctor --json`
Expected: Both PASS (`ok: true`, Python 3.14 inside `.venv`).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt scripts/install_skill.py skills/youtube-clipper/scripts/ tests/test_models_and_doctor.py
git commit -m "feat: add models, venv installer, run launcher, and doctor CLI"
```

---

### Task 2: YouTube Most Replayed Heatmap Peak Extraction (`lib/heatmap.py`)

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/heatmap.py`
- Test: `tests/test_heatmap.py`

**Interfaces:**
- Consumes: `lib.models.HeatmapPeak`
- Produces:
  - `extract_heatmap_peaks(heatmap: list[dict] | None, threshold: float = 0.65, merge_gap_sec: float = 20.0, pad_sec: float = 10.0, max_peaks: int = 15, duration_sec: float | None = None) -> list[HeatmapPeak]`
  - `score_time_range(peaks: list[HeatmapPeak], start: float, end: float) -> float`

- [ ] **Step 1: Write the failing test `tests/test_heatmap.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.heatmap import extract_heatmap_peaks, score_time_range  # noqa: E402


def test_extract_heatmap_peaks_merges_adjacent_and_sorts_by_score() -> None:
    heatmap = [
        {"start_time": 0.0, "end_time": 10.0, "value": 0.2},
        {"start_time": 100.0, "end_time": 110.0, "value": 0.72},
        {"start_time": 110.0, "end_time": 120.0, "value": 0.85},
        {"start_time": 125.0, "end_time": 135.0, "value": 0.70},
        {"start_time": 400.0, "end_time": 410.0, "value": 0.96},
    ]
    peaks = extract_heatmap_peaks(heatmap, threshold=0.65, merge_gap_sec=20.0, pad_sec=10.0, duration_sec=600.0)
    assert len(peaks) == 2
    assert peaks[0].score == 0.96
    assert peaks[0].start == 390.0
    assert peaks[0].end == 420.0
    assert peaks[1].score == 0.85
    assert peaks[1].start == 90.0
    assert peaks[1].end == 145.0


def test_extract_heatmap_peaks_handles_empty_and_clamps_bounds() -> None:
    assert extract_heatmap_peaks(None) == []
    assert extract_heatmap_peaks([]) == []
    peaks = extract_heatmap_peaks(
        [{"start_time": 2.0, "end_time": 8.0, "value": 0.9}],
        pad_sec=10.0,
        duration_sec=12.0,
    )
    assert peaks[0].start == 0.0
    assert peaks[0].end == 12.0


def test_score_time_range_returns_highest_overlapping_peak() -> None:
    peaks = extract_heatmap_peaks(
        [
            {"start_time": 50.0, "end_time": 60.0, "value": 0.75},
            {"start_time": 200.0, "end_time": 210.0, "value": 0.92},
        ],
        pad_sec=5.0,
    )
    assert score_time_range(peaks, 48.0, 55.0) == 0.75
    assert score_time_range(peaks, 10.0, 20.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_heatmap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.heatmap'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/heatmap.py`**

```python
from __future__ import annotations

from typing import Any

from lib.models import HeatmapPeak


def extract_heatmap_peaks(
    heatmap: list[dict[str, Any]] | None,
    threshold: float = 0.65,
    merge_gap_sec: float = 20.0,
    pad_sec: float = 10.0,
    max_peaks: int = 15,
    duration_sec: float | None = None,
) -> list[HeatmapPeak]:
    if not heatmap:
        return []

    hot = [
        (float(b["start_time"]), float(b["end_time"]), float(b["value"]))
        for b in heatmap
        if float(b.get("value", 0.0)) >= threshold
    ]
    if not hot:
        return []

    hot.sort(key=lambda item: item[0])
    merged: list[list[float]] = []
    for start, end, val in hot:
        mid = (start + end) / 2.0
        if not merged or start - merged[-1][1] > merge_gap_sec:
            merged.append([start, end, mid, val])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            if val > merged[-1][3]:
                merged[-1][2] = mid
                merged[-1][3] = val

    peaks: list[HeatmapPeak] = []
    for start, end, peak_time, score in merged:
        padded_start = max(0.0, start - pad_sec)
        padded_end = end + pad_sec
        if duration_sec is not None:
            padded_end = min(duration_sec, padded_end)
        peaks.append(
            HeatmapPeak(
                start=round(padded_start, 2),
                end=round(padded_end, 2),
                peak_time=round(peak_time, 2),
                score=round(score, 4),
            )
        )

    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks[:max_peaks]


def score_time_range(peaks: list[HeatmapPeak], start: float, end: float) -> float:
    best = 0.0
    for peak in peaks:
        if peak.end >= start and peak.start <= end:
            best = max(best, peak.score)
    return best
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\pytest.exe tests/test_heatmap.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/heatmap.py tests/test_heatmap.py
git commit -m "feat: add YouTube heatmap peak extraction and scoring"
```

---

### Task 3: Subtitle Parsing (`json3` / `vtt`), Word Slicing & Long-VOD Windowing (`lib/subtitles.py`)

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/subtitles.py`
- Test: `tests/test_subtitles.py`

**Interfaces:**
- Consumes: `lib.models.CaptionSegment`, `lib.models.WordTiming`, `lib.models.TranscriptBlock`, `lib.models.HeatmapPeak`, `lib.heatmap.score_time_range`
- Produces:
  - `parse_json3(content: str) -> tuple[list[CaptionSegment], list[WordTiming]]`
  - `parse_vtt(content: str) -> list[CaptionSegment]`
  - `slice_words_for_clip(words: list[WordTiming], start: float, end: float) -> list[WordTiming]` (re-zeroed to clip start `0.0`)
  - `segments_to_proportional_words(segments: list[CaptionSegment], start: float, end: float) -> list[WordTiming]` (re-zeroed to clip start `0.0`)
  - `build_transcript_blocks(segments: list[CaptionSegment], heatmap_peaks: list[HeatmapPeak], block_sec: float = 18.0, duration_sec: float = 0.0, max_windows: int = 15, full_transcript: bool = False) -> tuple[list[TranscriptBlock], bool]`

- [ ] **Step 1: Write the failing test `tests/test_subtitles.py`**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import CaptionSegment, HeatmapPeak  # noqa: E402
from lib.subtitles import (  # noqa: E402
    build_transcript_blocks,
    parse_json3,
    parse_vtt,
    segments_to_proportional_words,
    slice_words_for_clip,
)


def test_parse_json3_extracts_segments_and_word_offsets() -> None:
    payload = {
        "events": [
            {
                "tStartMs": 10000,
                "dDurationMs": 2000,
                "segs": [
                    {"utf8": "Nobody"},
                    {"utf8": " talks", "tOffsetMs": 400},
                    {"utf8": " about", "tOffsetMs": 800},
                    {"utf8": " this", "tOffsetMs": 1300},
                ],
            },
            {"tStartMs": 12000, "dDurationMs": 500, "segs": [{"utf8": "\n"}]},
        ]
    }
    segments, words = parse_json3(json.dumps(payload))
    assert len(segments) == 1
    assert segments[0].text == "Nobody talks about this"
    assert segments[0].start == 10.0
    assert [w.word for w in words] == ["Nobody", "talks", "about", "this"]
    assert words[1].start == 10.4
    assert words[0].end == 10.4


def test_parse_vtt_deduplicates_rolling_auto_captions() -> None:
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.000 --> 00:00:05.000
Hello world
this is a test

00:00:05.000 --> 00:00:07.000
<c>this</c> is a test
final line
"""
    segments = parse_vtt(vtt)
    assert [s.text for s in segments] == ["Hello world", "this is a test", "final line"]
    assert segments[0].start == 1.0
    assert segments[2].end == 7.0


def test_slice_words_and_proportional_fallback_rezero_timestamps() -> None:
    payload = {
        "events": [
            {
                "tStartMs": 10000,
                "dDurationMs": 2000,
                "segs": [{"utf8": "First"}, {"utf8": " second", "tOffsetMs": 1000}],
            }
        ]
    }
    _, words = parse_json3(json.dumps(payload))
    sliced = slice_words_for_clip(words, start=10.5, end=12.5)
    assert len(sliced) == 1
    assert sliced[0].word == "second"
    assert sliced[0].start == 0.5

    segs = [CaptionSegment(start=10.0, end=12.0, text="one two")]
    prop = segments_to_proportional_words(segs, start=10.0, end=12.0)
    assert [(w.word, w.start, w.end) for w in prop] == [("one", 0.0, 1.0), ("two", 1.0, 2.0)]


def test_build_transcript_blocks_windows_long_vods_by_heatmap() -> None:
    segs = [
        CaptionSegment(start=10.0, end=15.0, text="intro talk"),
        CaptionSegment(start=3600.0, end=3605.0, text="huge moment"),
        CaptionSegment(start=5000.0, end=5005.0, text="quiet part"),
    ]
    peaks = [HeatmapPeak(start=3590.0, end=3640.0, peak_time=3602.0, score=0.95)]
    blocks, truncated = build_transcript_blocks(
        segs,
        peaks,
        duration_sec=6000.0,
        max_windows=1,
        full_transcript=False,
    )
    assert truncated is True
    assert len(blocks) == 1
    assert "huge moment" in blocks[0].text
    assert blocks[0].heatmap_score == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_subtitles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.subtitles'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/subtitles.py`**

```python
from __future__ import annotations

import html
import json
import re

from lib.heatmap import score_time_range
from lib.models import CaptionSegment, HeatmapPeak, TranscriptBlock, WordTiming

VTT_TS_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _ts_to_sec(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h) if h else 0
    return round(hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0, 3)


def parse_json3(content: str) -> tuple[list[CaptionSegment], list[WordTiming]]:
    data = json.loads(content)
    events = data.get("events", [])
    segments: list[CaptionSegment] = []
    words: list[WordTiming] = []

    for event in events:
        segs = event.get("segs")
        if not segs or "tStartMs" not in event:
            continue
        start_ms = float(event["tStartMs"])
        dur_ms = float(event.get("dDurationMs", 0.0))
        end_ms = start_ms + max(dur_ms, 200.0)

        raw_pieces: list[tuple[str, float]] = []
        for piece in segs:
            token = _clean_text(str(piece.get("utf8", "")))
            if not token:
                continue
            offset_ms = float(piece.get("tOffsetMs", 0.0))
            raw_pieces.append((token, (start_ms + offset_ms) / 1000.0))

        if not raw_pieces:
            continue

        line_text = " ".join(p[0] for p in raw_pieces).strip()
        if not line_text:
            continue

        segments.append(
            CaptionSegment(
                start=round(start_ms / 1000.0, 3),
                end=round(end_ms / 1000.0, 3),
                text=line_text,
            )
        )

        # Build word timings: use exact tOffsetMs when present, or distribute within piece span
        for p_idx, (token, p_start) in enumerate(raw_pieces):
            p_end = (
                raw_pieces[p_idx + 1][1]
                if p_idx + 1 < len(raw_pieces) and raw_pieces[p_idx + 1][1] > p_start
                else max(p_start + 0.15, end_ms / 1000.0)
            )
            sub_tokens = token.split()
            span = max(0.1, p_end - p_start)
            step = span / max(1, len(sub_tokens))
            for s_idx, sub in enumerate(sub_tokens):
                w_start = p_start + s_idx * step
                w_end = p_start + (s_idx + 1) * step
                words.append(WordTiming(word=sub, start=round(w_start, 3), end=round(w_end, 3)))

    # Fix any overlapping word end times across consecutive events
    for i in range(len(words) - 1):
        if words[i].end > words[i + 1].start and words[i + 1].start > words[i].start:
            words[i].end = words[i + 1].start

    return segments, words


def parse_vtt(content: str) -> list[CaptionSegment]:
    lines = content.replace("\r\n", "\n").split("\n")
    segments: list[CaptionSegment] = []
    last_emitted = ""
    i = 0
    while i < len(lines):
        match = VTT_TS_RE.search(lines[i])
        if not match:
            i += 1
            continue
        start = _ts_to_sec(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _ts_to_sec(match.group(5), match.group(6), match.group(7), match.group(8))
        i += 1
        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip() != "":
            cleaned = _clean_text(lines[i])
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        # Deduplicate rolling captions where first line repeats previous cue
        filtered = [ln for ln in cue_lines if ln != last_emitted]
        text = " ".join(filtered).strip()
        if text:
            segments.append(CaptionSegment(start=start, end=end, text=text))
            last_emitted = filtered[-1]
    return segments


def slice_words_for_clip(words: list[WordTiming], start: float, end: float) -> list[WordTiming]:
    result: list[WordTiming] = []
    clip_dur = max(0.0, end - start)
    for w in words:
        if w.end <= start or w.start >= end:
            continue
        rel_start = max(0.0, round(w.start - start, 3))
        rel_end = min(clip_dur, round(w.end - start, 3))
        if rel_end > rel_start:
            result.append(WordTiming(word=w.word, start=rel_start, end=rel_end))
    return result


def segments_to_proportional_words(segments: list[CaptionSegment], start: float, end: float) -> list[WordTiming]:
    words: list[WordTiming] = []
    clip_dur = max(0.0, end - start)
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        tokens = seg.text.split()
        if not tokens:
            continue
        seg_start = max(start, seg.start)
        seg_end = min(end, seg.end)
        span = max(0.1, seg_end - seg_start)
        step = span / len(tokens)
        for idx, tok in enumerate(tokens):
            w_start = round((seg_start - start) + idx * step, 3)
            w_end = min(clip_dur, round((seg_start - start) + (idx + 1) * step, 3))
            if w_end > w_start:
                words.append(WordTiming(word=tok, start=w_start, end=w_end))
    return words


def build_transcript_blocks(
    segments: list[CaptionSegment],
    heatmap_peaks: list[HeatmapPeak],
    block_sec: float = 18.0,
    duration_sec: float = 0.0,
    max_windows: int = 15,
    full_transcript: bool = False,
) -> tuple[list[TranscriptBlock], bool]:
    if not segments:
        return [], False

    truncated = False
    selected_segments = segments
    if duration_sec > 5400.0 and not full_transcript and heatmap_peaks:
        windows = sorted(heatmap_peaks[:max_windows], key=lambda p: p.start)
        filtered = [
            seg
            for seg in segments
            if any(seg.end >= (w.start - 30.0) and seg.start <= (w.end + 30.0) for w in windows)
        ]
        if filtered:
            selected_segments = filtered
            truncated = True

    blocks: list[TranscriptBlock] = []
    cur_start = selected_segments[0].start
    cur_end = selected_segments[0].end
    cur_texts: list[str] = [selected_segments[0].text]

    for seg in selected_segments[1:]:
        if (seg.end - cur_start) <= block_sec and (seg.start - cur_end) <= 4.0:
            cur_end = max(cur_end, seg.end)
            cur_texts.append(seg.text)
        else:
            blocks.append(
                TranscriptBlock(
                    start=round(cur_start, 2),
                    end=round(cur_end, 2),
                    text=" ".join(cur_texts),
                    heatmap_score=score_time_range(heatmap_peaks, cur_start, cur_end),
                )
            )
            cur_start = seg.start
            cur_end = seg.end
            cur_texts = [seg.text]

    blocks.append(
        TranscriptBlock(
            start=round(cur_start, 2),
            end=round(cur_end, 2),
            text=" ".join(cur_texts),
            heatmap_score=score_time_range(heatmap_peaks, cur_start, cur_end),
        )
    )
    return blocks, truncated
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\pytest.exe tests/test_subtitles.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/subtitles.py tests/test_subtitles.py
git commit -m "feat: add json3/vtt subtitle parser, word slicing, and transcript windowing"
```

---

### Task 4: Styled `.ass` Karaoke Caption Generator (`lib/captions.py`)

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/captions.py`
- Test: `tests/test_captions.py`

**Interfaces:**
- Consumes: `lib.models.WordTiming`
- Produces:
  - `CaptionChunk(words: list[WordTiming], start: float, end: float)`
  - `group_words_into_chunks(words: list[WordTiming], max_words: int = 4, max_chars: int = 26, max_gap_sec: float = 0.6) -> list[CaptionChunk]`
  - `format_ass_time(seconds: float) -> str`
  - `generate_ass(words: list[WordTiming], style: str = "bold-highlight", play_res_x: int = 1080, play_res_y: int = 1920) -> str`

- [ ] **Step 1: Write the failing test `tests/test_captions.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.captions import format_ass_time, generate_ass, group_words_into_chunks  # noqa: E402
from lib.models import WordTiming  # noqa: E402


def test_format_ass_time() -> None:
    assert format_ass_time(0.0) == "0:00:00.00"
    assert format_ass_time(65.25) == "0:01:05.25"
    assert format_ass_time(3661.5) == "1:01:01.50"


def test_group_words_respects_max_words_and_pause_gap() -> None:
    words = [
        WordTiming("This", 0.0, 0.3),
        WordTiming("is", 0.3, 0.5),
        WordTiming("wild", 0.5, 0.9),
        WordTiming("Wait", 2.0, 2.3),
        WordTiming("what", 2.3, 2.6),
    ]
    chunks = group_words_into_chunks(words, max_words=4, max_gap_sec=0.6)
    assert len(chunks) == 2
    assert [w.word for w in chunks[0].words] == ["This", "is", "wild"]
    assert [w.word for w in chunks[1].words] == ["Wait", "what"]


def test_generate_ass_bold_highlight_and_clean_styles() -> None:
    words = [
        WordTiming("Watch", 0.0, 0.4),
        WordTiming("{this}", 0.4, 0.8),
    ]
    ass_bold = generate_ass(words, style="bold-highlight", play_res_x=1080, play_res_y=1920)
    assert "PlayResX: 1080" in ass_bold
    assert "PlayResY: 1920" in ass_bold
    assert "&H0000E6FF&" in ass_bold  # yellow active highlight in ASS BGR
    assert r"\{this\}" in ass_bold    # braces escaped
    assert ass_bold.count("Dialogue:") == 2

    ass_clean = generate_ass(words, style="clean", play_res_x=1280, play_res_y=720)
    assert "PlayResX: 1280" in ass_clean
    assert ass_clean.count("Dialogue:") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_captions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.captions'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/captions.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from lib.models import WordTiming

# ASS color format is &HAABBGGRR&
ASS_WHITE = "&H00FFFFFF&"
ASS_YELLOW = "&H0000E6FF&"  # #FFE600 in BGR


@dataclass
class CaptionChunk:
    words: list[WordTiming]
    start: float
    end: float


def escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def format_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def group_words_into_chunks(
    words: list[WordTiming],
    max_words: int = 4,
    max_chars: int = 26,
    max_gap_sec: float = 0.6,
) -> list[CaptionChunk]:
    if not words:
        return []
    chunks: list[CaptionChunk] = []
    current: list[WordTiming] = []
    cur_len = 0

    for w in words:
        word_len = len(w.word)
        gap = (w.start - current[-1].end) if current else 0.0
        exceeds = (
            len(current) >= max_words
            or (cur_len + word_len + (1 if current else 0)) > max_chars
            or gap > max_gap_sec
        )
        if current and exceeds:
            chunks.append(CaptionChunk(words=current, start=current[0].start, end=current[-1].end))
            current = []
            cur_len = 0
        current.append(w)
        cur_len += word_len + (1 if len(current) > 1 else 0)
        if w.word.endswith((".", "!", "?")):
            chunks.append(CaptionChunk(words=current, start=current[0].start, end=current[-1].end))
            current = []
            cur_len = 0

    if current:
        chunks.append(CaptionChunk(words=current, start=current[0].start, end=current[-1].end))
    return chunks


def generate_ass(
    words: list[WordTiming],
    style: str = "bold-highlight",
    play_res_x: int = 1080,
    play_res_y: int = 1920,
) -> str:
    is_vertical = play_res_y >= play_res_x
    font_size = 76 if (style == "bold-highlight" and is_vertical) else (54 if is_vertical else 44)
    outline = 6 if style == "bold-highlight" else 3
    shadow = 3 if style == "bold-highlight" else 1
    margin_v = 360 if is_vertical else 60
    max_words = 4 if style == "bold-highlight" else 7
    max_chars = 24 if style == "bold-highlight" else 44

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {play_res_x}",
            f"PlayResY: {play_res_y}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,Arial,{font_size},{ASS_WHITE},{ASS_YELLOW},&H00000000&,&H80000000&,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,60,60,{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    if not words or style == "none":
        return header + "\n"

    chunks = group_words_into_chunks(words, max_words=max_words, max_chars=max_chars)
    events: list[str] = []

    if style == "clean":
        for chunk in chunks:
            text = " ".join(escape_ass_text(w.word) for w in chunk.words)
            events.append(
                f"Dialogue: 0,{format_ass_time(chunk.start)},{format_ass_time(chunk.end)},Default,,0,0,0,,{text}"
            )
    else:
        for chunk in chunks:
            for active_idx, active_word in enumerate(chunk.words):
                w_start = active_word.start
                w_end = (
                    chunk.words[active_idx + 1].start
                    if active_idx + 1 < len(chunk.words) and chunk.words[active_idx + 1].start > w_start
                    else max(active_word.end, w_start + 0.05)
                )
                rendered_tokens: list[str] = []
                for idx, tok in enumerate(chunk.words):
                    safe = escape_ass_text(tok.word.upper())
                    if idx == active_idx:
                        rendered_tokens.append(f"{{\\c{ASS_YELLOW}\\fscx108\\fscy108}}{safe}{{\\c{ASS_WHITE}\\fscx100\\fscy100}}")
                    else:
                        rendered_tokens.append(safe)
                line = " ".join(rendered_tokens)
                events.append(
                    f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Default,,0,0,0,,{line}"
                )

    return header + "\n" + "\n".join(events) + "\n"
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\pytest.exe tests/test_captions.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/captions.py tests/test_captions.py
git commit -m "feat: add word-highlighted ASS karaoke caption generator"
```

---

### Task 5: Audio Energy Peaks & Local `faster-whisper` Fallback (`lib/audio_peaks.py`)

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/audio_peaks.py`
- Test: `tests/test_audio_peaks.py`

**Interfaces:**
- Consumes: `lib.models.HeatmapPeak`, `lib.models.CaptionSegment`, `lib.models.WordTiming`
- Produces:
  - `parse_ffmpeg_rms_levels(stderr_text: str, window_sec: float = 5.0) -> list[tuple[float, float]]`
  - `peaks_from_rms_windows(windows: list[tuple[float, float]], duration_sec: float, window_sec: float = 5.0, max_peaks: int = 15) -> list[HeatmapPeak]`
  - `detect_audio_peaks(audio_path: Path, duration_sec: float, window_sec: float = 5.0, max_peaks: int = 15) -> list[HeatmapPeak]`
  - `transcribe_audio(audio_path: Path, model_size: str = "base", language: str | None = None, clip_offset: float = 0.0) -> tuple[list[CaptionSegment], list[WordTiming], str]`

- [ ] **Step 1: Write the failing test `tests/test_audio_peaks.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.audio_peaks import (  # noqa: E402
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_audio_peaks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.audio_peaks'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/audio_peaks.py`**

```python
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

from lib.models import CaptionSegment, HeatmapPeak, WordTiming

PTS_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
RMS_RE = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+(?:\.\d+)?|-inf)")


def parse_ffmpeg_rms_levels(stderr_text: str, window_sec: float = 5.0) -> list[tuple[float, float]]:
    buckets: dict[int, list[float]] = {}
    current_t = 0.0
    for line in stderr_text.splitlines():
        pts_m = PTS_RE.search(line)
        if pts_m:
            current_t = float(pts_m.group(1))
            continue
        rms_m = RMS_RE.search(line)
        if rms_m:
            raw = rms_m.group(1)
            db = -80.0 if raw == "-inf" else max(-80.0, float(raw))
            bucket = int(current_t // window_sec)
            buckets.setdefault(bucket, []).append(db)

    return [
        (round(b * window_sec, 2), sum(vals) / len(vals))
        for b, vals in sorted(buckets.items())
    ]


def peaks_from_rms_windows(
    windows: list[tuple[float, float]],
    duration_sec: float,
    window_sec: float = 5.0,
    max_peaks: int = 15,
) -> list[HeatmapPeak]:
    if not windows:
        return []
    db_vals = [w[1] for w in windows]
    lo, hi = min(db_vals), max(db_vals)
    span = max(hi - lo, 1.0)

    normalized = [(t, (db - lo) / span) for t, db in windows]
    hot = [(t, score) for t, score in normalized if score >= 0.7]
    if not hot:
        return []

    merged: list[list[float]] = []
    for t, score in hot:
        end = t + window_sec
        mid = t + window_sec / 2.0
        if not merged or t - merged[-1][1] > 20.0:
            merged.append([t, end, mid, score])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            if score > merged[-1][3]:
                merged[-1][2] = mid
                merged[-1][3] = score

    peaks = [
        HeatmapPeak(
            start=round(max(0.0, s - 10.0), 2),
            end=round(min(duration_sec, e + 10.0), 2),
            peak_time=round(mid, 2),
            score=round(sc, 4),
        )
        for s, e, mid, sc in merged
    ]
    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks[:max_peaks]


def detect_audio_peaks(
    audio_path: Path,
    duration_sec: float,
    window_sec: float = 5.0,
    max_peaks: int = 15,
) -> list[HeatmapPeak]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        "asetnsamples=n=44100,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    levels = parse_ffmpeg_rms_levels(proc.stderr, window_sec=window_sec)
    return peaks_from_rms_windows(levels, duration_sec=duration_sec, window_sec=window_sec, max_peaks=max_peaks)


@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(
    audio_path: Path,
    model_size: str = "base",
    language: str | None = None,
    clip_offset: float = 0.0,
) -> tuple[list[CaptionSegment], list[WordTiming], str]:
    model = _load_whisper_model(model_size)
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    segments: list[CaptionSegment] = []
    words: list[WordTiming] = []
    for seg in raw_segments:
        text = str(seg.text).strip()
        if not text:
            continue
        segments.append(
            CaptionSegment(
                start=round(float(seg.start) + clip_offset, 3),
                end=round(float(seg.end) + clip_offset, 3),
                text=text,
            )
        )
        for w in getattr(seg, "words", None) or []:
            tok = str(w.word).strip()
            if tok:
                words.append(
                    WordTiming(
                        word=tok,
                        start=round(float(w.start) + clip_offset, 3),
                        end=round(float(w.end) + clip_offset, 3),
                    )
                )
    detected_lang = str(getattr(info, "language", language or "en"))
    return segments, words, detected_lang
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\pytest.exe tests/test_audio_peaks.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/audio_peaks.py tests/test_audio_peaks.py
git commit -m "feat: add FFmpeg audio peak scanner and faster-whisper CPU fallback"
```

---

### Task 6: YouTube `yt-dlp` Wrapper (`lib/youtube.py`) & Stage 1 `probe.py`

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/youtube.py`
- Create: `skills/youtube-clipper/scripts/probe.py`
- Test: `tests/test_probe_cli.py`

**Interfaces:**
- Consumes: `lib.models.*`, `lib.heatmap.extract_heatmap_peaks`, `lib.subtitles.*`, `lib.audio_peaks.*`
- Produces in `lib/youtube.py`:
  - `fetch_video_info(url: str, cookies_from_browser: str | None = None) -> dict`
  - `download_subtitles(url: str, out_dir: Path, lang: str = "en", cookies_from_browser: str | None = None) -> tuple[Path | None, str, str]`
  - `download_audio_only(url: str, out_dir: Path, cookies_from_browser: str | None = None) -> Path`
  - `download_video_section(url: str, start: float, end: float, out_path: Path, cookies_from_browser: str | None = None) -> Path`
- Produces in `probe.py`:
  - `run_probe(url: str, out_dir: Path | None = None, lang: str = "en", max_windows: int = 15, full_transcript: bool = False, cookies_from_browser: str | None = None) -> dict`

- [ ] **Step 1: Write the failing test `tests/test_probe_cli.py`**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import ClipError  # noqa: E402
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_probe_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'probe'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/youtube.py` and `skills/youtube-clipper/scripts/probe.py`**

Create `skills/youtube-clipper/scripts/lib/youtube.py`:
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.models import ClipError


def _ytdlp_base_cmd(cookies_from_browser: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings"]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    return cmd


def _classify_ytdlp_error(stderr: str, url: str) -> ClipError:
    low = stderr.lower()
    if "sign in to confirm you" in low or "bot" in low:
        return ClipError(
            "BOT_CHECK",
            f"YouTube requested bot verification for {url}.",
            hint="Retry with --cookies-from-browser chrome (or edge/firefox).",
        )
    if "sign in to confirm your age" in low or "members-only" in low or "private video" in low:
        return ClipError(
            "AUTH_REQUIRED",
            f"Video requires authentication or membership: {url}",
            hint="Retry with --cookies-from-browser chrome (or edge/firefox).",
        )
    return ClipError("UNAVAILABLE", f"yt-dlp failed for {url}: {stderr.strip()[:400]}")


def fetch_video_info(url: str, cookies_from_browser: str | None = None) -> dict[str, Any]:
    cmd = _ytdlp_base_cmd(cookies_from_browser) + ["--dump-single-json", "--skip-download", url]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise _classify_ytdlp_error(proc.stderr, url)
    return json.loads(proc.stdout)


def download_subtitles(
    url: str,
    out_dir: Path,
    lang: str = "en",
    cookies_from_browser: str | None = None,
) -> tuple[Path | None, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt, source_kind in (("json3", "youtube_json3"), ("vtt", "youtube_vtt")):
        template = str(out_dir / f"subtitles.%(ext)s")
        cmd = _ytdlp_base_cmd(cookies_from_browser) + [
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            f"{lang}.*,{lang}",
            "--sub-format",
            fmt,
            "-o",
            template,
            url,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        matches = sorted(out_dir.glob(f"subtitles*.{fmt}"))
        if matches:
            chosen = matches[0]
            detected_lang = chosen.stem.split(".")[-1] if "." in chosen.stem else lang
            return chosen, source_kind, detected_lang
    return None, "none", lang


def download_audio_only(
    url: str,
    out_dir: Path,
    cookies_from_browser: str | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "source_audio.%(ext)s")
    cmd = _ytdlp_base_cmd(cookies_from_browser) + [
        "-f",
        "bestaudio[ext=m4a]/bestaudio/best",
        "-o",
        template,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise _classify_ytdlp_error(proc.stderr, url)
    matches = sorted(out_dir.glob("source_audio.*"))
    if not matches:
        raise ClipError("DOWNLOAD_FAILED", "Audio download completed but no output file was found.")
    return matches[0]


def download_video_section(
    url: str,
    start: float,
    end: float,
    out_path: Path,
    cookies_from_browser: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start:.2f}-{end:.2f}"
    formats = [
        "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "best[height<=1080]/best",
    ]
    last_stderr = ""
    for fmt in formats:
        if out_path.exists():
            out_path.unlink()
        cmd = _ytdlp_base_cmd(cookies_from_browser) + [
            "-f",
            fmt,
            "--download-sections",
            section,
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out_path),
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and out_path.exists():
            return out_path
        last_stderr = proc.stderr
    raise _classify_ytdlp_error(last_stderr, url)
```

Create `skills/youtube-clipper/scripts/probe.py`:
```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.audio_peaks import detect_audio_peaks, transcribe_audio
from lib.heatmap import extract_heatmap_peaks
from lib.models import CaptionSegment, ClipError, HeatmapPeak
from lib.subtitles import build_transcript_blocks, parse_json3, parse_vtt
from lib.youtube import download_audio_only, download_subtitles, fetch_video_info


def _transcribe_fallback(
    audio_path: Path,
    duration_sec: float,
    peaks: list[HeatmapPeak],
    max_windows: int,
    work_dir: Path,
) -> tuple[list[CaptionSegment], str]:
    if duration_sec <= 1200.0 or not peaks:
        segments, _, lang = transcribe_audio(audio_path, model_size="base")
        return segments, lang

    all_segments: list[CaptionSegment] = []
    detected_lang = "en"
    tmp_dir = work_dir / "cache" / "whisper_windows"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, peak in enumerate(peaks[:max_windows]):
        w_start = max(0.0, peak.peak_time - 45.0)
        w_dur = min(90.0, max(10.0, duration_sec - w_start))
        slice_wav = tmp_dir / f"win_{idx:02d}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{w_start:.2f}",
                "-t",
                f"{w_dur:.2f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(slice_wav),
            ],
            check=True,
        )
        segs, _, detected_lang = transcribe_audio(slice_wav, model_size="base", clip_offset=w_start)
        all_segments.extend(segs)
    all_segments.sort(key=lambda s: s.start)
    return all_segments, detected_lang


def run_probe(
    url: str,
    out_dir: Path | None = None,
    lang: str = "en",
    max_windows: int = 15,
    full_transcript: bool = False,
    cookies_from_browser: str | None = None,
) -> dict[str, Any]:
    info = fetch_video_info(url, cookies_from_browser=cookies_from_browser)
    if info.get("is_live"):
        raise ClipError(
            "LIVE_STREAM_ACTIVE",
            "This YouTube broadcast is currently live. youtube-clipper supports finished videos and livestream VODs.",
            hint="Wait until the stream finishes or clip the VOD once YouTube publishes it.",
        )

    video_id = str(info.get("id") or "video")
    duration_sec = float(info.get("duration") or 0.0)
    work_dir = out_dir or (Path.cwd() / "clips" / video_id / "work")
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    heatmap_peaks = extract_heatmap_peaks(
        info.get("heatmap"),
        max_peaks=max_windows,
        duration_sec=duration_sec if duration_sec > 0 else None,
    )
    chapters = [
        {
            "title": str(ch.get("title", "")),
            "start": round(float(ch.get("start_time", 0.0)), 2),
            "end": round(float(ch.get("end_time", 0.0)), 2),
        }
        for ch in (info.get("chapters") or [])
    ]

    sub_path, caption_source, detected_lang = download_subtitles(
        url, cache_dir, lang=lang, cookies_from_browser=cookies_from_browser
    )
    segments: list[CaptionSegment] = []
    has_word_timestamps = False
    audio_path: Path | None = None
    audio_peaks: list[HeatmapPeak] = []

    if sub_path and caption_source == "youtube_json3":
        segments, words = parse_json3(sub_path.read_text(encoding="utf-8"))
        has_word_timestamps = len(words) > 0
    elif sub_path and caption_source == "youtube_vtt":
        segments = parse_vtt(sub_path.read_text(encoding="utf-8"))
    else:
        audio_path = download_audio_only(url, cache_dir, cookies_from_browser=cookies_from_browser)
        audio_peaks = detect_audio_peaks(audio_path, duration_sec=duration_sec, max_peaks=max_windows)
        candidate_peaks = heatmap_peaks or audio_peaks
        segments, detected_lang = _transcribe_fallback(
            audio_path, duration_sec, candidate_peaks, max_windows, work_dir
        )
        caption_source = "faster_whisper"

    signal_peaks = heatmap_peaks or audio_peaks
    blocks, truncated = build_transcript_blocks(
        segments=segments,
        heatmap_peaks=signal_peaks,
        duration_sec=duration_sec,
        max_windows=max_windows,
        full_transcript=full_transcript,
    )

    analysis = {
        "video": {
            "id": video_id,
            "url": url,
            "title": str(info.get("title", "")),
            "channel": str(info.get("channel") or info.get("uploader") or ""),
            "duration_sec": duration_sec,
            "upload_date": str(info.get("upload_date", "")),
            "is_live": False,
            "was_live": bool(info.get("was_live", False)),
        },
        "signals": {
            "caption_source": caption_source,
            "language": detected_lang,
            "has_heatmap": len(heatmap_peaks) > 0,
            "has_word_timestamps": has_word_timestamps,
            "truncated_to_windows": truncated,
            "chapters": chapters,
            "heatmap_peaks": [p.to_dict() for p in heatmap_peaks],
            "audio_peaks": [p.to_dict() for p in audio_peaks],
        },
        "transcript_blocks": [b.to_dict() for b in blocks],
        "cache_paths": {
            "subtitles": str(sub_path.resolve()) if sub_path else None,
            "audio": str(audio_path.resolve()) if audio_path else None,
        },
    }

    analysis_path = work_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "ok": True,
        "video_id": video_id,
        "title": analysis["video"]["title"],
        "duration_sec": duration_sec,
        "caption_source": caption_source,
        "has_heatmap": len(heatmap_peaks) > 0,
        "heatmap_peaks_count": len(heatmap_peaks),
        "transcript_blocks_count": len(blocks),
        "analysis_path": str(analysis_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe a YouTube URL for metadata, heatmap peaks, and captions.")
    parser.add_argument("url")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-windows", type=int, default=15)
    parser.add_argument("--full-transcript", action="store_true")
    parser.add_argument("--cookies-from-browser", default=None)
    args = parser.parse_args(argv)

    try:
        summary = run_probe(
            url=args.url,
            out_dir=args.out,
            lang=args.lang,
            max_windows=args.max_windows,
            full_transcript=args.full_transcript,
            cookies_from_browser=args.cookies_from_browser,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except ClipError as err:
        print(json.dumps(err.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\pytest.exe tests/test_probe_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/youtube.py skills/youtube-clipper/scripts/probe.py tests/test_probe_cli.py
git commit -m "feat: add yt-dlp wrapper and Stage 1 probe CLI"
```

---

### Task 7: Scene-Aware Smoothed Face Tracking (`lib/tracking.py`)

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/tracking.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `lib.models.TrackingPlan`
- Produces:
  - `compute_vertical_crop_width(src_w: int, src_h: int) -> int` (even integer `<= src_w`, target `9/16 * src_h`)
  - `detect_scene_cuts(histograms: list, threshold: float = 0.45) -> set[int]`
  - `smooth_crop_x0(face_centers: list[float | None], scene_cuts: set[int], src_w: int, crop_w: int, alpha: float = 0.22, deadzone_px: float = 18.0) -> list[int]`
  - `interpolate_x_positions(sample_x0: list[int], sample_indices: list[int], total_frames: int) -> list[int]`
  - `analyze_video_tracking(video_path: Path, sample_fps: float = 5.0) -> TrackingPlan`

- [ ] **Step 1: Write the failing test `tests/test_tracking.py`**

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.tracking import (  # noqa: E402
    compute_vertical_crop_width,
    detect_scene_cuts,
    interpolate_x_positions,
    smooth_crop_x0,
)


def test_compute_vertical_crop_width_is_even_and_clamped() -> None:
    assert compute_vertical_crop_width(1920, 1080) == 608
    assert compute_vertical_crop_width(1280, 720) == 404
    assert compute_vertical_crop_width(500, 1000) == 500


def test_smooth_crop_applies_deadzone_ema_and_snaps_on_scene_cut() -> None:
    # 1920 wide, crop_w=608 -> valid x0 in [0, 1312], center x0 = 656
    centers: list[float | None] = [500.0, 505.0, 510.0, 1500.0, 1500.0]
    cuts = {3}
    x0s = smooth_crop_x0(centers, cuts, src_w=1920, crop_w=608, alpha=0.25, deadzone_px=15.0)
    # Frame 0 snaps to target 500 - 304 = 196; frames 1-2 stay in deadzone
    assert x0s[0] == 196
    assert x0s[1] == 196
    assert x0s[2] == 196
    # Scene cut at index 3 snaps immediately to 1500 - 304 = 1196 instead of lagging via EMA
    assert x0s[3] == 1196
    assert x0s[4] == 1196


def test_detect_scene_cuts_and_interpolation() -> None:
    h_a = np.zeros((16, 16), dtype=np.float32)
    h_a[0, 0] = 1.0
    h_b = np.zeros((16, 16), dtype=np.float32)
    h_b[15, 15] = 1.0
    cuts = detect_scene_cuts([h_a, h_a, h_b, h_b], threshold=0.45)
    assert cuts == {2}

    interp = interpolate_x_positions([100, 200], [0, 4], total_frames=6)
    assert interp == [100, 125, 150, 175, 200, 200]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\pytest.exe tests/test_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.tracking'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/tracking.py`**

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lib.models import TrackingPlan


def compute_vertical_crop_width(src_w: int, src_h: int) -> int:
    target = int(round(src_h * 9.0 / 16.0))
    clamped = min(src_w, max(2, target))
    return clamped if clamped % 2 == 0 else max(2, clamped - 1)


def detect_scene_cuts(histograms: list[np.ndarray], threshold: float = 0.45) -> set[int]:
    cuts: set[int] = set()
    for i in range(1, len(histograms)):
        dist = float(cv2.compareHist(histograms[i - 1], histograms[i], cv2.HISTCMP_BHATTACHARYYA))
        if dist >= threshold:
            cuts.add(i)
    return cuts


def smooth_crop_x0(
    face_centers: list[float | None],
    scene_cuts: set[int],
    src_w: int,
    crop_w: int,
    alpha: float = 0.22,
    deadzone_px: float = 18.0,
) -> list[int]:
    max_x0 = max(0, src_w - crop_w)
    default_center = src_w / 2.0
    if not face_centers:
        return []

    cur_center = face_centers[0] if face_centers[0] is not None else default_center
    x0s: list[int] = []

    for idx, det in enumerate(face_centers):
        if idx in scene_cuts:
            cur_center = det if det is not None else default_center
        elif det is not None:
            if abs(det - cur_center) > deadzone_px:
                cur_center = (1.0 - alpha) * cur_center + alpha * det

        x0 = int(round(cur_center - crop_w / 2.0))
        x0s.append(max(0, min(max_x0, x0)))

    return x0s


def interpolate_x_positions(
    sample_x0: list[int],
    sample_indices: list[int],
    total_frames: int,
    scene_cuts: set[int] | None = None,
) -> list[int]:
    if total_frames <= 0:
        return []
    if not sample_x0:
        return [0] * total_frames

    cuts = scene_cuts or set()
    out = [sample_x0[0]] * total_frames
    for i in range(len(sample_indices) - 1):
        f0, f1 = sample_indices[i], sample_indices[i + 1]
        x0, x1 = sample_x0[i], sample_x0[i + 1]
        span = max(1, f1 - f0)
        is_cut = (i + 1) in cuts
        for f in range(f0, min(f1, total_frames)):
            if is_cut:
                out[f] = x0
            else:
                t = (f - f0) / span
                out[f] = int(round(x0 + t * (x1 - x0)))
    for f in range(sample_indices[-1], total_frames):
        out[f] = sample_x0[-1]
    return out


def analyze_video_tracking(video_path: Path, sample_fps: float = 5.0) -> TrackingPlan:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for tracking: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    crop_w = compute_vertical_crop_width(src_w, src_h)

    frontal = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    profile = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"))

    step = max(1, int(round(fps / max(0.5, sample_fps))))
    sample_indices: list[int] = []
    face_centers: list[float | None] = []
    histograms: list[np.ndarray] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            scale = min(1.0, 640.0 / max(1, src_w))
            small = cv2.resize(frame, (max(1, int(src_w * scale)), max(1, int(src_h * scale)))) if scale < 1.0 else frame
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            histograms.append(hist)

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))
            if len(faces) == 0:
                faces = profile.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(36, 36))

            if len(faces) > 0:
                largest = max(faces, key=lambda box: int(box[2]) * int(box[3]))
                cx_small = float(largest[0] + largest[2] / 2.0)
                face_centers.append(cx_small / scale)
            else:
                face_centers.append(None)
            sample_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    actual_total = max(total_frames, frame_idx)
    detected_count = sum(1 for c in face_centers if c is not None)
    face_ratio = round(detected_count / max(1, len(face_centers)), 3)

    cuts = detect_scene_cuts(histograms)
    sample_x0 = smooth_crop_x0(face_centers, cuts, src_w=src_w, crop_w=crop_w)
    x0_per_frame = interpolate_x_positions(sample_x0, sample_indices, actual_total, scene_cuts=cuts)

    return TrackingPlan(
        src_width=src_w,
        src_height=src_h,
        crop_width=crop_w,
        fps=fps,
        total_frames=actual_total,
        face_ratio=face_ratio,
        x0_per_frame=x0_per_frame,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.\.venv\Scripts\pytest.exe tests/test_tracking.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/tracking.py tests/test_tracking.py
git commit -m "feat: add scene-aware smoothed OpenCV face tracking"
```

---

### Task 8: FFmpeg Filtergraphs, Windows-Safe Pipe Renderer (`lib/ffmpeg_ops.py`) & Stage 3 `render.py`

**Files:**
- Create: `skills/youtube-clipper/scripts/lib/ffmpeg_ops.py`
- Create: `skills/youtube-clipper/scripts/render.py`
- Test: `tests/test_ffmpeg_ops.py`
- Test: `tests/test_render_integration.py`

**Interfaces:**
- Consumes: `lib.models.*`, `lib.youtube.download_video_section`, `lib.subtitles.*`, `lib.audio_peaks.transcribe_audio`, `lib.captions.generate_ass`, `lib.tracking.analyze_video_tracking`
- Produces in `lib/ffmpeg_ops.py`:
  - `probe_media(path: Path) -> MediaInfo`
  - `trim_exact(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path`
  - `build_static_filtergraph(layout: str, src_w: int, src_h: int, has_captions: bool, split_cam_corner: str = "br") -> str`
  - `render_static_layout_clip(trimmed_path: Path, out_path: Path, layout: str, media: MediaInfo, ass_filename: str | None, work_dir: Path, split_cam_corner: str = "br") -> Path`
  - `render_face_tracked_clip(trimmed_path: Path, out_path: Path, tracking: TrackingPlan, media: MediaInfo, ass_filename: str | None, work_dir: Path) -> Path`
- Produces in `render.py`:
  - `run_render(plan_path: Path, out_dir: Path | None = None, local_source: Path | None = None, keep_temp: bool = False) -> dict`

- [ ] **Step 1: Write failing unit test `tests/test_ffmpeg_ops.py` and real-FFmpeg integration test `tests/test_render_integration.py`**

Create `tests/test_ffmpeg_ops.py`:
```python
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.ffmpeg_ops import build_static_filtergraph  # noqa: E402


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
```

Create `tests/test_render_integration.py`:
```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.ffmpeg_ops import probe_media  # noqa: E402
from render import run_render  # noqa: E402


def _make_synthetic_video(out_path: Path, duration: float = 4.0) -> Path:
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
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out_path),
    ]
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\pytest.exe tests/test_ffmpeg_ops.py tests/test_render_integration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.ffmpeg_ops'`

- [ ] **Step 3: Implement `skills/youtube-clipper/scripts/lib/ffmpeg_ops.py` and `skills/youtube-clipper/scripts/render.py`**

Create `skills/youtube-clipper/scripts/lib/ffmpeg_ops.py`:
```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2

from lib.models import ClipError, MediaInfo, TrackingPlan

AUDIO_LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"


def _parse_fps(rate_str: str) -> float:
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        return float(num) / max(1.0, float(den))
    return float(rate_str or 30.0)


def probe_media(path: Path) -> MediaInfo:
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
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ClipError("MEDIA_ERROR", f"No video stream found in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return MediaInfo(
        width=int(v_stream["width"]),
        height=int(v_stream["height"]),
        fps=round(_parse_fps(str(v_stream.get("r_frame_rate", "30/1"))), 3),
        duration=float(data.get("format", {}).get("duration", 0.0)),
        has_audio=has_audio,
    )


def trim_exact(src_path: Path, out_path: Path, start_offset: float, duration: float) -> Path:
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
    ass_suffix = f",ass={ass_filename}" if has_captions else ""

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
    ]
    if media.has_audio:
        cmd.extend(["-map", "0:a:0", "-af", AUDIO_LOUDNORM, "-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")
    cmd.extend(
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
            str(out_path.resolve()),
        ]
    )
    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
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
    vf_parts = ["scale=1080:1920:flags=lanczos", "setsar=1"]
    if ass_filename:
        vf_parts.append(f"ass={ass_filename}")
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
    ]
    if media.has_audio:
        cmd.extend(["-map", "1:a:0", "-af", AUDIO_LOUDNORM, "-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")
    cmd.extend(
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
            "-shortest",
            str(out_path.resolve()),
        ]
    )

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(work_dir))
    cap = cv2.VideoCapture(str(trimmed_path))
    if not cap.isOpened():
        proc.kill()
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
            cropped = frame[:, x0 : x0 + crop_w]
            if not cropped.flags["C_CONTIGUOUS"]:
                cropped = cropped.copy()
            proc.stdin.write(cropped.tobytes())
            frame_idx += 1
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()

    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    retcode = proc.wait()
    if retcode != 0:
        raise ClipError("FFMPEG_FAILED", f"Face-tracked render failed: {stderr_bytes.decode('utf-8', 'ignore')[:400]}")
    return out_path
```

Create `skills/youtube-clipper/scripts/render.py`:
```python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

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
    if not cache_path:
        return [], []
    p = Path(cache_path)
    if not p.exists():
        return [], []
    text = p.read_text(encoding="utf-8")
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
                tracking = analyze_video_tracking(trimmed, sample_fps=5.0)
                if clip.layout == "auto" and tracking.face_ratio < 0.30:
                    layout_used = "blurred-fit"
                    warnings.append(f"Auto layout switched to blurred-fit (face presence {tracking.face_ratio:.0%}).")
                else:
                    layout_used = "face-track"
                    if tracking.face_ratio == 0.0:
                        warnings.append("No faces detected; fell back to centered 9:16 crop. Consider blurred-fit for slides/gameplay.")

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
                    trimmed, out_mp4, layout_used, media, ass_filename, clip_temp, split_cam_corner=clip.split_cam_corner
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
```

- [ ] **Step 4: Run unit & real-FFmpeg integration tests**

Run: `.\.venv\Scripts\pytest.exe tests/test_ffmpeg_ops.py tests/test_render_integration.py -v`
Expected: PASS (all 4 tests green, all 4 layouts rendered & verified by `ffprobe`).

- [ ] **Step 5: Commit**

```bash
git add skills/youtube-clipper/scripts/lib/ffmpeg_ops.py skills/youtube-clipper/scripts/render.py tests/test_ffmpeg_ops.py tests/test_render_integration.py
git commit -m "feat: add Windows-safe FFmpeg renderer and Stage 3 render CLI"
```

---

### Task 9: Author `SKILL.md` & `references/clip-rubric.md` via Skill TDD (RED → GREEN → REFACTOR)

**Files:**
- Create: `skills/youtube-clipper/SKILL.md`
- Create: `skills/youtube-clipper/references/clip-rubric.md`

**Interfaces:**
- Consumes: `scripts/run.py` (`doctor`, `probe`, `render`), `analysis.json` schema, `clips_plan.json` schema

- [ ] **Step 1 (RED): Run baseline scenario WITHOUT the skill**

Dispatch a read-only subagent (`research` or `self` with no reference to `SKILL.md`) with a realistic prompt:
> "Clip the 3 best 45-second vertical 9:16 Shorts with bold word-highlighted captions from this 2-hour YouTube livestream VOD `https://www.youtube.com/watch?v=example123` on Windows. Outline the exact commands and pipeline you would execute right now."

Document exact baseline failures verbatim (e.g., downloading full 2-hour video via raw `yt-dlp`, passing raw Windows paths to `ffmpeg -vf ass=C:\...`, skipping YouTube Most Replayed heatmap, static unsmoothed center crop, missing loudness normalization, or improvising an external API key).

- [ ] **Step 2 (GREEN): Write `skills/youtube-clipper/references/clip-rubric.md` and `skills/youtube-clipper/SKILL.md`**

Create `skills/youtube-clipper/references/clip-rubric.md`:
```markdown
# Highlight Selection Rubric & Post-Kit Guide

## Scoring Rubric (100 Points)

Score candidate windows from `analysis.json` (`heatmap_peaks`, `audio_peaks`, `chapters`, and `transcript_blocks`):

| Criterion | Weight | What Scores High |
|---|---|---|
| **3-Second Hook** | 35 | Opens immediately with a bold claim, contrarian take, emotional reaction, surprising stat, or tension question—never "Hey guys" or slow throat-clearing. |
| **Standalone Payoff** | 30 | Complete micro-story or punchline understandable with zero context from earlier in the video. |
| **Signal Corroboration** | 20 | Overlaps a high YouTube Most Replayed `heatmap_score` (`>= 0.70`) or audio energy peak. |
| **Pacing & Boundaries** | 15 | 25–60 seconds (unless user requested another duration); starts cleanly at a sentence start and ends right after a complete payoff sentence. |

## Boundary & Overlap Rules
- Never overlap selected clips (`clip[i].end + 5s <= clip[i+1].start`).
- Trim filler intros ("uh, so basically...") by starting on the first high-energy clause.
- Add `0.3s` breathing room after the closing word so the final syllable is never clipped.

## Layout Selection Guide
- **`face-track` (default):** Podcasts, talking heads, interviews, IRL streams.
- **`auto`:** Unknown visual style—uses `face-track` when faces appear in `>= 30%` of frames, otherwise switches to `blurred-fit`.
- **`blurred-fit`:** Coding tutorials, slide decks, multi-person wide shots, or anything where cropping loses context.
- **`split`:** Gaming livestreams with a webcam overlay (`split_cam_corner`: `br`, `bl`, `tr`, or `tl`).
- **`landscape`:** User explicitly asks for 16:9 widescreen clips.

## `clips_plan.json` Schema
```json
{
  "video_url": "<youtube_url>",
  "video_id": "<id from analysis.json>",
  "video_title": "<title from analysis.json>",
  "cache_subtitles": "<cache_paths.subtitles or null>",
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
```

Create `skills/youtube-clipper/SKILL.md`:
```markdown
---
name: youtube-clipper
description: >-
  Use when the user asks to clip, cut, extract highlights, or create YouTube
  Shorts, Instagram Reels, or TikToks from a YouTube video, podcast, or
  finished livestream VOD URL.
---

# YouTube Clipper

Turn finished YouTube videos, podcasts, and livestream VODs into ready-to-post short clips with zero external API keys. All heavy media commands MUST run through [run.py](./scripts/run.py) so the isolated `.venv` (`yt-dlp`, `opencv`, `faster-whisper`) and Windows-safe FFmpeg paths are used.

## Workflow

1. **Verify setup (once per session if unsure):**
   ```powershell
   python "<SKILL_DIR>/scripts/run.py" doctor --json
   ```
   If `"ok": false`, report the missing tool from `checks` and stop.

2. **Probe without downloading the video:**
   ```powershell
   python "<SKILL_DIR>/scripts/run.py" probe "<YOUTUBE_URL>"
   ```
   Optional flags: `--out <work_dir>`, `--lang <code>`, `--full-transcript`, `--cookies-from-browser chrome|edge|firefox`.
   - If exit code is `2` with `LIVE_STREAM_ACTIVE`, inform the user that active in-progress broadcasts must finish before clipping.
   - If `BOT_CHECK` or `AUTH_REQUIRED`, ask whether to retry with `--cookies-from-browser`.

3. **Select clips (Agent Brain):**
   Read `analysis.json` (path printed by `probe`) and [clip-rubric.md](./references/clip-rubric.md).
   - Pick top `N` non-overlapping moments (default `3`, `25–60s` each) using heatmap peaks, audio peaks, and transcript hooks.
   - Choose the layout matching the request or content (`face-track` default, `auto`, `blurred-fit`, `split`, or `landscape`).
   - If the user asked to preview/choose candidates first, present the shortlist with timestamps and wait; otherwise write `clips_plan.json` into the work folder and proceed immediately.

4. **Render targeted clips:**
   ```powershell
   python "<SKILL_DIR>/scripts/run.py" render --plan "<WORK_DIR>/clips_plan.json"
   ```
   Optional flags: `--out <output_dir>`, `--local-source <video_path>`.

5. **Deliver results:**
   Read `render_report.json` and `post_kit.md`, then return clickable `file://` links to each rendered `.mp4` and `post_kit.md`, summarizing titles, timestamps, layouts, and any warnings.

## Red Flags — Never Do These

| Mistake | Do This Instead |
|---|---|
| Downloading the full multi-hour video with raw `yt-dlp` | Always run `run.py probe` first; `run.py render` downloads only winning sections. |
| Running raw `ffmpeg -vf ass=C:\...` on Windows | Always use `run.py render`, which runs FFmpeg with relative `ass=captions.ass`. |
| Cutting blindly on heatmap peak timestamps | Snap `start`/`end` to clean sentence boundaries in `analysis.json`. |
| Calling external OpenAI/Gemini APIs for scoring | Score highlights and write post copy directly from `analysis.json`. |
```

- [ ] **Step 3 (GREEN & REFACTOR): Verify word count and run WITH-skill subagent check**

Check `SKILL.md` word count is `<500` words.
Dispatch a fresh subagent given `SKILL.md` + `references/clip-rubric.md` with the exact same baseline prompt from Step 1. Verify it uses `run.py probe` -> reads `analysis.json` + `clip-rubric.md` -> writes `clips_plan.json` -> runs `run.py render`, with zero raw full-video downloads and zero raw Windows FFmpeg `ass=C:\...` bugs. If any ambiguity or loophole surfaces, tighten `SKILL.md` and re-test.

- [ ] **Step 4: Commit**

```bash
git add skills/youtube-clipper/SKILL.md skills/youtube-clipper/references/clip-rubric.md
git commit -m "feat: add TDD-verified youtube-clipper SKILL.md and clip rubric"
```

---

### Task 10: Global Skill Installation, Cross-Workspace Smoke Check & Full Verification

**Files:**
- Verify: `C:\Users\Admin\.gemini\config\skills\youtube-clipper\SKILL.md`
- Verify: `C:\Users\Admin\.gemini\config\skills\youtube-clipper\scripts\.venv_python`

- [ ] **Step 1: Run `scripts/install_skill.py` to sync the skill globally**

Run: `python scripts/install_skill.py`
Expected: Exits 0 and prints `Synced skill to C:\Users\Admin\.gemini\config\skills\youtube-clipper`.

- [ ] **Step 2: Verify cross-workspace execution from outside the repository**

Run: `Push-Location C:\Users\Admin\Desktop; python "C:\Users\Admin\.gemini\config\skills\youtube-clipper\scripts\run.py" doctor --json; Pop-Location`
Expected: Exit code `0`, `"ok": true`, `"python"` pointing to `c:\Users\Admin\Desktop\clipper\.venv\Scripts\python.exe`.

- [ ] **Step 3: Run the complete test suite**

Run: `.\.venv\Scripts\pytest.exe -v`
Expected: All tests across `test_models_and_doctor.py`, `test_heatmap.py`, `test_subtitles.py`, `test_captions.py`, `test_audio_peaks.py`, `test_probe_cli.py`, `test_tracking.py`, `test_ffmpeg_ops.py`, and `test_render_integration.py` PASS with 0 failures.

- [ ] **Step 4: Verify clean git status**

Run: `git status`
Expected: Clean working tree (`.venv/`, `clips/`, `.venv_python` ignored by `.gitignore`).
