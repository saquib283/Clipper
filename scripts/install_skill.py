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
            cmd + ["-m", "pip", "install", "--dry-run", "--only-binary=:all:", "faster-whisper", "opencv-python-headless>=4.10.0,<5", "yt-dlp"],
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
