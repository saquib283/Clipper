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
