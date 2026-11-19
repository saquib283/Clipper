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
