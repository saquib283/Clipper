from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_LAYOUTS = {"face-track", "blurred-fit", "split", "landscape", "auto"}
VALID_CORNERS = {"br", "bl", "tr", "tl"}
VALID_CAPTION_STYLES = {"bold-highlight", "clean", "none"}
VALID_VISUAL_EFFECTS = {"none", "zoom-punch", "shake", "bw-flash", "red-flash"}


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
class MemeCue:
    at: float
    sfx: str | None = None
    visual: str = "none"
    volume: float = 0.35
    duration: float = 0.4

    def validate(self, clip_id: str) -> None:
        if self.at < 0.0:
            raise ClipError("INVALID_PLAN", f"Clip {clip_id}: effect 'at' ({self.at}) must be >= 0.")
        if self.visual not in VALID_VISUAL_EFFECTS:
            raise ClipError(
                "INVALID_PLAN",
                f"Clip {clip_id}: unsupported visual effect '{self.visual}'. Valid: {sorted(VALID_VISUAL_EFFECTS)}",
            )
        if not (0.0 < self.volume <= 1.5):
            raise ClipError("INVALID_PLAN", f"Clip {clip_id}: effect volume ({self.volume}) must be in (0, 1.5].")
        if not (0.1 <= self.duration <= 2.0):
            raise ClipError("INVALID_PLAN", f"Clip {clip_id}: effect duration ({self.duration}) must be in [0.1, 2.0].")
        if self.sfx is not None:
            cleaned = self.sfx.strip().lower()
            self.sfx = cleaned or None


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
    effects: list[MemeCue] = field(default_factory=list)

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
        for cue in self.effects:
            cue.validate(self.id)
        self.slug = sanitize_slug(self.slug or self.id, fallback=self.id)


@dataclass
class ClipsPlan:
    video_url: str
    video_id: str
    video_title: str
    clips: list[ClipSpec]
    cache_subtitles: str | None = None
    cache_live_manifest: str | None = None
    cookies_from_browser: str | None = None

    @classmethod
    def from_json_file(cls, path: Path) -> "ClipsPlan":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ClipError("INVALID_PLAN", f"Failed to parse plan JSON at {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ClipError("INVALID_PLAN", "Plan JSON root must be an object.")

        raw_clips = raw.get("clips")
        if not isinstance(raw_clips, list) or not raw_clips:
            raise ClipError("INVALID_PLAN", "Plan must contain a non-empty 'clips' list.")

        clips: list[ClipSpec] = []
        for idx, item in enumerate(raw_clips, start=1):
            if not isinstance(item, dict):
                raise ClipError("INVALID_PLAN", f"Invalid clip entry #{idx}: expected an object.")
            raw_hashtags = item.get("hashtags")
            if raw_hashtags is None:
                hashtags_list: list[Any] = []
            elif isinstance(raw_hashtags, list):
                hashtags_list = raw_hashtags
            else:
                raise ClipError("INVALID_PLAN", f"Invalid clip entry #{idx}: 'hashtags' must be a list.")

            raw_effects = item.get("effects")
            if raw_effects is None:
                effects_raw_list: list[Any] = []
            elif isinstance(raw_effects, list):
                effects_raw_list = raw_effects
            else:
                raise ClipError("INVALID_PLAN", f"Invalid clip entry #{idx}: 'effects' must be a list.")

            try:
                parsed_effects: list[MemeCue] = []
                for e_idx, eff in enumerate(effects_raw_list, start=1):
                    if not isinstance(eff, dict):
                        raise ClipError("INVALID_PLAN", f"Invalid clip entry #{idx} effect #{e_idx}: expected an object.")
                    parsed_effects.append(
                        MemeCue(
                            at=float(eff["at"]),
                            sfx=None if eff.get("sfx") is None else str(eff.get("sfx")),
                            visual=str(eff.get("visual", "none")),
                            volume=float(eff.get("volume", 0.35)),
                            duration=float(eff.get("duration", 0.4)),
                        )
                    )
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
                    hashtags=[str(h) for h in hashtags_list],
                    why_selected=str(item.get("why_selected", "")),
                    effects=parsed_effects,
                )
            except ClipError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ClipError("INVALID_PLAN", f"Invalid clip entry #{idx}: {exc}") from exc
            spec.validate()
            clips.append(spec)

        return cls(
            video_url=str(raw.get("video_url", "")),
            video_id=str(raw.get("video_id", "video")),
            video_title=str(raw.get("video_title", "")),
            clips=clips,
            cache_subtitles=raw.get("cache_subtitles"),
            cache_live_manifest=raw.get("cache_live_manifest"),
            cookies_from_browser=raw.get("cookies_from_browser"),
        )
