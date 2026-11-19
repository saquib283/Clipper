from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from lib.models import MemeCue

SAMPLE_RATE = 44100
AUDIO_LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
SFX_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}

STARTER_SFX_NAMES = (
    "vine-boom",
    "metal-pipe",
    "alert",
    "sus",
    "rizz",
    "womp-womp",
    "record-scratch",
    "apple-pay",
    "shutter-snap",
    "soft-whoosh",
    "soft-pop",
)

SUBTLE_SFX = {"shutter-snap", "soft-whoosh", "soft-pop", "apple-pay", "rizz"}


def _write_wav_mono(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    peak = float(np.max(np.abs(samples))) if samples.size else 1.0
    if peak > 1e-6:
        samples = samples / peak * 0.89  # -1 dBFS ceiling
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return path


def _lowpass_smooth(x: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return x
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    return np.convolve(x, kernel, mode="same")


def _synth_vine_boom() -> np.ndarray:
    dur = 1.05
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq = 34.0 + 96.0 * np.exp(-t * 14.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    sub = np.sin(phase) * np.exp(-t * 3.1)
    harm = 0.28 * np.sin(2.0 * phase) * np.exp(-t * 6.0)
    rng = np.random.default_rng(42)
    punch = _lowpass_smooth(rng.standard_normal(t.size), 12) * np.exp(-t * 45.0) * 0.5
    attack = np.minimum(1.0, t / 0.004)
    return np.tanh((sub + harm + punch) * 1.45) * attack


def _synth_metal_pipe() -> np.ndarray:
    dur = 1.3
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    base = 415.0
    ratios = [(1.0, 1.0, 2.2), (2.76, 0.75, 2.8), (5.40, 0.55, 3.6), (8.93, 0.35, 4.5), (11.42, 0.25, 5.5)]
    sig = np.zeros_like(t)
    for r, amp, dec in ratios:
        sig += amp * np.sin(2.0 * np.pi * base * r * t) * np.exp(-t * dec)
    rng = np.random.default_rng(7)
    clang = rng.standard_normal(t.size) * np.exp(-t * 65.0) * 0.9
    return np.tanh((sig + clang) * 1.1)


def _synth_alert() -> np.ndarray:
    dur = 0.42
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sig = np.zeros_like(t)
    m1 = t < 0.075
    t1 = t[m1]
    sig[m1] = (np.sin(2 * np.pi * 659.25 * t1) + 0.4 * np.sin(4 * np.pi * 659.25 * t1)) * np.exp(-t1 * 12.0)
    m2 = t >= 0.075
    t2 = t[m2] - 0.075
    env2 = np.minimum(1.0, t2 / 0.004) * np.exp(-t2 * 7.5)
    sig[m2] = (
        np.sin(2 * np.pi * 987.77 * t2)
        + 0.45 * np.sin(2 * np.pi * 1975.53 * t2)
        + 0.2 * np.sin(2 * np.pi * 2963.3 * t2)
    ) * env2
    return np.tanh(sig * 1.2)


def _synth_sus() -> np.ndarray:
    dur = 1.0
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    swell = np.sin(np.pi * np.minimum(1.0, t / 0.85)) ** 1.4
    tail = np.where(t > 0.85, np.exp(-(t - 0.85) * 35.0), 1.0)
    sub = (np.sin(2 * np.pi * 55.0 * t) + 0.8 * np.sin(2 * np.pi * 57.5 * t)) * 0.7
    tritone = (np.sin(2 * np.pi * 220.0 * t) + 0.75 * np.sin(2 * np.pi * 311.13 * t)) * 0.35 * (t / dur)
    return np.tanh((sub + tritone) * swell * tail * 1.2)


def _synth_rizz() -> np.ndarray:
    dur = 0.95
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    notes = [349.23, 440.0, 523.25, 659.25, 783.99]  # Fmaj9
    sig = np.zeros_like(t)
    for idx, freq in enumerate(notes):
        onset = idx * 0.045
        mask = t >= onset
        dt = t[mask] - onset
        env = np.minimum(1.0, dt / 0.008) * np.exp(-dt * 3.2)
        sig[mask] += (np.sin(2 * np.pi * freq * dt) + 0.25 * np.sin(4 * np.pi * freq * dt)) * env
    return sig


def _synth_womp_womp() -> np.ndarray:
    notes = [(293.66, 0.26, False), (277.18, 0.26, False), (261.63, 0.26, False), (246.94, 0.65, True)]
    parts: list[np.ndarray] = []
    for freq, dur, vibrato in notes:
        t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        f = freq * (1.0 - 0.06 * (t / dur)) if vibrato else np.full_like(t, freq)
        if vibrato:
            f = f + 4.5 * np.sin(2 * np.pi * 6.5 * t)
        phase = 2.0 * np.pi * np.cumsum(f) / SAMPLE_RATE
        # Warm brassy tone via odd/even harmonics
        wave_sig = (
            np.sin(phase)
            + 0.45 * np.sin(2 * phase)
            + 0.25 * np.sin(3 * phase)
            + 0.12 * np.sin(4 * phase)
        )
        env = np.minimum(1.0, t / 0.015) * np.exp(-t * (2.6 if vibrato else 3.8))
        parts.append(wave_sig * env)
    return np.concatenate(parts)


def _synth_record_scratch() -> np.ndarray:
    dur = 0.45
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq = np.where(
        t < 0.18,
        1500.0 - 6500.0 * t,
        np.where(t < 0.28, 330.0 + 5500.0 * (t - 0.18), 880.0 - 4200.0 * (t - 0.28)),
    )
    freq = np.maximum(90.0, freq)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    rng = np.random.default_rng(13)
    grit = _lowpass_smooth(rng.standard_normal(t.size), 5) * 0.45
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 4.5)
    return (np.sin(phase) * 0.7 + grit) * env


def _synth_apple_pay() -> np.ndarray:
    dur = 0.48
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sig = np.zeros_like(t)
    m1 = t < 0.11
    t1 = t[m1]
    sig[m1] = (np.sin(2 * np.pi * 739.99 * t1) + 0.3 * np.sin(2 * np.pi * 1479.98 * t1)) * np.exp(-t1 * 14.0)
    m2 = t >= 0.09
    t2 = t[m2] - 0.09
    env2 = np.minimum(1.0, t2 / 0.004) * np.exp(-t2 * 9.0)
    sig[m2] += (np.sin(2 * np.pi * 987.77 * t2) + 0.35 * np.sin(2 * np.pi * 1975.53 * t2)) * env2
    return sig


def _synth_shutter_snap() -> np.ndarray:
    dur = 0.16
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    rng = np.random.default_rng(99)
    n = rng.standard_normal(t.size)
    c1 = _lowpass_smooth(n, 4) * np.exp(-t * 140.0)
    c2 = _lowpass_smooth(n, 6) * np.where(t >= 0.032, np.exp(-(t - 0.032) * 120.0), 0.0) * 0.75
    thud = np.sin(2 * np.pi * 95.0 * t) * np.exp(-t * 40.0) * 0.45
    return c1 + c2 + thud


def _synth_soft_whoosh() -> np.ndarray:
    dur = 0.32
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    rng = np.random.default_rng(21)
    noise = _lowpass_smooth(rng.standard_normal(t.size), 14)
    env = np.sin(np.pi * t / dur) ** 1.6
    return noise * env


def _synth_soft_pop() -> np.ndarray:
    dur = 0.14
    t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq = 190.0 + 360.0 * np.exp(-((t - 0.025) / 0.022) ** 2)
    phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 32.0)
    return np.sin(phase) * env


_SYNTH_BUILDERS = {
    "vine-boom": _synth_vine_boom,
    "metal-pipe": _synth_metal_pipe,
    "alert": _synth_alert,
    "sus": _synth_sus,
    "rizz": _synth_rizz,
    "womp-womp": _synth_womp_womp,
    "record-scratch": _synth_record_scratch,
    "apple-pay": _synth_apple_pay,
    "shutter-snap": _synth_shutter_snap,
    "soft-whoosh": _synth_soft_whoosh,
    "soft-pop": _synth_soft_pop,
}


def default_builtin_sfx_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "assets" / "sfx"


def default_custom_sfx_dir() -> Path:
    """Resolve workspace custom_sfx/ via .venv_python marker (works across workspaces)."""
    scripts_dir = Path(__file__).resolve().parent.parent
    marker = scripts_dir / ".venv_python"
    if marker.is_file():
        raw = marker.read_text(encoding="utf-8").strip()
        if raw:
            venv_py = Path(raw)
            # .../<repo>/.venv/Scripts/python.exe -> repo_root
            if len(venv_py.parents) >= 3:
                return venv_py.parents[2] / "custom_sfx"
    return Path.cwd() / "custom_sfx"


def generate_starter_sfx_pack(out_dir: Path | None = None, overwrite: bool = False) -> list[Path]:
    """Generate the built-in royalty-free starter SFX pack into out_dir."""
    target = out_dir or default_builtin_sfx_dir()
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in STARTER_SFX_NAMES:
        wav_path = target / f"{name}.wav"
        if overwrite or not wav_path.exists():
            samples = _SYNTH_BUILDERS[name]()
            _write_wav_mono(wav_path, samples)
        paths.append(wav_path)
    return paths


def discover_sfx_library(
    custom_dir: Path | None = None,
    builtin_dir: Path | None = None,
) -> dict[str, Path]:
    """Discover built-in starter SFX and custom drop-in files (custom overrides same-name built-ins)."""
    b_dir = builtin_dir or default_builtin_sfx_dir()
    c_dir = custom_dir or default_custom_sfx_dir()
    library: dict[str, Path] = {}

    for folder in (b_dir, c_dir):
        if not folder.is_dir():
            continue
        for item in sorted(folder.iterdir()):
            if item.is_file() and item.suffix.lower() in SFX_EXTENSIONS:
                library[item.stem.lower()] = item.resolve()
    return library


def enforce_restraint(
    effects: list[MemeCue],
    clip_duration: float,
    min_gap_sec: float = 6.0,
    max_per_minute: int = 3,
) -> tuple[list[MemeCue], list[str]]:
    """Enforce anti-bombardment guardrails: min spacing and max cues per minute."""
    if not effects:
        return [], []

    max_allowed = max(1, min(6, int(math.ceil(max(1.0, clip_duration) / 60.0)) * max_per_minute))
    ordered = sorted(effects, key=lambda c: c.at)
    kept: list[MemeCue] = []
    dropped_reasons: list[str] = []

    for cue in ordered:
        if cue.at >= max(0.0, clip_duration - 0.15):
            dropped_reasons.append(f"Dropped cue at {cue.at:.2f}s (too close to clip end).")
            continue
        if kept and (cue.at - kept[-1].at) < min_gap_sec:
            dropped_reasons.append(f"Dropped cue at {cue.at:.2f}s (<{min_gap_sec:.0f}s after previous cue).")
            continue
        if len(kept) >= max_allowed:
            dropped_reasons.append(f"Dropped cue at {cue.at:.2f}s (exceeded max {max_allowed} cues for clip).")
            continue
        kept.append(cue)

    return kept, dropped_reasons


def _between_expr(cues: list[MemeCue]) -> str:
    return "+".join(f"between(t,{c.at:.3f},{(c.at + c.duration):.3f})" for c in cues)


def build_visual_effects_chain(
    input_label: str,
    output_label: str,
    effects: list[MemeCue],
    width: int,
    height: int,
) -> str:
    """Build an FFmpeg filter_complex segment applying subtle visual meme punches before captions."""
    by_kind: dict[str, list[MemeCue]] = {}
    for cue in effects:
        if cue.visual and cue.visual != "none":
            by_kind.setdefault(cue.visual, []).append(cue)

    if not by_kind:
        return f"{input_label}null{output_label}"

    stages: list[str] = []
    cur = input_label
    step = 0

    def _next_label(is_last: bool) -> str:
        nonlocal step
        step += 1
        return output_label if is_last else f"[vfx_{step}]"

    ordered_kinds = [k for k in ("zoom-punch", "shake", "bw-flash", "red-flash") if k in by_kind]
    for idx, kind in enumerate(ordered_kinds):
        is_last = idx == len(ordered_kinds) - 1
        nxt = _next_label(is_last)
        expr = _between_expr(by_kind[kind])

        if kind == "bw-flash":
            stages.append(f"{cur}hue=s=0:enable='{expr}',eq=contrast=1.18:enable='{expr}'{nxt}")
        elif kind == "red-flash":
            stages.append(f"{cur}colorbalance=rs=0.35:gs=-0.15:bs=-0.15:enable='{expr}'{nxt}")
        elif kind == "zoom-punch":
            zw = max(2, int(round(width * 1.12)) // 2 * 2)
            zh = max(2, int(round(height * 1.12)) // 2 * 2)
            stages.append(
                f"{cur}split=2[zbase_{step}][zsrc_{step}];"
                f"[zsrc_{step}]scale={zw}:{zh}:flags=lanczos,crop={width}:{height}:(iw-{width})/2:(ih-{height})/2[zov_{step}];"
                f"[zbase_{step}][zov_{step}]overlay=0:0:enable='{expr}'{nxt}"
            )
        elif kind == "shake":
            sw = max(2, int(round(width * 1.05)) // 2 * 2)
            sh = max(2, int(round(height * 1.05)) // 2 * 2)
            stages.append(
                f"{cur}split=2[sbase_{step}][ssrc_{step}];"
                f"[ssrc_{step}]scale={sw}:{sh}:flags=lanczos[sov_{step}];"
                f"[sbase_{step}][sov_{step}]overlay=x='-(w-W)/2+12*sin(t*75)':y='-(h-H)/2+9*cos(t*90)':enable='{expr}'{nxt}"
            )
        cur = nxt

    return ";".join(stages)


def build_audio_mix_filtergraph(
    base_audio_input: str,
    sfx_entries: list[tuple[int, MemeCue, Path]],
    has_base_audio: bool,
    output_label: str = "[aout]",
) -> str:
    """Build FFmpeg filter_complex audio chain delaying and gain-staging SFX cues before loudnorm."""
    parts: list[str] = []
    mix_inputs: list[str] = []

    if has_base_audio:
        parts.append(f"[{base_audio_input}]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo[abase]")
        mix_inputs.append("[abase]")

    for idx, (input_idx, cue, _path) in enumerate(sfx_entries):
        ms = max(0, int(round(cue.at * 1000)))
        lp = "lowpass=f=9000," if (cue.sfx or "") in SUBTLE_SFX else ""
        lbl = f"[sfx_{idx}]"
        parts.append(
            f"[{input_idx}:a:0]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo,"
            f"{lp}alimiter=limit=0.85,volume={cue.volume:.3f},adelay={ms}|{ms}{lbl}"
        )
        mix_inputs.append(lbl)

    duration_mode = "first" if has_base_audio else "longest"
    parts.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration={duration_mode}:dropout_transition=0:normalize=0,"
        f"{AUDIO_LOUDNORM}{output_label}"
    )
    return ";".join(parts)

