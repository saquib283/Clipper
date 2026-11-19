from __future__ import annotations

import sys
import wave
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "youtube-clipper" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from lib.models import MemeCue  # noqa: E402
from lib.sfx import (  # noqa: E402
    STARTER_SFX_NAMES,
    build_audio_mix_filtergraph,
    build_visual_effects_chain,
    discover_sfx_library,
    enforce_restraint,
    generate_starter_sfx_pack,
)


def test_generate_starter_sfx_pack_creates_valid_wavs(tmp_path: Path) -> None:
    generated = generate_starter_sfx_pack(tmp_path)
    assert len(generated) == len(STARTER_SFX_NAMES)
    for name in STARTER_SFX_NAMES:
        wav_path = tmp_path / f"{name}.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 44100
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 2000


def test_discover_sfx_library_prefers_custom_over_builtin(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    custom = tmp_path / "custom"
    builtin.mkdir()
    custom.mkdir()
    (builtin / "vine-boom.wav").write_bytes(b"builtin")
    (builtin / "sus.wav").write_bytes(b"builtin-sus")
    (custom / "vine-boom.mp3").write_bytes(b"custom-boom")
    (custom / "brother-eww.mp3").write_bytes(b"custom-eww")

    lib = discover_sfx_library(custom_dir=custom, builtin_dir=builtin)
    assert set(lib.keys()) == {"vine-boom", "sus", "brother-eww"}
    assert lib["vine-boom"] == custom / "vine-boom.mp3"
    assert lib["sus"] == builtin / "sus.wav"


def test_enforce_restraint_caps_count_and_spacing() -> None:
    cues = [
        MemeCue(at=1.0, sfx="alert", visual="zoom-punch"),
        MemeCue(at=3.0, sfx="vine-boom", visual="shake"),       # too close (<6s) -> dropped
        MemeCue(at=12.0, sfx="sus", visual="bw-flash"),
        MemeCue(at=25.0, sfx="metal-pipe", visual="red-flash"),
        MemeCue(at=40.0, sfx="womp-womp", visual="none"),       # exceeds 3/min cap -> dropped
        MemeCue(at=59.95, sfx="soft-pop", visual="none"),       # past end margin -> dropped
    ]
    kept, warnings = enforce_restraint(cues, clip_duration=60.0, min_gap_sec=6.0, max_per_minute=3)
    assert [c.at for c in kept] == [1.0, 12.0, 25.0]
    assert len(warnings) >= 1


def test_build_visual_and_audio_filtergraphs() -> None:
    cues = [
        MemeCue(at=2.0, sfx="vine-boom", visual="zoom-punch", volume=0.4, duration=0.35),
        MemeCue(at=10.0, sfx="sus", visual="bw-flash", volume=0.35, duration=0.6),
        MemeCue(at=18.0, sfx="soft-pop", visual="red-flash", volume=0.3, duration=0.35),
        MemeCue(at=26.0, sfx="metal-pipe", visual="shake", volume=0.35, duration=0.35),
    ]
    vfg = build_visual_effects_chain("[vbase]", "[vfx]", cues, width=1080, height=1920)
    assert "hue=s=0:enable=" in vfg
    assert "colorbalance=" in vfg
    assert "sin(t*75)" in vfg
    assert "[vfx]" in vfg

    afg = build_audio_mix_filtergraph(
        base_audio_input="0:a:0",
        sfx_entries=[(1, cues[0], Path("vine-boom.wav")), (2, cues[2], Path("soft-pop.wav"))],
        has_base_audio=True,
    )
    assert "adelay=2000|2000" in afg
    assert "adelay=18000|18000" in afg
    assert "lowpass=f=9000" in afg
    assert "amix=inputs=3:duration=first:dropout_transition=0:normalize=0" in afg
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in afg

