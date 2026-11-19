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
    assert format_ass_time(-1.0) == "0:00:00.00"


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


def test_group_words_splits_on_punctuation_max_chars_and_empty() -> None:
    assert group_words_into_chunks([]) == []
    words = [
        WordTiming("Hello.", 0.0, 0.3),
        WordTiming("Supercalifragilistic", 0.3, 0.8),
        WordTiming("expialidocious", 0.8, 1.3),
    ]
    chunks = group_words_into_chunks(words, max_words=4, max_chars=26, max_gap_sec=0.6)
    assert len(chunks) == 3
    assert [w.word for w in chunks[0].words] == ["Hello."]
    assert [w.word for w in chunks[1].words] == ["Supercalifragilistic"]
    assert [w.word for w in chunks[2].words] == ["expialidocious"]


def test_generate_ass_bold_highlight_and_clean_styles() -> None:
    words = [
        WordTiming("Watch", 0.0, 0.4),
        WordTiming("{this}", 0.4, 0.8),
    ]
    ass_bold = generate_ass(words, style="bold-highlight", play_res_x=1080, play_res_y=1920)
    assert "PlayResX: 1080" in ass_bold
    assert "PlayResY: 1920" in ass_bold
    assert "&H0000E6FF&" in ass_bold  # yellow active highlight in ASS BGR
    assert r"\{THIS\}" in ass_bold    # braces escaped (bold-highlight uppercases tokens)
    assert ",360,1\n" in ass_bold     # MarginV=360 for vertical 1080x1920
    assert ass_bold.count("Dialogue:") == 2

    ass_clean = generate_ass(words, style="clean", play_res_x=1280, play_res_y=720)
    assert "PlayResX: 1280" in ass_clean
    assert r"\{this\}" in ass_clean   # braces escaped, original casing preserved
    assert ",60,1\n" in ass_clean     # MarginV=60 for landscape
    assert ass_clean.count("Dialogue:") == 1


def test_generate_ass_none_style_and_empty_words() -> None:
    words = [WordTiming("Hello", 0.0, 0.5)]
    ass_none = generate_ass(words, style="none", play_res_x=1080, play_res_y=1920)
    assert "PlayResX: 1080" in ass_none
    assert "Dialogue:" not in ass_none

    ass_empty = generate_ass([], style="bold-highlight", play_res_x=1080, play_res_y=1920)
    assert "Dialogue:" not in ass_empty
