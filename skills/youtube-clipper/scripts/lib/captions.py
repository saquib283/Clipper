from __future__ import annotations

from dataclasses import dataclass

from lib.models import WordTiming

# ASS color format is &HAABBGGRR&
ASS_WHITE = "&H00FFFFFF&"
ASS_YELLOW = "&H0000E6FF&"  # #FFE600 in BGR


@dataclass
class CaptionChunk:
    """A timed group of words displayed together as one caption phrase."""

    words: list[WordTiming]
    start: float
    end: float


def escape_ass_text(text: str) -> str:
    """Escape backslashes and ASS override braces in caption text."""
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc."""
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
    """Group word timings into short readable caption chunks by count, length, pause, and punctuation."""
    if not words:
        return []
    chunks: list[CaptionChunk] = []
    current: list[WordTiming] = []
    cur_len = 0

    def flush() -> None:
        nonlocal current, cur_len
        if current:
            chunks.append(CaptionChunk(words=current, start=current[0].start, end=current[-1].end))
            current = []
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
            flush()
        current.append(w)
        cur_len += word_len + (1 if len(current) > 1 else 0)
        if w.word.endswith((".", "!", "?")):
            flush()

    flush()
    return chunks


def _build_bold_highlight_events(chunks: list[CaptionChunk]) -> list[str]:
    events: list[str] = []
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
                    rendered_tokens.append(
                        f"{{\\c{ASS_YELLOW}\\fscx108\\fscy108}}{safe}{{\\c{ASS_WHITE}\\fscx100\\fscy100}}"
                    )
                else:
                    rendered_tokens.append(safe)
            line = " ".join(rendered_tokens)
            events.append(
                f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Default,,0,0,0,,{line}"
            )
    return events


def generate_ass(
    words: list[WordTiming],
    style: str = "bold-highlight",
    play_res_x: int = 1080,
    play_res_y: int = 1920,
) -> str:
    """Generate a complete styled ASS subtitle script from word-level timings."""
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
    if style == "clean":
        events = [
            f"Dialogue: 0,{format_ass_time(chunk.start)},{format_ass_time(chunk.end)},Default,,0,0,0,,"
            + " ".join(escape_ass_text(w.word) for w in chunk.words)
            for chunk in chunks
        ]
    else:
        events = _build_bold_highlight_events(chunks)

    return header + "\n" + "\n".join(events) + "\n"
