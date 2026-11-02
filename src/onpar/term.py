"""Terminal presentation: the one seam between onpar's text and a human's terminal.

Every loop command already prints sentences that are pinned byte-for-byte: the
goldens capture stdout through a pipe and the guide embeds recordings made the
same way. So this module has one hard rule. **When stdout is not a TTY, or
``NO_COLOR`` is set, or ``ONPAR_PLAIN=1``, nothing here changes a byte.** Every
component returns its input untouched in that mode, and the stdout styler is
never installed.

On a TTY the same content gets colour from one palette (the names and values
are the brand tokens in ``site/assets/brand/tokens.css``), a header naming the
command, coloured status marks, the regret block in alarm red, intervals dimmed
after their point estimate, the ``next:`` footer boxed, and a drawn progress
bar on stderr. ``unstyle`` inverts all of it, and the tests hold every styled
golden to ``unstyle(styled) == plain``: decoration may be added, content never.

Zero dependencies: ANSI escapes and ``shutil.get_terminal_size``, which honours
``$COLUMNS``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TextIO

#: Brand tokens, by their token names. Truecolor: every terminal onpar's users
#: run (iTerm2, Terminal.app, kitty, WezTerm, VS Code, Windows Terminal) has it.
#: ponytail: no 256-colour fallback; add one if a user reports a terminal without
#: truecolor.
PALETTE: dict[str, str] = {
    "incumbent": "#f0a340",
    "candidate": "#2fd4c4",
    "proof": "#8fb3ff",
    "alarm": "#ff6b81",
    "muted": "#5b7385",
    "ink2": "#93aab8",
}

MARKS = {"✓": "proof", "✗": "alarm", "⚠": "incumbent", "·": "muted"}
BRAND = "▮▮"

ANSI = re.compile(r"\x1b\[[0-9;]*[mK]")
_RESET = "\x1b[0m"
#: The blank line under the header when the command's own output does not open
#: with one. A bare reset, so it is invisible yet distinguishable from a blank
#: line the command printed, and `unstyle` can drop exactly it.
SPACER = _RESET
_BOX = "╭╮╰╯─│"
#: A point estimate's interval: `[95–99]`, `[0.91–0.97]`, `[−3–+2]`.
_INTERVAL = re.compile(r"\[[-+−]?\d[\d.,]*%?\s?[–-]\s?[-+−]?\d[\d.,]*%?\]")
_RULE = re.compile(r"^\s*[-─=]{3,}\s*$")
_MARK_LINE = re.compile(r"^(\s*)([✓✗⚠·])(\s)")
_SIDE = re.compile(r"^(\s*)(incumbent|candidate)\b")
_HEADING = re.compile(r"^\s*[A-Z][A-Z0-9 /&()'—-]{3,}:?\s*$")
_CAPS_KEY = re.compile(r"^(\s*)([A-Z]{3,})( {2,}\S)")
_KEY = re.compile(r"^(\s*)([A-Z][a-z]+(?: [a-z]+)?:)(\s)")
#: Lines whose opening words carry the verdict, and the colour they get. The
#: ``block`` ones colour every line after them until a blank line: the regret
#: block and the must-stay block are read as one unit.
_LEADS: tuple[tuple[str, str, bool], ...] = (
    ("REGRET", "alarm", True),
    ("MUST STAY", "alarm", True),
    ("RECEIPT VOID", "alarm", True),
    ("[void]", "alarm", False),
    ("NOT PROVEN", "incumbent", False),
    ("NOT FEASIBLE", "incumbent", False),
    ("receipt still holds", "proof", False),
    ("[note]", "ink2", False),
)


def opens_block(text: str) -> bool:
    """Whether this line starts an alarm block that runs to the next blank line."""
    t = text.lstrip()
    return any(block and t.startswith(lead) for lead, _, block in _LEADS)


def plain_requested() -> bool:
    """Whether the environment asked for plain text regardless of the terminal."""
    return (
        "NO_COLOR" in os.environ
        or os.environ.get("ONPAR_PLAIN") == "1"
        or os.environ.get("TERM") == "dumb"
    )


def is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # no isatty, or a closed stream
        return False


def strip(s: str) -> str:
    """Remove ANSI escapes."""
    return ANSI.sub("", s)


class Term:
    """Styling for one stream. ``Term(None)`` follows ``sys.stdout`` as it is
    *now*, so a test's ``redirect_stdout`` is honoured; ``on`` may be forced."""

    def __init__(self, stream: TextIO | None = None, *, on: bool | None = None) -> None:
        self._stream = stream
        self._on = on

    @property
    def stream(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stdout

    @property
    def on(self) -> bool:
        if self._on is not None:
            return self._on
        return not plain_requested() and is_tty(self.stream)

    @property
    def width(self) -> int:
        return shutil.get_terminal_size((100, 24)).columns

    # -- components: each returns its plain input when styling is off --------

    def paint(self, text: str, role: str = "", *, bold: bool = False, dim: bool = False) -> str:
        if not self.on or not text:
            return text
        codes = []
        if bold:
            codes.append("1")
        if dim:
            codes.append("2")
        if role:
            h = PALETTE[role].lstrip("#")
            codes.append(f"38;2;{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}")
        return f"\x1b[{';'.join(codes)}m{text}{_RESET}" if codes else text

    def badge(self, mark: str) -> str:
        """A status mark (✓ ✗ ⚠ ·) in its colour."""
        return self.paint(mark, MARKS.get(mark, ""), bold=mark != "·")

    def rule(self, width: int, char: str = "─") -> str:
        return self.paint(char * width, "muted")

    def title(self, command: str, purpose: str) -> str:
        """The per-command header: brand mark, command, one-line purpose."""
        mark = self.paint(BRAND[0], "incumbent") + self.paint(BRAND[1], "candidate")
        name = self.paint(f"onpar {command}", bold=True)
        return f"  {mark} {name} {self.paint(f'· {purpose}', 'muted')}"

    def kv(self, key: str, value: str, width: int = 0) -> str:
        return f"{self.paint(key.ljust(width), 'ink2')}{value}"

    def table(
        self,
        header: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        indent: str = "",
        right: Sequence[bool] = (),
    ) -> list[str]:
        """Aligned columns, two spaces apart, a rule under the header.

        Widths come from the plain text, so styled cells (``paint``) align too.
        """
        grid = [list(header), *(list(r) for r in rows)]
        n = max(len(r) for r in grid)
        widths = [max(len(strip(r[i])) if i < len(r) else 0 for r in grid) for i in range(n)]

        def line(cells: list[str], head: bool = False) -> str:
            parts = []
            for i, w in enumerate(widths):
                cell = cells[i] if i < len(cells) else ""
                pad = " " * (w - len(strip(cell)))
                is_right = i < len(right) and right[i]
                parts.append(pad + cell if is_right else cell + pad)
            text = "  ".join(parts).rstrip()
            return indent + (self.paint(text, bold=True) if head else text)

        total = sum(widths) + 2 * (n - 1)
        return [line(grid[0], True), indent + self.rule(total), *(line(r) for r in grid[1:])]

    def box(self, lines: Sequence[str], role: str = "candidate") -> list[str]:
        """A closed box around lines that share a leading indent. The right edge
        is dropped when the box would not fit the terminal."""
        if not self.on or not lines:
            return list(lines)
        indent = min(len(ln) - len(ln.lstrip()) for ln in lines if ln.strip())
        inner = [ln[indent:].rstrip() for ln in lines]
        w = max(len(strip(s)) for s in inner)
        pad = " " * indent
        closed = indent + w + 4 <= self.width
        edge = self.paint("│", role)
        top = "╭" + "─" * (w + 2) + ("╮" if closed else "")
        bot = "╰" + "─" * (w + 2) + ("╯" if closed else "")
        body = [
            f"{pad}{edge} {s}{' ' * (w - len(strip(s))) + ' ' + edge if closed else ''}"
            for s in inner
        ]
        return [pad + self.paint(top, role), *body, pad + self.paint(bot, role)]

    def progress(self, done: int, total: int, text: str, *, bar: int = 20) -> str:
        """One in-place progress line, without the leading ``\\r``.

        Plain: ``text`` exactly, the line pipes and logs already get. Styled: a
        drawn bar before it, clipped to the terminal and clearing to the end of
        the line, so a shorter redraw leaves no residue.
        """
        if not self.on:
            return text
        frac = min(1.0, done / total) if total else 0.0
        fill = round(frac * bar)
        drawn = self.paint("█" * fill, "candidate") + self.paint("░" * (bar - fill), "muted")
        body = text.strip()
        room = max(0, self.width - bar - 6)
        if len(body) > room:
            body = body[: max(0, room - 1)] + "…"
        return f"  {drawn} {body}\x1b[K"

    # -- whole-line styling of existing output --------------------------------

    def line(self, text: str, alarm: bool = False) -> str:
        """Style one already-rendered line by its shape. Never adds characters."""
        if not self.on or not text.strip():
            return text
        if alarm:
            return self.paint(text, "alarm", bold=opens_block(text))
        t = text.lstrip()
        for lead, role, block in _LEADS:
            if t.startswith(lead):
                return self.paint(text, role, bold=block or t.isupper())
        if _RULE.match(text):
            return self.paint(text, "muted")
        if _HEADING.match(text):
            return self.paint(text, bold=True)
        if m := _MARK_LINE.match(text):
            mark = m.group(2)
            rest = text[m.end() :]
            if mark == "✗":
                rest = self.paint(rest, "alarm")
            return m.group(1) + self.badge(mark) + m.group(3) + self._intervals(rest)
        if m := _CAPS_KEY.match(text):
            key = self.paint(m.group(2), "ink2", bold=True)
            return m.group(1) + key + self._intervals(text[m.end(2) :])
        if m := _KEY.match(text):
            return m.group(1) + self.paint(m.group(2), "ink2") + self._intervals(text[m.end(2) :])
        if m := _SIDE.match(text):
            return (
                m.group(1) + self.paint(m.group(2), m.group(2)) + self._intervals(text[m.end() :])
            )
        return self._intervals(text)

    def _intervals(self, text: str) -> str:
        return _INTERVAL.sub(lambda m: self.paint(m.group(0), "muted"), text)


out = Term()
err = Term(sys.stderr)


def unstyle(styled: str, purpose_header: bool = True) -> str:
    """Invert the styling: strip ANSI, drop the header and box borders, and
    restore boxed lines to their original indent. ``unstyle(styled) == plain``
    is the contract the tests hold every command to."""
    lines = [strip(ln) for ln in styled.split("\n") if ln != SPACER]
    if purpose_header and lines and lines[0].lstrip().startswith(BRAND):
        lines = lines[1:]
    kept = []
    for ln in lines:
        s = ln.strip()
        if s and s[0] in "╭╰" and set(s[1:]) <= set(_BOX):
            continue
        if m := re.match(r"^(\s*)│ (.*?)(?: *│)?$", ln):
            kept.append((m.group(1) + m.group(2)).rstrip())
            continue
        kept.append(ln)
    return "\n".join(kept)


class _Styler:
    """A stdout stand-in that styles each complete line on its way to a TTY.

    Installed only when ``out.on``. Lines pass straight through after colour;
    the one thing held back is a ``next:`` block, until its first non-
    continuation line (or a flush) says where the box ends.
    """

    def __init__(self, raw: TextIO, term: Term) -> None:
        self._raw = raw
        self._term = term
        self._partial = ""
        self._next: list[str] = []
        self._alarm = False
        self._first = True

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw, name)

    def isatty(self) -> bool:
        return is_tty(self._raw)

    def write(self, s: str) -> int:
        self._partial += s
        *done, self._partial = self._partial.split("\n")
        for ln in done:
            self._emit(ln)
        return len(s)

    def _emit(self, ln: str) -> None:
        if self._first:
            self._first = False
            if ln.strip():
                self._raw.write(SPACER + "\n")
        if self._next:
            indent = len(self._next[0]) - len(self._next[0].lstrip())
            if ln.strip() and len(ln) - len(ln.lstrip()) > indent:
                self._next.append(ln)
                return
            self._close_box()
        if ln.lstrip().startswith("next:"):
            self._next = [ln]
            return
        if not ln.strip():
            self._alarm = False
        elif opens_block(ln):
            self._alarm = True
        self._raw.write(self._term.line(ln, alarm=self._alarm) + "\n")

    def _close_box(self) -> None:
        if self._next:
            for b in self._term.box(self._next):
                self._raw.write(b + "\n")
            self._next = []

    def flush(self) -> None:
        self._close_box()
        if self._partial:
            self._raw.write(self._partial)
            self._partial = ""
        self._raw.flush()


@contextmanager
def styled_stdout(command: str, purpose: str) -> Iterator[None]:
    """Style stdout for one command, if and only if styling is on.

    Off (a pipe, ``NO_COLOR``, ``ONPAR_PLAIN=1``): yields with stdout untouched,
    which is how the plain path stays byte-for-byte what it was.
    """
    if not out.on:
        yield
        return
    raw = sys.stdout
    styler = _Styler(raw, Term(raw, on=True))
    raw.write(styler._term.title(command, purpose) + "\n")
    sys.stdout = styler  # type: ignore[assignment]
    try:
        yield
    finally:
        styler.flush()
        sys.stdout = raw


def demo() -> None:
    """Render every component both ways; plain must equal the input."""
    import io

    off, on = Term(io.StringIO(), on=False), Term(io.StringIO(), on=True)
    text = "  ✓ hardware   M4 Max · 98% [95–99]"
    assert off.paint(text, "alarm", bold=True) == text
    assert off.badge("✓") == "✓" and off.line(text) == text
    assert off.box(["  next: onpar doctor"]) == ["  next: onpar doctor"]
    assert off.progress(3, 10, "  candidate: 3/10 replies   ") == "  candidate: 3/10 replies   "
    assert off.rule(5) == "─────"
    assert off.table(["a", "bb"], [["1", "2"]]) == ["a  bb", "─────", "1  2"]

    styled = on.line(text)
    assert "\x1b[" in styled and strip(styled) == text
    assert strip(on.line("REGRET — keep the incumbent for these:")).startswith("REGRET")
    assert strip(on.paint("x", "proof")) == "x"
    assert strip(on.table(["a", "bb"], [[on.paint("1", "alarm"), "2"]])[2]) == "1  2"
    bar = on.progress(5, 10, "  candidate: 5/10 replies")
    assert "█" in bar and "candidate: 5/10 replies" in strip(bar)

    # The styler: header + box are decoration, and unstyle() removes exactly that.
    buf = io.StringIO()
    s = _Styler(buf, on)
    plain = (
        "  ✓ hardware   ok\n  ⚠ engine   none\n\nREGRET — keep:\n  billing  (20%)\n\n"
        "  next: onpar prove x.json\n        (needs an endpoint)\n\ntail [1–2]\n"
    )
    s.write(plain)
    s.flush()
    assert "╭" in buf.getvalue() and "\x1b[" in buf.getvalue()
    assert unstyle(buf.getvalue(), purpose_header=False) == plain
    assert buf.getvalue().startswith(SPACER + "\n")  # output opened with text
    head = on.title("doctor", "what this machine can do") + "\n"
    assert unstyle(head + buf.getvalue()) == plain
    blank_first = io.StringIO()
    s = _Styler(blank_first, on)
    s.write("\n  ✓ ok\n")
    s.flush()
    assert unstyle(head + blank_first.getvalue()) == "\n  ✓ ok\n"


if __name__ == "__main__":
    demo()
    print("term: ok")
