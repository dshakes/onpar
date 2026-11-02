"""The terminal layer adds decoration on a TTY and nothing anywhere else.

Every golden in `test_cli_golden` is replayed with stdout forced to look like a
TTY. Styled, it must carry ANSI; unstyled (ANSI stripped, header and box
borders removed) it must be the golden byte for byte. So colour can be added
and content cannot. `NO_COLOR` and `ONPAR_PLAIN=1` must give the golden itself.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout

import pytest

from onpar import cli, hardware, term
from onpar.prove.equivalence import CandidateReport, Matrix, score_cluster
from onpar.prove.graders import EvalItem, grade
from test_cli_golden import CASES, GOLDEN, MACHINE

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TEXT_CASES = [(n, a) for n, a in CASES if "--json" not in a and n != "catalog_sources"]


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("ONPAR_CATALOG", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
    buf = _Tty()
    with redirect_stdout(buf):
        cli.main(argv)
    return buf.getvalue()


@pytest.fixture
def tty_env(monkeypatch):
    for var in ("NO_COLOR", "ONPAR_PLAIN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


def test_demo():
    term.demo()


@pytest.mark.parametrize("name,argv", TEXT_CASES, ids=[n for n, _ in TEXT_CASES])
def test_tty_styles_and_unstyles_to_the_golden(name, argv, monkeypatch, tmp_path, tty_env):
    got = _run(argv, monkeypatch, tmp_path)
    want = (GOLDEN / f"{name}.txt").read_text()
    assert "\x1b[" in got, f"`onpar {' '.join(argv)}` printed no colour on a TTY"
    assert ANSI.sub("", got.splitlines()[0]).lstrip().startswith(f"{term.BRAND} onpar {argv[0]}")
    assert term.unstyle(got) == want
    # The lead's check: stripping ANSI alone leaves the golden's content intact,
    # in order, with only the header and box lines added.
    stripped = ANSI.sub("", got)
    for line in want.splitlines():
        assert line.strip() in stripped


@pytest.mark.parametrize("env", [{"NO_COLOR": "1"}, {"NO_COLOR": ""}, {"ONPAR_PLAIN": "1"}])
@pytest.mark.parametrize("name,argv", TEXT_CASES[:3], ids=[n for n, _ in TEXT_CASES[:3]])
def test_plain_switches_win_over_a_tty(name, argv, env, monkeypatch, tmp_path, tty_env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _run(argv, monkeypatch, tmp_path) == (GOLDEN / f"{name}.txt").read_text()


def test_json_is_never_styled(monkeypatch, tmp_path, tty_env):
    got = _run(["fit", "--context", "32k", "--concurrency", "8", "--json"], monkeypatch, tmp_path)
    assert got == (GOLDEN / "fit_json.txt").read_text()


def test_progress_is_the_plain_text_off_a_tty():
    text = "  candidate: 3/197 replies · ~4m left   "
    assert term.Term(io.StringIO(), on=False).progress(3, 197, text) == text
    drawn = term.Term(io.StringIO(), on=True).progress(3, 197, text)
    assert "█" in drawn or "░" in drawn
    assert "candidate: 3/197 replies" in ANSI.sub("", drawn)


def _wide_matrix() -> Matrix:
    def results(cluster: str, n_pass: int, n_fail: int) -> list:
        ok = [grade(EvalItem(f"{cluster}-p{i}", cluster, "p", "1", "1")) for i in range(n_pass)]
        bad = [grade(EvalItem(f"{cluster}-f{i}", cluster, "p", "1", "x")) for i in range(n_fail)]
        return ok + bad

    clusters = tuple(
        score_cluster(f"c{i}", f"cluster number {i}", 1 / 7, results(f"c{i}", 40 - i, i))
        for i in range(7)
    )
    return Matrix([CandidateReport("qwen3-30b-a3b", clusters)], incumbent="gpt-4o-mini")


def test_narrow_tty_turns_the_matrix_cluster_per_row():
    m = _wide_matrix()
    wide = m.render()
    assert m.render(width=10_000) == wide  # wide enough: the pipe layout, untouched
    narrow = m.render(width=100)
    assert narrow != wide
    rows = [ln for ln in narrow.splitlines() if ln.startswith("cluster number")]
    assert len(rows) == 7, narrow
    # Same cells, same headline: nothing is lost by transposing.
    for c in m.candidates[0].clusters:
        assert c.render_cell() in narrow
    assert Matrix._weighted(m.candidates[0]) in narrow
    assert "weighted" in wide and "cluster number 6" in narrow
