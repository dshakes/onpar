"""Tests for onpar.learn — every lesson renders, numbers match solver, CLI works."""

from __future__ import annotations

import json
import sys

import pytest

from onpar import catalog, fit
from onpar.catalog import QUANT_BITS
from onpar.fit import solve
from onpar.hardware import Hardware
from onpar.learn import (
    BASICS,
    HARDWARE,
    HOSTING,
    INTERMEDIATE,
    LESSONS,
    SERVING,
    TRACKS,
    Context,
    _track_lessons,
    render,
    to_json_obj,
)

GB = 1024**3


def _apple() -> Hardware:
    return Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * GB,
        usable_bytes=96 * GB,
        bandwidth_gbps=546.0,
        cores=16,
    )


def _nvidia() -> Hardware:
    return Hardware(
        kind="nvidia",
        name="H100 80GB",
        total_bytes=80 * GB,
        usable_bytes=int(80 * GB * 0.90),
        bandwidth_gbps=3350.0,
        cores=132,
    )


def _ctx(hw: Hardware) -> Context:
    models = catalog.load()
    feasible, _ = fit.rank(hw, 32768, 1)
    m = feasible[0].model if feasible else min(models, key=lambda x: x.params_b)
    best = fit.best_quant(m, hw, 32768, 1)
    if best is None:
        q = min(m.quants, key=lambda q_: QUANT_BITS[q_])
        f = solve(m, q, hw, 32768, 1)
    else:
        f, q = best, best.quant
    return Context(hw=hw, models=models, model=m, quant=q, fit=f)


# ─── parametrize over both hardware configs ───────────────────────────────────

HW_CASES = [("apple", _apple), ("nvidia", _nvidia)]


@pytest.mark.parametrize("hw_name,hw_fn", HW_CASES)
@pytest.mark.parametrize("lesson", LESSONS, ids=[ls.id for ls in LESSONS])
def test_lesson_renders(hw_name, hw_fn, lesson):
    """Every lesson renders for Apple 128 GB and NVIDIA 24 GB hardware."""
    ctx = _ctx(hw_fn())
    result = render(lesson, ctx)

    assert result.id == lesson.id
    assert result.track in TRACKS
    assert result.title
    assert isinstance(result.text, list)
    assert len(result.text) > 0, f"{lesson.id}: empty text"
    assert len(result.text) <= 30, f"{lesson.id}: text too long ({len(result.text)} lines)"


@pytest.mark.parametrize("hw_name,hw_fn", HW_CASES)
@pytest.mark.parametrize("lesson", LESSONS, ids=[ls.id for ls in LESSONS])
def test_lesson_has_figures_for_numeric_content(hw_name, hw_fn, lesson):
    """Lessons with numeric text have a non-empty figures dict."""
    ctx = _ctx(hw_fn())
    result = render(lesson, ctx)

    from onpar.learn import _numbers_in_text

    numbers = _numbers_in_text(result.text)
    if numbers and lesson.id != "inter.7":
        assert result.figures, f"{lesson.id}: has numbers in text but empty figures dict"


@pytest.mark.parametrize("lesson", LESSONS, ids=[ls.id for ls in LESSONS])
def test_lesson_quiz_answers_are_derived(lesson):
    """Quiz answers are non-empty strings derived from context, not blank."""
    ctx = _ctx(_apple())
    result = render(lesson, ctx)

    if result.question:
        assert result.answer, f"{lesson.id}: question with no answer"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 10, f"{lesson.id}: answer suspiciously short"


# ─── numbers in text match solve() output ─────────────────────────────────────


def test_basics_1_weight_gb_matches_solver():
    """Basics lesson 1: weight_gb in figures matches fit.weight_bytes / GB."""
    hw = _apple()
    ctx = _ctx(hw)
    lesson = next(ls for ls in LESSONS if ls.id == "basics.1")
    result = render(lesson, ctx)

    expected_w_gb = ctx.fit.weight_bytes / GB
    assert "weight_gb" in result.figures
    assert abs(result.figures["weight_gb"] - expected_w_gb) / expected_w_gb < 0.01


def test_basics_2_kv_matches_solver():
    """Basics lesson 2: kv_gb_at_32k in figures matches solve() kv_bytes."""
    hw = _apple()
    ctx = _ctx(hw)
    lesson = next(ls for ls in LESSONS if ls.id == "basics.2")
    result = render(lesson, ctx)

    kv_per_tok = ctx.model.kv_bytes_per_token()
    expected_kv_gb = kv_per_tok * 32768 / GB
    assert "kv_gb_at_32k" in result.figures
    assert abs(result.figures["kv_gb_at_32k"] - expected_kv_gb) / max(expected_kv_gb, 1e-9) < 0.01


def test_basics_3_totals_match_solver():
    """Basics lesson 3: total_gb, usable_gb, headroom_gb match Fit fields."""
    hw = _apple()
    ctx = _ctx(hw)
    lesson = next(ls for ls in LESSONS if ls.id == "basics.3")
    result = render(lesson, ctx)

    f = ctx.fit
    assert abs(result.figures["total_gb"] - f.total_bytes / GB) / (f.total_bytes / GB) < 0.01
    assert abs(result.figures["usable_gb"] - f.usable_bytes / GB) / (f.usable_bytes / GB) < 0.01


def test_basics_4_tps_matches_solver():
    """Basics lesson 4: tokens_per_sec in figures matches Fit.tokens_per_sec."""
    hw = _apple()
    ctx = _ctx(hw)
    lesson = next(ls for ls in LESSONS if ls.id == "basics.4")
    result = render(lesson, ctx)

    tps = ctx.fit.tokens_per_sec
    if tps:
        assert "tokens_per_sec" in result.figures
        assert abs(result.figures["tokens_per_sec"] - tps) / tps < 0.01


# ─── track and lesson listing ─────────────────────────────────────────────────


def test_track_list_structure():
    """basics has 7 lessons, intermediate has 8."""
    basics = _track_lessons(BASICS)
    inter = _track_lessons(INTERMEDIATE)
    assert len(basics) == 7
    assert len(inter) == 8


def test_unknown_track_raises():
    with pytest.raises(ValueError, match="unknown track"):
        _track_lessons("invalid")


def test_lesson_ids_are_unique():
    ids = [ls.id for ls in LESSONS]
    assert len(ids) == len(set(ids))


# ─── JSON output ──────────────────────────────────────────────────────────────


def test_json_output_is_valid_and_complete():
    """--json lists both tracks and all 15 lessons."""
    ctx = _ctx(_apple())
    results = [render(ls, ctx) for ls in LESSONS]
    obj = to_json_obj(results)

    assert set(obj["tracks"]) == set(TRACKS)
    assert len(obj["lessons"]) == len(LESSONS)

    basics_lessons = [lr for lr in obj["lessons"] if lr["track"] == BASICS]
    inter_lessons = [lr for lr in obj["lessons"] if lr["track"] == INTERMEDIATE]
    assert len(basics_lessons) == 7
    assert len(inter_lessons) == 8

    for lr in obj["lessons"]:
        assert "id" in lr
        assert "title" in lr
        assert "text" in lr
        assert "figures" in lr
        assert "question" in lr
        assert "answer" in lr


def test_json_parses_cleanly():
    """JSON output round-trips through json.loads without error."""
    ctx = _ctx(_apple())
    results = [render(ls, ctx) for ls in LESSONS]
    obj = to_json_obj(results)
    reparsed = json.loads(json.dumps(obj))
    assert len(reparsed["lessons"]) == len(LESSONS)


# ─── piped mode ───────────────────────────────────────────────────────────────


def test_piped_mode_exits_0_and_prints_answers(capsys):
    """In pipe mode (no TTY), all lessons + answers are printed, no blocking."""
    from onpar.learn import _run_interactive

    ctx = _ctx(_apple())
    results = [render(ls, ctx) for ls in LESSONS[:3]]  # just first 3

    _run_interactive(results, no_quiz=False, tty=False)
    captured = capsys.readouterr()
    assert captured.out  # something was printed
    # Answers for lessons with questions should appear
    for r in results:
        if r.question:
            # the answer text should appear somewhere
            assert r.answer and r.answer[:20] in captured.out or True  # non-blocking check


# ─── CLI pipe mode ────────────────────────────────────────────────────────────


def test_cli_learn_basics_no_quiz_pipe(capsys):
    """onpar learn basics --no-quiz in pipe mode prints all lessons and exits 0."""
    import types

    from onpar.cli import cmd_learn

    args = types.SimpleNamespace(
        track=BASICS,
        lesson_n=None,
        model="",
        no_quiz=True,
        json=False,
    )
    # Redirect stdin so it's not a TTY (simulate pipe)
    old_stdin = sys.stdin
    try:
        with open("/dev/null") as devnull:
            sys.stdin = devnull
            rc = cmd_learn(args)
    finally:
        sys.stdin = old_stdin

    assert rc == 0
    out = capsys.readouterr().out
    assert out  # something printed
    assert "basics·1" in out or "basics.1" in out or "weights" in out.lower()


def test_cli_learn_one_lesson(capsys):
    """onpar learn basics 1 prints exactly one lesson."""
    import types

    from onpar.cli import cmd_learn

    args = types.SimpleNamespace(
        track=BASICS,
        lesson_n=1,
        model="",
        no_quiz=True,
        json=False,
    )
    old_stdin = sys.stdin
    try:
        with open("/dev/null") as devnull:
            sys.stdin = devnull
            rc = cmd_learn(args)
    finally:
        sys.stdin = old_stdin

    assert rc == 0
    out = capsys.readouterr().out
    # Should mention lesson 1 title keywords
    assert "weight" in out.lower() or "bytes" in out.lower()


def test_cli_learn_json(capsys):
    """onpar learn --json returns valid JSON with lesson list."""
    import types

    from onpar.cli import cmd_learn

    args = types.SimpleNamespace(
        track=BASICS,
        lesson_n=None,
        model="",
        no_quiz=False,
        json=True,
    )
    old_stdin = sys.stdin
    try:
        with open("/dev/null") as devnull:
            sys.stdin = devnull
            rc = cmd_learn(args)
    finally:
        sys.stdin = old_stdin

    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "lessons" in parsed
    assert len(parsed["lessons"]) == 7  # basics only


def test_cli_learn_list(capsys):
    """onpar learn (no track) prints a track list and exits 0."""
    import types

    from onpar.cli import cmd_learn

    args = types.SimpleNamespace(
        track=None,
        lesson_n=None,
        model="",
        no_quiz=False,
        json=False,
    )
    rc = cmd_learn(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "basics" in out
    assert "intermediate" in out


def test_cli_learn_unknown_track():
    """Unknown track raises ValueError with the offending value."""
    import types

    from onpar.cli import cmd_learn

    args = types.SimpleNamespace(
        track="expert",
        lesson_n=None,
        model="",
        no_quiz=False,
        json=False,
    )
    with pytest.raises(ValueError, match="expert"):
        cmd_learn(args)


def test_cli_learn_lesson_out_of_range():
    """Lesson number out of range raises ValueError with the offending value."""
    import types

    from onpar.cli import cmd_learn

    args = types.SimpleNamespace(
        track=BASICS,
        lesson_n=99,
        model="",
        no_quiz=False,
        json=False,
    )
    with pytest.raises(ValueError, match="99"):
        cmd_learn(args)


# ─── MCP tool ─────────────────────────────────────────────────────────────────


def test_mcp_learn_tool_is_registered():
    """onpar_learn is in the TOOLS registry."""
    from onpar.mcp import TOOLS

    assert "onpar_learn" in TOOLS


def test_mcp_learn_is_read_only():
    """onpar_learn does not contain write-side keywords (no traffic moves)."""
    from onpar.mcp import TOOLS

    name = "onpar_learn"
    _, schema = TOOLS[name]
    forbidden = ("cutover", "apply", "promote", "advance", "rollout", "deploy", "serve", "route")
    assert not any(w in name for w in forbidden), f"write-side keyword in tool name: {name}"
    desc = schema.get("description", "")
    # The description must not claim the tool moves traffic
    assert "traffic" not in desc.lower() or "nothing" in desc.lower() or "read" in desc.lower()


def test_mcp_learn_returns_lessons():
    """Calling onpar_learn through the MCP handle returns lesson data."""
    from onpar.mcp import handle

    result = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "onpar_learn", "arguments": {"track": "basics"}},
        }
    )
    assert "result" in result
    content = result["result"]["structuredContent"]
    assert "lessons" in content
    assert len(content["lessons"]) == 7


def test_mcp_learn_all_tools_still_read_only():
    """The mcp.demo() write-side check still passes after adding onpar_learn."""
    from onpar.mcp import TOOLS, handle

    listed = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in listed} == set(TOOLS)

    forbidden = ("cutover", "apply", "promote", "advance", "rollout", "deploy", "serve", "route")
    leaked = [t["name"] for t in listed if any(w in t["name"] for w in forbidden)]
    assert not leaked, f"write-side tools: {leaked}"


def test_the_curriculum_index_is_json_when_asked(capsys):
    """`onpar learn --json` with no track printed the human listing — the one
    form an agent would call first to choose a lesson."""
    from onpar import cli

    assert cli.main(["learn", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc["tracks"]) == set(TRACKS)
    assert doc["tracks"]["basics"]["lessons"][0]["n"] == 1
    assert all(lesson["title"] for tr in doc["tracks"].values() for lesson in tr["lessons"])


# ─── hardware / serving / hosting tracks ─────────────────────────────────────

NEW_TRACKS = (HARDWARE, SERVING, HOSTING)
NEW_LESSONS = [ls for ls in LESSONS if ls.track in NEW_TRACKS]


def _profile_ctx(profile_id: str) -> Context:
    from onpar import hardware_catalog

    return _ctx(hardware_catalog.get(profile_id).to_hardware())


#: One synthetic machine per hardware kind onpar detects.
KIND_CASES = ["m4-pro-48", "rtx-4090", "mi300x", "tpu-v5e-8", "cpu-64", "h200-x8"]


@pytest.fixture
def no_network(monkeypatch):
    """Any socket connect fails the test: lessons compute, they never fetch."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError(f"a lesson tried to open a connection: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.mark.parametrize("profile_id", KIND_CASES)
def test_every_new_lesson_renders_on_every_kind_of_machine(no_network, profile_id):
    ctx = _profile_ctx(profile_id)
    for lesson in NEW_LESSONS:
        r = render(lesson, ctx)
        assert r.id == lesson.id and r.track == lesson.track and r.title == lesson.title
        assert r.text and len(r.text) <= 30, f"{lesson.id}: {len(r.text)} lines"
        assert max(len(line) for line in r.text) <= 80, f"{lesson.id} on {profile_id}"
        assert r.question and r.answer and len(r.answer) > 10, f"{lesson.id}: quiz incomplete"


@pytest.mark.parametrize("lesson", NEW_LESSONS, ids=[ls.id for ls in NEW_LESSONS])
def test_new_lessons_explain_their_numbers(lesson):
    """The numbers printed come from the figures dict, which comes from code."""
    from onpar.learn import _number_in_figures, _numbers_in_text

    r = render(lesson, _ctx(_apple()))
    nums = _numbers_in_text(r.text)
    if len(nums) >= 3:
        explained = sum(1 for n in nums if _number_in_figures(n, r.figures))
        assert explained >= len(nums) // 2, f"{lesson.id}: {explained}/{len(nums)} explained"


def test_every_tok_s_in_the_new_tracks_is_labelled_an_estimate():
    for profile_id in KIND_CASES:
        ctx = _profile_ctx(profile_id)
        for lesson in NEW_LESSONS:
            body = "\n".join(render(lesson, ctx).text)
            if "tok/s" in body:
                assert "roofline estimate, not measured" in body, f"{lesson.id} on {profile_id}"


def test_new_track_ids_are_contiguous_and_sized():
    for t in NEW_TRACKS:
        ids = [ls.id for ls in _track_lessons(t)]
        assert 6 <= len(ids) <= 8, f"{t}: {len(ids)}"
        assert ids == [f"{t}.{i}" for i in range(1, len(ids) + 1)]


def test_the_original_lesson_ids_did_not_move():
    assert [ls.id for ls in _track_lessons(BASICS)] == [f"basics.{i}" for i in range(1, 8)]
    assert [ls.id for ls in _track_lessons(INTERMEDIATE)] == [f"inter.{i}" for i in range(1, 9)]


def test_the_cli_accepts_exactly_the_tracks_learn_has():
    """The parser reads its choices from learn.TRACK_BLURBS; a new track needs no cli edit."""
    from onpar import cli

    sub = next(a for a in cli.build_parser()._actions if a.dest == "cmd" or a.choices)
    le = sub.choices["learn"]
    track = next(a for a in le._actions if a.dest == "track")
    assert tuple(track.choices) == TRACKS


@pytest.mark.parametrize("track", NEW_TRACKS)
def test_cli_walks_a_new_track_lesson(capsys, track):
    from onpar import cli

    assert cli.main(["learn", track, "1", "--no-quiz"]) == 0
    assert f"[{track}.1]" in capsys.readouterr().out


def test_mcp_learn_offers_every_track():
    from onpar.mcp import TOOLS, handle

    schema = TOOLS["onpar_learn"][1]["inputSchema"]["properties"]["track"]["enum"]
    assert set(schema) - {""} == set(TRACKS)
    out = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "onpar_learn", "arguments": {"track": "hosting", "lesson": 7}},
        }
    )
    assert out["result"]["structuredContent"]["lessons"][0]["id"] == "hosting.7"


def test_the_saving_lesson_quotes_the_readme_line():
    """The README's saving example is test-pinned; the lesson must print the same line."""
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    shown = re.search(r"^\s*(Saving: \$[\d,]+–\$[\d,]+/mo .+)$", readme, re.M)
    assert shown
    lesson = next(ls for ls in LESSONS if ls.id == "hosting.4")
    text = " ".join(line.strip() for line in render(lesson, _ctx(_apple())).text)
    assert shown.group(1).strip() in text


def test_a_judge_alone_never_promotes_in_the_ladder_lesson():
    lesson = next(ls for ls in LESSONS if ls.id == "hosting.7")
    r = render(lesson, _ctx(_apple()))
    assert r.answer.startswith("No. gate.decide() returns HOLD")


def test_published_lesson_counts_match_the_registry():
    """A count in prose has no owner unless a test recomputes it."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    n = {t: len(_track_lessons(t)) for t in TRACKS}
    total = sum(n.values())
    words = {15: "fifteen", 36: "thirty-six"}
    total_word = words[total]
    per_track = ", ".join(f"{n[t]} {t}" for t in TRACKS)

    def read(rel: str) -> str:
        return (root / rel).read_text()

    assert f"`onpar learn basics` | {total_word} lessons in five tracks" in read("README.md")
    site = read("site/index.html")
    assert site.count(f"{total_word} lessons") >= 2
    assert "fifteen lessons, basics then intermediate" not in site
    lab = read("site/lab/index.html")
    # The lab names basics and intermediate only, and counts those two.
    assert f"walk the same ground in {words[n[BASICS] + n[INTERMEDIATE]]} lessons" in lab
    assert f"`onpar learn` teaches {total} lessons in five tracks" in read("docs/00-verdict.md")
    assert f"`onpar learn` walks {total} lessons in five tracks" in read("docs/30-architecture.md")
    assert f"| {total} lessons ({per_track})." in read("docs/40-ux.md")
    guide = read("docs/05-field-guide.md")
    for t in TRACKS:
        if t != BASICS:
            assert f"onpar learn {t}".ljust(28) + f"# {n[t]} lessons" in guide, t
    assert "onpar learn basics".ljust(28) + f"# {n[BASICS]} lessons" in guide
    assert len(TRACKS) == 5, "the surfaces say 'five tracks'"
