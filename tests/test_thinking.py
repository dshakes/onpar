"""Reasoning models: thinking off at serve time, and a reasoning-only reply is not an answer.

Found on a real run: `onpar run qwen3-30b-a3b` served a model whose chat
template thinks by default, so `max_tokens: 200` came back as
`{"content": "", "reasoning": "..."}` and `onpar prove` graded every item as a
wrong answer. Two halves, both pinned here: the planner turns thinking off (in
each engine's own cited spelling, or says it cannot), and the collector names a
reasoning-only reply as a failure instead of grading "".
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from onpar import catalog, launch
from onpar.cli import build_parser, main
from onpar.engines import SOURCES, Setting, Translated, Unsupported, adapter_for
from onpar.hardware import Hardware
from onpar.plan import Requirements, Workload, plan
from onpar.prove.collect import THINKING_EXHAUSTED, Collected, Collection, _Failed, _reply

GB = 1024**3
M4 = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * GB,
    usable_bytes=96 * GB,
    bandwidth_gbps=546.0,
    cores=16,
)
H100 = Hardware(
    kind="nvidia",
    name="H100 80GB",
    total_bytes=80 * GB,
    usable_bytes=76 * GB,
    bandwidth_gbps=3350.0,
    cores=132,
)
OFF = '{"enable_thinking":false}'
REASONING = ("qwen3-30b-a3b", "qwen3-32b", "qwen3-235b-a22b")


# --- the adapters ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "argv", "source"),
    [
        ("vllm", ("--default-chat-template-kwargs", OFF), "vllm_thinking"),
        ("vllm-tpu", ("--default-chat-template-kwargs", OFF), "vllm_thinking"),
        ("llm-d", ("--default-chat-template-kwargs", OFF), "vllm_thinking"),
        ("sglang", ("--default-chat-template-kwargs", OFF), "sglang_thinking"),
        ("mlx", ("--chat-template-args", OFF), "mlx_thinking"),
        (
            "llama.cpp",
            ("--chat-template-kwargs", '{"enable_thinking":false}', "--reasoning-budget", "0"),
            "llamacpp_thinking",
        ),
    ],
)
def test_thinking_off_is_each_engines_cited_spelling(engine, argv, source):
    a = adapter_for(engine)
    assert a is not None
    assert a.translate(Setting.THINKING, False).argv == argv
    assert "checked 2026-11-0" in SOURCES[source], "an uncited spelling is not emitted"
    # The value is JSON the engine parses, not a string that merely looks like it.
    assert json.loads(argv[1]) == {"enable_thinking": False}


def test_ollama_has_no_serve_time_switch_and_says_what_to_do_instead():
    gap = adapter_for("ollama").translate(Setting.THINKING, False)
    assert isinstance(gap, Unsupported)
    assert gap.setting is Setting.THINKING
    assert '"think": false' in gap.reason and "each" in gap.reason


@pytest.mark.parametrize("engine", ["vllm", "sglang", "mlx", "llama.cpp", "ollama", "llm-d"])
def test_thinking_on_is_every_engines_default_so_nothing_is_emitted(engine):
    out = adapter_for(engine).translate(Setting.THINKING, True)
    assert isinstance(out, Translated) and out.argv == () and out.env == ()


# --- the catalogue ---------------------------------------------------------------


def test_exactly_the_qwen3_family_is_marked_reasoning():
    """Each was read off the published chat template; nothing else is guessed in."""
    marked = {m.id for m in catalog.load() if m.reasoning}
    assert marked == set(REASONING)


def test_a_quoted_reasoning_flag_is_refused(tmp_path, monkeypatch):
    """`"reasoning": "false"` is a truthy string: it must not load as yes."""
    row = {
        **json.loads(catalog.CATALOG_PATH.read_text())["models"][0],
        "id": "x",
        "reasoning": "false",
    }
    f = tmp_path / "extra.json"
    f.write_text(json.dumps({"models": [row]}))
    monkeypatch.setenv("ONPAR_CATALOG", str(f))
    with pytest.raises(ValueError, match="reasoning"):
        catalog.load()


# --- the planner -----------------------------------------------------------------


@pytest.mark.parametrize("model_id", REASONING)
def test_a_reasoning_model_is_planned_with_thinking_off(model_id):
    knob = plan(H100, Requirements(Workload.INTERACTIVE), catalog.get(model_id), "q4").get(
        Setting.THINKING
    )
    assert knob is not None and knob.value is False
    assert "--thinking" in knob.why, "the knob says how to keep it on"


def test_thinking_kept_on_when_asked():
    req = Requirements(Workload.INTERACTIVE, thinking=True)
    knob = plan(H100, req, catalog.get("qwen3-32b"), "q4").get(Setting.THINKING)
    assert knob is not None and knob.value is True


@pytest.mark.parametrize("thinking", [False, True])
def test_a_model_without_a_reasoning_phase_gets_no_thinking_knob(thinking):
    req = Requirements(Workload.INTERACTIVE, thinking=thinking)
    assert plan(H100, req, catalog.get("llama-3.1-8b"), "q4").get(Setting.THINKING) is None
    assert plan(H100, req).get(Setting.THINKING) is None, "no model, nothing to switch off"


def test_vllm_plan_command_carries_the_flag_and_no_gap():
    p = plan(H100, Requirements(Workload.INTERACTIVE), catalog.get("qwen3-32b"), "q4")
    argv, gaps = p.command("Qwen/Qwen3-32B")
    assert argv[argv.index("--default-chat-template-kwargs") + 1] == OFF
    assert not [g for g in gaps if g.startswith("thinking")]


def test_ollama_plan_reports_the_gap_rather_than_dropping_it():
    p = plan(M4, Requirements(Workload.INTERACTIVE), catalog.get("qwen3-30b-a3b"), "q4")
    p = replace(p, engine=type(p.engine)("ollama"))
    _, gaps = p.command("qwen3:30b-a3b")
    assert any(g.startswith("thinking:") and "think" in g for g in gaps), gaps


# --- run and build ---------------------------------------------------------------


def _launch(tmp_path, **kw):
    return launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        exists=lambda repo: True,
        cache=tmp_path / "w.json",
        **kw,
    )


def test_run_on_mlx_turns_thinking_off_in_the_command(tmp_path):
    """The command from the real run, now with the flag it was missing."""
    p = _launch(tmp_path)
    assert isinstance(p, launch.LaunchPlan), p
    assert p.argv[0] == "mlx_lm.server"
    assert p.argv[p.argv.index("--chat-template-args") + 1] == OFF
    assert "--chat-template-args" in p.render()


def test_run_with_thinking_leaves_the_template_default(tmp_path):
    p = _launch(tmp_path, thinking=True)
    assert isinstance(p, launch.LaunchPlan), p
    assert "--chat-template-args" not in p.argv


def test_the_cli_flags_parse_and_default_off():
    parser = build_parser()
    assert parser.parse_args(["run", "qwen3-30b-a3b"]).thinking is False
    assert parser.parse_args(["run", "qwen3-30b-a3b", "--thinking"]).thinking is True
    assert parser.parse_args(["build", "a chatbot", "--thinking"]).thinking is True


def test_run_forwards_thinking_to_the_launcher(monkeypatch, capsys):
    seen = {}

    def fake(model_id, **kw):
        seen.update(kw)
        return launch.Refusal(reason="stub", next_step="none")

    monkeypatch.setattr(launch, "plan", fake)
    main(["run", "qwen3-30b-a3b", "--dry-run", "--thinking"])
    assert seen["thinking"] is True
    main(["run", "qwen3-30b-a3b", "--dry-run"])
    assert seen["thinking"] is False


def test_build_carries_thinking_into_the_session(capsys):
    main(["build", "chatbot for 10 people", "--on", "h100", "--thinking", "--json"])
    state = json.loads(capsys.readouterr().out)["state"]
    assert json.dumps(state).count('"thinking": true') == 1, state


# --- the collector ---------------------------------------------------------------


def _body(message: dict, usage: dict | None = None) -> dict:
    return {"choices": [{"message": message, "finish_reason": "length"}], "usage": usage or {}}


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_a_reasoning_only_reply_is_a_named_failure_not_an_empty_answer(key):
    with pytest.raises(_Failed) as e:
        _reply(_body({"content": "", key: "Okay, the user wants..."}, {"completion_tokens": 200}))
    assert str(e.value) == f"{THINKING_EXHAUSTED}; 200 reasoning tokens, no answer"


def test_null_content_with_reasoning_is_the_same_failure():
    with pytest.raises(_Failed, match=THINKING_EXHAUSTED):
        _reply(_body({"content": None, "reasoning": "hmm"}))


def test_the_reasoning_token_count_prefers_the_servers_own_breakdown():
    usage = {"completion_tokens": 210, "completion_tokens_details": {"reasoning_tokens": 200}}
    with pytest.raises(_Failed, match="; 200 reasoning tokens"):
        _reply(_body({"content": "", "reasoning": "hmm"}, usage))


def test_an_unreported_count_stays_unreported():
    with pytest.raises(_Failed, match="reasoning tokens not reported"):
        _reply(_body({"content": "", "reasoning": "hmm"}))


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content"])
def test_with_an_answer_only_the_answer_is_graded(key):
    assert _reply(_body({"content": "Paris", key: "France's capital is..."})) == ("Paris", ())


def test_tool_calls_after_thinking_are_still_a_reply():
    call = {"type": "function", "function": {"name": "f", "arguments": "{}"}}
    assert _reply(_body({"content": "", "reasoning": "use f", "tool_calls": [call]})) == (
        "",
        (call,),
    )


def test_empty_reasoning_does_not_turn_an_empty_answer_into_a_failure():
    """Only a reply that actually thought is reclassified."""
    assert _reply(_body({"content": "", "reasoning": "  "})) == ("", ())


def test_the_collection_counts_and_explains_them_apart_from_other_failures():
    got = Collection(
        base="http://127.0.0.1:8000/v1",
        model="m",
        side="candidate",
        replies=(
            Collected("i0", reason=f"{THINKING_EXHAUSTED}; 200 reasoning tokens, no answer"),
            Collected("i1", reason=f"{THINKING_EXHAUSTED}; 187 reasoning tokens, no answer"),
            Collected("i2", reason="HTTP 500"),
            Collected("i3", text="ok"),
        ),
        items=(),
        seconds=0.1,
    )
    assert got.thinking_exhausted == 2
    text = got.render()
    assert "2 of them spent the whole token budget thinking" in text
    assert "thinking off" in text and "max_tokens" in text
    assert "3 item(s) never got a reply and are NOT scored" in text


def test_a_collection_with_no_thinking_failures_says_nothing_about_thinking():
    got = Collection(
        base="b",
        model="m",
        side="candidate",
        replies=(Collected("i0", reason="x"),),
        items=(),
        seconds=0,
    )
    assert got.thinking_exhausted == 0
    assert "thinking" not in got.render()


def test_an_inline_think_block_is_stripped_before_grading_and_counted():
    """llama.cpp with --reasoning-format none, and older mlx builds, put the
    reasoning inline in `content`. Only what follows the block is the answer;
    the collection summary says how many replies carried one."""
    from onpar.prove.collect import Collected, Collection, _strip_inline_think

    assert _strip_inline_think("<think>\nlet me see\n</think>\n\n4") == ("4", True)
    assert _strip_inline_think("</think>4") == ("4", True)
    assert _strip_inline_think("<think>never closed") == ("", True)
    assert _strip_inline_think("4, and <think> is a word here") == (
        "4, and <think> is a word here",
        False,
    )

    c = Collection(
        base="http://x/v1",
        model="m",
        side="candidate",
        replies=(Collected("1", text="4", inline_think=True), Collected("2", text="5")),
        items=(),
        seconds=0.0,
    )
    assert c.inline_think == 1
    assert "1 replies carried an inline <think> block" in c.render()


@pytest.mark.parametrize(
    "content,want_text,want_reason",
    [
        ("<think>\n2+2 is 4\n</think>\n\n4", "4", ""),
        ("plain 4", "plain 4", ""),
        ("<think>still going when max_tokens ran out", "", THINKING_EXHAUSTED),
    ],
)
def test_ask_strips_an_inline_think_reply_and_fails_an_unclosed_one(
    content, want_text, want_reason, monkeypatch
):
    """Through the real request path: a closed block is stripped and flagged, a
    reply without one is untouched, and a block that never closed is the token-
    budget failure, not an empty answer graded as wrong."""
    from onpar.prove import collect

    body = {"choices": [{"message": {"content": content}}], "model": "m"}
    monkeypatch.setattr(collect, "_once", lambda *a, **k: body)
    got = collect._ask(
        "1",
        b"{}",
        url="http://x/v1/chat/completions",
        headers={},
        timeout=1,
        retries=0,
        backoff=0,
        sleep=lambda _s: None,
    )
    assert got.text == want_text
    assert got.inline_think == (content.startswith("<think>") and not want_reason)
    assert want_reason in (got.reason or "")
    assert got.ok == (not want_reason)
