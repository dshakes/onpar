"""Agentic evals: the step is replayed with everything the incumbent saw.

The property under test: a candidate asked to take one step of an agent loop
must be handed the same system prompt, the same prior turns, the same tool
schemas and the same output-format demand the incumbent was. Before this, the
collector sent the user text alone — so every system-prompted workload was
replayed as a different task, every tool-using cluster graded as a regression
by construction, and the judge compared two empty strings on tool-call steps
and called them a tie.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from onpar.distill import trajectory
from onpar.distill.shape import Capture
from onpar.observe import distill
from onpar.prove import EvalItem, Outcome, grade, judge_item
from onpar.prove.collect import _cache_key, request_body
from onpar.prove.graders import ToolArgsSchema, schema_problems
from onpar.prove.judge import Comparison, Reply, render_answer, render_request

SYSTEM = {"role": "system", "content": "You are a refunds agent. Use tools."}
USER = {"role": "user", "content": "refund order 42"}
REFUND_TOOL = {
    "type": "function",
    "function": {
        "name": "refund",
        "parameters": {
            "type": "object",
            "required": ["order_id"],
            "properties": {
                "order_id": {"type": "integer"},
                "reason": {"type": "string", "enum": ["damaged", "late"]},
            },
            "additionalProperties": False,
        },
    },
}


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def item(**kw: Any) -> EvalItem:
    base = {
        "item_id": "i",
        "cluster": "c",
        "prompt": USER["content"],
        "baseline": "",
        "candidate": "",
        "messages": (SYSTEM, USER),
        "tools": (REFUND_TOOL,),
    }
    base.update(kw)
    return EvalItem(**base)


# --- replay ------------------------------------------------------------------------


def test_the_candidate_is_sent_the_whole_conversation_and_the_tools():
    body = request_body(item(response_format="json_object"), model="m")
    assert body["messages"] == [SYSTEM, USER], "system prompt and turns, as captured"
    assert body["tools"] == [REFUND_TOOL]
    # A captured json_object goes out as the schema form of the same contract:
    # that is what grammar-constrained engines enforce (see JSON_OBJECT_FORMAT).
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["stream"] is False and body["temperature"] == 0.0


def test_an_item_without_captured_messages_falls_back_to_the_bare_prompt():
    body = request_body(EvalItem("i", "c", "just text", "", ""), model="m")
    assert body["messages"] == [{"role": "user", "content": "just text"}]
    assert "tools" not in body and "response_format" not in body


def test_the_resume_cache_is_bound_to_the_context_not_just_the_prompt():
    """A regenerated eval set with a new system prompt is a new question. Reusing
    the old reply would score the old task's answer against the new task."""
    a = _cache_key(base="b", model="m", side="candidate", item=item())
    other_system = {"role": "system", "content": "You are a translator."}
    b = _cache_key(base="b", model="m", side="candidate", item=item(messages=(other_system, USER)))
    c = _cache_key(base="b", model="m", side="candidate", item=item(tools=()))
    assert len({a, b, c}) == 3


def test_unfaithful_replays_are_counted_and_said_out_loud():
    from onpar.prove.collect import Collection

    col = Collection(base="b", model="m", side="candidate", replies=(), items=(), seconds=0.0)
    assert "replayed as bare user text" not in col.render()
    col = Collection(
        base="b", model="m", side="candidate", replies=(), items=(), seconds=0.0, unfaithful=3
    )
    assert "3 item(s)" in col.render() and "bare user text" in col.render()


# --- the schema grader -----------------------------------------------------------


def schema_outcome(calls: tuple[dict[str, Any], ...], tools=(REFUND_TOOL,)) -> tuple[Outcome, str]:
    s = ToolArgsSchema().grade(item(candidate_tool_calls=calls, tools=tools))
    return s.outcome, s.detail


def test_well_formed_arguments_pass():
    assert schema_outcome((call("refund", order_id=42, reason="late"),))[0] is Outcome.PASS


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({"order_id": "42"}, "expected integer, got str"),
        ({"reason": "late"}, "order_id: required"),
        ({"order_id": 42, "reason": "whim"}, "not one of"),
        ({"order_id": 42, "amount": 5}, "not in schema"),
        ({"order_id": True}, "expected integer, got bool"),
    ],
)
def test_each_way_an_argument_object_can_violate_its_schema_is_named(args, needle):
    outcome, detail = schema_outcome((call("refund", **args),))
    assert outcome is Outcome.FAIL
    assert needle in detail, detail


def test_arguments_that_are_not_json_fail_rather_than_hiding_as_empty():
    bad = {"type": "function", "function": {"name": "refund", "arguments": "{not json"}}
    outcome, detail = schema_outcome((bad,))
    assert outcome is Outcome.FAIL and "not JSON" in detail


def test_calling_a_tool_the_request_never_offered_fails():
    """ToolChoice misses this when the baseline called nothing. A hallucinated
    tool is the one argument error no harness can recover from."""
    outcome, detail = schema_outcome((call("delete_account", id=1),))
    assert outcome is Outcome.FAIL and "never offered" in detail


def test_one_hallucinated_tool_beside_a_valid_call_still_fails():
    """Found in review: a valid call plus a made-up one passed, because the
    unknown call was skipped and the all-unknown branch never fired."""
    outcome, detail = schema_outcome((call("refund", order_id=1), call("delete_account", id=1)))
    assert outcome is Outcome.FAIL and "delete_account" in detail and "never offered" in detail


def test_the_schema_grader_abstains_when_it_has_nothing_to_check():
    assert schema_outcome(())[0] is Outcome.NOT_APPLICABLE
    assert schema_outcome((call("refund", order_id=1),), tools=())[0] is Outcome.NOT_APPLICABLE


def test_the_schema_grader_is_in_the_default_stack():
    r = grade(
        item(
            baseline_tool_calls=(call("refund", order_id=1),),
            candidate_tool_calls=(call("refund", order_id="x"),),
        )
    )
    assert any(s.grader == "tool-args-schema" and not s.passed for s in r.applicable)
    assert not r.passed


def test_the_validator_covers_nested_objects_and_arrays():
    schema = {
        "type": "object",
        "properties": {
            "lines": {"type": "array", "items": {"type": "object", "required": ["sku"]}},
            "meta": {"type": ["object", "null"]},
        },
    }
    assert schema_problems({"lines": [{"sku": 1}], "meta": None}, schema) == []
    assert schema_problems({"lines": [{"qty": 1}]}, schema) == ["$.lines[0].sku: required"]
    assert schema_problems({"meta": 3}, schema) == ["$.meta: expected object/null, got int"]


def test_unknown_schema_keywords_are_accepted_not_refused():
    """A validator that rejected `$ref` would fail every call to a tool that used
    it, which reads exactly like a model that cannot call tools."""
    assert schema_problems({"a": 1}, {"$ref": "#/defs/x", "allOf": []}) == []


# --- the judge sees tool calls -----------------------------------------------------


def test_tool_calls_are_rendered_into_the_answer_the_judge_compares():
    assert render_answer("", (call("refund", order_id=42),)) == '→ refund({"order_id": 42})'
    assert render_answer("", ()) == "(no text, no tool calls)"
    assert render_answer("done", ("lookup",)) == "done\n→ lookup()"


def test_the_judge_is_shown_the_system_prompt_tools_and_recent_turns():
    text = render_request(item())
    assert "[system] You are a refunds agent" in text
    assert "[tools offered] refund" in text
    assert "[user] refund order 42" in text
    assert text.endswith("[the next assistant step is what is being judged]")


def test_long_conversations_show_the_tail_and_say_what_was_cut():
    turns = tuple({"role": "user", "content": f"t{i}"} for i in range(10))
    text = render_request(item(messages=(SYSTEM, *turns)))
    assert "[… 4 earlier turns omitted]" in text
    assert "[user] t9" in text and "[user] t0" not in text


def test_the_judge_no_longer_compares_two_empty_strings_on_a_tool_step():
    seen: list[Comparison] = []

    def judge(c: Comparison) -> Reply:
        seen.append(c)
        return Reply("tie")

    judge_item(
        item(
            baseline_tool_calls=(call("refund", order_id=42),),
            candidate_tool_calls=(call("refund", order_id=7),),
        ),
        judge,
        "j",
    )
    assert len(seen) == 2
    assert all("refund" in c.a and "refund" in c.b for c in seen)
    assert all("[tools offered] refund" in c.prompt for c in seen)


def test_old_eval_sets_judge_exactly_as_before():
    seen: list[Comparison] = []

    def judge(c: Comparison) -> Reply:
        seen.append(c)
        return Reply("a")

    judge_item(EvalItem("i", "c", "the prompt", "base answer", "cand answer"), judge, "j")
    assert seen[0] == Comparison("the prompt", a="base answer", b="cand answer")


# --- distill carries it all into the file ------------------------------------------


def rows() -> list[dict[str, Any]]:
    """An agent loop of three steps plus one unrelated request, as the bridge
    hands them over."""
    a1 = {"role": "assistant", "tool_calls": [call("refund", order_id=42)]}
    t1 = {"role": "tool", "content": "ok"}

    def row(rid: str, msgs: list[dict[str, Any]], tokens: int, calls=(), response=""):
        return {
            "request_id": rid,
            "model": "gpt-5",
            "messages": msgs,
            "response": response,
            "prompt_tokens": tokens,
            "tools": [REFUND_TOOL],
            "tool_calls": list(calls),
            "response_format": None,
        }

    return [
        row("r1", [SYSTEM, USER], 100, calls=["refund"]),
        row("r2", [SYSTEM, USER, a1, t1], 250, response="Refunded."),
        row("r3", [SYSTEM, {"role": "user", "content": "status of 9"}], 90, response="Shipped."),
    ]


def test_eval_items_carry_the_conversation_tools_and_session():
    doc, rep = distill(rows(), budget=10, min_per_cluster=1)
    by_id = {i["item_id"]: i for i in doc["items"]}
    assert by_id["r1"]["messages"] == [SYSTEM, USER]
    assert by_id["r1"]["tools"] == [REFUND_TOOL]
    assert by_id["r1"]["session"] == by_id["r2"]["session"] != by_id["r3"]["session"]
    assert (by_id["r1"]["step"], by_id["r2"]["step"], by_id["r3"]["step"]) == (1, 2, 1)
    assert doc["provenance"]["agent_sessions"] == 1
    assert "agent sessions" in rep.render()


def test_the_prove_loader_reads_what_distill_wrote(tmp_path):
    """The file is the contract. If the loader dropped a key the writer added,
    the replay would silently be the old unfaithful one."""
    from onpar import cli

    doc, _ = distill(rows(), budget=10, min_per_cluster=1)
    for it in doc["items"]:
        it["candidate"] = it["baseline"] or "x"
    path = tmp_path / "e.json"
    path.write_text(json.dumps(doc))
    rc = cli.main(["prove", str(path), "--json"])
    assert rc in (0, 1)


def test_the_eval_item_knows_whether_it_is_agentic():
    assert item().agentic
    assert not EvalItem("i", "c", "p", "b", "c").agentic
    assert EvalItem("i", "c", "p", "b", "c", baseline_tool_calls=("lookup",)).agentic


def test_sessions_link_structurally_regardless_of_log_order():
    caps = [
        Capture(request_id=r["request_id"], model="m", messages=tuple(r["messages"]), response="")
        for r in reversed(rows())
    ]
    linked = dict(zip((c.request_id for c in caps), trajectory.link(caps), strict=True))
    assert linked["r1"] == (linked["r2"][0], 1) and linked["r2"][1] == 2
    assert linked["r3"][0] != linked["r1"][0]


# --- the human-agreement loop ------------------------------------------------------


def test_the_judge_ledger_records_what_the_judge_saw_and_a_human_can_measure_it(tmp_path):
    from onpar.prove.label import agreement_from, label, load, recording

    ledger = tmp_path / "judged.jsonl"
    rec = recording(lambda c: Reply("a"), ledger)
    rec(Comparison("q", a="x", b="y"))
    rec(Comparison("q", a="y", b="x"))
    assert [r["judge"] for r in load(ledger)] == ["a", "a"]

    shown: list[str] = []
    answers = iter(["a", "b"])
    assert label(ledger, ask=lambda _: next(answers), show=shown.append) == 2
    # Blind: the judge's verdict is never printed to the labeller.
    assert not any('"judge"' in s or "judge:" in s for s in shown[:-1])
    ag = agreement_from(ledger, "j")
    assert (ag.agreed, ag.total) == (1, 2)


def test_prove_wires_the_ledger_in_and_refuses_two_sources_for_one_rate(tmp_path):
    import argparse

    from onpar import cli
    from onpar.prove.label import load

    ledger = tmp_path / "judged.jsonl"
    ns = argparse.Namespace(
        judge_endpoint="http://127.0.0.1:9/v1",
        judge_model="j",
        judge_agreed=0,
        judge_samples=0,
        judge_sample=str(ledger),
        judge_labels=None,
        collect_timeout=0.2,
    )
    judge, model, agreement = cli._judge_from(ns)
    assert model == "j" and agreement is not None and agreement.rate is None
    judge(Comparison("q", a="x", b="y"))  # unreachable endpoint: recorded as undecided
    rows = load(ledger)
    assert len(rows) == 1 and rows[0]["judge"] == ""

    ledger.write_text(
        json.dumps({"key": "k", "prompt": "q", "a": "x", "b": "y", "judge": "a"})
        + "\n"
        + json.dumps({"key": "k", "human": "a"})
        + "\n"
    )
    ns.judge_sample, ns.judge_labels = None, str(ledger)
    _, _, agreement = cli._judge_from(ns)
    assert (agreement.agreed, agreement.total) == (1, 1)

    ns.judge_samples = 20
    with pytest.raises(ValueError, match="two sources"):
        cli._judge_from(ns)


def test_label_in_a_pipe_refuses_rather_than_reading_stdin_as_a_person(
    tmp_path, monkeypatch, capsys
):
    from onpar import cli

    ledger = tmp_path / "judged.jsonl"
    assert cli.main(["label", str(ledger)]) == 1, "a missing ledger is an error with a sentence"
    ledger.write_text(
        json.dumps({"key": "k", "prompt": "q", "a": "x", "b": "y", "judge": "a"}) + "\n"
    )
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    assert cli.main(["label", str(ledger)]) == 2
    assert "needs a terminal" in capsys.readouterr().err


# --- a tool call that arrived as text ---------------------------------------------


def test_a_tool_call_leaked_into_text_is_a_failed_call_with_the_server_fix_named():
    from onpar.prove.graders import ToolLeak

    leaked = item(
        candidate='<tool_call>{"name": "refund", "arguments": {"order_id": 42}}</tool_call>'
    )
    s = ToolLeak().grade(leaked)
    assert s.outcome is Outcome.FAIL and "parser" in s.detail
    # Parsed properly, or no tools offered, there is nothing to leak.
    assert ToolLeak().grade(item(candidate="Refunded.")).outcome is Outcome.PASS
    assert (
        ToolLeak()
        .grade(item(candidate="<tool_call>", candidate_tool_calls=(call("refund"),)))
        .outcome
        is Outcome.NOT_APPLICABLE
    )
    assert (
        ToolLeak().grade(EvalItem("i", "c", "p", "b", "<tool_call>x")).outcome
        is Outcome.NOT_APPLICABLE
    )
    assert "tool-leak" in {s.grader for s in grade(leaked).scores}


# --- the receipt carries the agentic block, and old receipts still verify ---------


def test_the_receipt_carries_the_agentic_block_as_numbers():
    from onpar.prove import suite

    items = [
        item(
            item_id=f"i{n}",
            baseline="",
            candidate="",
            baseline_tool_calls=(call("refund", order_id=n),),
            candidate_tool_calls=(call("refund", order_id=n),) if n % 4 else (),
        )
        for n in range(12)
    ]
    r = suite(items, shares={"c": 1.0}, issued="2026-10-31", candidate="cand", incumbent="inc")
    a = r.receipt.agentic
    assert a["steps"] == 12.0 and a["agreed"] == 9.0 and a["talked_instead"] == 3.0
    assert 0.0 <= a["agreement_low"] < a["agreement_high"] <= 1.0
    assert "Agentic:" in r.receipt.render()
    # Round-trips through JSON with the digest intact.
    from onpar.prove.receipt import Receipt

    back = Receipt.from_json(r.receipt.to_json())
    assert back.agentic == a and back.format == "onpar.receipt/v3"


def test_a_v2_receipt_sealed_before_agentic_existed_still_verifies():
    """Adding a field to the digest of an already-issued format is a false
    accusation of forgery against every receipt on disk. v2 digests over v2
    fields, exactly as v1 does over v1."""
    from onpar.prove.receipt import Receipt

    r = Receipt(
        incumbent="inc",
        candidate="cand",
        issued="2026-08-01",
        eval_set="e" * 64,
        bar=0.9,
        proven=(),
        regret=(),
        unproven=(),
        format="onpar.receipt/v2",
    )
    blob = json.loads(r.to_json())
    assert "agentic" not in json.dumps(blob["receipt"]) or blob["receipt"]["agentic"] == {}
    # Simulate the v2 file as written by 1.3.x: no `agentic` key at all.
    del blob["receipt"]["agentic"]
    back = Receipt.from_json(json.dumps(blob))
    assert back.digest() == blob["digest"]


# --- whole sessions, replayed from the recording ------------------------------------


def test_prove_replays_sessions_against_the_candidate_and_reports_the_path(tmp_path, capsys):
    """The stub candidate agrees at step one (calls refund) and answers at step
    two once it sees a tool result — so the session finishes. The request it
    receives at step two must carry its own call id on the recorded result."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    seen: list[dict[str, Any]] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if any(m.get("role") == "tool" for m in body["messages"]):
                msg = {"role": "assistant", "content": "Refunded."}
            else:
                msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [call("refund", order_id=42) | {"id": "cand-9"}],
                }
            out = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    Thread(target=srv.serve_forever, daemon=True).start()
    try:
        doc, _ = distill(rows(), budget=10, min_per_cluster=1)
        path = tmp_path / "e.json"
        path.write_text(json.dumps(doc))
        from onpar import cli

        rc = cli.main(
            [
                "prove",
                str(path),
                "--candidate-endpoint",
                f"http://127.0.0.1:{srv.server_address[1]}/v1",
                "--replay-sessions",
                "--collect-timeout",
                "5",
            ]
        )
        out = capsys.readouterr().out
    finally:
        srv.shutdown()
        srv.server_close()
    assert rc in (0, 1)
    assert "SESSION REPLAY" in out and "1/1 sessions followed" in out
    # Two requests carry a tool result: the per-step collection of step two (the
    # recording as captured, no call id on its tool message) and the replay's
    # step two, which must point the recorded result at the candidate's call.
    step_two = [b for b in seen if any(m.get("role") == "tool" for m in b["messages"])]
    assert len(step_two) == 2
    assert step_two[-1]["messages"][-1]["tool_call_id"] == "cand-9"


# --- an upstream error is not an answer ---------------------------------------------


def test_an_upstream_error_capture_is_dropped_from_the_eval_set_and_said_so():
    """Found on the first live run: a 401 before the key was set became an eval
    item whose baseline was "". Older gateways wrote those; distill drops them."""
    bad = {
        "request_id": "e1",
        "model": "gpt-4o-mini",
        "messages": [USER],
        "response": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "tools": [],
        "tool_calls": [],
        "response_format": None,
    }
    doc, rep = distill([*rows(), bad], budget=10, min_per_cluster=1)
    assert rep.errors == 1 and "upstream errors" in rep.render()
    assert all(i["item_id"] != "e1" for i in doc["items"])
    # A legitimate empty reply that called a tool is an answer, and stays.
    assert any(i["baseline_tool_calls"] for i in doc["items"])
    with pytest.raises(ValueError, match="every one of the 1 captures"):
        distill([bad], budget=10, min_per_cluster=1)


def test_the_distill_hint_plans_for_structured_output_when_the_corpus_asked_for_it():
    """The real proof on 2026-11-02 lost a whole JSON cluster to an engine with no
    constrained decoding, and nothing had said so beforehand. The session report
    now carries the count of captures that asked for a response format and puts
    `--structured-output` in the plan hint, with and without a prefix hint."""
    from onpar.distill.trajectory import SessionReport

    with_prefix = SessionReport(
        captures=100,
        sessions=20,
        multi_step=10,
        longest=3,
        tool_captures=5,
        prompt_tokens=10_000,
        shared_prefix_tokens=6_000,
        multi_step_captures=30,
        structured_captures=40,
    )
    hint = "\n".join(with_prefix.render())
    assert "--structured-output" in hint and "NOT EXPRESSED" in hint

    no_sessions = SessionReport(
        captures=40,
        sessions=40,
        multi_step=0,
        longest=1,
        tool_captures=0,
        prompt_tokens=0,
        shared_prefix_tokens=0,
        structured_captures=40,
    )
    lines = no_sessions.render()
    assert len(lines) == 1 and "onpar build --structured-output" in lines[0]

    plain = SessionReport(
        captures=40,
        sessions=40,
        multi_step=0,
        longest=1,
        tool_captures=0,
        prompt_tokens=0,
        shared_prefix_tokens=0,
    )
    assert plain.render() == []
