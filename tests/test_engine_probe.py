"""tools/engine_probe.py: the verdicts, against a stub endpoint.

The probe exists because two defects reached a real run before anyone saw
them: a reasoning model that spent its token budget thinking and answered
nothing, and a server that did not enforce `response_format`. The real-engine
run lives in CI (.github/workflows/engine-probe.yml); this pins what each
verdict says for each reply shape, without a model.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import engine_probe  # noqa: E402


def _serve(reply_for):
    """Serve /v1/models and /v1/chat/completions; `reply_for(payload) -> message dict`."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102 - quiet
            pass

        def do_GET(self):  # noqa: N802
            body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            msg = reply_for(payload)
            body = json.dumps(
                {
                    "choices": [{"message": msg, "finish_reason": msg.pop("_finish", "stop")}],
                    "usage": {"completion_tokens": msg.pop("_tokens", 3)},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def _good(payload):
    if payload.get("tools"):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup_order", "arguments": "{}"},
                }
            ],
        }
    if payload.get("response_format"):
        return {"role": "assistant", "content": '{"sentence": "The sea is wide."}'}
    return {"role": "assistant", "content": "billing"}


def test_a_well_behaved_server_passes_all_three():
    srv, url = _serve(_good)
    try:
        assert engine_probe.probe_answers(url, "m", 5)[0]
        assert engine_probe.probe_json(url, "m", 5)[0]
        assert engine_probe.probe_tool(url, "m", 5)[0]
    finally:
        srv.shutdown()


def test_thinking_that_eats_the_budget_is_named_as_such():
    def reply(payload):
        return {
            "role": "assistant",
            "content": "",
            "reasoning": "Let me think about billing...",
            "_finish": "length",
            "_tokens": 24,
        }

    srv, url = _serve(reply)
    try:
        ok, why = engine_probe.probe_answers(url, "m", 5)
    finally:
        srv.shutdown()
    assert not ok and "thinking is on" in why


def test_prose_under_json_object_means_the_format_is_not_enforced():
    srv, url = _serve(lambda p: {"role": "assistant", "content": "The sea is wide and blue."})
    try:
        ok, why = engine_probe.probe_json(url, "m", 5)
    finally:
        srv.shutdown()
    assert not ok and "not enforced" in why


def test_an_array_is_valid_json_but_not_the_promised_object():
    srv, url = _serve(
        lambda p: {"role": "assistant", "content": '[{"name": "Widget"}, "total", 1]'}
    )
    try:
        ok, why = engine_probe.probe_json(url, "m", 5)
    finally:
        srv.shutdown()
    assert not ok and "not the object" in why


def test_a_tool_call_written_as_text_is_a_missing_parser():
    tool_text = '<tool_call>{"name":"lookup_order","arguments":{"order_id":"A12345"}}</tool_call>'
    srv, url = _serve(lambda p: {"role": "assistant", "content": tool_text})
    try:
        ok, why = engine_probe.probe_tool(url, "m", 5)
    finally:
        srv.shutdown()
    assert not ok and "as text" in why


@pytest.mark.parametrize("expect,code", [("answers,tool-call", 0), ("answers,json,tool-call", 1)])
def test_exit_code_follows_the_expected_set(monkeypatch, capsys, expect, code):
    """json fails on this stub; it only fails the run when it was expected to pass."""

    def reply(payload):
        if payload.get("tools"):
            return _good(payload)
        if payload.get("response_format"):
            return {"role": "assistant", "content": "prose"}
        return {"role": "assistant", "content": "billing"}

    srv, url = _serve(reply)
    try:
        monkeypatch.setattr(sys, "argv", ["engine_probe", "--endpoint", url, "--expect", expect])
        assert engine_probe.main() == code
    finally:
        srv.shutdown()
    out = capsys.readouterr().out
    assert "stub-model" in out and ("warn" in out or "FAIL" in out)


def test_reasoning_prose_in_content_is_not_an_answer():
    """llama.cpp with --reasoning-budget 0 and a 0.6B model returned "Okay, the
    user mentioned they were charged twice…" as content; the first CI run
    counted that as an answer. A label is an answer; prose is not."""
    srv, url = _serve(
        lambda p: {
            "role": "assistant",
            "content": "Okay, the user mentioned they were charged twice.",
        }
    )
    try:
        ok, why = engine_probe.probe_answers(url, "m", 5)
    finally:
        srv.shutdown()
    assert not ok and "not a label" in why
