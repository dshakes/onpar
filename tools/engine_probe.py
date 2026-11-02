#!/usr/bin/env python3
"""Probe a running OpenAI-compatible endpoint for the three behaviours that a
proof silently depends on, and that unit tests with stub servers cannot see.

    python3 tools/engine_probe.py --endpoint http://127.0.0.1:8000/v1 [--model NAME]

Each probe is a real request. The verdict per probe is one of:

  answers       a short classification prompt returns a non-empty answer within
                a small token budget (a reasoning model served with thinking on
                spends the budget thinking and returns nothing — found on
                Qwen3-30B-A3B on mlx_lm.server, 2026-11-02)
  json          `response_format: json_object` yields valid JSON that is an
                object; otherwise the engine does not enforce the format and a
                structured-output cluster will fail for a server reason
  tool-call     a request offering one tool yields a structured `tool_calls`
                entry rather than the call written out as text

Exit 0 when every probe passes, 1 otherwise, with one line per probe. Zero
dependencies. The CI job in .github/workflows/engine-probe.yml serves a small
model with the real engine and runs this, which is how the two defects above
stop being things a person notices later.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Fetch an order by id.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}


def chat(endpoint: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - caller-supplied, by design
        return json.load(r)


def served_model(endpoint: str, timeout: float) -> str:
    with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=timeout) as r:  # noqa: S310
        data = json.load(r).get("data") or []
    if not data:
        raise RuntimeError("the endpoint lists no models")
    return data[0]["id"]


def probe_answers(endpoint: str, model: str, timeout: float) -> tuple[bool, str]:
    r = chat(
        endpoint,
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Reply with exactly one word: billing, login or bug.",
                },
                {"role": "user", "content": "I was charged twice this month."},
            ],
            "max_tokens": 24,
            "temperature": 0,
        },
        timeout,
    )
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    if content:
        return (
            True,
            f"answered {content!r} in {r.get('usage', {}).get('completion_tokens', '?')} tokens",
        )
    if reasoning:
        return False, "empty answer: the budget went to a reasoning phase (thinking is on)"
    return False, f"empty answer, finish_reason={r['choices'][0].get('finish_reason')}"


def probe_json(endpoint: str, model: str, timeout: float) -> tuple[bool, str]:
    """Enforcement, not luck: a prompt that invites prose, with json_object asked.

    A server with constrained decoding returns a JSON object anyway. One without
    it returns the prose, and a structured-output cluster then fails for a
    server reason. The real proof on 2026-11-02 saw 40 of 40 invoice items fail
    on mlx_lm.server while a two-item probe happened to come back valid, which
    is why this asks for something the model would not write as JSON on its own.
    """
    r = chat(
        endpoint,
        {
            "model": model,
            "messages": [{"role": "user", "content": "Write one sentence about the sea."}],
            "response_format": {"type": "json_object"},
            "max_tokens": 80,
            "temperature": 0,
        },
        timeout,
    )
    text = (r["choices"][0]["message"].get("content") or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return (
            False,
            f"prose came back: {text[:60]!r} (response_format is not enforced by this server)",
        )
    if not isinstance(parsed, dict):
        return (
            False,
            f"valid JSON but a {type(parsed).__name__}, not the object json_object promises",
        )
    return True, "a JSON object came back even for a prose prompt: the server enforces the format"


def probe_tool(endpoint: str, model: str, timeout: float) -> tuple[bool, str]:
    r = chat(
        endpoint,
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Use the tool to look the order up before answering.",
                },
                {"role": "user", "content": "What is the status of order A12345?"},
            ],
            "tools": [TOOL],
            "tool_choice": "auto",
            "max_tokens": 120,
            "temperature": 0,
        },
        timeout,
    )
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        name = calls[0].get("function", {}).get("name")
        return True, f"structured tool call to {name!r}"
    text = (msg.get("content") or "").strip()
    if "lookup_order" in text:
        return (
            False,
            f"the call came back as text: {text[:80]!r} (no tool-call parser on the server)",
        )
    return False, f"no tool call and no mention of the tool: {text[:80]!r}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--endpoint",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1",
    )
    ap.add_argument(
        "--model", default="", help="served model id; defaults to the first the endpoint lists"
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--expect", default="answers,json,tool-call", help="comma-separated probes that must pass"
    )
    args = ap.parse_args()
    try:
        model = args.model or served_model(args.endpoint, args.timeout)
    except (urllib.error.URLError, RuntimeError, OSError) as e:
        print(f"error: cannot reach {args.endpoint}: {e}", file=sys.stderr)
        return 1
    expected = {p.strip() for p in args.expect.split(",") if p.strip()}
    probes = {"answers": probe_answers, "json": probe_json, "tool-call": probe_tool}
    failed = 0
    print(f"probing {args.endpoint} · model {model}")
    for name, fn in probes.items():
        try:
            ok, why = fn(args.endpoint, model, args.timeout)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            ok, why = False, f"request failed: {e}"
        mark = "ok  " if ok else ("FAIL" if name in expected else "warn")
        if not ok and name in expected:
            failed += 1
        print(f"  {mark}  {name:<9} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
