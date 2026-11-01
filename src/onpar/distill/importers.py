"""Import LLM traces from external observability platforms as onpar capture rows.

All functions are pure (no I/O except :func:`load`). Each record importer
returns a row dict in the shape :func:`~onpar.distill.shape.from_capture_row`
reads, or ``None`` to skip the record, with a reason counted in skip_counts.

Format auto-detection is from the **record shape alone**, never from filename
or file extension.

Verified schemas
----------------
**OpenAI chat completions** (``openai_pair`` format):
    ``{"request": {...chat.completions request...}, "response": {...}}``
    This is the portable escape hatch — produce it from any platform not
    natively supported.
    Source: https://platform.openai.com/docs/api-reference/chat/create

**Anthropic Messages API normalisation**:
    ``system`` / content blocks with ``tool_use`` / ``tool_result`` are
    normalised to the chat shape, mirroring the rules in
    ``onpar-gateway/src/capture/normalize.rs``.
    Source: https://docs.anthropic.com/en/api/messages

Partially verified / field-level unverified
-------------------------------------------
**Langfuse generations export** (``langfuse`` format):
    Shape verified per task spec and common Langfuse SDK output:
    ``{"type": "GENERATION", "id": ..., "traceId": ..., "model": ...,
    "input": ..., "output": ..., "usage": ...,
    "startTime": ..., "endTime": ...}``

    Field-level status:
    - ``type``, ``id``, ``traceId``, ``model``, ``input``, ``output``,
      ``startTime``, ``endTime``: per task spec / common Langfuse SDK output.
    - ``usage.input`` / ``usage.output``: per task spec.
    - ``usage.promptTokens`` / ``usage.completionTokens``: **UNVERIFIED** —
      included as common alternate spellings; importer accepts both.
    - Tool schema location (``input.tools``, ``metadata.tools``,
      ``modelParameters.tools``): **UNVERIFIED** — Langfuse has no
      documented canonical export field for tool schemas; importer checks
      all three locations in that order and tolerates absence.
    Schema not independently verified with WebFetch (tool unavailable at
    implementation time); records with unrecognised shapes are skipped with a
    counted reason rather than raised.

    Cite: https://langfuse.com/docs/integrations/sdk/python (unconfirmed)

**Kitaru session export**: NOT IMPLEMENTED.
    The on-disk schema could not be verified from public documentation at
    implementation time (github.com/zenml-io/kitaru, docs.zenml.io/kitaru).
    Use the ``openai_pair`` format as an alternative.

Redaction
---------
Every text field is passed through :func:`onpar.core.redact` before becoming
a row (invariant 4 — redaction happens before persistence, fails closed).

When the compiled extension is absent, :func:`load` raises ``RuntimeError``
naming ``pip install onpar-core`` — unless *unredacted=True* is passed, in
which case it prints a one-line warning and skips redaction.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["load", "demo"]

# ponytail: plain dict for skip counts — no dataclass for a 2-field counter
SkipCounts = dict[str, int]


def _bump(counts: SkipCounts, reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


# ── Anthropic normalisation ────────────────────────────────────────────────────
# Pure Python mirror of onpar-gateway/src/capture/normalize.rs:
#   messages_request_fields / messages_body_text / messages_body_tool_calls


def _anthropic_system_text(sys_val: Any) -> str:
    if isinstance(sys_val, str):
        return sys_val
    if isinstance(sys_val, list):
        return "\n".join(
            str(b.get("text") or "")
            for b in sys_val
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        )
    return ""


def _tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _tool_use_to_call(block: dict[str, Any]) -> dict[str, Any]:
    inp = block.get("input") or {}
    return {
        "id": str(block.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(block.get("name") or ""),
            "arguments": json.dumps(inp) if isinstance(inp, dict) else str(inp),
        },
    }


def _anthropic_content_to_messages(role: str, content: Any) -> list[dict[str, Any]]:
    """One Anthropic message's content → zero or more OpenAI-shape messages."""
    if content is None:
        return [{"role": role, "content": ""}]
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return []

    has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)

    if has_tool_result:
        return [
            {
                "role": "tool",
                "tool_call_id": str(b.get("tool_use_id") or ""),
                "content": _tool_result_text(b),
            }
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]

    if has_tool_use:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text"):
                text_parts.append(str(b["text"]))
            elif b.get("type") == "tool_use":
                tool_calls.append(_tool_use_to_call(b))
        m: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
        if text_parts:
            m["content"] = "\n".join(text_parts)
        return [m]

    # plain text blocks
    text = "\n".join(
        str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )
    return [{"role": role, "content": text}]


def _anthropic_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        f: dict[str, Any] = {}
        for key in ("name", "description"):
            if key in t:
                f[key] = t[key]
        if "input_schema" in t:
            f["parameters"] = t["input_schema"]
        out.append({"type": "function", "function": f})
    return out


def _normalize_anthropic_body(
    body: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anthropic Messages request body → (messages, tools)."""
    msgs: list[dict[str, Any]] = []
    sys_val = body.get("system")
    if sys_val is not None:
        text = _anthropic_system_text(sys_val)
        if text:
            msgs.append({"role": "system", "content": text})
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            msgs.extend(
                _anthropic_content_to_messages(str(msg.get("role") or "user"), msg.get("content"))
            )
    raw_tools = body.get("tools")
    tools = _anthropic_tools_to_openai(raw_tools) if isinstance(raw_tools, list) else []
    return msgs, tools


def _anthropic_response_to_parts(
    output: Any,
) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic response body → (text, [{name, arguments}...])."""
    if isinstance(output, str):
        return output, []
    if not isinstance(output, dict):
        return "", []
    content = output.get("content") or []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text_parts.append(str(b.get("text") or ""))
        elif b.get("type") == "tool_use":
            inp = b.get("input") or {}
            tool_calls.append(
                {
                    "name": str(b.get("name") or ""),
                    "arguments": json.dumps(inp) if isinstance(inp, dict) else str(inp),
                }
            )
    return "\n".join(filter(None, text_parts)), tool_calls


# ── Redaction helper ───────────────────────────────────────────────────────────


def _redact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply core.redact() to every text field in a row."""
    from onpar import core  # lazy — keeps importers importable without the extension

    def r(s: str) -> str:
        if not s:
            return s
        cleaned, _ = core.redact(s)
        return cleaned

    messages = []
    for msg in row.get("messages") or []:
        if not isinstance(msg, dict):
            messages.append(msg)
            continue
        m = dict(msg)
        if isinstance(m.get("content"), str):
            m["content"] = r(m["content"])
        messages.append(m)

    tool_calls = []
    for tc in row.get("tool_calls") or []:
        if not isinstance(tc, dict):
            tool_calls.append(tc)
            continue
        t = dict(tc)
        if isinstance(t.get("arguments"), str):
            t["arguments"] = r(t["arguments"])
        tool_calls.append(t)

    return {
        **row,
        "messages": messages,
        "response": r(str(row.get("response") or "")),
        "tool_calls": tool_calls,
    }


# ── Format detection ───────────────────────────────────────────────────────────


def _detect_format(rec: dict[str, Any]) -> str | None:
    """Record shape → format name, or None if unrecognised.

    Never inspects the filename — format is entirely determined by record shape.
    """
    if not isinstance(rec, dict):
        return None
    if rec.get("type") == "GENERATION":
        return "langfuse"
    if "request" in rec and "response" in rec:
        return "openai_pair"
    return None


# ── Langfuse importer ──────────────────────────────────────────────────────────


def _ms_between(start: Any, end: Any) -> int | None:
    if not (start and end):
        return None
    try:
        t0 = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        return max(0, int((t1 - t0).total_seconds() * 1000))
    except (ValueError, OverflowError):
        return None


def _import_langfuse(rec: dict[str, Any], counts: SkipCounts) -> dict[str, Any] | None:
    """Import one Langfuse GENERATION observation.

    See module docstring for field-level verification status.
    """
    raw_input = rec.get("input")
    raw_output = rec.get("output")
    model = str(rec.get("model") or "unknown")

    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    is_anthropic_shaped = False

    if isinstance(raw_input, list):
        # Already an OpenAI-style messages array
        messages = [m for m in raw_input if isinstance(m, dict)]
    elif isinstance(raw_input, dict):
        # Anthropic-shaped when it has a 'system' key: must be checked before
        # 'messages', because Anthropic request bodies have both and the system
        # prompt is in a top-level 'system' field that the 'messages' branch
        # would silently ignore.
        if "system" in raw_input:
            is_anthropic_shaped = True
            messages, tools = _normalize_anthropic_body(raw_input)
        elif "messages" in raw_input:
            msgs_val = raw_input["messages"]
            if isinstance(msgs_val, list):
                messages = [m for m in msgs_val if isinstance(m, dict)]
            # tools nested in the input wrapper
            inp_tools = raw_input.get("tools")
            if isinstance(inp_tools, list):
                tools = [t for t in inp_tools if isinstance(t, dict)]
        else:
            # Generic dict — treat it as a single user message
            messages = [{"role": "user", "content": json.dumps(raw_input)}]
    elif isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    elif raw_input is None:
        _bump(counts, "langfuse: missing input")
        return None
    else:
        _bump(counts, "langfuse: unrecognised input shape")
        return None

    # Tools in other locations (UNVERIFIED field names — see module docstring)
    if not tools:
        for container in (
            rec.get("metadata"),
            rec.get("modelParameters"),
        ):
            if isinstance(container, dict):
                candidate = container.get("tools")
                if isinstance(candidate, list):
                    tools = [t for t in candidate if isinstance(t, dict)]
                    break

    if not messages:
        _bump(counts, "langfuse: no messages resolved from input")
        return None

    # Output → response text + tool_calls
    response_text = ""
    tool_calls: list[dict[str, Any]] = []

    if is_anthropic_shaped:
        response_text, tool_calls = _anthropic_response_to_parts(raw_output)
    elif isinstance(raw_output, str):
        response_text = raw_output
    elif isinstance(raw_output, dict):
        # Could be an Anthropic-shaped response (content list) or OpenAI message dict
        content_val = raw_output.get("content")
        if isinstance(content_val, list):
            response_text, tool_calls = _anthropic_response_to_parts(raw_output)
        else:
            response_text = str(content_val or "")
            tc_list = raw_output.get("tool_calls")
            if isinstance(tc_list, list):
                for tc in tc_list:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        name = str(fn.get("name") or tc.get("name") or "")
                        args = str(fn.get("arguments") or tc.get("arguments") or "")
                        if name:
                            tool_calls.append({"name": name, "arguments": args})
    elif raw_output is None:
        pass  # no-op response, valid for tool-call-only outputs
    else:
        _bump(counts, "langfuse: unrecognised output shape")
        return None

    # Token counts — accept both Langfuse spellings
    # (usage.input/output verified; promptTokens/completionTokens UNVERIFIED)
    usage = rec.get("usage") or {}
    pt = usage.get("input") or usage.get("promptTokens") or usage.get("prompt_tokens")
    ct = usage.get("output") or usage.get("completionTokens") or usage.get("completion_tokens")

    return {
        "request_id": str(rec.get("id") or rec.get("traceId") or ""),
        "model": model,
        "messages": messages,
        "response": response_text,
        "prompt_tokens": int(pt) if isinstance(pt, (int, float)) else None,
        "completion_tokens": int(ct) if isinstance(ct, (int, float)) else None,
        "tools": tools,
        "tool_calls": tool_calls,
        "response_format": None,
        "latency_ms": _ms_between(rec.get("startTime"), rec.get("endTime")),
    }


# ── OpenAI request/response pair importer ─────────────────────────────────────


def _import_openai_pair(rec: dict[str, Any], counts: SkipCounts) -> dict[str, Any] | None:
    """Import an OpenAI-compatible request/response pair.

    Shape: ``{"request": {...chat.completions...}, "response": {...}}``
    This is the recommended portable format for platforms not natively supported.
    """
    request = rec.get("request")
    response = rec.get("response")
    if not isinstance(request, dict) or not isinstance(response, dict):
        _bump(counts, "openai_pair: request or response is not a dict")
        return None

    messages = request.get("messages")
    if not isinstance(messages, list):
        _bump(counts, "openai_pair: request.messages missing or not a list")
        return None
    messages = [m for m in messages if isinstance(m, dict)]

    tools = request.get("tools") or []
    if not isinstance(tools, list):
        tools = []
    tools = [t for t in tools if isinstance(t, dict)]

    # response_format: "json_object", "json_schema", etc.
    rf = request.get("response_format")
    if isinstance(rf, dict):
        rf = rf.get("type") or None
    elif not isinstance(rf, str):
        rf = None

    # Extract the assistant reply from choices[0].message
    choices = response.get("choices") or []
    message: dict[str, Any] = {}
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or {}

    response_text = str(message.get("content") or "")

    # tool_calls in the response message
    tool_calls: list[dict[str, Any]] = []
    tc_list = message.get("tool_calls")
    if isinstance(tc_list, list):
        for tc in tc_list:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            args = str(fn.get("arguments") or "")
            if name:
                tool_calls.append({"name": name, "arguments": args})

    usage = response.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")

    model = str(request.get("model") or response.get("model") or "unknown")
    req_id = str(response.get("id") or request.get("id") or "")

    return {
        "request_id": req_id,
        "model": model,
        "messages": messages,
        "response": response_text,
        "prompt_tokens": int(pt) if isinstance(pt, (int, float)) else None,
        "completion_tokens": int(ct) if isinstance(ct, (int, float)) else None,
        "tools": tools,
        "tool_calls": tool_calls,
        "response_format": rf,
        "latency_ms": None,
    }


# ── Top-level loader ───────────────────────────────────────────────────────────


def _parse_file(path: str | Path) -> list[Any]:
    """Read a JSONL or JSON file, returning a list of records.

    JSON arrays (pretty-printed or compact) and JSONL are both handled.
    Detection is from content, not filename: a file starting with ``[`` is
    tried as a JSON array first; otherwise each non-empty line is parsed as
    a separate JSON object.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    # JSON array: try first when the content starts with '[' — avoids the
    # JSONL path treating `[{...}, {...}]` as one record that is a list.
    if text.startswith("["):
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass  # not a valid JSON array; try JSONL
    # JSONL: one JSON object per non-empty line
    records: list[Any] = []
    errors = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            errors += 1
    if errors and not records:
        # Last resort: try parsing the whole thing as a single JSON value
        records = [json.loads(text)]
        if not isinstance(records, list):
            records = [records]
    return records


def load(path: str | Path, *, unredacted: bool = False) -> tuple[list[dict[str, Any]], SkipCounts]:
    """Load a trace export file as onpar capture rows.

    Reads JSONL (one record per line) or a JSON array. Format is auto-detected
    from each record's shape. Returns (rows, skip_counts).

    Raises:
        RuntimeError: the compiled extension is absent and *unredacted* is False
            — invariant 4, fails closed.
        json.JSONDecodeError: the file is not valid JSON or JSONL.
        OSError: the file cannot be read.
    """
    from onpar import core

    if not unredacted and not core.available():
        raise RuntimeError(
            f"importing traces requires the compiled redactor "
            f"({core.why_unavailable()}). "
            "Pass unredacted=True to skip redaction, or run: pip install onpar-core"
        )

    if unredacted:
        print(
            "  WARNING: --unredacted — the eval set will contain unredacted text "
            "and must not leave this machine.",
            file=sys.stderr,
        )

    records = _parse_file(path)
    counts: SkipCounts = {}
    rows: list[dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            _bump(counts, "not a JSON object")
            continue

        fmt = _detect_format(rec)
        if fmt is None:
            _bump(counts, "unrecognised format")
            continue

        if fmt == "langfuse":
            row = _import_langfuse(rec, counts)
        elif fmt == "openai_pair":
            row = _import_openai_pair(rec, counts)
        else:
            _bump(counts, f"unhandled format: {fmt}")
            continue

        if row is None:
            continue

        if not unredacted:
            row = _redact_row(row)

        rows.append(row)

    return rows, counts


# ── Self-check ─────────────────────────────────────────────────────────────────


def demo() -> None:
    """Assert-based self-check: run with `python -m onpar.distill.importers`."""
    # ── Format detection ──────────────────────────────────────────────────────
    assert _detect_format({"type": "GENERATION", "id": "x"}) == "langfuse"
    assert _detect_format({"request": {}, "response": {}}) == "openai_pair"
    assert _detect_format({"foo": "bar"}) is None
    assert _detect_format("not a dict") is None  # type: ignore[arg-type]

    # ── Anthropic normalisation ───────────────────────────────────────────────
    # system string
    assert _anthropic_system_text("Be helpful.") == "Be helpful."
    # system block list
    blocks = [{"type": "text", "text": "A"}, {"type": "image"}, {"type": "text", "text": "B"}]
    assert _anthropic_system_text(blocks) == "A\nB"

    # tool_use block → OpenAI tool_call
    msgs = _anthropic_content_to_messages(
        "assistant",
        [{"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "hi"}}],
    )
    assert len(msgs) == 1 and msgs[0]["role"] == "assistant"
    tc = msgs[0]["tool_calls"][0]
    assert tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"])["q"] == "hi"

    # tool_result blocks → separate role:tool messages
    msgs = _anthropic_content_to_messages(
        "user",
        [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "result A"},
            {"type": "tool_result", "tool_use_id": "tu2", "content": "result B"},
        ],
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool" and msgs[0]["tool_call_id"] == "tu1"

    # ── Langfuse importer ─────────────────────────────────────────────────────
    counts: SkipCounts = {}

    # Messages list input
    lf_rec = {
        "type": "GENERATION",
        "id": "obs1",
        "traceId": "tr1",
        "model": "gpt-4",
        "input": [{"role": "user", "content": "hello"}],
        "output": "Hi there",
        "usage": {"input": 10, "output": 5},
        "startTime": "2024-01-01T00:00:00.000Z",
        "endTime": "2024-01-01T00:00:01.000Z",
    }
    row = _import_langfuse(lf_rec, counts)
    assert row is not None
    assert row["request_id"] == "obs1"
    assert row["model"] == "gpt-4"
    assert row["messages"] == [{"role": "user", "content": "hello"}]
    assert row["response"] == "Hi there"
    assert row["prompt_tokens"] == 10
    assert row["latency_ms"] == 1000

    # Dict input with messages key
    lf_rec2 = {
        "type": "GENERATION",
        "id": "obs2",
        "model": "claude-3",
        "input": {"messages": [{"role": "user", "content": "translate"}]},
        "output": "hola",
        "usage": {"promptTokens": 8, "completionTokens": 2},
    }
    row2 = _import_langfuse(lf_rec2, {})
    assert row2 is not None
    assert row2["prompt_tokens"] == 8  # alternate spelling (UNVERIFIED field)

    # Skipped: missing input
    bad = {"type": "GENERATION", "id": "x", "model": "m"}
    r = _import_langfuse(bad, counts)
    assert r is None
    assert counts.get("langfuse: missing input", 0) == 1

    # ── OpenAI pair importer ──────────────────────────────────────────────────
    oai_rec = {
        "request": {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "summarise"}],
            "tools": [{"type": "function", "function": {"name": "fetch"}}],
        },
        "response": {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Here is the summary.",
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "fetch", "arguments": '{"url":"x"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        },
    }
    row3 = _import_openai_pair(oai_rec, {})
    assert row3 is not None
    assert row3["model"] == "gpt-4o"
    assert row3["messages"][0]["content"] == "summarise"
    assert row3["prompt_tokens"] == 20
    assert row3["tool_calls"][0]["name"] == "fetch"
    assert row3["response_format"] is None

    # ── Skipped records counting ──────────────────────────────────────────────
    counts2: SkipCounts = {}
    _import_openai_pair({"request": "not a dict", "response": {}}, counts2)
    assert counts2.get("openai_pair: request or response is not a dict") == 1

    # ── _parse_file handles both JSONL and JSON array ─────────────────────────
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in [lf_rec, lf_rec2]:
            f.write(json.dumps(rec) + "\n")
        jsonl_path = f.name
    records_from_jsonl = _parse_file(jsonl_path)
    assert len(records_from_jsonl) == 2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([lf_rec], f)
        json_path = f.name
    records_from_json = _parse_file(json_path)
    assert len(records_from_json) == 1

    print("importers: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
    sys.exit(0)
