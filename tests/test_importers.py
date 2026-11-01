"""Tests for onpar.distill.importers — trace import from external platforms."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onpar.distill.importers import (
    _anthropic_content_to_messages,
    _anthropic_system_text,
    _detect_format,
    _import_langfuse,
    _import_openai_pair,
    _parse_file,
    load,
)

# ── Format detection ──────────────────────────────────────────────────────────


def test_detect_format_langfuse():
    assert _detect_format({"type": "GENERATION", "id": "x"}) == "langfuse"


def test_detect_format_openai_pair():
    assert _detect_format({"request": {}, "response": {}}) == "openai_pair"


def test_detect_format_unknown_returns_none():
    assert _detect_format({"model": "gpt-4"}) is None


def test_detect_format_non_dict_returns_none():
    assert _detect_format("not a dict") is None  # type: ignore[arg-type]


# ── Anthropic normalisation ───────────────────────────────────────────────────


def test_anthropic_system_string_passthrough():
    assert _anthropic_system_text("Be helpful.") == "Be helpful."


def test_anthropic_system_block_list_joins_text_skips_other():
    blocks = [
        {"type": "text", "text": "Part 1"},
        {"type": "image", "source": {}},
        {"type": "text", "text": "Part 2"},
    ]
    assert _anthropic_system_text(blocks) == "Part 1\nPart 2"


def test_anthropic_system_unknown_type_returns_empty():
    assert _anthropic_system_text(42) == ""


def test_anthropic_string_content_passes_through():
    msgs = _anthropic_content_to_messages("user", "hello world")
    assert msgs == [{"role": "user", "content": "hello world"}]


def test_anthropic_none_content_becomes_empty_string():
    msgs = _anthropic_content_to_messages("user", None)
    assert msgs == [{"role": "user", "content": ""}]


def test_anthropic_text_block_list_joins_text():
    content = [
        {"type": "text", "text": "A"},
        {"type": "image"},
        {"type": "text", "text": "B"},
    ]
    msgs = _anthropic_content_to_messages("user", content)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "A\nB"


def test_anthropic_tool_use_becomes_assistant_tool_calls():
    content = [
        {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "rust"}},
    ]
    msgs = _anthropic_content_to_messages("assistant", content)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    tc = msgs[0]["tool_calls"][0]
    assert tc["function"]["name"] == "search"
    assert json.loads(tc["function"]["arguments"])["q"] == "rust"


def test_anthropic_tool_result_blocks_become_separate_tool_messages():
    content = [
        {"type": "tool_result", "tool_use_id": "tu1", "content": "result A"},
        {"type": "tool_result", "tool_use_id": "tu2", "content": "result B"},
    ]
    msgs = _anthropic_content_to_messages("user", content)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool" and msgs[0]["tool_call_id"] == "tu1"
    assert msgs[1]["role"] == "tool" and msgs[1]["tool_call_id"] == "tu2"


def test_anthropic_tool_result_with_block_list_joins_text():
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "tu3",
            "content": [{"type": "text", "text": "X"}, {"type": "text", "text": "Y"}],
        }
    ]
    msgs = _anthropic_content_to_messages("user", content)
    assert msgs[0]["content"] == "X\nY"


# ── Langfuse importer ─────────────────────────────────────────────────────────

_LANGFUSE_GENERATION = {
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


def test_langfuse_messages_list_input():
    row = _import_langfuse(_LANGFUSE_GENERATION, {})
    assert row is not None
    assert row["request_id"] == "obs1"
    assert row["model"] == "gpt-4"
    assert row["messages"] == [{"role": "user", "content": "hello"}]
    assert row["response"] == "Hi there"
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    assert row["latency_ms"] == 1000


def test_langfuse_dict_input_with_messages_key():
    rec = {
        **_LANGFUSE_GENERATION,
        "input": {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        },
    }
    row = _import_langfuse(rec, {})
    assert row is not None
    assert len(row["messages"]) == 2
    assert row["messages"][0]["role"] == "system"


def test_langfuse_alternate_token_spellings():
    rec = {
        **_LANGFUSE_GENERATION,
        "usage": {"promptTokens": 8, "completionTokens": 3},
    }
    row = _import_langfuse(rec, {})
    assert row is not None
    assert row["prompt_tokens"] == 8
    assert row["completion_tokens"] == 3


def test_langfuse_string_input_becomes_user_message():
    rec = {**_LANGFUSE_GENERATION, "input": "just a string"}
    row = _import_langfuse(rec, {})
    assert row is not None
    assert row["messages"] == [{"role": "user", "content": "just a string"}]


def test_langfuse_tools_in_input_dict():
    tool_def = {"type": "function", "function": {"name": "calc"}}
    rec = {
        **_LANGFUSE_GENERATION,
        "input": {
            "messages": [{"role": "user", "content": "add"}],
            "tools": [tool_def],
        },
    }
    row = _import_langfuse(rec, {})
    assert row is not None
    assert row["tools"] == [tool_def]


def test_langfuse_tools_in_metadata():
    tool_def = {"type": "function", "function": {"name": "fetch"}}
    rec = {
        **_LANGFUSE_GENERATION,
        "metadata": {"tools": [tool_def]},
    }
    row = _import_langfuse(rec, {})
    assert row is not None
    assert row["tools"] == [tool_def]


def test_langfuse_anthropic_shaped_input():
    rec = {
        "type": "GENERATION",
        "id": "a1",
        "model": "claude-3",
        "input": {
            "model": "claude-3",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object"},
                }
            ],
        },
        "output": {
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "x"}},
            ]
        },
        "usage": {"input": 15, "output": 8},
    }
    row = _import_langfuse(rec, {})
    assert row is not None
    # system message is first
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][0]["content"] == "You are helpful."
    # tool in openai shape
    assert row["tools"][0]["function"]["name"] == "search"
    assert row["tools"][0]["function"]["parameters"] == {"type": "object"}
    # response text from text block
    assert "Hello!" in row["response"]
    # tool call from tool_use block
    assert row["tool_calls"][0]["name"] == "search"


def test_langfuse_skip_missing_input():
    counts: dict[str, int] = {}
    rec = {"type": "GENERATION", "id": "x", "model": "m", "output": "y"}
    row = _import_langfuse(rec, counts)
    assert row is None
    assert counts.get("langfuse: missing input", 0) == 1


def test_langfuse_skip_unrecognised_input_shape():
    counts: dict[str, int] = {}
    rec = {**_LANGFUSE_GENERATION, "input": 12345}
    row = _import_langfuse(rec, counts)
    assert row is None
    assert counts.get("langfuse: unrecognised input shape", 0) == 1


# ── OpenAI pair importer ──────────────────────────────────────────────────────

_OAI_PAIR = {
    "request": {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "summarise this"}],
        "tools": [{"type": "function", "function": {"name": "fetch"}}],
        "response_format": {"type": "json_object"},
    },
    "response": {
        "id": "chatcmpl-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here is the summary.",
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    },
}


def test_openai_pair_basic_fields():
    row = _import_openai_pair(_OAI_PAIR, {})
    assert row is not None
    assert row["model"] == "gpt-4o"
    assert row["messages"] == [{"role": "user", "content": "summarise this"}]
    assert row["response"] == "Here is the summary."
    assert row["prompt_tokens"] == 20
    assert row["completion_tokens"] == 10
    assert row["response_format"] == "json_object"
    assert row["tools"] == [{"type": "function", "function": {"name": "fetch"}}]


def test_openai_pair_tool_calls_in_response():
    rec = {
        "request": {"model": "gpt-4o", "messages": [{"role": "user", "content": "add"}]},
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    }
    row = _import_openai_pair(rec, {})
    assert row is not None
    assert row["tool_calls"] == [{"name": "add", "arguments": '{"a":1,"b":2}'}]


def test_openai_pair_skip_bad_request():
    counts: dict[str, int] = {}
    row = _import_openai_pair({"request": "string", "response": {}}, counts)
    assert row is None
    assert counts.get("openai_pair: request or response is not a dict") == 1


def test_openai_pair_skip_missing_messages():
    counts: dict[str, int] = {}
    row = _import_openai_pair({"request": {"model": "m"}, "response": {}}, counts)
    assert row is None
    assert counts.get("openai_pair: request.messages missing or not a list") == 1


# ── _parse_file ───────────────────────────────────────────────────────────────


def test_parse_file_jsonl(tmp_path: Path):
    p = tmp_path / "traces.jsonl"
    lines = [_LANGFUSE_GENERATION, _OAI_PAIR]
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    records = _parse_file(p)
    assert len(records) == 2


def test_parse_file_json_array(tmp_path: Path):
    p = tmp_path / "traces.json"
    p.write_text(json.dumps([_LANGFUSE_GENERATION, _OAI_PAIR]))
    records = _parse_file(p)
    assert len(records) == 2


def test_parse_file_json_single_object(tmp_path: Path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(_LANGFUSE_GENERATION))
    records = _parse_file(p)
    assert len(records) == 1


# ── load() — redaction and fail-closed behaviour ──────────────────────────────


def test_load_fails_closed_when_core_unavailable(monkeypatch, tmp_path: Path):
    """Invariant 4: refuses to import without redaction when core is absent."""
    from onpar import core

    monkeypatch.setattr(core, "available", lambda: False)
    monkeypatch.setattr(
        core,
        "why_unavailable",
        lambda: "the compiled extension is not installed (pip install onpar-core).",
    )

    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps(_LANGFUSE_GENERATION) + "\n")

    with pytest.raises(RuntimeError, match="pip install onpar-core"):
        load(p)


def test_load_unredacted_skips_redaction_with_warning(monkeypatch, tmp_path: Path, capsys):
    """--unredacted skips the core check and prints a one-line warning."""
    from onpar import core

    monkeypatch.setattr(core, "available", lambda: False)

    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps(_LANGFUSE_GENERATION) + "\n")

    rows, counts = load(p, unredacted=True)
    assert len(rows) == 1
    assert "unredacted" in capsys.readouterr().err.lower()


def test_load_returns_skip_counts_for_unrecognised_records(monkeypatch, tmp_path: Path):
    """Unrecognised records are counted, not raised."""
    from onpar import core

    monkeypatch.setattr(core, "available", lambda: True)
    # patch redact to identity so we don't need the extension built
    monkeypatch.setattr(core, "redact", lambda s: (s, {}))

    p = tmp_path / "traces.jsonl"
    records = [
        _LANGFUSE_GENERATION,
        {"some": "unknown format"},
        _OAI_PAIR,
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    rows, counts = load(p)
    assert len(rows) == 2
    assert counts.get("unrecognised format") == 1


def test_load_skips_non_generation_langfuse_records(monkeypatch, tmp_path: Path):
    """Langfuse records with type != GENERATION are not GENERATION — unrecognised."""
    from onpar import core

    monkeypatch.setattr(core, "available", lambda: True)
    monkeypatch.setattr(core, "redact", lambda s: (s, {}))

    p = tmp_path / "traces.jsonl"
    records = [
        {"type": "SPAN", "id": "s1"},
        _LANGFUSE_GENERATION,
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    rows, counts = load(p)
    assert len(rows) == 1
    assert counts.get("unrecognised format") == 1


# ── CLI integration test ──────────────────────────────────────────────────────


def test_cli_distill_import_produces_eval_set(monkeypatch, tmp_path: Path):
    """onpar distill --import f.jsonl --out e.json writes eval set with expected keys."""
    from onpar import cli, core

    monkeypatch.setattr(core, "available", lambda: True)
    monkeypatch.setattr(core, "redact", lambda s: (s, {}))

    # Build a JSONL with enough records to form at least one cluster
    records = []
    for i in range(6):
        records.append(
            {
                "type": "GENERATION",
                "id": f"obs{i}",
                "model": "gpt-4",
                "input": [{"role": "user", "content": f"summarise doc {i}"}],
                "output": "a summary",
                "usage": {"input": 40, "output": 10},
            }
        )
    # Add a tool-calling record to get a second cluster
    records.append(
        {
            "request": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "refund order 1"}],
                "tools": [{"type": "function", "function": {"name": "refund"}}],
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tc1",
                                    "type": "function",
                                    "function": {"name": "refund", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 5},
            },
        }
    )

    import_file = tmp_path / "traces.jsonl"
    import_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out_file = tmp_path / "evalset.json"

    code = cli.main(
        [
            "distill",
            "--import",
            str(import_file),
            "--unredacted",
            "--out",
            str(out_file),
            "--budget",
            "20",
            "--min-per-cluster",
            "2",
        ]
    )
    assert code == 0, f"distill returned {code}"
    assert out_file.exists()

    doc = json.loads(out_file.read_text())
    assert "items" in doc
    assert len(doc["items"]) > 0

    # Every item carries the fields the rest of the chain needs
    for item in doc["items"]:
        assert "messages" in item, "item missing messages"
        assert "tools" in item, "item missing tools"
        assert "session" in item, "item missing session"
        assert "step" in item, "item missing step"


def test_cli_distill_without_import_and_no_log_errors(monkeypatch, tmp_path: Path, capsys):
    """Existing behaviour preserved: no --import and no log → error."""
    from onpar import cli, core

    monkeypatch.setattr(core, "available", lambda: True)

    code = cli.main(
        [
            "distill",
            "--capture",
            str(tmp_path / "nonexistent.log"),
            "--key",
            str(tmp_path / "nonexistent.key"),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code != 0


def test_cli_distill_import_fails_closed_without_unredacted(monkeypatch, tmp_path: Path):
    """Without --unredacted, importing without core returns nonzero."""
    from onpar import cli, core

    monkeypatch.setattr(core, "available", lambda: False)
    monkeypatch.setattr(core, "why_unavailable", lambda: "not installed (pip install onpar-core).")

    p = tmp_path / "traces.jsonl"
    p.write_text(json.dumps(_LANGFUSE_GENERATION) + "\n")

    code = cli.main(
        [
            "distill",
            "--import",
            str(p),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert code != 0


# ── demo() self-check ─────────────────────────────────────────────────────────


def test_demo_runs_without_error():
    """demo() should pass all its own assertions."""
    from onpar.distill.importers import demo

    demo()
