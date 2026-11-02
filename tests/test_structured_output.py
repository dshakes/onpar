"""Structured output: say when the server must enforce it, and notice when it did not.

Found on a real proof: the eval item asked for `response_format: json_object`,
the candidate on mlx_lm.server answered `[{"name": ...}, ..., "total": 84.99]`,
and the structured-JSON cluster scored 0% with nothing pointing at the engine.
mlx has no constrained decoding; vLLM does. Two halves: `--structured-output`
makes the planner state the requirement (and its gap), and the collector names
unenforced format separately — while the item still fails, so the proof stays
honest.
"""

from __future__ import annotations

import json

import pytest

from onpar import catalog, launch
from onpar.cli import build_parser, main
from onpar.hardware import Hardware
from onpar.plan import Engine, Requirements, Workload, plan
from onpar.prove.collect import Collected, Collection
from onpar.prove.graders import EvalItem, JsonValid

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

#: The reply from the real run: a list that ends in a key/value pair. Not JSON.
MALFORMED = (
    '[{"name": "Widget", "price": 19.99}, {"name": "Gadget", "price": 65.0}, "total": 84.99]'
)
BASELINE = '{"items": [{"name": "Widget", "price": 19.99}], "total": 84.99}'


def _item(candidate: str, response_format: object = "json_object", n: int = 0) -> EvalItem:
    return EvalItem(
        item_id=f"i{n}",
        cluster="structured-json",
        prompt="price these",
        baseline=BASELINE,
        candidate=candidate,
        response_format=response_format,
    )


def _collection(*items: EvalItem) -> Collection:
    return Collection(
        base="http://127.0.0.1:8000/v1",
        model="m",
        side="candidate",
        replies=tuple(Collected(i.item_id, text=i.candidate) for i in items),
        items=items,
        seconds=0.1,
    )


# --- the collector ---------------------------------------------------------------


def test_the_real_malformed_reply_is_named_as_unenforced_format_and_still_fails():
    got = _collection(_item(MALFORMED), _item(BASELINE, n=1))
    assert got.format_unenforced == 1 and got.structured == 2
    line = next(x for x in got.render().splitlines() if "⚠" in x)
    assert "1 of 2 structured-output replies" in line
    assert "format not enforced by the server" in line
    assert "server setting, not a model verdict" in line
    # Still scored, still failed: the warning says where to look, it does not excuse.
    assert len(got.items) == 2 and not got.failures
    assert not JsonValid().grade(got.items[0]).passed


@pytest.mark.parametrize(
    ("response_format", "candidate", "counted"),
    [
        ("json_object", '[{"a": 1}]', True),  # valid JSON, but not the object asked for
        ({"type": "json_object"}, MALFORMED, True),  # the OpenAI request shape
        ("json", "Sure! Here is the JSON:", True),
        ({"type": "json_schema", "json_schema": {}}, '[{"a": 1}]', False),  # any JSON
        ({"type": "json_schema", "json_schema": {}}, MALFORMED, True),
        ("json_object", ' {"a": 1}\n', False),
        ("enum", "not json at all", False),  # asks for no JSON
        (None, MALFORMED, False),
    ],
)
def test_only_replies_that_miss_the_json_asked_for_are_counted(response_format, candidate, counted):
    assert _collection(_item(candidate, response_format)).format_unenforced == int(counted)


def test_a_clean_collection_says_nothing_about_format():
    got = _collection(_item(BASELINE), _item("free text", None, n=1))
    assert got.format_unenforced == 0
    assert "⚠" not in got.render()


def test_the_baseline_side_is_checked_against_the_baseline_text():
    item = _item(candidate=MALFORMED)
    got = Collection(
        base="b",
        model="m",
        side="baseline",
        replies=(Collected("i0", text=BASELINE),),
        items=(item,),
        seconds=0,
    )
    assert got.format_unenforced == 0


# --- the planner -----------------------------------------------------------------


def test_vllm_emits_its_cited_structured_output_flag():
    p = plan(
        H100, Requirements(Workload.INTERACTIVE, structured_output=True), catalog.get("qwen3-32b")
    )
    assert p.engine is Engine.VLLM
    argv, gaps = p.command("Qwen/Qwen3-32B")
    assert json.loads(argv[argv.index("--structured-outputs-config") + 1]) == {
        "backend": "xgrammar"
    }
    assert not [g for g in gaps if g.startswith("structured_output")]


def test_mlx_reports_it_under_not_expressed(tmp_path):
    p = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        exists=lambda repo: True,
        cache=tmp_path / "w.json",
        structured_output=True,
    )
    assert isinstance(p, launch.LaunchPlan), p
    assert "structured_output: no grammar or structured-output flag" in p.render()


def test_without_the_flag_nothing_is_said_about_structured_output(tmp_path):
    p = launch.plan("qwen3-30b-a3b", hw=M4, exists=lambda repo: True, cache=tmp_path / "w.json")
    assert isinstance(p, launch.LaunchPlan), p
    assert "structured_output" not in p.render()
    assert "--structured-outputs-config" not in p.argv


# --- run and build ---------------------------------------------------------------


def test_the_cli_flags_parse_and_default_off():
    parser = build_parser()
    assert parser.parse_args(["run", "qwen3-30b-a3b"]).structured_output is False
    assert parser.parse_args(["run", "m", "--structured-output"]).structured_output is True
    assert parser.parse_args(["build", "x", "--structured-output"]).structured_output is True


def test_run_forwards_structured_output_to_the_launcher(monkeypatch, capsys):
    seen = {}

    def fake(model_id, **kw):
        seen.update(kw)
        return launch.Refusal(reason="stub", next_step="none")

    monkeypatch.setattr(launch, "plan", fake)
    main(["run", "qwen3-30b-a3b", "--dry-run", "--structured-output"])
    assert seen["structured_output"] is True
    main(["run", "qwen3-30b-a3b", "--dry-run"])
    assert seen["structured_output"] is False


def test_build_carries_structured_output_into_the_session(capsys):
    main(["build", "chatbot for 10 people", "--on", "h100", "--structured-output", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["state"]["requirements"]["structured_output"] is True, out["state"]
    assert "--structured-outputs-config" in out["answer"]


def test_sglang_emits_its_cited_grammar_backend():
    req = Requirements(
        Workload.INTERACTIVE, concurrency=8, prefix_sharing=0.9, structured_output=True
    )
    p = plan(H100, req, catalog.get("qwen3-32b"))
    assert p.engine is Engine.SGLANG
    argv, gaps = p.command("Qwen/Qwen3-32B")
    assert argv[argv.index("--grammar-backend") + 1] == "xgrammar"
    assert not [g for g in gaps if g.startswith("structured_output")]
