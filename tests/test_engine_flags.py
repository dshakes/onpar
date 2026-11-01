"""The flags we emit, checked against the engine that has to accept them.

`engines.py` cites a docs URL for every flag. Docs have been wrong twice already
in ways that passed every test in this repo and could not start a server:
`--speculative-config eagle3` (needs JSON) and `--quantization q4` (needs a
method name, not a precision). MLX escaped both, because MLX is the one engine
the author can install and `installed_flags` reads its `--help` off the binary.

This does the same for vLLM and SGLang. It cannot run on a Mac — neither engine
builds there — so it runs on Linux in CI (`.github/workflows/engine-flags.yml`)
and for free on any developer machine where the engine happens to be installed.

No GPU is needed for either, which took a while to establish: vLLM's CUDA image
dies at `Failed to infer device type` before printing help, and that was read as
"vLLM cannot be asked without a GPU". The device check belongs to the image's
platform plugin, so the same binary in `vllm/vllm-tpu:latest` answers fine.

## The skip is the dangerous part

`installed_flags` returns `None` for "could not ask", which is deliberately not
the same as "accepts nothing". Locally that is a skip. In CI a skip would be a
green tick over an unasked question — the exact defect this repo removed from
its own CI once already — so `ONPAR_REQUIRE_ENGINES=1` turns "could not ask"
into a failure. That variable is what makes the check able to fail, and a check
that cannot fail is worse than none.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from onpar import catalog
from onpar.engines import (
    SGLANG_KV_DTYPE,
    LoraFleet,
    SglangAdapter,
    Translated,
    Unsupported,
    adapter_for,
)
from onpar.hardware import Hardware
from onpar.plan import Requirements, Setting, Workload, plan

#: Synthetic accelerators. The CI runner has no GPU and never will — this check
#: is about what the engine's argument parser accepts, not about running a model.
H100 = Hardware(
    kind="nvidia",
    name="H100 80GB",
    total_bytes=80 * 2**30,
    usable_bytes=76 * 2**30,
    bandwidth_gbps=3350.0,
    cores=132,
)
QUAD = replace(H100, devices=4)

#: Set in CI. Makes "the engine could not be interrogated" a failure rather than
#: a skip, so an install that silently broke cannot read as a pass.
REQUIRE = "ONPAR_REQUIRE_ENGINES"

#: (label, model id, quantisation, hardware, requirements). Chosen to cover every
#: flag the planner can emit, not to be representative of demand: tensor
#: parallelism, structured output, both LoRA dialects, fp8 quantisation, the MTP
#: speculative config, and both workloads' opposite prefill settings.
_CASES = (
    ("interactive 8B", "llama-3.1-8b", "fp16", H100, Requirements(Workload.INTERACTIVE, 8)),
    ("batch 32B", "qwen3-32b", "q4", H100, Requirements(Workload.BATCH, 32)),
    (
        "realtime 70B, 4-way, structured",
        "llama-3.3-70b",
        "q4",
        QUAD,
        Requirements(Workload.REALTIME, 8, itl_ms=50, structured_output=True),
    ),
    (
        # The only case that emits --speculative-config: this family carries its
        # own MTP head, so the planner has a method it can name without a draft
        # checkpoint. Also the only one that emits --quantization.
        "deepseek-v3 fp8, mtp speculative",
        "deepseek-v3",
        "fp8",
        QUAD,
        Requirements(Workload.INTERACTIVE, 4),
    ),
    (
        "vllm multi-LoRA",
        "llama-3.1-8b",
        "fp16",
        H100,
        Requirements(
            Workload.INTERACTIVE,
            2,
            lora=LoraFleet((("support", "org/support"), ("sql", "org/sql")), 48, 2),
        ),
    ),
    (
        # prefix_sharing above PREFIX_SHARING_FOR_RADIX is what selects SGLang;
        # below it the planner picks vLLM and this file would never test SGLang.
        "sglang multi-LoRA, high prefix sharing",
        "qwen3-32b",
        "q4",
        H100,
        Requirements(
            Workload.INTERACTIVE,
            8,
            prefix_sharing=0.8,
            lora=LoraFleet((("support", "org/support"), ("sql", "org/sql")), 48, 2),
        ),
    ),
    (
        "sglang batch, high prefix sharing",
        "mistral-small-24b",
        "fp16",
        H100,
        Requirements(Workload.BATCH, 32, prefix_sharing=0.9),
    ),
    (
        # vLLM tool calling: a known family so --enable-auto-tool-choice and
        # --tool-call-parser must appear. Also checks --tool-strict-level.
        "vllm tool calling, llama3",
        "llama-3.1-8b",
        "fp16",
        H100,
        Requirements(Workload.INTERACTIVE, 8, tool_calling=True),
    ),
    (
        # SGLang tool calling on a Qwen model: --tool-call-parser qwen25.
        "sglang tool calling, qwen3, high prefix sharing",
        "qwen3-32b",
        "q4",
        H100,
        Requirements(Workload.INTERACTIVE, 8, prefix_sharing=0.8, tool_calling=True),
    ),
)


def cases() -> list[tuple[str, str, list[str]]]:
    """(label, engine, argv) for every case, generated exactly as the CLI does.

    Through `plan()` and `Plan.command()` rather than by hand: a hand-written
    argv would test this file's idea of the flags, which is the thing already
    known to drift. `--host`/`--port` are appended because `launch` appends them,
    and they are flags the engine has to accept too.
    """
    out = []
    for label, model_id, quant, hw, req in _CASES:
        model = catalog.get(model_id)
        p = plan(hw, req, model, quant)
        argv, _gaps = p.command(model.repo or model.id)
        if not argv:
            # No verified dialect for the chosen engine (llm-d). Nothing to check.
            continue
        out.append((label, p.engine.value, argv + ["--host", "127.0.0.1", "--port", "8000"]))
    return out


def _for(engine: str) -> list[tuple[str, list[str]]]:
    return [(label, argv) for label, eng, argv in cases() if eng == engine]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_every_generated_flag_is_one_the_installed_engine_accepts(engine: str):
    """Every `--flag` we emit appears in the installed engine's own `--help`.

    Not a docs check. `installed_flags` runs the real binary, so this fails the
    day an engine renames a flag underneath us — which is how the last two got
    found, months after they shipped.
    """
    from onpar.engines import _installed_flags_detail, unknown_flags

    adapter = adapter_for(engine)
    assert adapter is not None, engine

    flags, reason = _installed_flags_detail(adapter)
    if flags is None:
        asked = " ".join(adapter.help_argv)
        why = f"{engine} could not be interrogated: `{asked}` did not run — {reason}"
        if os.environ.get(REQUIRE):
            pytest.fail(
                f"{why}. {REQUIRE} is set, so this is a failure rather than a skip: "
                f"an engine that cannot be asked has not been checked, and a green "
                f"tick over an unasked question is what this job exists to prevent"
            )
        pytest.skip(f"{why} (set {REQUIRE}=1 to make this a failure)")

    checked = _for(engine)
    assert checked, f"no case in this file plans onto {engine}; it is going unchecked"

    bad = {label: rejected for label, argv in checked if (rejected := unknown_flags(adapter, argv))}
    assert not bad, (
        f"the installed {engine} rejects flags this repo generates:\n"
        + "\n".join(f"  {label}: {', '.join(flags)}" for label, flags in bad.items())
        + f"\n{engine}'s own --help is the authority; the dialect in engines.py has drifted"
    )


def test_the_cases_cover_the_flags_worth_covering():
    """A case list that stopped exercising a flag would pass while proving less.

    Runs everywhere, including on the Mac where neither engine installs, so a
    change that quietly narrows the coverage is caught before CI is reached.
    """
    emitted = {a for _, _, argv in cases() for a in argv if a.startswith("--")}
    for flag in (
        "--max-model-len",  # vllm
        "--context-length",  # sglang: same intent, different name
        "--enable-chunked-prefill",
        "--chunked-prefill-size",
        "--tensor-parallel-size",
        "--structured-outputs-config",  # the flag that replaced --guided-decoding-backend
        "--speculative-config",  # the JSON one that shipped broken
        "--quantization",
        "--lora-modules",  # vllm dialect
        "--lora-paths",  # sglang dialect
        "--max-loras",
        "--max-loras-per-batch",
        "--kv-cache-dtype",
        "--host",
        "--enable-auto-tool-choice",  # vllm tool calling
        "--tool-call-parser",  # both engines (different values per family)
        "--tool-strict-level",  # vllm envelope constraint
    ):
        assert flag in emitted, f"no case emits {flag}; it is validated by nothing"

    engines = {eng for _, eng, _ in cases()}
    assert engines == {"vllm", "sglang"}, engines


if __name__ == "__main__":
    # `python3 tests/test_engine_flags.py` prints what CI is about to validate.
    # The list belongs in the job's log: a check whose subject is invisible is
    # one nobody can tell has narrowed.
    for label, engine, argv in cases():
        print(f"[{engine}] {label}\n  {' '.join(argv)}\n")


# --- the two most-installed runtimes -------------------------------------------


def test_llamacpp_divides_context_among_slots_so_we_multiply_it():
    """`--ctx-size` is the TOTAL context llama.cpp shares across `--parallel`
    slots, not the per-sequence figure the planner means.

    Passed straight through, every slot silently gets a fraction of what was
    asked for: request 8192 across 4 slots and each sequence has 2048. The
    model would serve, answer, and truncate long prompts — the quiet kind of
    wrong, since nothing errors.
    """
    from onpar.engines import Setting, adapter_for

    a = adapter_for("llama.cpp")
    assert a is not None
    argv, _ = a.command("/m.gguf", {Setting.CONTEXT_LENGTH: 8192, Setting.MAX_CONCURRENT: 4})
    assert "--ctx-size" in argv
    assert argv[argv.index("--ctx-size") + 1] == "32768", f"per-slot context was not scaled: {argv}"
    assert argv[argv.index("--parallel") + 1] == "4"

    # One slot needs no multiplication, and must not get one.
    solo, _ = a.command("/m.gguf", {Setting.CONTEXT_LENGTH: 8192, Setting.MAX_CONCURRENT: 1})
    assert solo[solo.index("--ctx-size") + 1] == "8192"


def test_ollama_is_configured_by_environment_and_says_so():
    """`ollama serve` has exactly one flag — `--help`. Everything it can be told
    is told through OLLAMA_*, so an adapter that only speaks argv would have to
    call every setting unsupported, which would be false: Ollama honours
    concurrency and context as well as any engine here.
    """
    from onpar.engines import Setting, adapter_for

    a = adapter_for("ollama")
    assert a is not None
    argv, gaps = a.command("llama3.1:8b", {Setting.CONTEXT_LENGTH: 8192, Setting.MAX_CONCURRENT: 4})
    line = " ".join(argv)
    assert line == "ollama serve", line
    assert not gaps, [g.reason for g in gaps]

    # The settings are honoured through the environment, which is a separate
    # channel from argv — this assertion used to demand they were IN argv, which
    # is not runnable and is the defect both reviewers caught.
    assert a.environment({Setting.CONTEXT_LENGTH: 8192, Setting.MAX_CONCURRENT: 4}) == (
        ("OLLAMA_CONTEXT_LENGTH", "8192"),
        ("OLLAMA_NUM_PARALLEL", "4"),
    )
    # And the model is NOT on the command line: `serve` starts a daemon and the
    # model is requested by name over the API. Emitting it would not run.
    assert "llama3.1:8b" not in line


@pytest.mark.parametrize("engine", ["llama.cpp", "ollama"])
def test_the_new_adapters_emit_only_flags_their_real_binary_accepts(engine):
    """The discipline that caught `--speculative-config eagle3`: ask the binary,
    do not trust a table in this file. Skipped where the engine is absent unless
    ONPAR_REQUIRE_ENGINES makes that a failure."""
    from onpar.engines import Setting, adapter_for, unknown_flags

    a = adapter_for(engine)
    assert a is not None
    argv, _ = a.command(
        "m",
        {
            Setting.CONTEXT_LENGTH: 8192,
            Setting.MAX_CONCURRENT: 4,
            Setting.PREFIX_REUSE: True,
        },
    )
    bad = unknown_flags(a, argv)
    if bad is None:
        if os.environ.get("ONPAR_REQUIRE_ENGINES"):
            pytest.fail(f"{engine} could not be interrogated and that was required")
        pytest.skip(f"{engine} is not installed here")
    assert bad == [], f"{engine} rejects: {bad}"


def test_quantisation_is_refused_by_both_because_it_is_baked_into_the_file():
    """GGUF is already Q4_K_M on disk and an Ollama tag already names its
    precision. A serve-time `--quantization` flag would be invented, which is
    the failure this adapter layer exists to prevent."""
    from onpar.engines import Setting, Unsupported, adapter_for

    for engine, word in (("llama.cpp", "gguf"), ("ollama", "tag")):
        a = adapter_for(engine)
        out = a.translate(Setting.QUANTIZATION, "q4")
        assert isinstance(out, Unsupported), f"{engine} claimed to convert at start-up"
        assert word in out.reason.lower(), out.reason


def test_ollama_env_is_never_argv_because_argv_gets_executed():
    """Both reviewers caught this, and the docstring was the tell: it claimed the
    output was "native, standalone and pasteable" and the code backed none of it.

    The first cut returned `["OLLAMA_CONTEXT_LENGTH=8192", "ollama", "serve"]`.
    `subprocess.Popen` would try to exec a binary named
    `OLLAMA_CONTEXT_LENGTH=8192`; the printed form `shlex.quote`s each element,
    so a paste into bash gives `command not found`. Right-looking, runnable
    neither way.
    """
    from onpar.engines import Setting, adapter_for

    a = adapter_for("ollama")
    st = {Setting.CONTEXT_LENGTH: 8192, Setting.MAX_CONCURRENT: 4}
    argv, _ = a.command("llama3.1:8b", st)

    assert argv[0] == "ollama", f"argv[0] must be executable, got {argv[0]!r}"
    assert not any("=" in tok for tok in argv), f"environment leaked into argv: {argv}"
    assert a.environment(st) == (
        ("OLLAMA_CONTEXT_LENGTH", "8192"),
        ("OLLAMA_NUM_PARALLEL", "4"),
    )
    # Flag-driven engines have no environment, and must not grow one by accident.
    assert adapter_for("llama.cpp").environment(st) == ()
    assert adapter_for("vllm").environment(st) == ()


def test_llamacpp_never_emits_another_engines_kv_cache_spelling():
    """The planner speaks vLLM's KV vocabulary. `llama-server --help` allows
    f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 — and nothing named
    fp8. Passed straight through, the server rejects it at start-up: the
    cross-dialect leak this whole layer exists to prevent, committed inside the
    layer itself.
    """
    from onpar.engines import LLAMACPP_KV_DTYPE, Setting, Translated, Unsupported, adapter_for

    a = adapter_for("llama.cpp")
    allowed = {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}

    # Whatever the planner asks for, we either emit something llama.cpp accepts
    # or refuse — never a passthrough.
    for asked in ("fp8_e4m3", "fp8_e5m2", "fp8", "auto", "fp16", "int8", "nonsense_dtype"):
        out = a.translate(Setting.KV_CACHE_DTYPE, asked)
        if isinstance(out, Unsupported):
            continue
        assert isinstance(out, Translated)
        emitted = out.argv[1]
        assert emitted in allowed, f"{asked!r} became {emitted!r}, which llama.cpp rejects"

    assert set(LLAMACPP_KV_DTYPE.values()) <= allowed
    assert isinstance(a.translate(Setting.KV_CACHE_DTYPE, "nonsense_dtype"), Unsupported)


@pytest.mark.parametrize("engine", ["vllm", "sglang", "llama.cpp", "mlx", "ollama"])
def test_speculative_off_is_honoured_not_reported_as_a_gap(engine):
    """ "off" is the planner's decision, not the user's request.

    `plan()` attaches a SPECULATIVE knob to every plan and sets it "off" for
    every BATCH workload and whenever concurrency exceeds the spec-decode
    ceiling — so adapters see it constantly, unasked. vLLM and SGLang returned
    `Translated((), ...)`, a successful no-op. llama.cpp, MLX and Ollama
    returned `Unsupported`, which becomes a gap, and every renderer that
    surfaces gaps showed the user an optimisation the engine "could not
    express" when nothing had been wanted.

    Those three are the most-installed engines in the catalogue, and the suite
    missed it because the tests that drive the real `plan()` path assert they
    only ever produce vllm/sglang plans — so this branch was never reached.
    """
    from onpar.engines import Setting, Translated, adapter_for

    got = adapter_for(engine).translate(Setting.SPECULATIVE, "off")
    assert isinstance(got, Translated), f"{engine} reports a gap for an unrequested setting"
    assert got.argv == (), f"{engine} emitted flags for speculative=off: {got.argv}"


def test_a_real_draft_request_is_still_answered_on_its_merits():
    """The control: silencing "off" must not silence a genuine request.

    Ollama exposes no speculative setting at all, so a real draft model is still
    an honest `Unsupported` — that gap is information, not noise.
    """
    from onpar.engines import Setting, Unsupported, adapter_for

    assert isinstance(
        adapter_for("ollama").translate(Setting.SPECULATIVE, "org/draft-model"), Unsupported
    )


# --- SGLang shares vLLM's flag name, not its accepted values ---------------------


@pytest.mark.parametrize("dtype", sorted(SGLANG_KV_DTYPE))
def test_sglang_emits_every_kv_dtype_it_documents(dtype):
    r = SglangAdapter().translate(Setting.KV_CACHE_DTYPE, dtype)
    assert isinstance(r, Translated)
    assert r.argv == ("--kv-cache-dtype", dtype)


@pytest.mark.parametrize(
    ("dtype", "whose"),
    [
        ("fp8_inc", "vLLM's Gaudi-only value"),
        ("fp8", "a vLLM alias SGLang does not document"),
        ("q8_0", "llama.cpp's vocabulary"),
        ("f16", "llama.cpp's vocabulary"),
    ],
)
def test_sglang_refuses_a_kv_dtype_from_another_engines_dialect(dtype, whose):
    # The flag name is shared with vLLM and the accepted values are not. The
    # code carried a comment saying exactly that — "passing vLLM's fp8_inc here
    # would be rejected at startup" — with no check behind it, which is the
    # cross-dialect leak this layer exists to prevent, committed inside the
    # layer itself.
    r = SglangAdapter().translate(Setting.KV_CACHE_DTYPE, dtype)
    assert isinstance(r, Unsupported), whose
    assert dtype in r.reason
    assert "fp8_e4m3" in r.reason, "a refusal must name the set it will take"


def test_the_value_the_planner_actually_emits_is_accepted():
    # `_kv_cache_dtype` emits "fp8_e4m3". If the allowlist ever stopped
    # containing it, every KV-pressured SGLang plan would silently lose the
    # setting it was built to apply — a refusal is only correct for values the
    # planner does not produce.
    r = SglangAdapter().translate(Setting.KV_CACHE_DTYPE, "fp8_e4m3")
    assert isinstance(r, Translated)


# --- tool calling: parser must match the model family --------------------------


@pytest.mark.parametrize(
    ("repo", "vllm_parser", "sglang_parser"),
    [
        ("Qwen/Qwen3-32B", "hermes", "qwen25"),
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "qwen3_xml", "qwen3_coder"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama3_json", "llama3"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama3_json", "llama3"),
        ("mistralai/Mistral-Small-24B-Instruct-2501", "mistral", "mistral"),
        ("deepseek-ai/DeepSeek-V3", "deepseek_v3", "deepseekv3"),
        ("THUDM/glm-4.5-9b", "glm45", "glm45"),
        ("moonshotai/Kimi-K2-Instruct", "kimi_k2", "kimi_k2"),
    ],
)
def test_known_family_maps_to_cited_parser_on_both_engines(repo, vllm_parser, sglang_parser):
    """The parser name is what makes tool calls arrive as tool_calls; a wrong
    name means every call comes back as plain text."""
    from onpar.engines import Setting, SglangAdapter, Translated, VllmAdapter

    v_out = VllmAdapter().translate(Setting.TOOL_CALLING, repo)
    assert isinstance(v_out, Translated), f"vLLM refused {repo!r}: {v_out}"
    argv = list(v_out.argv)
    assert "--enable-auto-tool-choice" in argv
    assert "--tool-call-parser" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == vllm_parser, (
        f"vLLM: expected {vllm_parser!r} for {repo!r}, got {argv}"
    )
    assert "--tool-strict-level" in argv

    s_out = SglangAdapter().translate(Setting.TOOL_CALLING, repo)
    assert isinstance(s_out, Translated), f"SGLang refused {repo!r}: {s_out}"
    assert s_out.argv == ("--tool-call-parser", sglang_parser), (
        f"SGLang: expected {sglang_parser!r} for {repo!r}, got {s_out.argv}"
    )


@pytest.mark.parametrize(
    "repo",
    [
        "google/gemma-3-27b-it",
        "microsoft/phi-4",
        "MiniMaxAI/MiniMax-M3",
        "unknown-org/some-random-model",
    ],
)
def test_unknown_repo_yields_unsupported_on_both_engines(repo):
    """A family not in the verified table must not get a guessed parser name.

    A wrong parser is worse than no parser: the engine starts, parses nothing,
    and every tool call leaks back as text with no indication why.
    """
    from onpar.engines import Setting, SglangAdapter, Unsupported, VllmAdapter

    v_out = VllmAdapter().translate(Setting.TOOL_CALLING, repo)
    assert isinstance(v_out, Unsupported), f"vLLM guessed a parser for {repo!r}"
    assert "no verified tool-call parser" in v_out.reason

    s_out = SglangAdapter().translate(Setting.TOOL_CALLING, repo)
    assert isinstance(s_out, Unsupported), f"SGLang guessed a parser for {repo!r}"
    assert "no verified tool-call parser" in s_out.reason


def test_llamacpp_emits_jinja_for_tool_calling():
    """--jinja is the one flag llama-server exposes for tool calling.
    Verified: tools/server/README.md (2026-10-31).
    """
    from onpar.engines import Setting, Translated, adapter_for

    a = adapter_for("llama.cpp")
    assert a is not None
    out = a.translate(Setting.TOOL_CALLING, "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(out, Translated), out
    assert "--jinja" in out.argv


@pytest.mark.parametrize("engine", ["mlx", "ollama"])
def test_mlx_and_ollama_tool_calling_is_unsupported_with_reason(engine):
    """Neither engine exposes a verified server flag for tool calling.
    The reason must name why so the caller knows what to do instead.
    """
    from onpar.engines import Setting, Unsupported, adapter_for

    a = adapter_for(engine)
    assert a is not None
    out = a.translate(Setting.TOOL_CALLING, "meta-llama/Llama-3.1-8B-Instruct")
    assert isinstance(out, Unsupported), f"{engine} unexpectedly translated tool_calling"
    assert len(out.reason) > 20, f"{engine}: reason is too short to be informative"


def test_deepseek_v31_does_not_match_v3_parser():
    """DeepSeek-V3.1 repos contain 'DeepSeek-V3' as a substring. The table must
    be ordered so V3.1 matches first, or both models get the wrong parser."""
    from onpar.engines import Setting, SglangAdapter, Translated, VllmAdapter

    v31_repo = "deepseek-ai/DeepSeek-V3.1"
    v_out = VllmAdapter().translate(Setting.TOOL_CALLING, v31_repo)
    assert isinstance(v_out, Translated), v_out
    argv = list(v_out.argv)
    assert argv[argv.index("--tool-call-parser") + 1] == "deepseek_v31", (
        "V3.1 repo matched the V3 entry first; check table ordering"
    )

    s_out = SglangAdapter().translate(Setting.TOOL_CALLING, v31_repo)
    assert isinstance(s_out, Translated), s_out
    assert s_out.argv[s_out.argv.index("--tool-call-parser") + 1] == "deepseekv31"
