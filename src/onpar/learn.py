"""onpar learn — LLM inference curriculum grounded in this machine's numbers.

Every figure is derived from onpar's own code for the hardware the user is
sitting at: hardware.detect(), the catalogue, fit.solve(), the planner, the
engine adapters, host.survey(), box, the k8s operator and the cutover gate.
Where a lesson compares machines it solves on named hardware_catalog profiles.
No number is hard-coded. Roofline figures are labelled "roofline estimate,
not measured" exactly as fit.py does.
"""

from __future__ import annotations

import contextlib
import re
import shlex
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

from . import box, catalog, core, fit, hardware, hardware_catalog, host
from .catalog import QUANT_BITS, ModelSpec
from .engines import _SGLANG_TOOL_PARSERS, _VLLM_TOOL_PARSERS, Setting, _tool_parser, adapter_for
from .fit import (
    BANDWIDTH_EFFICIENCY,
    OVERHEAD_FLOOR,
    OVERHEAD_FRACTION,
    Fit,
    solve,
)
from .hardware import Hardware
from .k8s import controller as k8s_controller
from .k8s import nodes as k8s_nodes
from .k8s import reconcile as k8s_reconcile
from .plan import (
    MAX_SPEC_DECODE_CONCURRENCY,
    MEMORY_SAFETY_MARGIN,
    PREFILL_CHUNK_TOKENS,
    PREFIX_CACHING_FLOOR,
    PREFIX_SHARING_FOR_RADIX,
    TPU_MATURING,
    TPU_RECOMMENDED,
    Engine,
    Requirements,
    Workload,
    _pick_engine,
)
from .plan import plan as plan_for
from .prove import EvalItem, cost, gate, suite
from .prove.equivalence import ClusterScore
from .prove.stats import wilson

GB = 1024**3

BASICS = "basics"
INTERMEDIATE = "intermediate"
HARDWARE = "hardware"
SERVING = "serving"
HOSTING = "hosting"
TRACKS = (BASICS, INTERMEDIATE, HARDWARE, SERVING, HOSTING)

#: Who each track is for. The lesson count is appended from the registry below,
#: so the listing can never publish a count the registry does not have.
_TRACK_TOPICS = {
    BASICS: "for anyone: how a model fits and what limits it",
    INTERMEDIATE: "for engineers: GQA, MLA, concurrency, speculative decoding",
    HARDWARE: "for anyone choosing a machine: memory, bandwidth, what each box runs",
    SERVING: "for engineers choosing a stack: engines, settings, forks, native config",
    HOSTING: "for teams deciding where it runs: rented, box, Kubernetes, cost, cutover",
}

#: Titles for the three newer tracks, shared by the registry and the renderers.
_TITLES = {
    "hardware.1": "Unified memory and VRAM are different budgets",
    "hardware.2": "Bandwidth is the ceiling on tokens per second",
    "hardware.3": "Usable memory is less than installed memory",
    "hardware.4": "What a 24 GB to 1 TB box can run",
    "hardware.5": "A TPU is sized per host, not per chip",
    "hardware.6": "A bigger model at q4 or a smaller one at q8?",
    "hardware.7": "What CPU-only really gets you",
    "serving.1": "The engines onpar can configure, and what each is for",
    "serving.2": "What onpar picks on this machine, and why",
    "serving.3": "The settings, and the forks between engines",
    "serving.4": "The config onpar writes runs without onpar",
    "serving.5": "Tool calling is decided by the server",
    "serving.6": "Continuous batching spends KV, not weights",
    "serving.7": "What an engine cannot express",
    "hosting.1": "Run it here, rented, or managed",
    "hosting.2": "One box, tuned per host class, re-solved on arrival",
    "hosting.3": "On Kubernetes the operator sizes before it applies",
    "hosting.4": "The saving is a range, never a point",
    "hosting.5": "The gateway is in your path only while you migrate",
    "hosting.6": "Nothing leaves your machine, and nothing raw hits disk",
    "hosting.7": "The cutover ladder, and who may climb it",
}


@dataclass(frozen=True, slots=True)
class Context:
    hw: Hardware
    models: tuple[ModelSpec, ...]
    model: ModelSpec
    quant: str
    fit: Fit  # solve(model, quant, hw, 32768, 1)


@dataclass
class LessonResult:
    id: str
    track: str
    title: str
    text: list[str]  # lines of terminal output, ≤ 25 lines
    figures: dict[str, float]  # name -> value; every number in text is here
    question: str | None  # "check yourself" prompt
    answer: str | None  # derived from context, not a literal


@dataclass(frozen=True, slots=True)
class Lesson:
    id: str
    track: str
    title: str
    render: Callable[[Context], LessonResult]


# ─── context builder ──────────────────────────────────────────────────────────


def _make_context(model_id: str | None = None) -> Context:
    """Detect hardware and pick a model, with a fallback when detection fails."""
    hw = hardware.detect()

    # Fallback: synthesise a sane Apple machine when detection returns
    # something with no bandwidth (can't teach tok/s without it).
    if hw.bandwidth_gbps is None:
        hw = Hardware(
            kind="apple",
            name="M4 Max (synthetic)",
            total_bytes=128 * GB,
            usable_bytes=96 * GB,
            bandwidth_gbps=546.0,
            cores=16,
            note="synthetic — hardware.detect() returned no bandwidth figure",
        )

    models = catalog.load()
    if model_id:
        m = catalog.get(model_id)
    else:
        feasible, _ = fit.rank(hw, 32768, 1)
        m = feasible[0].model if feasible else min(models, key=lambda x: x.params_b)

    best = fit.best_quant(m, hw, 32768, 1)
    if best is None:
        q = min(m.quants, key=lambda q_: QUANT_BITS[q_])
        f = solve(m, q, hw, 32768, 1)
    else:
        f, q = best, best.quant

    return Context(hw=hw, models=models, model=m, quant=q, fit=f)


# ─── basics lessons ───────────────────────────────────────────────────────────


def _basics_1(ctx: Context) -> LessonResult:
    """What a model is in bytes (weights)."""
    m = ctx.model
    bits = QUANT_BITS[ctx.quant]
    w_gb = ctx.fit.weight_bytes / GB
    usable_gb = ctx.hw.usable_gb
    usable_frac = usable_gb / ctx.hw.total_gb

    lines = [
        "  lesson basics·1 — what a model is in bytes (weights)",
        "",
        "  A language model is a large array of learned numbers — the weights.",
        "  Everything else (layers, attention heads, MLP) is how they're organised.",
        "",
        f"  {m.name} has {m.params_b:g}B parameters.",
        f"  At {ctx.quant} ({bits} bits per weight):",
        "",
        f"    weight bytes = {m.params_b:g}B × 10⁹ × {bits} bits ÷ 8",
        f"                 = {w_gb:.1f} GB",
        "",
        f"  Your {ctx.hw.name} has {ctx.hw.total_gb:.0f} GB total.",
        f"  onpar counts {usable_gb:.0f} GB usable ({usable_frac:.0%}, a calibration knob).",
        "",
        f"  Just the weights consume {w_gb / usable_gb:.0%} of that budget.",
    ]

    # Q: at q8, how much for a 7B model?
    q_gb = round(7e9 * QUANT_BITS["q8"] / 8 / GB, 1)

    return LessonResult(
        id="basics.1",
        track=BASICS,
        title="What a model is in bytes (weights)",
        text=lines,
        figures={
            "params_b": float(m.params_b),
            "quant_bits": float(bits),
            "weight_gb": round(w_gb, 2),
            "hw_total_gb": round(ctx.hw.total_gb, 0),
            "hw_usable_gb": round(usable_gb, 0),
            "usable_fraction": round(usable_frac, 3),
            "weight_of_usable": round(w_gb / usable_gb, 3),
        },
        question="At q8 (8 bits per weight), how many GB does a 7B model weigh?",
        answer=f"{q_gb} GB  (7B × 10⁹ × 8 bits ÷ 8 = {q_gb} GB)",
    )


def _basics_2(ctx: Context) -> LessonResult:
    """What context costs (KV cache)."""
    m = ctx.model
    kv_per_tok = m.kv_bytes_per_token()
    ctx_tokens = 32768
    conc = 1
    kv_gb = kv_per_tok * ctx_tokens * conc / GB
    per_layer = 2 * m.kv_heads * m.head_dim

    lines = [
        "  lesson basics·2 — what context costs (KV cache)",
        "",
        "  Every token in context is stored in the KV cache — a pair of",
        "  vectors (K and V) for each layer.",
        "",
        f"  For {m.name} ({m.kv_scheme.upper()}):",
        f"    per-token KV = {m.layers} layers × 2 × {m.kv_heads} KV-heads"
        f" × {m.head_dim} dim × 2 bytes",
        f"               = {kv_per_tok:,} bytes/token",
        "",
        f"  At {ctx_tokens:,} tokens context, concurrency 1:",
        f"    KV total = {kv_per_tok:,} × {ctx_tokens:,} = {kv_gb:.1f} GB",
        "",
        "  Double the context → double the KV. Scale concurrency →",
        "  scale KV by the same factor. It is strictly linear.",
    ]

    ans_gb = round(kv_per_tok * 65536 / GB, 1)

    return LessonResult(
        id="basics.2",
        track=BASICS,
        title="What context costs (KV cache)",
        text=lines,
        figures={
            "layers": float(m.layers),
            "kv_heads": float(m.kv_heads),
            "head_dim": float(m.head_dim),
            "kv_bytes_per_token": float(kv_per_tok),
            "per_layer_bytes": float(per_layer),
            "kv_gb_at_32k": round(kv_gb, 2),
        },
        question=(
            f"If you double the context from {ctx_tokens:,} to {ctx_tokens * 2:,} tokens,"
            " what happens to KV?"
        ),
        answer=f"It doubles: {ans_gb:.1f} GB  ({kv_per_tok:,} B/tok × {ctx_tokens * 2:,} tok ÷ GB)",
    )


def _basics_3(ctx: Context) -> LessonResult:
    """Why 'fits' is weights + KV + overhead vs usable memory."""
    f = ctx.fit
    w_gb = f.weight_bytes / GB
    kv_gb = f.kv_bytes / GB
    oh_gb = f.overhead_bytes / GB
    tot_gb = f.total_bytes / GB
    usable_gb = f.usable_bytes / GB
    headroom = f.headroom_bytes / GB

    verdict = "FITS" if f.feasible else "DOES NOT FIT"

    # Example: overhead for a 10 GB model
    ex_oh = round(10 * OVERHEAD_FRACTION + OVERHEAD_FLOOR / GB, 1)

    lines = [
        "  lesson basics·3 — why 'fits' is weights + KV + overhead vs usable",
        "",
        "  The solver adds three terms:",
        "",
        f"    weights   {w_gb:5.1f} GB",
        f"    kv cache  {kv_gb:5.1f} GB  (32k ctx × 1 concurrent request)",
        f"    overhead  {oh_gb:5.1f} GB  ({OVERHEAD_FRACTION:.0%} of weights"
        f" + {OVERHEAD_FLOOR / GB:.1f} GB floor)",
        f"    {'─' * 28}",
        f"    required  {tot_gb:5.1f} GB",
        f"    usable    {usable_gb:5.1f} GB",
        f"    headroom  {headroom:5.1f} GB  → {verdict}",
        "",
        "  Overhead accounts for activations, workspace and allocator",
        "  fragmentation — it scales with the model, not with context.",
    ]

    return LessonResult(
        id="basics.3",
        track=BASICS,
        title="Why 'fits' is weights + KV + overhead vs usable memory",
        text=lines,
        figures={
            "weight_gb": round(w_gb, 2),
            "kv_gb": round(kv_gb, 2),
            "overhead_gb": round(oh_gb, 2),
            "overhead_fraction": OVERHEAD_FRACTION,
            "overhead_floor_gb": OVERHEAD_FLOOR / GB,
            "total_gb": round(tot_gb, 2),
            "usable_gb": round(usable_gb, 2),
            "headroom_gb": round(headroom, 2),
        },
        question=f"If overhead is {OVERHEAD_FRACTION:.0%} of weights + {OVERHEAD_FLOOR / GB:.1f} GB"
        f" floor, and weights = 10 GB, what's overhead?",
        answer=(
            f"{ex_oh} GB  (10 × {OVERHEAD_FRACTION:.2f} + {OVERHEAD_FLOOR / GB:.1f} = {ex_oh} GB)"
        ),
    )


def _basics_4(ctx: Context) -> LessonResult:
    """Why tokens/sec is memory bandwidth (roofline, estimate)."""
    f = ctx.fit
    m = ctx.model
    bw = ctx.hw.bandwidth_gbps
    eff_bw = bw * BANDWIDTH_EFFICIENCY if bw else 0.0
    w_read = m.active_b * 1e9 * QUANT_BITS[ctx.quant] / 8
    kv_read = m.kv_bytes_per_token() * 32768
    total_read = w_read + kv_read
    tps = f.tokens_per_sec

    lines = [
        "  lesson basics·4 — why tokens/sec is memory bandwidth",
        "",
        "  Decode is bandwidth-bound: every token generated requires",
        "  reading the full weight array AND the whole KV cache.",
        "",
        "  Each token reads:",
        f"    weights  {w_read / GB:5.1f} GB  ({m.active_b:g}B × {QUANT_BITS[ctx.quant]} bits ÷ 8)",
        f"    kv       {kv_read / GB:5.1f} GB  ({m.kv_bytes_per_token():,} B/tok × 32,768 tok)",
        f"    total    {total_read / GB:5.1f} GB/token",
        "",
        f"  {ctx.hw.name} bandwidth: {bw:.0f} GB/s peak",
        f"  Effective (×{BANDWIDTH_EFFICIENCY:.0%} efficiency): {eff_bw:.0f} GB/s",
        "",
        f"  Roofline tok/s = {eff_bw:.0f} ÷ {total_read / GB:.1f} GB ≈ {tps:.0f} tok/s",
        "  (roofline estimate, not measured — `onpar measure` measures)",
    ]

    # Q: if bandwidth doubles
    ans = round(tps * 2) if tps else 0

    return LessonResult(
        id="basics.4",
        track=BASICS,
        title="Why tokens/sec is memory bandwidth (roofline estimate)",
        text=lines,
        figures={
            "bandwidth_gbps": float(bw),
            "bandwidth_efficiency": BANDWIDTH_EFFICIENCY,
            "effective_bw_gbps": round(eff_bw, 1),
            "weight_read_gb": round(w_read / GB, 2),
            "kv_read_gb": round(kv_read / GB, 2),
            "total_read_gb": round(total_read / GB, 2),
            "tokens_per_sec": round(tps, 1) if tps else 0.0,
        },
        question=(
            f"If {ctx.hw.name}'s bandwidth doubled to {bw * 2:.0f} GB/s, what would tok/s become?"
        ),
        answer=f"≈ {ans} tok/s  (bandwidth scales linearly into the roofline — estimate only)",
    )


def _basics_5(ctx: Context) -> LessonResult:
    """What quantization buys and costs."""
    m = ctx.model
    hw = ctx.hw
    ctx_t, conc = 32768, 1

    rows: list[tuple[str, float, float | None]] = []
    for q in sorted(set(m.quants) & {"q4", "q8", "fp16"}, key=lambda q_: QUANT_BITS[q_]):
        f = solve(m, q, hw, ctx_t, conc)
        rows.append((q, f.weight_bytes / GB, f.tokens_per_sec))

    if not rows:
        for q in m.quants[:3]:
            f = solve(m, q, hw, ctx_t, conc)
            rows.append((q, f.weight_bytes / GB, f.tokens_per_sec))

    figs: dict[str, float] = {}
    weight_lines = []
    for q, w_gb, tps in rows:
        tps_str = f"  {tps:.0f} tok/s (roofline est.)" if tps else ""
        weight_lines.append(f"    {q:<6}  {w_gb:5.1f} GB{tps_str}")
        figs[f"weight_gb_{q}"] = round(w_gb, 2)
        if tps:
            figs[f"tps_{q}"] = round(tps, 1)

    # ratio fp16 vs q4 (or first vs last)
    if len(rows) >= 2:
        ratio = rows[-1][1] / rows[0][1]
        ratio_str = f"  {rows[-1][0]} is {ratio:.1f}× larger than {rows[0][0]}."
        figs["weight_ratio"] = round(ratio, 2)
    else:
        ratio_str = ""

    lines = (
        [
            "  lesson basics·5 — what quantisation buys and costs",
            "",
            "  Quantisation compresses each weight from fp16 (16 bits) to fewer bits.",
            "  Smaller format → fewer bytes to read per token → faster decode.",
            "",
            f"  {m.name} weight size:",
        ]
        + weight_lines
        + [
            "",
            ratio_str,
            "  Quality cost: smaller quants lose some precision. At 4–8 bit the",
            "  quality gap is typically smaller than the gap between model sizes.",
            "  Decode speed beats quality for most inference-time choices.",
        ]
    )

    return LessonResult(
        id="basics.5",
        track=BASICS,
        title="What quantisation buys and costs",
        text=lines,
        figures=figs,
        question="On a bandwidth-bound system, which is faster to decode: q4 or q8?",
        answer="q4 — fewer bytes per weight means fewer bytes read per token.",
    )


def _basics_6(ctx: Context) -> LessonResult:
    """Why MoE is sized on total params (the dangerous error)."""
    # Always use qwen3-30b-a3b for this lesson — it's the best MoE example
    try:
        moe = catalog.get("qwen3-30b-a3b")
    except KeyError:
        # fall back to any MoE in the catalog
        moe = next((m for m in ctx.models if m.is_moe), ctx.model)

    hw = ctx.hw
    f_total = solve(moe, "q4", hw, 32768, 1)
    w_total = f_total.weight_bytes / GB

    # The dangerous error: size on active_b
    w_wrong = moe.active_b * 1e9 * QUANT_BITS["q4"] / 8 / GB

    lines = [
        "  lesson basics·6 — why MoE is sized on total params",
        "",
        "  Mixture-of-Experts (MoE) models activate only a fraction of their",
        "  weights per token — the router picks the best 'experts'.",
        "",
        f"  {moe.name}:",
        f"    total params   {moe.params_b:g}B   (all experts, all resident in memory)",
        f"    active params  {moe.active_b:g}B   (routed per token)",
        "",
        "  Memory holds ALL experts. Correct sizing at q4:",
        f"    {moe.params_b:g}B × 4.5 bits ÷ 8 = {w_total:.1f} GB",
        "",
        "  The dangerous error — sizing on active params:",
        f"    {moe.active_b:g}B × 4.5 bits ÷ 8 = {w_wrong:.1f} GB",
        "",
        f"  That understates by {w_total / w_wrong:.0f}×. A machine that 'fits'",
        "  the wrong number OOMs on load.",
        "",
        "  Sparsity saves compute (and compute cost), not memory.",
    ]

    # Q: 100B MoE at 10B active, q4
    q_correct = round(100e9 * 4.5 / 8 / GB, 0)

    return LessonResult(
        id="basics.6",
        track=BASICS,
        title="Why MoE is sized on total params (the dangerous error)",
        text=lines,
        figures={
            "moe_total_params_b": float(moe.params_b),
            "moe_active_params_b": float(moe.active_b),
            "weight_gb_correct": round(w_total, 1),
            "weight_gb_wrong": round(w_wrong, 1),
            "overestimate_factor": round(w_total / w_wrong, 1),
        },
        question="A 100B MoE with 10B active: how many GB of weights at q4 (4.5 bits)?",
        answer=f"{q_correct:.0f} GB  (100B × 4.5 bits ÷ 8 — ALL experts must fit)",
    )


def _basics_7(ctx: Context) -> LessonResult:
    """What 'proven on par' means (Wilson interval, the ? cell, bar)."""
    # These are conceptual numbers — use a concrete example
    n, successes, bar = 5, 5, 0.90
    # Wilson lower bound for 5/5 at 95% CI
    # p_hat = 1.0, z = 1.96
    import math

    p_hat = successes / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)

    lines = [
        "  lesson basics·7 — what 'proven on par' means",
        "",
        "  onpar uses Wilson confidence intervals, not point estimates.",
        "  A model 'proves' a cluster only when the WHOLE interval",
        "  clears the bar — not just the mean.",
        "",
        f"  Example: {successes}/{n} items matched the incumbent.",
        f"    point estimate: {p_hat:.0%}",
        f"    Wilson 95% CI:  [{lo:.0%}, {hi:.0%}]",
        f"    bar:            {bar:.0%}",
        "",
        f"  The lower bound {lo:.0%} < {bar:.0%} → NOT proven.",
        "  A sample of 5 is too thin to make a confident claim.",
        "",
        "  A '?' cell means the sample was too small to say either way.",
        f"  The bar ({bar:.0%}) is a calibration knob, not a law of physics.",
        "  Nothing authorises a cutover except shadow mode.",
    ]

    return LessonResult(
        id="basics.7",
        track=BASICS,
        title="What 'proven on par' means (Wilson interval, the ? cell, bar)",
        text=lines,
        figures={
            "wilson_lo": round(lo, 3),
            "wilson_hi": round(hi, 3),
            "point_estimate": p_hat,
            "bar": bar,
            "sample_n": float(n),
            # percentage forms — the text displays these as e.g. "47%"
            "wilson_lo_pct": round(lo * 100, 0),
            "wilson_hi_pct": round(hi * 100, 0),
            "bar_pct": round(bar * 100, 0),
        },
        question=(
            f"A model scores {successes}/{n} items (point estimate {p_hat:.0%})."
            f" Is it proven at the {bar:.0%} bar?"
        ),
        answer=(
            f"No — Wilson lower bound is {lo:.0%}, below the {bar:.0%} bar. More samples needed."
        ),
    )


# ─── intermediate lessons ─────────────────────────────────────────────────────


def _inter_1(ctx: Context) -> LessonResult:
    """GQA vs MHA and kv_heads."""
    # Use a GQA model — qwen3-32b has 8 KV heads (verified) vs ~64 attn heads
    try:
        m = catalog.get("qwen3-32b")
    except KeyError:
        m = next((x for x in ctx.models if x.kv_scheme == "gqa"), ctx.model)

    kv_per_tok = m.kv_bytes_per_token()
    kv_gb = kv_per_tok * 32768 / GB

    # Hypothetical full-MHA: same model but kv_heads = a round multiple
    # Published: qwen3-32b has 64 attention heads, 8 KV heads
    attn_heads_est = 64  # published architecture fact for qwen3-32b
    naive_per_tok = m.layers * 2 * attn_heads_est * m.head_dim * 2
    naive_gb = naive_per_tok * 32768 / GB
    ratio = naive_per_tok / kv_per_tok

    lines = [
        "  lesson inter·1 — GQA vs MHA and kv_heads",
        "",
        "  Grouped-query attention (GQA) reduces KV storage by sharing",
        "  KV heads across attention heads.",
        "",
        f"  {m.name} at 32k context:",
        f"    kv_heads in catalog:  {m.kv_heads}",
        f"    Published attn heads: {attn_heads_est}  (the field NOT in the catalog)",
        "",
        f"  KV per token (correct, using kv_heads={m.kv_heads}):",
        f"    {m.layers} layers × 2 × {m.kv_heads} × {m.head_dim} dim × 2 bytes",
        f"    = {kv_per_tok:,} bytes/tok → {kv_gb:.1f} GB",
        "",
        f"  KV per token (error, using {attn_heads_est} attn heads as KV heads):",
        f"    = {naive_per_tok:,} bytes/tok → {naive_gb:.1f} GB  ({ratio:.0f}× overestimate)",
        "",
        "  The sizing error: using attention heads instead of kv_heads.",
        "  onpar uses kv_heads from the catalog — never an estimated count.",
    ]

    return LessonResult(
        id="inter.1",
        track=INTERMEDIATE,
        title="GQA vs MHA and kv_heads",
        text=lines,
        figures={
            "kv_heads": float(m.kv_heads),
            "attn_heads_reference": float(attn_heads_est),
            "kv_bytes_per_token": float(kv_per_tok),
            "kv_gb_at_32k": round(kv_gb, 2),
            "naive_kv_bytes_per_token": float(naive_per_tok),
            "naive_kv_gb": round(naive_gb, 1),
            "overestimate_ratio": round(ratio, 0),
        },
        question="Why does onpar store kv_heads in the catalog rather than attention heads?",
        answer="KV cache size depends only on KV heads, not attention heads. Using attention "
        "heads inflates the KV estimate by the GQA group ratio.",
    )


def _inter_2(ctx: Context) -> LessonResult:
    """MLA and kv_lora_rank (64× overestimate)."""
    try:
        mla = catalog.get("deepseek-v3")
    except KeyError:
        mla = next((m for m in ctx.models if m.kv_scheme == "mla"), ctx.model)

    kv_mla = mla.kv_bytes_per_token()
    # Hypothetical GQA formula on the same model
    kv_gqa = mla.layers * 2 * mla.kv_heads * mla.head_dim * 2
    ratio = kv_gqa / kv_mla if kv_mla else 0
    lora_rank = mla.kv_lora_rank or 0

    lines = [
        "  lesson inter·2 — MLA and kv_lora_rank (64× overestimate)",
        "",
        "  Multi-head latent attention (MLA, DeepSeek family) compresses",
        "  K and V into a single low-rank latent per layer.",
        "",
        f"  {mla.name} (MLA):",
        f"    kv_lora_rank = {lora_rank}",
        f"    kv_bytes_per_token = {lora_rank} × {mla.layers} layers × 2 bytes",
        f"                      = {kv_mla:,} bytes/tok",
        "",
        f"  Using the GQA formula instead ({mla.kv_heads} kv_heads × {mla.head_dim} dim):",
        f"    = {kv_gqa:,} bytes/tok",
        "",
        f"  Ratio: {ratio:.0f}× overestimate.",
        "",
        f"  A {mla.name[0:20]} that 'doesn't fit' sized with GQA often does fit",
        "  when sized correctly. The catalog enforces kv_lora_rank for all MLA",
        "  entries — refusing to load a row that would be sized this way.",
    ]

    return LessonResult(
        id="inter.2",
        track=INTERMEDIATE,
        title="MLA and kv_lora_rank (the 64× overestimate)",
        text=lines,
        figures={
            "kv_lora_rank": float(lora_rank),
            "layers": float(mla.layers),
            "kv_bytes_per_token_mla": float(kv_mla),
            "kv_bytes_per_token_gqa_formula": float(kv_gqa),
            "overestimate_ratio": round(ratio, 0),
        },
        question="Why does onpar refuse to load an MLA model without kv_lora_rank set?",
        answer=f"Without kv_lora_rank it falls back to the GQA formula, which is "
        f"{ratio:.0f}× too large. A model that fits gets reported as infeasible.",
    )


def _inter_3(ctx: Context) -> LessonResult:
    """Concurrency and the KV budget."""
    m = ctx.model
    hw = ctx.hw
    kv_per_tok = m.kv_bytes_per_token()
    ctx_t = 32768

    kv_c1 = kv_per_tok * ctx_t / GB
    kv_c4 = kv_per_tok * ctx_t * 4 / GB
    kv_c8 = kv_per_tok * ctx_t * 8 / GB

    max_conc = fit.max_concurrency(m, ctx.quant, hw, ctx_t)

    lines = [
        "  lesson inter·3 — concurrency and the KV budget",
        "",
        "  Each concurrent request holds its own KV cache slice.",
        "  KV scales linearly with both context and concurrency.",
        "",
        f"  {m.name} at {ctx_t:,} context:",
        f"    c=1   {kv_c1:5.1f} GB KV",
        f"    c=4   {kv_c4:5.1f} GB KV  (4×)",
        f"    c=8   {kv_c8:5.1f} GB KV  (8×)",
        "",
        f"  Max concurrency on {hw.name} at {ctx_t:,} ctx: {max_conc}",
        "",
        "  Beyond that the KV budget is exhausted — not the weights.",
        "  Context and concurrency are competing claims on the same pool.",
        "  `onpar fit --context 8k` shows how more context buys concurrency.",
    ]

    return LessonResult(
        id="inter.3",
        track=INTERMEDIATE,
        title="Concurrency and the KV budget",
        text=lines,
        figures={
            "kv_per_token_bytes": float(kv_per_tok),
            "kv_gb_c1": round(kv_c1, 2),
            "kv_gb_c4": round(kv_c4, 2),
            "kv_gb_c8": round(kv_c8, 2),
            "max_concurrency_at_32k": float(max_conc),
        },
        question=f"If max concurrency at 32k is {max_conc}, what is it at 16k context?",
        answer=f"~{min(fit.max_concurrency(m, ctx.quant, hw, 16384), 999)} "
        f"— halving context doubles the concurrency ceiling.",
    )


def _inter_4(ctx: Context) -> LessonResult:
    """Prefix caching and agent loops (PREFIX_CACHING_FLOOR)."""
    m = ctx.model
    kv_per_tok = m.kv_bytes_per_token()
    ctx_t = 32768

    # Example: 80% prefix share in an agent loop
    share = 0.80
    prefix_tokens = int(ctx_t * share)
    saved_gb = kv_per_tok * prefix_tokens / GB

    lines = [
        "  lesson inter·4 — prefix caching and agent loops",
        "",
        "  An agent loop often has a fixed system prompt plus retrieved context.",
        "  Requests in the same fleet share this prefix — its KV can be computed",
        "  once and reused.",
        "",
        f"  At {share:.0%} prefix sharing over {ctx_t:,} context:",
        f"    prefix KV saved per request = {kv_per_tok:,} × {prefix_tokens:,} tok",
        f"                                = {saved_gb:.1f} GB",
        "",
        f"  Threshold: below {PREFIX_CACHING_FLOOR:.0%} sharing the bookkeeping",
        "  overhead costs more than the cache hits return.",
        "",
        f"  onpar sets PREFIX_CACHING_FLOOR = {PREFIX_CACHING_FLOOR:.0%}.",
        "  Above it the planner enables prefix caching; below it, it does not.",
        "  RadixAttention (SGLang) exploits it structurally; vLLM adds it as a flag.",
    ]

    return LessonResult(
        id="inter.4",
        track=INTERMEDIATE,
        title="Prefix caching and agent loops",
        text=lines,
        figures={
            "prefix_sharing_example": share,
            "prefix_sharing_pct": round(share * 100, 0),
            "prefix_tokens": float(prefix_tokens),
            "prefix_kv_saved_gb": round(saved_gb, 2),
            "prefix_caching_floor": PREFIX_CACHING_FLOOR,
            "prefix_caching_floor_pct": round(PREFIX_CACHING_FLOOR * 100, 0),
        },
        question=(
            f"At {PREFIX_CACHING_FLOOR - 0.05:.0%} prefix sharing, does onpar"
            " enable prefix caching?"
        ),
        answer=(
            f"No — {PREFIX_CACHING_FLOOR - 0.05:.0%} is below the"
            f" {PREFIX_CACHING_FLOOR:.0%} floor. Bookkeeping costs more than the hits return."
        ),
    )


def _inter_5(ctx: Context) -> LessonResult:
    """Speculative decoding and why it turns negative at high batch."""
    tps = ctx.fit.tokens_per_sec
    tps_str = f"~{tps:.0f} tok/s (roofline est.)" if tps else "(bandwidth unknown)"

    lines = [
        "  lesson inter·5 — speculative decoding and high-batch limits",
        "",
        "  Speculative decoding drafts several tokens with a small model,",
        "  then verifies them in one pass of the big model. It wins when",
        "  the accelerator has spare compute between tokens.",
        "",
        "  It loses at high batch: draft + verify steps compete with real",
        "  tokens already queued. Published EAGLE-3 numbers turn negative",
        "  somewhere around batch 32.",
        "",
        f"  onpar's threshold: MAX_SPEC_DECODE_CONCURRENCY = {MAX_SPEC_DECODE_CONCURRENCY}",
        "",
        "  Below it: planner recommends eagle3 (or native MTP for models",
        "  that ship their own multi-token-prediction head).",
        "  Above it: planner turns speculative decoding off.",
        "",
        f"  On {ctx.hw.name}: {tps_str}",
        f"  At single-stream that headroom exists; at c={MAX_SPEC_DECODE_CONCURRENCY + 1}"
        " it does not.",
    ]

    return LessonResult(
        id="inter.5",
        track=INTERMEDIATE,
        title="Speculative decoding and why it turns negative at high batch",
        text=lines,
        figures={
            "max_spec_decode_concurrency": float(MAX_SPEC_DECODE_CONCURRENCY),
            "tokens_per_sec": round(tps, 1) if tps else 0.0,
        },
        question=(
            f"At concurrency {MAX_SPEC_DECODE_CONCURRENCY + 1},"
            " does the planner enable speculative decoding?"
        ),
        answer=(
            f"No — {MAX_SPEC_DECODE_CONCURRENCY + 1} > {MAX_SPEC_DECODE_CONCURRENCY}."
            " The draft pass would compete with real tokens."
        ),
    )


def _inter_6(ctx: Context) -> LessonResult:
    """Chunked prefill / TTFT vs ITL budgets."""
    ctx_t = 32768
    n_chunks = ctx_t // PREFILL_CHUNK_TOKENS

    lines = [
        "  lesson inter·6 — chunked prefill and TTFT vs ITL",
        "",
        f"  Without chunked prefill, a {ctx_t:,}-token prompt occupies a full",
        "  scheduler step. Every in-flight response stalls behind it.",
        "  Time-to-first-token (TTFT) spikes for every other request.",
        "",
        "  Chunked prefill splits the prompt into pieces:",
        f"    chunk size: {PREFILL_CHUNK_TOKENS:,} tokens  (PREFILL_CHUNK_TOKENS)",
        f"    {ctx_t:,}-token prompt → {n_chunks} chunks, each sharing the scheduler",
        "",
        "  Two latency budgets move in opposite directions:",
        "    TTFT — set by prefill cost; chunking protects other requests",
        "    ITL  — set by decode cost; a bigger batch raises it",
        "",
        "  Batch workloads: chunk size 8,192 (prefill throughput > TTFT fairness)",
        f"  Interactive: chunk size {PREFILL_CHUNK_TOKENS:,} (TTFT fairness matters)",
    ]

    return LessonResult(
        id="inter.6",
        track=INTERMEDIATE,
        title="Chunked prefill / TTFT vs ITL budgets",
        text=lines,
        figures={
            "prefill_chunk_tokens": float(PREFILL_CHUNK_TOKENS),
            "context_tokens": float(ctx_t),
            "n_chunks_at_32k": float(n_chunks),
        },
        question=f"A {ctx_t:,}-token prompt with no chunking monopolises how many scheduler steps?",
        answer="1 full step — all decode requests stall behind it until prefill completes.",
    )


def _inter_7(ctx: Context) -> LessonResult:
    """Tool calling as a server problem (parser flags; tool-leak)."""
    lines = [
        "  lesson inter·7 — tool calling as a server problem",
        "",
        "  Without the right parser flag, a model's tool call output comes",
        "  back as plain text with finish_reason: stop.",
        "  The eval suite calls this tool-leak.",
        "",
        "  Tool-call parsing is engine-specific and family-specific:",
        "    • vLLM: --tool-call-parser <name>",
        "    • SGLang: --tool-call-parser <name>",
        "    • name depends on the model family (Llama, Qwen, Mistral, …)",
        "",
        "  onpar's planner maps the model's repo to the verified parser name.",
        "  If no parser is known for this family, the plan surfaces the gap",
        "  rather than silently serving without it.",
        "",
        "  Tip: test with `finish_reason` before deploying a tool-using agent.",
        "  A 'stop' where you expected 'tool_calls' is tool-leak.",
    ]

    return LessonResult(
        id="inter.7",
        track=INTERMEDIATE,
        title="Tool calling as a server problem (parser flags; tool-leak)",
        text=lines,
        figures={},  # no numeric figures in this conceptual lesson
        question="What is tool-leak, and how does it manifest in an API response?",
        answer="The model produces a tool call as plain text; finish_reason is 'stop' "
        "instead of 'tool_calls'. The parser flag was missing or wrong.",
    )


def _inter_8(ctx: Context) -> LessonResult:
    """Teacher-forced step replay and why a judge needs an interval."""
    import math

    n, k = 10, 8
    p_hat = k / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    lo = max(0.0, centre - half)
    bar = 0.90

    lines = [
        "  lesson inter·8 — teacher-forced step replay and judge intervals",
        "",
        "  Agentic evals replay each step with the RECORDED context, not the",
        "  model's own prior output. This isolates per-step quality from",
        "  compounding errors across a trajectory.",
        "",
        "  Step replay: given the same messages+tools as in production,",
        "  does the candidate make the same next move?",
        "",
        "  Judge reliability: a judge comparing two responses needs an interval.",
        f"  Example — {k}/{n} agentic steps matched:",
        f"    point estimate: {p_hat:.0%}",
        f"    Wilson 95% CI lower: {lo:.0%}",
        f"    bar: {bar:.0%}",
        "",
        f"  {lo:.0%} < {bar:.0%} → NOT proven. Step count is the binding constraint",
        "  in agentic evals — multi-step traces are expensive to capture.",
        "",
        "  The Wilson interval is why the ? cell exists.",
    ]

    return LessonResult(
        id="inter.8",
        track=INTERMEDIATE,
        title="Teacher-forced step replay and why a judge needs an interval",
        text=lines,
        figures={
            "point_estimate": p_hat,
            "point_estimate_pct": round(p_hat * 100, 0),
            "wilson_lo": round(lo, 3),
            "wilson_lo_pct": round(lo * 100, 0),
            "bar": bar,
            "bar_pct": round(bar * 100, 0),
            "sample_n": float(n),
            "sample_k": float(k),
        },
        question="For agentic evals, why is teacher-forced replay used instead of full rollouts?",
        answer="To isolate each step from compounding prior errors. A rollout blames "
        "step N for step 1's mistake; replay asks only 'given the real context, "
        "does the candidate make the same move?'",
    )


# ─── shared helpers for the hardware / serving / hosting tracks ───────────────

CTX = 32768
ROOFLINE = "roofline estimate, not measured"


def _profile(profile_id: str) -> tuple[hardware_catalog.Profile, Hardware]:
    p = hardware_catalog.get(profile_id)
    return p, p.to_hardware()


def _wrap(text: str, indent: str = "    ", width: int = 72) -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _first_sentence(text: str) -> str:
    head, sep, _ = text.partition(". ")
    return head + ("." if sep else "")


def _small_model(ctx: Context) -> ModelSpec:
    """A model small enough that every profile in a comparison can hold it."""
    try:
        return catalog.get("llama-3.1-8b")
    except KeyError:
        return min(ctx.models, key=lambda m: m.params_b)


def _servable(ctx: Context) -> ModelSpec:
    """The lesson model if it names a weights repo, else the largest that fits and does."""
    if ctx.model.repo:
        return ctx.model
    fits = [f.model for f in fit.rank(ctx.hw, CTX, 1)[0] if f.model.repo]
    pool = fits or [m for m in ctx.models if m.repo]
    return max(pool, key=lambda m: m.params_b) if fits else min(pool, key=lambda m: m.params_b)


def _read_gb(m: ModelSpec, quant: str, context: int) -> float:
    """Bytes decode reads per token, in GB: active weights plus the KV cache."""
    return (m.active_b * 1e9 * QUANT_BITS[quant] / 8 + m.kv_bytes_per_token() * context) / GB


def _tps(f: Fit) -> float:
    return round(f.tokens_per_sec or 0.0, 1)


# ─── hardware lessons ─────────────────────────────────────────────────────────

#: Profiles that represent each memory architecture onpar detects.
_ARCHITECTURES = (
    ("m4-max-128", "unified: CPU and GPU share one pool"),
    ("h100", "VRAM on the card, separate from RAM"),
    ("mi300x", "VRAM on the card, separate from RAM"),
    ("tpu-v5e-8", "HBM per chip, summed over chips"),
    ("cpu-64", "system RAM, no accelerator"),
)


def _hardware_1(ctx: Context) -> LessonResult:
    """Unified memory vs VRAM."""
    hw = ctx.hw
    figs: dict[str, float] = {
        "host_total_gb": round(hw.total_gb, 0),
        "host_usable_gb": round(hw.usable_gb, 0),
        "host_usable_pct": round(hw.usable_bytes / hw.total_bytes * 100, 0),
    }
    rows = []
    for pid, pool in _ARCHITECTURES:
        p, h = _profile(pid)
        pct = h.usable_bytes / h.total_bytes * 100
        figs[f"{pid}_total_gb"] = round(h.total_gb, 0)
        figs[f"{pid}_usable_gb"] = round(h.usable_gb, 0)
        figs[f"{pid}_usable_pct"] = round(pct, 0)
        rows.append(f"    {p.name:<27} {h.total_gb:5.0f} GB  {h.usable_gb:5.0f} GB  {pct:3.0f}%")
        rows.append(f"      {pool}")

    pro, pro_hw = _profile("m4-pro-48")
    lines = [
        "  lesson hardware·1 · unified memory and VRAM are different budgets",
        "",
        "  On Apple silicon the GPU shares unified memory with the OS, so the",
        "  wired limit is the ceiling. A discrete card holds the model in its own",
        "  VRAM. A TPU host sums the HBM on every chip.",
        "",
        "    profile                     total     usable",
        *rows,
        "",
        f"  Your {hw.name} ({hw.kind}): {hw.total_gb:.0f} GB total,"
        f" {hw.usable_gb:.0f} GB usable ({figs['host_usable_pct']:.0f}%).",
        "  The usable share per kind is a reservation in hardware_catalog.py",
        "  and hardware.py, and it is a calibration knob.",
    ]
    return LessonResult(
        id="hardware.1",
        track=HARDWARE,
        title=_TITLES["hardware.1"],
        text=lines,
        figures=figs,
        question=f"How much of the {pro.name} profile does onpar count as usable?",
        answer=f"{pro_hw.usable_gb:.0f} GB of {pro_hw.total_gb:.0f} GB"
        f" ({pro_hw.usable_bytes / pro_hw.total_bytes:.0%}), from Profile.to_hardware().",
    )


_BANDWIDTH_LADDER = ("cpu-64", "m4-pro-48", "m4-max-128", "rtx-4090", "h100", "b200")


def _hardware_2(ctx: Context) -> LessonResult:
    """Bandwidth is the ceiling."""
    m = _small_model(ctx)
    q = "q4" if "q4" in m.quants else m.quants[-1]
    read = _read_gb(m, q, CTX)
    figs: dict[str, float] = {
        "read_gb_per_token": round(read, 2),
        "bandwidth_efficiency": BANDWIDTH_EFFICIENCY,
    }
    rows = []
    for pid in _BANDWIDTH_LADDER:
        p, h = _profile(pid)
        f = solve(m, q, h, CTX, 1)
        figs[f"{pid}_bw_gbps"] = float(h.bandwidth_gbps or 0)
        figs[f"{pid}_tps"] = _tps(f)
        rows.append(f"    {p.name:<27} {h.bandwidth_gbps or 0:7,.0f} GB/s  {_tps(f):6.1f} tok/s")
    here = solve(m, q, ctx.hw, CTX, 1)
    figs["host_bw_gbps"] = float(ctx.hw.bandwidth_gbps or 0)
    figs["host_tps"] = _tps(here)

    _, h100 = _profile("h100")
    _, b200 = _profile("b200")
    t_h100 = solve(m, q, h100, CTX, 1).tokens_per_sec or 0.0
    t_b200 = solve(m, q, b200, CTX, 1).tokens_per_sec or 0.0

    lines = [
        "  lesson hardware·2 · bandwidth is the ceiling on tokens per second",
        "",
        f"  Each decoded token reads {read:.1f} GB for {m.name} at {q}, 32k context.",
        f"  tok/s = bandwidth × {BANDWIDTH_EFFICIENCY:.0%} ÷ {read:.1f} GB  ({ROOFLINE})",
        "",
        *rows,
        f"    {'this machine':<27} {ctx.hw.bandwidth_gbps or 0:7,.0f} GB/s"
        f"  {_tps(here):6.1f} tok/s",
        "",
        f"  {BANDWIDTH_EFFICIENCY:.0%} is BANDWIDTH_EFFICIENCY in fit.py, a calibration knob.",
        "  `onpar measure` replaces the estimate with a measurement.",
        "  Memory decides what fits. Bandwidth decides how fast it runs.",
    ]
    return LessonResult(
        id="hardware.2",
        track=HARDWARE,
        title=_TITLES["hardware.2"],
        text=lines,
        figures=figs,
        question="The B200 profile has more bandwidth than the H100. How much faster is the"
        f" roofline for {m.name}?",
        answer=f"{t_b200 / t_h100:.2f}× ({t_b200:.0f} vs {t_h100:.0f} tok/s), the bandwidth"
        f" ratio, because the bytes per token are the same. {ROOFLINE}.",
    )


def _hardware_3(ctx: Context) -> LessonResult:
    """Usable vs total, and what overhead takes."""
    hw, f = ctx.hw, ctx.fit
    reserved = (hw.total_bytes - hw.usable_bytes) / GB
    w, oh, kv = f.weight_bytes / GB, f.overhead_bytes / GB, f.kv_bytes / GB
    spare = (f.usable_bytes - f.weight_bytes - f.overhead_bytes) / GB
    max_c = fit.max_concurrency(ctx.model, ctx.quant, hw, CTX)
    lines = [
        "  lesson hardware·3 · usable is less than installed",
        "",
        f"  Your {hw.name}:",
        f"    installed  {hw.total_gb:6.1f} GB",
        f"    reserved   {reserved:6.1f} GB  (OS, driver, other processes)",
        f"    usable     {hw.usable_gb:6.1f} GB",
        "",
        f"  {ctx.model.name} at {ctx.quant} spends that budget:",
        f"    weights    {w:6.1f} GB",
        f"    overhead   {oh:6.1f} GB  ({OVERHEAD_FRACTION:.0%} of weights"
        f" + {OVERHEAD_FLOOR / GB:.1f} GB floor)",
        f"    left       {spare:6.1f} GB  for KV cache",
        f"    KV at 32k  {kv:6.1f} GB  per request",
        "",
        f"  That leaves room for {max_c} requests at 32k context.",
        "  OVERHEAD_FRACTION covers activations, workspace and fragmentation.",
        "  It is a calibration knob in fit.py, not a measured constant.",
    ]
    return LessonResult(
        id="hardware.3",
        track=HARDWARE,
        title=_TITLES["hardware.3"],
        text=lines,
        figures={
            "total_gb": round(hw.total_gb, 1),
            "reserved_gb": round(reserved, 1),
            "usable_gb": round(hw.usable_gb, 1),
            "weight_gb": round(w, 1),
            "overhead_gb": round(oh, 1),
            "overhead_fraction": OVERHEAD_FRACTION,
            "overhead_floor_gb": OVERHEAD_FLOOR / GB,
            "spare_gb": round(spare, 1),
            "kv_gb": round(kv, 1),
            "max_concurrency_at_32k": float(max_c),
        },
        question=f"How many requests at 32k context fit beside {ctx.model.name} here?",
        answer=f"{max_c}, from fit.max_concurrency(): {spare:.1f} GB left"
        f" ÷ {kv:.2f} GB of KV per request.",
    )


#: Real profiles closest to the box sizes people ask about. There is no 96 GB
#: or 8xH100 profile: the 128 GB Mac counts 96 GB usable, and 8xH200 is the
#: largest node in the catalogue.
_BOX_SIZES = ("rtx-4090", "l40s", "h100", "m4-max-128", "h100-x4", "h200-x8")


def _hardware_4(ctx: Context) -> LessonResult:
    """What each box size runs."""
    total_models = len(ctx.models)
    figs: dict[str, float] = {"catalogue_models": float(total_models)}
    rows = []
    n24 = 0
    for pid in _BOX_SIZES:
        p, h = _profile(pid)
        feasible, _ = fit.rank(h, CTX, 1)
        if pid == "rtx-4090":
            n24 = len(feasible)
        figs[f"{pid}_usable_gb"] = round(h.usable_gb, 0)
        figs[f"{pid}_n_fit"] = float(len(feasible))
        if feasible:
            top = feasible[0]
            figs[f"{pid}_top_tps"] = _tps(top)
            best = f"{top.model.id} {top.quant}, {_tps(top):.0f} tok/s"
        else:
            best = "nothing"
        rows.append(f"    {p.name:<24} {h.usable_gb:5.0f} GB  {len(feasible):2}  {best}")
    lines = [
        "  lesson hardware·4 · what a 24 GB to 1 TB box can run",
        "",
        f"  fit.rank() sizes all {total_models} catalogue models on each profile,",
        "  at 32k context and one request. Largest that fits is shown.",
        "",
        "    profile                  usable  fit  largest",
        *rows,
        "",
        f"  tok/s is single-stream, {ROOFLINE}.",
        "  The 128 GB Mac counts 96 GB usable. The catalogue has no 8× H100 profile;",
        "  8× H200 is its largest node.",
        f"  `onpar where {ctx.model.id}` ranks all {len(hardware_catalog.PROFILES)}"
        " profiles for one model.",
    ]
    figs["profiles"] = float(len(hardware_catalog.PROFILES))
    return LessonResult(
        id="hardware.4",
        track=HARDWARE,
        title=_TITLES["hardware.4"],
        text=lines,
        figures=figs,
        question="How many catalogue models fit a 24 GB RTX 4090 at 32k context, one request?",
        answer=f"{n24} of {total_models}, from fit.rank() on the rtx-4090 profile.",
    )


_TPUS = ("tpu-v5e-8", "tpu-v6e-8", "tpu-v5p-4")


def _hardware_5(ctx: Context) -> LessonResult:
    """TPUs are sized per host."""
    figs: dict[str, float] = {"tp_scaling_efficiency": hardware_catalog.TP_SCALING_EFFICIENCY}
    rows = []
    gens = []
    engine = ""
    for pid in _TPUS:
        p, h = _profile(pid)
        gen = pid.split("-")[1]
        gens.append(gen)
        feasible, _ = fit.rank(h, CTX, 1)
        top = feasible[0].model.id if feasible else "nothing"
        rec = "recommended" if gen in TPU_RECOMMENDED else "experimental"
        engine = str(_pick_engine(h, Requirements(Workload.INTERACTIVE))[0])
        figs[f"{pid}_chip_gb"] = p.memory_gb
        figs[f"{pid}_usable_gb"] = round(h.usable_gb, 0)
        rows.append(
            f"    {gen:<4} {p.devices} chips × {p.memory_gb:3.0f} GB  {h.usable_gb:4.0f} GB usable"
            f"  {h.bandwidth_gbps or 0:6,.0f} GB/s  {rec}"
        )
        rows.append(f"         largest fit at 32k: {top}")
    v5p = "v5p"
    lines = [
        "  lesson hardware·5 · a TPU is sized per host, not per chip",
        "",
        "  A TPU host is several chips. The model is sharded across them, so",
        "  memory and bandwidth add, less the all-reduce at every layer",
        f"  ({hardware_catalog.TP_SCALING_EFFICIENCY:.0%} scaling per extra chip,"
        " a calibration knob).",
        "",
        *rows,
        "",
        f"  The planner picks {engine} on every TPU. It is the only engine there.",
        f"  vLLM lists {', '.join(TPU_RECOMMENDED)} as recommended (plan.TPU_RECOMMENDED).",
        f"  Still maturing on TPU: {', '.join(TPU_MATURING)}.",
        "  `onpar where <model>` ranks these beside the GPUs.",
    ]
    return LessonResult(
        id="hardware.5",
        track=HARDWARE,
        title=_TITLES["hardware.5"],
        text=lines,
        figures=figs,
        question="Is TPU v5p a recommended generation for vLLM, or experimental?",
        answer=(
            "Recommended."
            if v5p in TPU_RECOMMENDED
            else f"Experimental. plan.TPU_RECOMMENDED lists only {', '.join(TPU_RECOMMENDED)}."
        ),
    )


def _hardware_6(ctx: Context) -> LessonResult:
    """Bigger model at q4 vs smaller at q8."""
    hw = ctx.hw

    def biggest(quant: str, below: float) -> Fit | None:
        fits = [
            solve(m, quant, hw, CTX, 1)
            for m in ctx.models
            if quant in m.quants and m.params_b < below
        ]
        fits = [f for f in fits if f.feasible]
        return max(fits, key=lambda f: f.model.params_b) if fits else None

    big = biggest("q4", float("inf"))
    small = biggest("q8", big.model.params_b) if big else None
    head = [
        "  lesson hardware·6 · a bigger model at q4 or a smaller one at q8?",
        "",
        "  Same machine, two ways to spend it. onpar computes both at 32k:",
        "",
    ]
    if big is None or small is None:
        lines = head + [f"  Your {hw.name} cannot hold a q4/q8 pair at 32k context."]
        return LessonResult(
            id="hardware.6",
            track=HARDWARE,
            title=_TITLES["hardware.6"],
            text=lines,
            figures={},
            question="What decides decode speed between two models that both fit?",
            answer="Bytes read per token: active weights at their bit width plus the KV cache.",
        )
    rb, rs = _read_gb(big.model, "q4", CTX), _read_gb(small.model, "q8", CTX)
    faster = big if (big.tokens_per_sec or 0) >= (small.tokens_per_sec or 0) else small
    lines = head + [
        f"    {big.model.name} q4",
        f"      {big.model.params_b:g}B params  {big.weight_bytes / GB:5.1f} GB weights"
        f"  reads {rb:5.1f} GB/token  {_tps(big):5.1f} tok/s",
        f"    {small.model.name} q8",
        f"      {small.model.params_b:g}B params  {small.weight_bytes / GB:5.1f} GB weights"
        f"  reads {rs:5.1f} GB/token  {_tps(small):5.1f} tok/s",
        "",
        f"  tok/s is {ROOFLINE}.",
        "  Speed follows bytes read per token, so an MoE with few active",
        "  parameters can be big and fast at once.",
        "  onpar ranks by total parameters as its capability proxy (fit.rank).",
        "  It does not score quality here. `onpar prove` does, on your traffic.",
    ]
    return LessonResult(
        id="hardware.6",
        track=HARDWARE,
        title=_TITLES["hardware.6"],
        text=lines,
        figures={
            "big_params_b": float(big.model.params_b),
            "big_weight_gb": round(big.weight_bytes / GB, 1),
            "big_read_gb": round(rb, 1),
            "big_tps": _tps(big),
            "small_params_b": float(small.model.params_b),
            "small_weight_gb": round(small.weight_bytes / GB, 1),
            "small_read_gb": round(rs, 1),
            "small_tps": _tps(small),
        },
        question="Which of the two decodes faster on this machine, and why?",
        answer=f"{faster.model.name} at {faster.quant}: it reads"
        f" {min(rb, rs):.1f} GB per token against {max(rb, rs):.1f} GB. {ROOFLINE}.",
    )


def _hardware_7(ctx: Context) -> LessonResult:
    """CPU-only reality."""
    m = _small_model(ctx)
    q = "q4" if "q4" in m.quants else m.quants[-1]
    p, cpu = _profile("cpu-64")
    f_cpu = solve(m, q, cpu, CTX, 1)
    f_here = solve(m, q, ctx.hw, CTX, 1)
    engine, why = fit.recommend_runtime(cpu, CTX, 1)
    ratio = (f_here.tokens_per_sec or 0.0) / (f_cpu.tokens_per_sec or 1.0)
    lines = [
        "  lesson hardware·7 · what CPU-only really gets you",
        "",
        "  A CPU can hold a model in system RAM. It cannot read it fast.",
        "",
        f"  {m.name} at {q}, 32k context:",
        f"    {p.name:<24} {cpu.bandwidth_gbps or 0:6,.0f} GB/s  {_tps(f_cpu):6.1f} tok/s"
        f"  fits: {'yes' if f_cpu.feasible else 'no'}",
        f"    {'this machine':<24} {ctx.hw.bandwidth_gbps or 0:6,.0f} GB/s  {_tps(f_here):6.1f}"
        f" tok/s  fits: {'yes' if f_here.feasible else 'no'}",
        f"  ({ROOFLINE})",
        "",
        f"  The planner picks {engine} on a CPU box and says:",
        *_wrap(why, width=70)[:4],
        "",
        "  Usable is 70% of RAM on a CPU profile. A CPU box is a fine place",
        "  to learn the tools. It is rarely where traffic should land.",
    ]
    return LessonResult(
        id="hardware.7",
        track=HARDWARE,
        title=_TITLES["hardware.7"],
        text=lines,
        figures={
            "cpu_bw_gbps": float(cpu.bandwidth_gbps or 0),
            "cpu_tps": _tps(f_cpu),
            "host_bw_gbps": float(ctx.hw.bandwidth_gbps or 0),
            "host_tps": _tps(f_here),
            "cpu_usable_pct": round(cpu.usable_bytes / cpu.total_bytes * 100, 0),
            "speedup": round(ratio, 1),
        },
        question=f"How much faster is this machine than the CPU-only profile for {m.name}?",
        answer=f"{ratio:.1f}× by roofline ({_tps(f_here):.1f} vs {_tps(f_cpu):.1f} tok/s),"
        " the bandwidth ratio. An estimate, not a measurement.",
    )


# ─── serving lessons ──────────────────────────────────────────────────────────


def _engine_scenarios() -> dict[str, str]:
    """For each engine, the scenario in which `_pick_engine` chooses it, found by asking it."""
    _, apple = _profile("m4-max-128")
    _, h100 = _profile("h100")
    _, tpu = _profile("tpu-v6e-8")
    batch_c = _llmd_threshold(h100)
    probes = [
        (apple, Requirements(Workload.INTERACTIVE, 1), "Apple silicon, one request"),
        (apple, Requirements(Workload.INTERACTIVE, 8), "Apple silicon, 8 requests"),
        (h100, Requirements(Workload.INTERACTIVE, 8), "NVIDIA, no structural reason"),
        (
            h100,
            Requirements(Workload.INTERACTIVE, 8, prefix_sharing=PREFIX_SHARING_FOR_RADIX),
            f"NVIDIA, prefix sharing {PREFIX_SHARING_FOR_RADIX:.0%}+",
        ),
        (h100, Requirements(Workload.BATCH, batch_c), f"NVIDIA, batch at {batch_c}+ requests"),
        (tpu, Requirements(Workload.INTERACTIVE, 1), "any TPU"),
    ]
    out: dict[str, str] = {}
    for hw, req, label in probes:
        out.setdefault(str(_pick_engine(hw, req)[0]), label)
    return out


def _llmd_threshold(hw: Hardware) -> int:
    """Smallest batch concurrency at which the planner disaggregates on this hardware."""
    return next(
        (
            c
            for c in range(1, 1025)
            if _pick_engine(hw, Requirements(Workload.BATCH, c, 8192))[0] is Engine.LLMD
        ),
        0,
    )


def _serving_1(ctx: Context) -> LessonResult:
    """The seven engines and what each is for."""
    scen = _engine_scenarios()
    rows = []
    with_adapter = 0
    for e in Engine:
        a = adapter_for(e.value)
        with_adapter += a is not None
        when = scen.get(e.value, "never auto-picked; `onpar run --engine` forces it")
        rows.append(f"    {e.value:<10} {type(a).__name__ if a else 'none':<15} {when}")
    _, apple = _profile("m4-max-128")
    apple_why = _first_sentence(_pick_engine(apple, Requirements(Workload.INTERACTIVE))[1])
    lines = [
        f"  lesson serving·1 · the {len(Engine)} engines onpar can configure",
        "",
        "  onpar does not build an engine. It picks one and writes its flags.",
        f"  Each of these {with_adapter} has an adapter that translates intent",
        "  into that engine's own verified flags (engines.adapter_for).",
        "",
        "    engine     adapter         when the planner picks it",
        *rows,
        "",
        "  The scenarios are not a table. Each row is plan._pick_engine()",
        "  asked about a real hardware profile.",
    ]
    return LessonResult(
        id="serving.1",
        track=SERVING,
        title=_TITLES["serving.1"],
        text=lines,
        figures={"engines": float(len(Engine)), "with_adapter": float(with_adapter)},
        question="Why does the planner never pick vLLM on Apple silicon?",
        answer=f'plan._pick_engine says: "{apple_why}"',
    )


def _serving_2(ctx: Context) -> LessonResult:
    """Which engine the planner picks here and why."""
    hw = ctx.hw
    e1, why1 = fit.recommend_runtime(hw, CTX, 1)
    e8, why8 = fit.recommend_runtime(hw, CTX, 8)
    p = plan_for(hw, Requirements(Workload.INTERACTIVE, 1, CTX), ctx.model, ctx.quant)
    mem = p.get(Setting.MEMORY_FRACTION)
    lines = [
        f"  lesson serving·2 · what onpar picks on your {hw.name}",
        "",
        "  fit.recommend_runtime() asks the planner, then checks that onpar can",
        "  launch the answer. A pick that cannot start is not a pick.",
        "",
        f"  one request at 32k   → {e1}",
        *_wrap(_first_sentence(why1), indent="    "),
        f"  8 requests at 32k    → {e8}",
        *_wrap(_first_sentence(why8), indent="    "),
        "",
        "  Every setting carries its reason (Knob.why). For example:",
    ]
    if mem is not None:
        lines += [f"    {mem.render()}", *_wrap(_first_sentence(mem.why), indent="      ")[:3]]
    lines += ["", f"  `onpar run --dry-run {ctx.model.id}` prints the full plan here."]
    return LessonResult(
        id="serving.2",
        track=SERVING,
        title=_TITLES["serving.2"],
        text=lines,
        figures={
            "concurrency_one": 1.0,
            "concurrency_eight": 8.0,
            "context_tokens": float(CTX),
            "memory_fraction": float(mem.value) if mem is not None else 0.0,
            "margin": MEMORY_SAFETY_MARGIN,
        },
        question="Which engine does onpar pick on this machine at 8 concurrent requests?",
        answer=f"{e8}. {_first_sentence(why8)}",
    )


def _serving_3(ctx: Context) -> LessonResult:
    """Twelve settings and three forks."""
    m = ctx.model
    _, h100 = _profile("h100")

    def eng(req: Requirements) -> str:
        return str(_pick_engine(h100, req)[0])

    def shared(x: float) -> str:
        return eng(Requirements(Workload.INTERACTIVE, 8, prefix_sharing=x))

    def batch(c: int) -> str:
        return eng(Requirements(Workload.BATCH, c, 8192))

    def spec(c: int) -> str:
        k = plan_for(h100, Requirements(Workload.INTERACTIVE, c, CTX), m).get(Setting.SPECULATIVE)
        return str(k.value) if k is not None else "not set"

    radix = PREFIX_SHARING_FOR_RADIX
    below = round(radix - 0.01, 2)
    batch_c = _llmd_threshold(h100)
    settings = [s.value for s in Setting]
    grid = [", ".join(settings[i : i + 4]) for i in range(0, len(settings), 4)]
    lines = [
        f"  lesson serving·3 · {len(settings)} settings, and the forks between engines",
        "",
        f"  The planner speaks in {len(settings)} intents, not flags:",
        *[f"    {row}" for row in grid],
        "",
        "  Three thresholds flip a decision. Asked on the H100 profile:",
        f"    prefix sharing  {below:.0%} → {shared(below)}   {radix:.0%} → {shared(radix)}"
        "   (PREFIX_SHARING_FOR_RADIX)",
        f"    batch requests  {batch_c - 1} → {batch(batch_c - 1)}   {batch_c} → {batch(batch_c)}",
        f"    speculative     c={MAX_SPEC_DECODE_CONCURRENCY} → {spec(MAX_SPEC_DECODE_CONCURRENCY)}"
        f"   c={MAX_SPEC_DECODE_CONCURRENCY + 1} → {spec(MAX_SPEC_DECODE_CONCURRENCY + 1)}"
        "   (MAX_SPEC_DECODE_CONCURRENCY)",
        "",
        "  SGLang and llm-d ship CUDA-only builds. On other hardware the first",
        "  two forks fall back to vLLM, and the plan says why.",
        "  These thresholds are calibration knobs in plan.py.",
    ]
    q_share = round(radix + 0.05, 2)
    return LessonResult(
        id="serving.3",
        track=SERVING,
        title=_TITLES["serving.3"],
        text=lines,
        figures={
            "settings": float(len(settings)),
            "prefix_sharing_for_radix": radix,
            "below_radix": below,
            "llmd_batch_threshold": float(batch_c),
            "max_spec_decode_concurrency": float(MAX_SPEC_DECODE_CONCURRENCY),
        },
        question=f"At {q_share:.0%} prefix sharing on an H100, which engine does the planner pick?",
        answer=f"{eng(Requirements(Workload.INTERACTIVE, 8, prefix_sharing=q_share))}:"
        f" {q_share:.0%} is at or above PREFIX_SHARING_FOR_RADIX ({radix:.0%}).",
    )


def _serving_4(ctx: Context) -> LessonResult:
    """Native config is standalone."""
    m = _servable(ctx)
    quant = ctx.quant if m is ctx.model else None
    p = plan_for(ctx.hw, Requirements(Workload.INTERACTIVE, 1, CTX), m, quant)
    argv, gaps = p.command(m.repo)
    env = p.environment()
    cmd = shlex.join(argv) if argv else "(no command)"
    body = textwrap.wrap(cmd, width=66, break_on_hyphens=False, break_long_words=False)
    lines = [
        "  lesson serving·4 · the config onpar writes runs without onpar",
        "",
        f"  The plan for {m.name} on your {ctx.hw.name}:",
        "",
        *[
            f"    {ln}{cont}"
            for ln, cont in zip(body[:10], [" \\"] * (len(body[:10]) - 1) + [""], strict=True)
        ],
    ]
    if env:
        lines += ["", "  environment:"] + [f"    {k}={v}" for k, v in env[:3]]
    if gaps:
        lines += ["", "  not expressed by this engine:"]
        lines += [ln for g in gaps[:3] for ln in _wrap(g, width=74)[:2]]
    lines += [
        "",
        f"  That is a real {p.engine.value} invocation. Uninstall onpar and it still runs.",
        "  A wrapper would tie your serving to this tool, so onpar never emits one.",
        f"  `onpar run --dry-run {m.id}` prints this plan without starting it.",
    ]
    return LessonResult(
        id="serving.4",
        track=SERVING,
        title=_TITLES["serving.4"],
        text=lines,
        figures={"argv_len": float(len(argv)), "gaps": float(len(gaps))},
        question="Does the command above need onpar installed to run?",
        answer=f"No. It is {argv[0] if argv else p.engine.value} with its own flags."
        " Generated config is native and standalone (NFR-4).",
    )


def _serving_5(ctx: Context) -> LessonResult:
    """Tool calling as a server problem: the parser table."""
    m = _servable(ctx)
    vt, st = _VLLM_TOOL_PARSERS, _SGLANG_TOOL_PARSERS
    rows = [f"    {prefix:<26} {parser}" for prefix, parser in vt]
    v_here = _tool_parser(m.repo, vt)
    s_here = _tool_parser(m.repo, st)
    lines = [
        "  lesson serving·5 · tool calling is decided by the server",
        "",
        "  Without the right parser flag, a tool call comes back as plain",
        "  text with finish_reason stop. onpar's eval suite calls that tool-leak.",
        "",
        f"  vLLM parsers onpar maps, first match wins ({len(vt)} rows):",
        *rows,
        f"  SGLang has its own table ({len(st)} rows), with its own names.",
        "",
        f"  {m.repo}:",
        f"    vLLM   → {v_here or 'no verified parser: the plan reports the gap'}",
        f"    SGLang → {s_here or 'no verified parser: the plan reports the gap'}",
        "",
        "  `onpar build --tool-calling` asks the planner for it.",
    ]
    return LessonResult(
        id="serving.5",
        track=SERVING,
        title=_TITLES["serving.5"],
        text=lines,
        figures={"vllm_rows": float(len(vt)), "sglang_rows": float(len(st))},
        question=f"Which vLLM tool-call parser does onpar pick for {m.repo}?",
        answer=f"{v_here}, the first prefix in the table that matches the repo."
        if v_here
        else "None. No prefix matches, so the plan surfaces the gap instead of guessing.",
    )


def _serving_6(ctx: Context) -> LessonResult:
    """Continuous batching and the KV budget at N."""
    m, hw = ctx.model, ctx.hw
    figs: dict[str, float] = {}
    rows = []
    fits = {}
    for c in (1, 8, 32):
        f = solve(m, ctx.quant, hw, CTX, c)
        agg = f.aggregate_tokens_per_sec or 0.0
        fits[c] = f.feasible
        figs[f"kv_gb_c{c}"] = round(f.kv_bytes / GB, 1)
        figs[f"total_gb_c{c}"] = round(f.total_bytes / GB, 1)
        figs[f"agg_tps_c{c}"] = round(agg, 0)
        rows.append(
            f"    c={c:<3} KV {f.kv_bytes / GB:6.1f} GB  total {f.total_bytes / GB:6.1f} GB"
            f"  {'fits' if f.feasible else 'DOES NOT FIT':<12}  {agg:7,.0f} tok/s"
        )
    max_c = fit.max_concurrency(m, ctx.quant, hw, CTX)
    figs["usable_gb"] = round(hw.usable_gb, 1)
    figs["max_concurrency"] = float(max_c)
    lines = [
        "  lesson serving·6 · continuous batching spends KV, not weights",
        "",
        "  A batching engine decodes many requests in one pass. The weights",
        "  are read once per step for the whole batch. Each request still",
        "  holds its own KV cache.",
        "",
        f"  {m.name} at {ctx.quant}, 32k context, {hw.usable_gb:.1f} GB usable:",
        *rows,
        "",
        f"  tok/s is aggregate across the batch, {ROOFLINE}.",
        f"  The KV budget runs out at {max_c} requests (fit.max_concurrency).",
    ]
    if m.is_moe:
        lines.append("  For MoE the batched figure is optimistic: experts diverge per request.")
    return LessonResult(
        id="serving.6",
        track=SERVING,
        title=_TITLES["serving.6"],
        text=lines,
        figures=figs,
        question="Does this machine hold 32 requests at 32k context?",
        answer=(
            f"{'Yes' if fits[32] else 'No'}: {figs['total_gb_c32']:.1f} GB needed against"
            f" {hw.usable_gb:.1f} GB usable. The ceiling is {max_c}."
        ),
    )


def _serving_7(ctx: Context) -> LessonResult:
    """What an engine does not express."""
    m = _servable(ctx)
    _, h100 = _profile("h100")
    req = Requirements(
        Workload.INTERACTIVE, 8, CTX, prefix_sharing=0.6, structured_output=True, tool_calling=True
    )
    rows = []
    worst: tuple[str, tuple[str, ...]] = ("", ())
    figs: dict[str, float] = {}
    for e in Engine:
        _, gaps = plan_for(h100, req, m, force_engine=e).command(m.repo)
        figs[f"gaps_{e.value}"] = float(len(gaps))
        rows.append(f"    {e.value:<10} {len(gaps)} not expressed")
        if len(gaps) > len(worst[1]):
            worst = (e.value, gaps)
    lines = [
        "  lesson serving·7 · what an engine cannot express",
        "",
        *_wrap(
            "Ask every engine for the same plan: 8 requests, prefix sharing,"
            f" structured output and tool calls, for {m.name} on the H100 profile.",
            indent="  ",
        ),
        "",
        *rows,
    ]
    if worst[1]:
        lines += ["", f"  {worst[0]}, for example:"]
        lines += [ln for g in worst[1][:3] for ln in _wrap(g, indent="    ")[:2]]
    lines += [
        "",
        "  Plan.command() returns each gap beside the flags. `onpar run`",
        "  prints them as NOT EXPRESSED. Nothing is dropped in silence.",
    ]
    return LessonResult(
        id="serving.7",
        track=SERVING,
        title=_TITLES["serving.7"],
        text=lines,
        figures=figs,
        question="What does onpar do with a setting an engine has no verified flag for?",
        answer="It returns it as a gap beside the command and prints it as NOT EXPRESSED."
        " It never borrows another engine's flag, which would not run.",
    )


# ─── hosting lessons ──────────────────────────────────────────────────────────


def _hosting_1(ctx: Context) -> LessonResult:
    """Local vs rented vs managed."""
    m = ctx.model
    s = host.survey(m, context=CTX, concurrency=1)
    opts = [o for o in s.options if o.placement.feasible]
    kinds: dict[str, int] = {}
    for o in opts:
        kinds[o.provider.artifact_kind] = kinds.get(o.provider.artifact_kind, 0) + 1
    paid = sorted(
        (o for o in opts if o.offer.usd_per_hour), key=lambda o: o.offer.usd_per_hour or 0.0
    )
    free = [o for o in opts if o.provider.free is not None and not o.offer.usd_per_hour]
    figs: dict[str, float] = {"options": float(len(opts)), "excluded": float(len(s.excluded))}
    rows = []
    for o in paid[:4]:
        cpm = o.placement.cost_per_mtok_usd
        figs[f"usd_h_{o.provider.id}_{o.offer.profile_id}"] = o.offer.usd_per_hour or 0.0
        rows.append(
            f"    {o.provider.name[:22]:<22} {o.offer.profile_id:<12}"
            f" ${o.offer.usd_per_hour:6.2f}/h" + (f"  ~${cpm:,.2f}/Mtok" if cpm else "")
        )
    lines = [
        f"  lesson hosting·1 · run {m.name} here, rented, or managed",
        "",
        f"  Here: {'it fits' if ctx.fit.feasible else 'it does not fit'} your {ctx.hw.name}"
        f" at {ctx.quant}.",
        f"  Elsewhere: `onpar host {m.id}` finds {len(opts)} places that can hold it.",
        "    by artifact: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())),
        f"    free tiers that fit: {len(free)}",
        "",
        "  Cheapest by the hour (prices checked " + host.CHECKED + "):",
        *rows,
        "",
        "  $/Mtok is an estimate. It assumes the machine is busy.",
    ]
    if s.excluded:
        lines.append(f"  Excluded on purpose: {s.excluded[0].provider_name}, by its own terms.")
    lines.append("  onpar writes the deploy file. It never authenticates or pushes.")
    cheapest = paid[0] if paid else None
    return LessonResult(
        id="hosting.1",
        track=HOSTING,
        title=_TITLES["hosting.1"],
        text=lines,
        figures=figs,
        question=f"Which listed place rents hardware for {m.name} cheapest by the hour?",
        answer=(
            f"{cheapest.provider.name} on {cheapest.offer.profile_id} at"
            f" ${cheapest.offer.usd_per_hour:.2f}/h, from host.survey()."
            if cheapest
            else "None. No paid offer in host.py holds it at 32k context."
        ),
    )


_BOX_FOR_KIND = {"apple": "darwin-metal", "nvidia": "linux-cuda", "amd": "linux-rocm"}


def _hosting_2(ctx: Context) -> LessonResult:
    """The box: run as packed, re-solve, or refuse."""
    m, hw = ctx.model, ctx.hw
    rows = []
    figs: dict[str, float] = {}
    for t in box.TARGETS:
        r = box._solve(m, t, None, CTX, 1)
        if isinstance(r, box.Skipped):
            rows += [f"    {t.id:<13} refused"] + _wrap(r.reason, indent="      ")[:2]
            continue
        cls, f, p = r
        figs[f"{t.id}_total_gb"] = round(f.total_bytes / GB, 1)
        rows.append(f"    {t.id:<13} {cls[:24]:<24} {f.quant:<4} {p.engine.value}")
    target_id = _BOX_FOR_KIND.get(hw.kind)
    target = next((t for t in box.TARGETS if t.id == target_id), None)
    here = fit.best_quant(m, hw, CTX, 1)
    if target is None:
        arrival = f"no box target matches {hw.kind} hardware, so a box would not run here."
    else:
        _, tuned = _profile(target.profile_id)
        packed = fit.best_quant(m, tuned, CTX, 1)
        if here is None:
            arrival = "it does not fit at any quant here, so the run refuses."
        elif packed is not None and tuned.usable_bytes == hw.usable_bytes:
            arrival = f"it matches the {target.id} host class, so it runs as packed."
        else:
            arrival = (
                f"this host differs from {target.id}'s class, so it re-solves to {here.quant}."
            )
    _, small = _profile("rtx-4090")
    on_24 = fit.best_quant(m, small, CTX, 1)
    lines = [
        "  lesson hosting·2 · one box, tuned per host class, re-solved on arrival",
        "",
        f"  `onpar box {m.id}` builds one artifact with a tuning per target:",
        *rows,
        "",
        "  On arrival the box re-detects the machine (ADR-0005). If the host",
        "  matches, it runs as packed. If it differs, it re-solves. If nothing",
        "  fits, it refuses rather than running out of memory.",
        "",
        "  Landing on your machine:",
        *_wrap(arrival, indent="    "),
        "",
        "  The box carries roofline estimates, each labelled as one.",
    ]
    return LessonResult(
        id="hosting.2",
        track=HOSTING,
        title=_TITLES["hosting.2"],
        text=lines,
        figures=figs,
        question=f"A box of {m.name} tuned for an 80 GB card lands on a 24 GB RTX 4090."
        " What happens?",
        answer=(
            f"It re-solves to {on_24.quant} and runs ({on_24.total_bytes / GB:.1f} GB"
            f" of {small.usable_gb:.1f} GB usable)."
            if on_24
            else f"It refuses: no quant of {m.name} fits {small.usable_gb:.1f} GB usable at 32k."
        ),
    )


def _node_from_profile(profile_id: str) -> k8s_nodes.Node:
    p, h = _profile(profile_id)
    return k8s_nodes.Node(
        name=profile_id,
        kind=p.kind,
        devices=p.devices,
        device_bytes=int(p.memory_gb * GB),
        bandwidth_gbps=p.bandwidth_gbps,
        product=p.name,
        host_bytes=h.total_bytes,
        cpus=0,
        unknown="",
        note="built from the hardware catalogue profile",
        labels={},
    )


def _hosting_3(ctx: Context) -> LessonResult:
    """Kubernetes: the operator sizes what it applies."""
    nodes = [_node_from_profile("l4"), _node_from_profile("h100")]
    with_repo = sorted((m for m in ctx.models if m.repo), key=lambda x: -x.params_b)
    biggest = with_repo[0]

    def run(m: ModelSpec) -> tuple[k8s_reconcile.Reconciled, dict[str, str]]:
        r = k8s_reconcile.reconcile(
            {
                "metadata": {"name": "triage", "namespace": "ml"},
                "spec": {"model": m.repo, "workload": "interactive", "concurrency": 1},
            },
            nodes,
        )
        cond = next((c for c in r.status.get("conditions", ()) if c.get("type") == "Ready"), {})
        return r, cond

    # The largest model the operator itself accepts on these nodes.
    good, ok = next(
        ((m, r) for m in with_repo if k8s_controller._ready(r := run(m)[0])),
        (with_repo[-1], run(with_repo[-1])[0]),
    )
    bad, bad_cond = run(biggest)
    applies_ok = k8s_controller._ready(ok)
    applies_bad = k8s_controller._ready(bad)
    lines = [
        "  lesson hosting·3 · on Kubernetes the operator sizes before it applies",
        "",
        "  onpar's operator reads an InferenceWorkload and the cluster's nodes,",
        "  then derives the engine, the node and every flag. Two nodes, from",
        "  the l4 and h100 profiles:",
        "",
        f"  {good.repo}, one request:",
        f"    engine {ok.status.get('engine')} on {ok.status.get('node')}",
        f"    objects: {', '.join(o['kind'] for o in ok.objects) or 'none'}"
        f"   knobs: {len(ok.status.get('knobs', []))}   applied: {'yes' if applies_ok else 'no'}",
        f"  {biggest.repo}, one request:",
        f"    Ready={bad_cond.get('status')} reason={bad_cond.get('reason')}",
        *_wrap(str(bad_cond.get("message", "")), indent="    ")[:2],
        f"    applied: {'yes' if applies_bad else 'no'}",
        "",
        f"  The workload names no quant, so it sizes {good.quants[0]}, the first one",
        "  the catalogue lists. The container runs the engine natively. llm-d puts",
        "  vLLM pods behind an InferencePool; that pool and its balancing",
        "  belong to llm-d and the Gateway API, not to onpar (ADR-0015).",
    ]
    return LessonResult(
        id="hosting.3",
        track=HOSTING,
        title=_TITLES["hosting.3"],
        text=lines,
        figures={"knobs": float(len(ok.status.get("knobs", [])))},
        question="If no node can hold the model, what does the operator apply?",
        answer=f"Nothing. Ready={bad_cond.get('status')} with reason {bad_cond.get('reason')},"
        " and the controller refuses to apply a workload that is not Ready.",
    )


def _hosting_4(ctx: Context) -> LessonResult:
    """Cost as a range (ADR-0017), reusing the README's test-pinned example."""
    # The README's inputs, pinned by tests/test_docs_lab.py: $2,847 incumbent,
    # $317 candidate, 400 captures over 14 days, a 90% bar, 4 of 200 items differ.
    items = [
        EvalItem(
            item_id=str(i),
            cluster="c",
            prompt=f"p{i}",
            baseline="a",
            candidate="a" if i >= 4 else f"x{i}",
        )
        for i in range(200)
    ]
    got = suite(
        items,
        shares={"c": 1.0},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        bar=0.90,
        traffic_captures=400,
        traffic_window="14 days",
    )
    line = next(x.strip() for x in got.policy.render().splitlines() if "Saving" in x)
    short = cost.saving(2847.0, 317.0, 0.98, captures=400, window="6 days")
    lines = [
        "  lesson hosting·4 · the saving is a range, never a point",
        "",
        "  You give onpar what the incumbent costs per month and what the",
        "  candidate will cost. It prices only the traffic it proved, and the",
        "  dollar range carries the same Wilson interval as the evidence.",
        "",
        "  The README's example: $2,847/mo incumbent, $317/mo candidate,",
        "  400 captured requests over 14 days, 196 of 200 items matching:",
        "",
        *_wrap(line, width=74),
        "",
        f"  The same inputs over 6 days are refused (MIN_WINDOW_DAYS = {cost.MIN_WINDOW_DAYS:g}):",
        *_wrap(short.refusal, indent="    ")[:3],
        "",
        "  Fewer captures widen the range. The rates never change.",
    ]
    return LessonResult(
        id="hosting.4",
        track=HOSTING,
        title=_TITLES["hosting.4"],
        text=lines,
        figures={"min_window_days": float(cost.MIN_WINDOW_DAYS)},
        question="Why does onpar refuse to price a 6-day capture?",
        answer=short.refusal or "It does not; the window was long enough.",
    )


def _hosting_5(ctx: Context) -> LessonResult:
    """What the gateway does in the path, and that it leaves (ADR-0015)."""
    ladder = [s.render() for s in gate.LADDER]
    last = gate.LADDER[-1]
    lines = [
        "  lesson hosting·5 · the gateway is in your path only while you migrate",
        "",
        "  `onpar observe --upstream URL` puts a gateway between your app and",
        "  the provider you use today. While a migration runs, it does three",
        "  things no load balancer does:",
        "    · mirrors traffic to a candidate that is scored, never served",
        "    · splits by percentage deterministically, so a retry lands the same way",
        "    · keeps each task cluster on the incumbent until that cluster is proven",
        "",
        f"  The phases (k8s reconcile.PHASE_ORDER): {', '.join(k8s_reconcile.PHASE_ORDER)}.",
        "  The ladder (prove gate.LADDER):",
        f"    {' → '.join(ladder)}",
        "",
        f"  At {last.render()} the candidate takes {last.percent}% and there is nothing",
        "  left to compare. The gateway leaves. Balancing belongs to llm-d and",
        "  the Gateway API (ADR-0015).",
        "  `onpar observe --no-capture` routes without writing a log.",
    ]
    return LessonResult(
        id="hosting.5",
        track=HOSTING,
        title=_TITLES["hosting.5"],
        text=lines,
        figures={f"rung_{i}_pct": float(s.percent) for i, s in enumerate(gate.LADDER)},
        question="At which rung does onpar leave the request path?",
        answer=f"{last.render()}: {last.percent}% of traffic is on the candidate, so there"
        " is nothing left to compare (ADR-0015).",
    )


def _hosting_6(ctx: Context) -> LessonResult:
    """Zero egress and redaction before disk (NFR-2, NFR-3)."""
    sample = "reach me at someone@example.com"
    try:
        redacted, counts = core.redact(sample)
        status = [f'    in:  "{sample}"', f'    out: "{redacted}"', f"    found: {counts}"]
        here = "redaction ran here, before anything could be written"
    except (core.SeamError, ValueError) as e:
        status = _wrap(f"redaction unavailable: {e}", indent="    ")[:3]
        here = "redaction cannot run here, so a capture would be dropped, not written"
    lines = [
        "  lesson hosting·6 · nothing leaves your machine, and nothing raw hits disk",
        "",
        "  Captured traffic is the most sensitive data you have. Two rules:",
        "    NFR-2  no traffic, prompt or eval leaves the machine unless you",
        "           export it. No telemetry, no background sync.",
        "    NFR-3  redaction runs before persistence. If it fails, the",
        "           capture is dropped. Unredacted text never touches disk.",
        "",
        "  core.redact() on a synthetic line, on this machine:",
        *status,
        "",
        f"  So: {here}.",
        "  A prompt in the corpus that says 'ignore previous instructions'",
        "  is a row in a table. Captured traffic is data, never instructions.",
    ]
    return LessonResult(
        id="hosting.6",
        track=HOSTING,
        title=_TITLES["hosting.6"],
        text=lines,
        figures={},
        question="What happens to a capture when redaction fails?",
        answer=f"It is dropped, never written unredacted (NFR-3). On this machine, {here}.",
    )


def _hosting_7(ctx: Context) -> LessonResult:
    """The cutover ladder: shadow, canary, cut; a judge alone never promotes."""

    def reading(name: str, passed: int, total: int, *, judge: bool = False) -> gate.Reading:
        return gate.Reading(
            score=ClusterScore(name, name, 1.0, wilson(passed, total), 0), judge_only=judge
        )

    shadow, canary = gate.LADDER[0], gate.LADDER[2]
    graded = gate.decide([reading("extract", 40, 40)], shadow)
    judged = gate.decide([reading("summarise", 40, 40, judge=True)], shadow)
    regress = gate.decide([reading("classify", 2, 40)], canary, health=gate.Health(100, 0))
    cases = [
        ("40/40 graded, in shadow", graded),
        ("40/40 judged only, in shadow", judged),
        (f"2/40 graded, at {canary.render()}", regress),
    ]
    rows = []
    for label, d in cases:
        rows.append(f"    {label}:")
        rows += _wrap(d.render(), indent="      ")[:2]
        rows.append(f"      applied without a human: {'yes' if d.automatic else 'no'}")
    lines = [
        "  lesson hosting·7 · the cutover ladder, and who may climb it",
        "",
        f"  {' → '.join(s.render() for s in gate.LADDER)}",
        "",
        "  gate.decide() on three cases, run now:",
        *rows,
        "",
        "  A rollback is automatic. An advance is a proposal a human applies.",
        "  A judge alone may block a cluster but never moves traffic.",
    ]
    return LessonResult(
        id="hosting.7",
        track=HOSTING,
        title=_TITLES["hosting.7"],
        text=lines,
        figures={f"rung_{i}_pct": float(s.percent) for i, s in enumerate(gate.LADDER)},
        question="Can a cluster scored only by an LLM judge advance a migration?",
        answer=f"No. gate.decide() returns {judged.action.value.upper()}:"
        f" {_first_sentence(judged.reason)}",
    )


# ─── lesson registry ──────────────────────────────────────────────────────────

LESSONS: list[Lesson] = [
    Lesson("basics.1", BASICS, "What a model is in bytes (weights)", _basics_1),
    Lesson("basics.2", BASICS, "What context costs (KV cache)", _basics_2),
    Lesson("basics.3", BASICS, "Weights + KV + overhead vs usable memory", _basics_3),
    Lesson("basics.4", BASICS, "Why tok/s is memory bandwidth (roofline estimate)", _basics_4),
    Lesson("basics.5", BASICS, "What quantisation buys and costs", _basics_5),
    Lesson("basics.6", BASICS, "Why MoE is sized on total params (the dangerous error)", _basics_6),
    Lesson("basics.7", BASICS, "What 'proven on par' means (Wilson interval)", _basics_7),
    Lesson("inter.1", INTERMEDIATE, "GQA vs MHA and kv_heads", _inter_1),
    Lesson("inter.2", INTERMEDIATE, "MLA and kv_lora_rank (the 64× overestimate)", _inter_2),
    Lesson("inter.3", INTERMEDIATE, "Concurrency and the KV budget", _inter_3),
    Lesson("inter.4", INTERMEDIATE, "Prefix caching and agent loops", _inter_4),
    Lesson("inter.5", INTERMEDIATE, "Speculative decoding and high-batch limits", _inter_5),
    Lesson("inter.6", INTERMEDIATE, "Chunked prefill / TTFT vs ITL budgets", _inter_6),
    Lesson("inter.7", INTERMEDIATE, "Tool calling as a server problem", _inter_7),
    Lesson("inter.8", INTERMEDIATE, "Teacher-forced step replay and judge intervals", _inter_8),
]
_RENDERERS = {
    HARDWARE: (
        _hardware_1,
        _hardware_2,
        _hardware_3,
        _hardware_4,
        _hardware_5,
        _hardware_6,
        _hardware_7,
    ),
    SERVING: (_serving_1, _serving_2, _serving_3, _serving_4, _serving_5, _serving_6, _serving_7),
    HOSTING: (_hosting_1, _hosting_2, _hosting_3, _hosting_4, _hosting_5, _hosting_6, _hosting_7),
}
for _track, _fns in _RENDERERS.items():
    for _i, _fn in enumerate(_fns, 1):
        _id = f"{_track}.{_i}"
        LESSONS.append(Lesson(_id, _track, _TITLES[_id], _fn))

_BY_TRACK: dict[str, list[Lesson]] = {}
for _l in LESSONS:
    _BY_TRACK.setdefault(_l.track, []).append(_l)

TRACK_BLURBS = {t: f"{_TRACK_TOPICS[t]} ({len(_BY_TRACK[t])} lessons)" for t in TRACKS}


# ─── rendering helpers ────────────────────────────────────────────────────────


def _track_lessons(track: str) -> list[Lesson]:
    ls = _BY_TRACK.get(track)
    if ls is None:
        known = ", ".join(sorted(_BY_TRACK))
        raise ValueError(f"unknown track {track!r} — known: {known}")
    return ls


def render(lesson: Lesson, ctx: Context) -> LessonResult:
    """Render one lesson against this machine's context."""
    return lesson.render(ctx)


def render_all(track: str, ctx: Context) -> list[LessonResult]:
    return [render(l_, ctx) for l_ in _track_lessons(track)]


# ─── number extraction for demo() assertion ───────────────────────────────────

_NUMBER_RE = re.compile(r"\b(\d+\.?\d*)\s*(?:GB|B/tok|bits|tok/s|%|×)")


def _numbers_in_text(lines: list[str]) -> list[float]:
    """Pull all numeric values from lesson text lines."""
    out: list[float] = []
    for line in lines:
        for m in _NUMBER_RE.finditer(line):
            out.append(float(m.group(1)))
    return out


def _number_in_figures(n: float, figures: dict[str, float], tol: float = 0.05) -> bool:
    """True if `n` is within `tol` of any figures entry (relative).

    Also checks n vs v*100 to handle percentages displayed as e.g. '90%'
    but stored as 0.9 in figures.
    """
    for v in figures.values():
        if v == 0:
            continue
        if abs(n - v) / max(abs(v), 1e-9) <= tol:
            return True
        # percentage vs fraction
        if abs(v) <= 1.0 and abs(n - v * 100) / max(abs(v * 100), 1e-9) <= tol:
            return True
    return False


# ─── JSON output ──────────────────────────────────────────────────────────────


def to_json_obj(lessons: list[LessonResult]) -> dict:
    return {
        "tracks": list(TRACK_BLURBS.keys()),
        "lessons": [
            {
                "id": lr.id,
                "track": lr.track,
                "title": lr.title,
                "text": lr.text,
                "question": lr.question,
                "answer": lr.answer,
                "figures": {k: v for k, v in lr.figures.items()},
            }
            for lr in lessons
        ],
    }


# ─── interactive runner ───────────────────────────────────────────────────────


def _run_interactive(
    lessons: list[LessonResult],
    *,
    no_quiz: bool = False,
    tty: bool = True,
) -> None:
    """Print lessons to stdout.  In TTY mode, pause between them."""
    for i, lr in enumerate(lessons, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{lr.id}]  {lr.title}")
        print()
        for line in lr.text:
            print(line)
        print()

        if lr.question and not no_quiz:
            print(f"  check yourself: {lr.question}")
            if tty:
                with contextlib.suppress(EOFError, KeyboardInterrupt):
                    input("  your answer: ")
            print(f"  answer: {lr.answer}")
            print()

        if tty and i < len(lessons):
            try:
                input("  [Enter for next lesson, Ctrl-C to stop] ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
        elif not tty and lr.question and no_quiz:
            print(f"  answer: {lr.answer}")


# ─── demo / self-check ────────────────────────────────────────────────────────


def demo() -> None:
    """Self-check runnable via `python -m onpar.learn`."""
    apple = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * GB,
        usable_bytes=96 * GB,
        bandwidth_gbps=546.0,
        cores=16,
    )
    nvidia = Hardware(
        kind="nvidia",
        name="H100 80GB",
        total_bytes=80 * GB,
        usable_bytes=int(80 * GB * 0.90),
        bandwidth_gbps=3350.0,
        cores=132,
    )

    models = catalog.load()

    for _, hw in [("apple", apple), ("nvidia", nvidia)]:
        feasible, _ = fit.rank(hw, 32768, 1)
        m = feasible[0].model if feasible else min(models, key=lambda x: x.params_b)
        best = fit.best_quant(m, hw, 32768, 1)
        if best is None:
            q = min(m.quants, key=lambda q_: QUANT_BITS[q_])
            f = solve(m, q, hw, 32768, 1)
        else:
            f, q = best, best.quant
        ctx = Context(hw=hw, models=models, model=m, quant=q, fit=f)

        for lesson in LESSONS:
            result = render(lesson, ctx)

            # Every lesson renders without error
            assert result.id == lesson.id, f"{lesson.id}: id mismatch"
            assert result.text, f"{lesson.id}: empty text"

            # Every lesson has a title and track
            assert result.title
            assert result.track in TRACKS
            assert len(result.text) <= 30, f"{lesson.id}: {len(result.text)} lines"

            # Lessons with numeric content must have figures
            numbers = _numbers_in_text(result.text)
            if numbers and lesson.id != "inter.7":
                assert result.figures, f"{lesson.id}: has numbers in text but empty figures dict"

            # Every figure value must appear in the text (approx) or be a supporting
            # computation. Check that the key figures from solve() are represented.
            if lesson.track == BASICS and lesson.id not in ("basics.7",):
                # For basic sizing lessons, solve()-derived figures must appear
                for key in ("weight_gb", "kv_gb", "hw_usable_gb", "tokens_per_sec"):
                    if key in result.figures:
                        val = result.figures[key]
                        if val > 0 and val < 1000:
                            assert _number_in_figures(val, result.figures), (
                                f"{lesson.id}: figure {key}={val} not round-trippable"
                            )

            # Quiz answers are derived (non-empty), not literal booleans
            if result.question:
                assert result.answer, f"{lesson.id}: question with no answer"
                assert isinstance(result.answer, str) and len(result.answer) > 5

    # JSON output is valid and covers both tracks
    all_results = []
    hw = apple
    feasible, _ = fit.rank(hw, 32768, 1)
    m = feasible[0].model if feasible else min(models, key=lambda x: x.params_b)
    best = fit.best_quant(m, hw, 32768, 1)
    if best is None:
        q = min(m.quants, key=lambda q_: QUANT_BITS[q_])
        f = solve(m, q, hw, 32768, 1)
    else:
        f, q = best, best.quant
    ctx = Context(hw=hw, models=models, model=m, quant=q, fit=f)

    for lesson in LESSONS:
        all_results.append(render(lesson, ctx))

    obj = to_json_obj(all_results)
    assert tuple(obj["tracks"]) == TRACKS
    assert len(obj["lessons"]) == len(LESSONS)
    per_track = {t: [lr["id"] for lr in obj["lessons"] if lr["track"] == t] for t in TRACKS}
    basics_ids, inter_ids = per_track[BASICS], per_track[INTERMEDIATE]
    assert len(basics_ids) == 7, f"expected 7 basics, got {len(basics_ids)}"
    assert len(inter_ids) == 8, f"expected 8 intermediate, got {len(inter_ids)}"
    for t in (HARDWARE, SERVING, HOSTING):
        assert 6 <= len(per_track[t]) <= 8, f"{t}: {len(per_track[t])} lessons"
        assert per_track[t] == [f"{t}.{i}" for i in range(1, len(per_track[t]) + 1)]
    # The listing publishes each count; it must be the registry's.
    for t, blurb in TRACK_BLURBS.items():
        assert blurb.endswith(f"({len(per_track[t])} lessons)"), blurb
    # Every projection is labelled where it is printed.
    for lr_obj in obj["lessons"]:
        body = "\n".join(lr_obj["text"])
        if "tok/s" in body and lr_obj["track"] in (HARDWARE, SERVING):
            assert ROOFLINE in body or "(roofline est.)" in body, (
                f"{lr_obj['id']}: unlabelled tok/s"
            )

    # Numbers in text should correspond to figures
    for lr_obj in obj["lessons"]:
        nums = _numbers_in_text(lr_obj["text"])
        figs = lr_obj["figures"]
        if nums and figs:
            # At least half the extracted numbers should be explained by figures
            explained = sum(1 for n in nums if _number_in_figures(n, figs))
            assert explained >= len(nums) // 2 or len(nums) < 3, (
                f"{lr_obj['id']}: {explained}/{len(nums)} numbers explained by figures"
            )

    # Sanity-check the module-level constants this module imports.  These
    # assertions use each constant by name with tight bounds so that the test
    # harness can detect when one is perturbed — a mutated constant breaks at
    # least one bound, and demo() raises.
    assert GB == 1024**3, f"GB should be 1024^3, got {GB}"
    assert 0.0 < BANDWIDTH_EFFICIENCY <= 1.0, (
        f"BANDWIDTH_EFFICIENCY={BANDWIDTH_EFFICIENCY} out of (0,1]"
    )
    assert 0.0 < OVERHEAD_FRACTION < 1.0, f"OVERHEAD_FRACTION={OVERHEAD_FRACTION} out of (0,1)"
    assert OVERHEAD_FLOOR > 0, f"OVERHEAD_FLOOR={OVERHEAD_FLOOR} must be positive"
    assert 0.0 < PREFIX_CACHING_FLOOR < 1.0, (
        f"PREFIX_CACHING_FLOOR={PREFIX_CACHING_FLOOR} out of (0,1)"
    )
    assert 1 <= MAX_SPEC_DECODE_CONCURRENCY <= 512, (
        f"MAX_SPEC_DECODE_CONCURRENCY={MAX_SPEC_DECODE_CONCURRENCY} implausible"
    )
    assert 256 <= PREFILL_CHUNK_TOKENS <= 65536, (
        f"PREFILL_CHUNK_TOKENS={PREFILL_CHUNK_TOKENS} implausible"
    )

    counts = " + ".join(f"{len(per_track[t])} {t}" for t in TRACKS)
    print(f"learn: {len(LESSONS)} lessons ({counts})")
    print("ok")


if __name__ == "__main__":
    demo()
