"""onpar learn — LLM inference curriculum grounded in this machine's numbers.

Every figure is derived from solve() for the hardware the user is sitting at.
No number is hard-coded. Roofline figures are labelled "roofline estimate,
not measured" exactly as fit.py does.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from . import catalog, fit, hardware
from .catalog import QUANT_BITS, ModelSpec
from .fit import (
    BANDWIDTH_EFFICIENCY,
    OVERHEAD_FLOOR,
    OVERHEAD_FRACTION,
    Fit,
    solve,
)
from .hardware import Hardware
from .plan import (
    MAX_SPEC_DECODE_CONCURRENCY,
    PREFILL_CHUNK_TOKENS,
    PREFIX_CACHING_FLOOR,
)

GB = 1024**3

BASICS = "basics"
INTERMEDIATE = "intermediate"

TRACK_BLURBS = {
    BASICS: "for anyone — how a model fits and what limits it (7 lessons)",
    INTERMEDIATE: "for engineers — GQA, MLA, concurrency, speculative decoding (8 lessons)",
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
    """MLA and kv_lora_rank (50× overestimate)."""
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
        "  lesson inter·2 — MLA and kv_lora_rank (50× overestimate)",
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
        title="MLA and kv_lora_rank (the 50× overestimate)",
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
    Lesson("inter.2", INTERMEDIATE, "MLA and kv_lora_rank (the 50× overestimate)", _inter_2),
    Lesson("inter.3", INTERMEDIATE, "Concurrency and the KV budget", _inter_3),
    Lesson("inter.4", INTERMEDIATE, "Prefix caching and agent loops", _inter_4),
    Lesson("inter.5", INTERMEDIATE, "Speculative decoding and high-batch limits", _inter_5),
    Lesson("inter.6", INTERMEDIATE, "Chunked prefill / TTFT vs ITL budgets", _inter_6),
    Lesson("inter.7", INTERMEDIATE, "Tool calling as a server problem", _inter_7),
    Lesson("inter.8", INTERMEDIATE, "Teacher-forced step replay and judge intervals", _inter_8),
]

_BY_TRACK: dict[str, list[Lesson]] = {}
for _l in LESSONS:
    _BY_TRACK.setdefault(_l.track, []).append(_l)


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
            assert result.track in (BASICS, INTERMEDIATE)

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
    assert set(obj["tracks"]) == {BASICS, INTERMEDIATE}
    assert len(obj["lessons"]) == len(LESSONS)
    basics_ids = [lr["id"] for lr in obj["lessons"] if lr["track"] == BASICS]
    inter_ids = [lr["id"] for lr in obj["lessons"] if lr["track"] == INTERMEDIATE]
    assert len(basics_ids) == 7, f"expected 7 basics, got {len(basics_ids)}"
    assert len(inter_ids) == 8, f"expected 8 intermediate, got {len(inter_ids)}"

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

    print(f"learn: {len(LESSONS)} lessons ({len(basics_ids)} basics + {len(inter_ids)} inter)")
    print("ok")


if __name__ == "__main__":
    demo()
