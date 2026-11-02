<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/lockup-stacked-on-dark.svg">
  <img src="docs/assets/brand/lockup-stacked.svg" alt="onpar" width="168">
</picture>

# Is the open model<br>on par?

<img src="docs/assets/loop-animated.svg" alt="The switch loop on 197 recorded captures: observe, distill into 7 task shapes, fit 5 of 14 models, prove with κ=1.00 over 36 agent steps and a 75% [70–77] weighted verdict, deploy with 13 knobs set, cut over moves 0% while shadow holds, guard keeps the receipt. Every figure is read from the recorded run." width="100%">

**onpar proves which open-weight model can replace your closed-model API, on your own traffic and hardware, then moves you across with a quality gate and a rollback.**

[![status](https://img.shields.io/badge/status-beta-22d3ee?style=flat-square)](docs/50-roadmap.md)
[![tests](https://img.shields.io/badge/tests-2772-34d399?style=flat-square)](#verification)
[![license](https://img.shields.io/badge/license-Apache--2.0-a78bfa?style=flat-square)](LICENSE)
[![docs](https://img.shields.io/badge/docs-read-fbbf24?style=flat-square)](https://dshakes.github.io/onpar/docs/)

**[Website](https://dshakes.github.io/onpar/) &nbsp;·&nbsp; [Docs](https://dshakes.github.io/onpar/docs/) &nbsp;·&nbsp; [Try it in ten seconds](#try-it-in-ten-seconds)**

</div>

| if you are… | run | you get |
|---|---|---|
| **new to all of this** | `onpar` | a conversation. Say what you are building. It asks one question at a time, only when the answer changes the plan, and ends with a model, a quantisation and the memory it needs. |
| **curious how it works** | `onpar learn basics` | thirty-six lessons in five tracks: basics, intermediate, hardware, serving, hosting. Every figure is computed by onpar's own code for the machine you are sitting at. |
| **an engineer with traffic** | `onpar doctor` | a check of the gateway, the extension and the engine, then the next command to run. |
| **an agent** | `onpar-mcp` | eleven read-only tools. None of them can move production traffic. |

The agent door, in full. No tool in this registry can move production traffic, and adding one fails the build:

```jsonc
// onpar-mcp — JSON-RPC over stdio, zero dependencies
onpar_fit       // what runs on this machine, at this context and concurrency
onpar_explain   // the full arithmetic behind one verdict — weights, KV, headroom
onpar_where     // the inverse: which hardware would run this, and at what cost
onpar_catalog   // parameters, MoE split, context, licence
onpar_advise    // what to change unprompted, and where production diverged
onpar_build     // the whole flow, multi-turn: pass state back to continue
onpar_distill   // captured traffic -> an eval set, so there is something to prove against
onpar_prove     // run the eval suite: verdict, traffic split, and a receipt
onpar_receipt   // read a proof: what is proven, what must stay, what is unknown
onpar_guard     // does that proof still hold — and if not, which of three ways
onpar_learn     // inference curriculum grounded in this machine's numbers
```

## One real run

Benchmarks are someone else's exam. This is what onpar said about one team's own traffic: gpt-4o-mini as the incumbent, Qwen3 30B-A3B served locally through MLX at q8 as the candidate. Every figure below is quoted from the recordings the docs embed.

| | What the run printed |
|---|---|
| Captured and distilled | 197 captures → 7 task shapes → 197 eval items |
| Agent traffic | 12 agent sessions · 36 steps (18% of captures) |
| Left out | 1 capture was an upstream error, and an error is not the incumbent's answer |
| Agent moves | decision agreement 100% [90–100] over 36 tool-context steps · κ=1.00 |
| Whole sessions | 12/12 followed the incumbent's path to the end |
| Structured JSON | 0% [0–9]: 40 of 40 replies were not the JSON asked for, because the server does not enforce `response_format` |
| Weighted verdict | 75% [70–77] |
| Decision | **Move 0% of traffic** |

Recorded 2026-11-02 on an M4 Max. The agent work looks perfect and still moves nothing, because 12 flawless items cannot clear a 90% bar. That is the point: the proof does not flatter. Walk the whole run in [your first proof, in ten minutes](docs/guide/start/30-walkthrough.md).

## Seven stages. Each one can say no.

1. **[Observe](docs/guide/loop/10-observe.md).** A Rust gateway in your request path records real traffic, redacted before it touches disk.
2. **[Distill](docs/guide/loop/20-distill.md).** Captures become task shapes and an eval set, with agent sessions linked step by step.
3. **[Fit](docs/guide/loop/30-fit.md).** Every catalogue model is sized on this machine, with the arithmetic one flag away.
4. **[Serve](docs/guide/loop/40-serve.md).** The engine of seven for your silicon, configured, as a native command.
5. **[Prove](docs/guide/loop/50-prove.md).** Each task is scored against the incumbent, with an interval or a `?`.
6. **[Migrate](docs/guide/loop/60-migrate.md).** Shadow first, then canary, then the cut, and only a human advances it.
7. **[Guard](docs/guide/loop/70-guard.md).** The receipt is voided when the model changes behind its name or traffic moves.

Stage one, as it prints. Capture starts with an explicit command and stops at Ctrl-C:

```bash
onpar observe --upstream https://api.openai.com/v1

  In your request path from now until Ctrl-C, and not after (ADR-0015).
  capture   ~/.onpar/captures.log (key: ~/.onpar/capture.key)
  upstream  https://api.openai.com/v1
  listening http://127.0.0.1:8787
  point your base_url at http://127.0.0.1:8787/v1 and nothing else changes.
```

The gateway captures OpenAI chat, OpenAI Responses and Anthropic Messages as one record. It adds **+0.07 ms** at p95 against a 15 ms budget, measured. Redaction runs before persistence and fails closed, records are encrypted with XChaCha20-Poly1305, and nothing leaves the machine.

## What it sets for you

- **Sizing done right.** MoE on total params, GQA on `kv_heads`, MLA on `kv_lora_rank`. Fit calculators get all three wrong.
- **The engine of seven for your silicon.** There are seven engines with adapters today: vLLM, vLLM-TPU, SGLang, llm-d, llama.cpp, MLX and Ollama. The output is a real `vllm serve` or `InferencePool` that runs with onpar uninstalled.
- **Thirteen knobs, each with its why.** Context length, concurrency, chunked prefill, prefix reuse, speculative decoding, memory fraction, tensor parallel, structured output, quantisation, KV-cache dtype, LoRA fleet, the tool-calling parser and thinking. Three forks decide most of them: prefix sharing picks the engine, concurrency turns drafting off, and the latency budget sets the prefill chunk.
- **What it cannot set, it lists.** Anything an engine has no verified flag for appears under `NOT EXPRESSED`, with the reason, never a guessed flag.

<img src="docs/assets/autotune.svg" alt="Autotune: your workload goes in and thirteen knobs come out, each with its reason. A prefix share of 0.55 or more picks the engine, more than 16 concurrent turns drafting off, and an unreachable latency budget becomes a plan warning, never a guessed flag." width="100%">

### Try it in ten seconds

```bash
uvx --from onpar onpar fit --context 32k --concurrency 8
```

No install, no config, no account, no telemetry. It reads the machine and tells you what fits:

```

  M4 Max · 16 cores · 128 GB · 546 GB/s
  usable for inference: 96 GB

  FEASIBLE at 32,768 context, concurrency 8

  model                     quant   weights      kv   total    free  ~tok/s  license
  ----------------------------------------------------------------------------------------
  Qwen3 32B                 q4        17.2G   64.0G   84.1G   11.9G      15 slow  Apache-2.0 OK
  Qwen3 30B-A3B (MoE)       q8        28.4G   24.0G   56.2G   39.8G      60  Apache-2.0 OK
  Mistral Small 24B         q8        22.0G   40.0G   65.2G   30.8G      14 slow  Apache-2.0 OK
  Phi-4 14B                 q8        13.7G   50.0G   66.3G   29.7G      18  MIT OK
  Llama 3.1 8B              q8         7.5G   32.0G   41.6G   54.4G      32  Llama 3.1 !

  NOT FEASIBLE

  Gemma 3 27B               weights fit (14 GB) but KV at 32,768 ctx x8 (124 GB) puts it 45 GB over the 96 GB available
  Llama 3.3 70B             weights fit (37 GB) but KV at 32,768 ctx x8 (80 GB) puts it 25 GB over the 96 GB available
  Qwen3 235B-A22B (MoE)     weights alone need 123 GB at q4 — MoE sparsity (22B of 235B active) cuts compute, not memory
  DeepSeek V3 (MoE, MLA)    weights alone need 352 GB at q4 — MoE sparsity (37B of 671B active) cuts compute, not memory
  GLM-5.2                   weights alone need 186 GB at q4 — MoE sparsity (32B of 355B active) cuts compute, not memory
  Kimi K2.7 Code            weights alone need 524 GB at q4 — MoE sparsity (32B of 1000B active) cuts compute, not memory
  Kimi K3 (2.8T MoE)        weights alone need 1,467 GB at q4 — MoE sparsity (50B of 2800B active) cuts compute, not memory
  DeepSeek V4 Pro           weights alone need 838 GB at q4 — MoE sparsity (45B of 1600B active) cuts compute, not memory
  MiniMax M3                weights alone need 239 GB at q4 — MoE sparsity (46B of 456B active) cuts compute, not memory

  runtime -> mlx
             Apple silicon: the CUDA engines cannot run here at all. MLX has the better
             batching story of the two Metal options.

  onpar fit --explain <model-id>   # show the arithmetic
```

`onpar fit --explain <model-id>` prints the derivation for any row, and `~tok/s` is a roofline estimate, not measured. When a row looks right, `onpar run qwen3-30b-a3b` serves it.

## What it refuses

Every stage can say no, and says why. Each line below is quoted from a recorded run.

| Stage | It refused | Because |
|---|---|---|
| Distill | an upstream error as an eval item | `1 capture(s) were upstream errors (empty reply, no usage) and are not in the eval set` |
| Serve | a latency budget the hardware cannot meet | `inter-token budget is 2ms, but decode is a roofline estimate of 60 tok/s` |
| Serve | a speed taken on a busy machine | `the samples disagree by 25% (35.3–44.3 tok/s), over the 20% limit` |
| Prove | a pass on thin evidence | `needs 35 graded items to clear the 90% bar at its observed 100% (has 12, 23 to go)` |
| Prove | a model verdict on a server fault | `40 of 40 structured-output replies were not the JSON response_format asked for` |
| Migrate | an advance on one item | `HOLD: free text · <=1k context #4 has 1 graded items; 7 more needed` |

It refuses a point estimate of money too. Give it what the incumbent costs and it prices the move as a range:

```bash
onpar prove evalset.json --incumbent-cost 2847 --candidate-cost 317 \
    --traffic-window '14 days'

  Saving: $2,506–$2,530/mo (~89%) on traffic proven at or above the 90% bar
    100% [99–100] of traffic moved, measured on 400 captured requests
    over 14 days; Wilson score, 95%
```

<img src="docs/assets/money-range.svg" alt="Money: the same migration costed at 4,000, 400 and 40 captured requests. The $2,847 and $317 rates never change, but the dollar range widens as the sample shrinks." width="100%">

That is an illustration with stated inputs, not the recorded run above. The recorded run passed no costs, so its receipt reads `Saving: unknown` rather than inventing one.

**Not a load balancer.** Not an inference engine, a chat UI, a RAG framework or hosted inference. The gateway is in your request path only while you are migrating, and not after ([ADR-0015](docs/adr/0015-in-the-path-only-while-migrating.md)). Unredacted prompt text never touches disk. Captured traffic is data, never instructions. Nothing authorises a cutover except shadow mode, and a judge alone never promotes.

## Install

```bash
uvx --from onpar onpar fit           # no install, no deps
pip install onpar                       # or install it
brew install dshakes/tap/onpar          # macOS, from the shared tap
npx onpar-cli@1.4.0 fit                        # same build, via npm
onpar version                              # what you have, and where it came from
```

**`fit` is one package. The evidence half is three:**

```bash
pip install onpar onpar-gateway onpar-core   # observe, distill, prove
```

`onpar doctor` checks all three in the order the loop needs them, and ends with the one command to run next. The commands above are unpinned and fetch the newest release — currently **1.4.0**. `npx onpar-cli@1.4.0` runs `onpar` 1.4.0 and nothing else. Pin it when you need a build to stay put:

```bash
uvx --from onpar==1.4.0 onpar fit   # exactly this build
```

**Versioning.** `onpar version` reads the installed metadata, not a hand-typed string. Through 0.1.4 receipts were stamped `0.1.0`. Fixed in 0.1.5, so a receipt now names the build that produced it.

## Verification

```bash
cargo test --all                                   # 297 Rust
cargo clippy --all-targets -- -D warnings
uv run --with pytest --with pyyaml --python 3.13 pytest -q   # 2475 Python
```

**2772 tests.** 2475 Python, 297 Rust. Every module's `demo()` is run by a test, the Rust core cannot panic on a sizing bug, and gateway tests run over real TCP. The fit transcript, the saving line, the install pins and the test counts on this page are checked by tests; the market figures are dated and sourced. Depth: [field guide](docs/05-field-guide.md#verification-in-depth).

## Docs

The guide follows the loop: [start here](docs/guide/start/10-install.md), [the seven stages](docs/guide/loop/10-observe.md), [for agents](docs/guide/agents/10-mcp.md), [learn](docs/guide/learn/10-curriculum.md) and [reference](docs/guide/reference/20-receipt.md). It is published at [dshakes.github.io/onpar/docs](https://dshakes.github.io/onpar/docs/), next to an interactive memory lab. The specs that built the product:

| | |
|---|---|
| [00 — Verdict](docs/00-verdict.md) | Build vs. buy, honestly. |
| [05 — Field guide](docs/05-field-guide.md) | The depth behind this page: sizing, evals, receipts, the architecture. |
| [10 — Landscape](docs/10-landscape.md) | Who else is in the loop, dated. |
| [20 — PRD](docs/20-prd.md) | Goals, personas, FR/NFR, epics, risks. |
| [30 — Architecture](docs/30-architecture.md) | Datapath/control-plane split, grader stack, fit math. |
| [40 — UX](docs/40-ux.md) | CLI, TUI, console, MCP. |
| [50 — Roadmap](docs/50-roadmap.md) | What is shipped, what is next, what is out of scope. |
| [60 — Pitch](docs/60-pitch.md) | The argument in one pass, for someone deciding. |
| [70 — Naming](docs/70-naming.md) | Why it is called this. |
| [80 — Plan](docs/80-implementation-plan.md) | M0–M10, acceptance criteria, risk gates. |
| [90 — CI gating](docs/90-ci-gating.md) | Gate a deploy on a proof that still holds. |
| [95 — GA readiness](docs/95-ga-readiness.md) | What blocks GA, ranked, assessed by running the published packages. |
| [ADRs](docs/adr/) | 20 decisions, including the two later reversed. |

Where onpar sits against Kitaru, Langfuse, Braintrust, promptfoo and LiteLLM, dated 31 Oct 2026, is in the [landscape review](docs/10-landscape.md). Conventions and invariants are in [`CLAUDE.md`](CLAUDE.md). Licence: Apache-2.0, see [LICENSE](LICENSE).

---

<sub>*On par* means on equal footing. "Is the open model on par with the one we pay for?" is the question this tool answers, and its reply uses the same words: on par, or off par. More in [70 — Naming](docs/70-naming.md).</sub>
