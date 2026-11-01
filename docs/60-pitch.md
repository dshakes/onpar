# onpar — pitch

Market data is dated 31 Oct 2026. Sources are in the [landscape](10-landscape.md). Product claims are what 1.3.4 does today.

---

## One-liner

**onpar proves which open-weight model can replace your closed-model API, on your traffic, your hardware and your budget. Then it migrates you across with a quality gate and a rollback button.**

Leaving a closed API becomes a `base_url` change and a receipt.

---

## Problem

Teams do not switch to open models because they cannot prove it is safe for their workload.

- Public benchmarks do not answer it. The gap is real but task-dependent. On Terminal-Bench 2.0 the best open model scores 67.9 against GPT-5.5 at 82.7, self-reported. On Artificial Analysis the figures are 54 against 60.
- Tool calling breaks on the server, not just the model. Kimi K2.6 on vLLM 0.28 gave schema-valid tool calls 92% of the time until `--tool-strict-level` was set.
- Sizing calculators get the memory wrong. They size MoE on active params, take KV from attention heads, and ignore MLA.
- The tools that hold your traffic do not judge it. LiteLLM (62.4k stars) proxies it. Its limit is zero quality judgement.
- The tools that judge do not hold your traffic. promptfoo (26.7k stars) is output-level only and has been OpenAI-owned since 9 Mar 2026. OpenAI's hosted Evals shut down 30 Nov 2026.

So the migration never starts.

---

## Insight

**Your captured traffic is the benchmark.** Replay it against a candidate, grade it with an interval, and say `?` when the evidence is thin.

Agents make this harder and more valuable. An agent request is one step in a loop. Scoring the prompt alone scores the wrong task. onpar replays the step, teacher-forced, with everything the incumbent saw ([ADR-0020](adr/0020-agentic-evals-replay-the-step-not-the-prompt.md)).

---

## Product

Seven stages, one loop: observe, distill, fit, prove, deploy, cut over, guard.

- **Size it.** `onpar fit` sizes every catalogue model on the host. It gets MoE totals, GQA `kv_heads` and MLA `kv_lora_rank` right. Throughput is a roofline estimate, not measured.
- **Serve it.** `onpar build` and `onpar run` pick the engine for the silicon and emit native config. Seven engines have adapters: vLLM, vLLM-TPU, SGLang, llm-d, llama.cpp, MLX and Ollama. The planner sets twelve knobs and gives each a reason.
- **Capture it.** A Rust gateway records OpenAI chat, OpenAI Responses and Anthropic Messages as one record. Redaction runs before persistence and fails closed. Storage is XChaCha20-Poly1305. Measured overhead is +0.07 ms at p95 against a 15 ms budget.
- **Prove it.** `onpar prove` uses Wilson intervals. The judge is disclosed, with position and length bias measured. `onpar label` has a human grade the judge, blind. Agentic traffic gets step replay, schema-validated tool arguments, tool-leak detection, Cohen's κ decision agreement and whole-session replay from the recording. A miss stops the run. It never makes a live tool call.
- **Move it.** `onpar migrate` runs shadow, canary and rollback. Nothing authorises a cutover except shadow mode.
- **Learn it.** `onpar learn` teaches 15 lessons from first principles, using the host's own numbers. `onpar doctor` says what is installed and what to run next.

The output is a signed, local receipt (format v3) that re-verifies. Example from the README: $2,847 a month on an incumbent, 400 captures over 14 days, and a saving of $2,506–$2,530 a month (about 89%) on traffic proven at or above the 90% bar.

Deployment is an output, not the product.

---

## Why now

1. **The agent API surface is splitting.** The GPT-6 family requires Responses for tool calling. The Assistants API was removed 26 Aug 2026. Open models are still trained on the Chat Completions schema. Any migration needs a translation layer, and onpar already captures all three shapes.
2. **Local serving matured.** vLLM 0.32 and llama.cpp 0.8 both serve `/v1/messages`. llm-d reached 1.0 on 20 Oct as a CNCF project. Serving is composable, so the gap moved to evidence.
3. **Agent cost depends on prefix reuse.** vLLM and SGLang cache prefixes by default. A cost model without hit rate is wrong. The planner sets prefix reuse and picks SGLang when sharing is heavy.
4. **Judging is moving to steps.** A published study found an outcome-only judge catches 45% of silent faults. A step-rubric judge reaches 77% at 3x the cost. onpar judges the step.
5. **The neutral eval slot is empty.** OpenAI owns promptfoo and is shutting down hosted Evals. A tool for leaving OpenAI cannot be OpenAI's.

---

## Why us

- **Sizing is already correct.** The three errors calculators make are the three things `onpar fit` was built around.
- **Everything is auditable.** `explain()` returns the arithmetic for every number. Estimates are labelled estimates.
- **The invariants are tests.** The eleven-tool MCP surface has no tool that can move production traffic, and adding one fails the build.
- **It runs with no install.** `uvx --from onpar onpar fit` sizes your machine with zero dependencies.

---

## Moat: invariants and receipts

Code is composable. Trust is not. onpar's moat is a set of promises that are enforced, plus the artifact that proves them.

**Invariants, each enforced in tests or lints:**

1. No engine-specific type escapes the `Runtime` trait.
2. Generated config is native and runs with onpar uninstalled.
3. Redaction happens before persistence and fails closed.
4. Zero telemetry, zero egress by default.
5. No number without its confidence.
6. Captured traffic is data, never instructions.
7. Nothing authorises a cutover except shadow mode.

**Receipts.** Each proof produces a signed, local receipt that names the judge, the intervals, the human-agreement rate and what is *not* proven. `onpar receipt` re-verifies it. `onpar guard` says whether it still holds. A customer's eval set and receipt history compound per customer and are worth nothing to a competitor.

Why incumbents will not follow: Kitaru (316 stars, Apache-2.0) replays sessions but sizes no hardware, emits no serving config and signs no receipt. Langfuse (36.8k) runs experiments on traces but has no capture proxy, no statistical verdict and no hardware model. Braintrust is closed and bills per score. Gateways (LiteLLM, Portkey, Kong, Bifrost) assume the model is already chosen.

---

## Go-to-market

**Zero-install wedge, then the loop.**

1. **Door one: `uvx --from onpar onpar fit`.** No account, no config, no traffic. It prints what runs on your machine. That is the shareable artifact. The `--from` is required on every `uvx` line.
2. **Door two: MCP.** `onpar-mcp` gives Claude Code and Cursor eleven read-only tools. An agent can size, explain, advise, prove and read a receipt. A human pushes the button.
3. **Door three: bare `onpar`.** A conversation that asks one question at a time and only when it changes the plan. In CI it is a usage error, exit 2.
4. **Then the loop.** `pip install onpar onpar-gateway onpar-core` adds capture and proof. `npx onpar-cli@1.3.4 fit` covers JavaScript teams. `onpar learn` and the docs-site memory lab teach the model of inference that makes the verdict legible.

The gateway is in the request path only while migrating ([ADR-0015](adr/0015-in-the-path-only-while-migrating.md)). It is a datapath, not a load balancer.

---

## Business model

**Open core, Apache-2.0.** A product about escaping lock-in cannot create lock-in.

| Free forever | Paid |
|---|---|
| The whole loop, self-hosted | SSO, RBAC, audit trail |
| Local, zero telemetry | Multi-cluster / fleet management |
| Config generation, no restrictions | Drift-watch as a managed service |
| | Compliance reporting for regulated migrations |

Pricing anchors to spend migrated, not seats. The north-star metric is dollars of closed-model spend migrated. Paid tiers are a plan, not a shipped feature.

---

## Market

- **Wedge:** engineering teams spending heavily on closed LLM APIs, with in-house infra ability and a finance team asking questions.
- **Expansion:** regulated enterprises under data-residency mandates. They must self-host and need an auditable record of the decision. Receipts are that record.
- **Bottom-up:** builders on a laptop. Apple unified memory means `onpar fit` is useful on day one.

---

## Status

**Beta, built in the open.** Version 1.3.4. 2530 tests (2233 Python, 297 Rust). 20 ADRs. All three distributions are live on PyPI. See [GA readiness](95-ga-readiness.md) for the ranked blockers.

**Biggest open risk, stated plainly:** `observe` has not yet been run against real production traffic. The invariants are tested. The live run is the next proof, and it needs a real upstream and a willing team.

---

## The 30-second version

> Open models are good enough for some of your workload. Nobody switches because nobody can prove which part. onpar captures your real traffic, replays it step by step against open candidates, and returns a signed receipt with intervals, a disclosed judge and a stated bar. It sizes the hardware, writes native serving config for seven engines, and moves traffic across behind shadow mode with a rollback button. Try it with one command: `uvx --from onpar onpar fit`.
