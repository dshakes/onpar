# Competitive landscape

**Dated 31 Oct 2026.** Stars and release dates come from the GitHub API on that day. Pricing is from the vendor page where one exists, otherwise a third-party tracker (marked 3P). onpar itself: v1.3.4, 2 stars.

**The finding:** no product below covers more than one of the six capabilities in the matrix, and three cover none. The nearest one, Kitaru, replays sessions but sizes no hardware and signs no receipt. Nobody ships the whole path from your traffic to a statistical verdict to a rollback.

<img src="assets/landscape-2026.svg" alt="Coverage matrix for the five closest products and onpar across six capabilities: capture proxy, session replay with recorded tool results, statistical verdict, hardware sizing, native serving config, and a cutover gate with rollback. Dated 31 Oct 2026." width="100%">

---

## Ranked: closest to onpar's thesis

| # | Product | Overlap | What it lacks |
|---|---|---|---|
| 1 | [Kitaru](https://github.com/zenml-io/kitaru) (ZenML) | Replay, model swap, regression cohorts | Sizing, self-host serving config, Wilson intervals, receipt, redaction proxy. Needs an SDK wrap or a trace platform |
| 2 | [Langfuse](https://github.com/langfuse/langfuse) | Experiments on production traces, CI gate, OSS | Capture proxy, statistics, hardware, cutover |
| 3 | Braintrust | Trace to eval to playground swap | Closed, hosted, per-score billing |
| 4 | [promptfoo](https://github.com/promptfoo/promptfoo) | CLI, local providers, OSS | Output-level only, no capture, OpenAI-owned |
| 5 | [LiteLLM](https://github.com/BerriAI/litellm) | The proxy already in place. Could add shadow | Zero quality judgement |

Kitaru (316 stars, Apache-2.0, v0.7.0 on 30 Oct) is the one to watch. It replays recorded sessions with tool results answered from the recording, swaps the model, and treats cohorts as regression gates. It can fail on a miss with `on_miss="fail"`. It [imports Langfuse, LangSmith, Braintrust, Logfire, Phoenix and MLflow traces](https://docs.zenml.io/kitaru). Cloud starts at [$39/mo (3P)](https://www.zenml.io/blog/opik-vs-langfuse). We could not verify Kitaru's hosted pricing page.

---

## 1. Gateways

| Product | Stars | Latest release | Pricing | Has, onpar lacks | Explicitly not |
|---|---|---|---|---|---|
| [LiteLLM](https://github.com/BerriAI/litellm) | 62.4k | v1.106.6, 28 Oct | MIT + annual Enterprise licence; 3P [$250/mo to $30k/yr](https://www.truefoundry.com/blog/litellm-pricing-guide) | 140+ providers, budgets, SSO/SCIM, [Lens agent tracing (Oct 2026)](https://www.litellm.ai/about) | No quality judgement, no sizing, no cutover gate |
| [Portkey](https://github.com/portkey-ai/gateway) | 13.3k | v1.15.2, 12 Jan | [Basic enterprise gateway free since 19 Feb 2026](https://atlan.com/know/litellm-vs-portkey-vs-bedrock-gateway/) | 50+ guardrails, pricing DB for 2,300 models | Same as LiteLLM |
| [Helicone](https://github.com/Helicone/helicone) | 6.2k | Aug 2025 | 3P Free / $79 / $799; [acquired by Mintlify Mar 2026, hosted in maintenance](https://aitoolpick.org/blog/helicone-pricing-2026/) | Observability + gateway | Effectively frozen |
| [Kong AI Gateway 2.2](https://www.prnewswire.com/news-releases/kong-ai-gateway-expands-with-new-capabilities-bringing-enterprise-governance-to-the-agentic-ai-era-302894215.html) | 44.3k (Kong/kong) | 2.2 GA 30 Sep | 3P [~$105/mo/service + $34/M req](https://vantaige.io/ai-tool/kong-ai-gateway) | MCP server bundling, A2A, identity-aware policy | Bills per request, so agent loops burn quota |
| [Bifrost](https://github.com/maximhq/bifrost) | 9.3k | ent-v2.4.0, 30 Oct | Apache-2.0 + Enterprise, 14-day trial | Under 100µs overhead, MCP/agent gateway | Same as LiteLLM |

All five assume the model is already chosen. None sizes hardware, emits a statistical verdict, or generates serving config.

## 2. Eval and observability

| Product | Stars | Latest | Pricing | Has, onpar lacks | Not |
|---|---|---|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | 26.7k | 0.125.1, 25 Oct | Free OSS; enterprise quote. [Acquired by OpenAI 9 Mar 2026, stays MIT](https://www.promptfoo.dev/blog/promptfoo-joining-openai/) | Red-team, 60+ providers, CLI | [Output-level only](https://www.braintrust.dev/articles/braintrust-vs-promptfoo). OpenAI-owned neutrality question |
| Braintrust | closed | n/a | Free / [Pro $249/mo, bills per GB and per score](https://www.voiceflow.com/blog/what-is-braintrust); $80M Series B Feb 2026 | Trace to dataset to experiment, playground model swap | No tool-history replay. Hosted. Self-host is enterprise only |
| [Langfuse](https://github.com/langfuse/langfuse) | 36.8k | v4.72.0, 29 Oct | [Hobby free / Core $29 / Pro $199 / Ent $2,499](https://costbench.com/software/ai-observability/langfuse/) | [Experiments on sampled production traces (Apr 2026), CI action](https://langfuse.com/changelog/2026-04-13-experiments-rebuild) | No capture proxy, no stats verdict, no hardware |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | 12.1k | client v3.6.0, 30 Oct | ELv2 free; AX Free / $50 | [ToolSelection, ToolInvocation, ToolResponseHandling evaluators](https://pypi.org/project/arize-phoenix-evals/) | Observability-first |
| [DeepEval](https://github.com/confident-ai/deepeval) | 19.2k | 4.2.4, 22 Sep | Confident AI [Free / $200 / $2,000](https://costbench.com/software/ai-evaluation/deepeval/) | 6 agent metrics incl. tool and argument correctness | Python only, judge-token cost |
| [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | 3.1k | 0.3.287, 31 Oct | MIT | Sandboxed react agents; [inspect-swe runs claude_code in a sandbox](https://benchmarkingagents.com/inspect-uk-aisi/) | A benchmark harness, not your traffic |
| OpenAI Evals | 19.7k repo, last push Apr 2026 | none | [Hosted evals read-only on 31 Oct, shut down 30 Nov 2026. OpenAI points to promptfoo](https://developers.openai.com/api/docs/deprecations) | | Dead |

## 3. Routing

Routers choose per prompt among hosted models. None proves that one open model matches your traffic, and none handles self-hosting. The category is consolidating, not a threat.

- [Not Diamond](https://github.com/Not-Diamond/notdiamond-python): repo archived Dec 2025. [Pivoted to a coding-agent router, waitlist](https://bito.ai/blog/best-ai-model-routers-for-coding-agents-in-2026/).
- Martian: active, [$20 per 5k requests (3P)](https://paritylayer.com/blog/best-llm-routers-2026).
- [RouteLLM](https://github.com/lm-sys/RouteLLM): 5.7k stars, last push Aug 2024.
- [OpenRouter Auto](https://openrouter.ai/openrouter/auto): ranks by community spend over 7 days, no router fee. [5.5% credit fee, not on the Free tier](https://openrouter.ai/pricing).
- [Unify](https://github.com/unifyai/unify): archived. [Site is now a continual-learning lab](https://www.clawrouters.com/blog/best-ai-model-router-free-2026).

## 4. Local inference: what 2026 shipped

onpar emits native config for seven engines. Here is where each stands.

- [vLLM](https://github.com/vllm-project/vllm/releases/tag/v0.32.0): 96k stars, v0.32.0 on 18 Oct. Generic `--tool-call-parser hf` reads the checkpoint's response template. NVFP4 KV cache. A `prompt_tokens_cached_by_source` metric. An [Anthropic `/v1/messages` endpoint](https://github.com/vllm-project/vllm/pull/22627). Fast Start weight caching since v0.30. `--tool-strict-level` since 0.31.
- [SGLang](https://github.com/sgl-project/sglang): 38.3k, v0.5.22 on 19 Oct. [Rust prefix-cache core by default, prefill/decode role switch, Decisions API](https://www.sglang.io/). RadixArk startup in Jan 2026.
- [llm-d](https://newsroom.ibm.com/2026-10-20-llm-d-1-0-advances-open,-cost-efficient-ai-inference-for-production-scale): 5.1k, v1.0.0 on 20 Oct, CNCF. Multi-tenant QoS, session-aware routing, v1 config API.
- [Ollama](https://github.com/ollama/ollama): 185k, v0.40.13 on 28 Oct. Cloud [Pro $20 / Max $100 / Team $500, pay-as-you-go since 26 Oct, peak-hour pricing](https://ollamatps.com/pricing/).
- [llama.cpp](https://github.com/ggml-org/llama.cpp): 133k, v0.8.0 on 28 Oct. [Serves Anthropic `/v1/messages`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
- **Desktop and packaging:** LM Studio is closed and free, with [`lms`](https://github.com/lmstudio-ai/lms) at 5.4k stars and [Bionic $20 / Pro $100, enterprise quote](https://toolradar.com/tools/lm-studio/pricing). [Ramalama](https://github.com/containers/ramalama) 3.1k, v0.25.0. [Docker Model Runner](https://github.com/docker/model-runner) 658 stars, v1.2.9. [Lemonade](https://github.com/lemonade-sdk/lemonade) 6.0k, v2026.44.1, OpenAI + Anthropic + Ollama APIs, AMD NPU.

## 5. Fit calculators

[canirun.ai](https://www.canirun.ai/), [apxml](https://apxml.com/tools/vram-calculator), [vram.asmirnov](https://vram.asmirnov.xyz/), [HF accelerate estimate-memory](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/model_size_estimator.md), [kvanta](https://github.com/vaclcer/kvanta).

What they get wrong:

- **MoE sized on active params.** [Qwen3-Coder-480B needs about 270 GB at Q4, not 35B worth](https://localaimaster.com/tools/vram-calculator).
- **KV from attention heads, not `kv_heads`.** [4 to 8x over](https://flaviocopes.com/tools/llm-vram/).
- **No MLA.**
- **Pre-quantized MXFP4 gpt-oss.**
- **No runtime overhead.** [accelerate reports 64 GB for Llama-3.1-70B against about 140 GB real](https://github.com/huggingface/accelerate/issues/3379).

None picks a runtime or emits config. `onpar fit` already handles MoE totals, GQA `kv_heads` and MLA `kv_lora_rank`. The gap there is distribution, not correctness.

---

## The agentic era

Agent traffic changes what a migration proof must check. Each point maps to something onpar does today.

- **API surface.** [Chat Completions is not deprecated, but the GPT-6 family requires Responses for tool calling. The Assistants API was removed 26 Aug 2026](https://developers.openai.com/api/docs/deprecations). [Codex dropped chat/completions](https://github.com/openai/codex/discussions/7782). [Open models are still trained on the Chat Completions schema](https://docs.nvidia.com/nemo/gym/infrastructure/engineering-notes/responses-api-evolution/), so a Responses-to-Chat translation layer is load-bearing in any migration. The onpar gateway captures OpenAI chat, OpenAI Responses and Anthropic Messages as one record.
- **MCP.** [Donated to the Linux Foundation AAIF in Dec 2025](https://en.wikipedia.org/wiki/Model_Context_Protocol). [286 members in Oct 2026](https://techfastforward.com/articles/mcp-97-million-installs-linux-foundation-agentic-ai-standard-2026). More than 10k public servers is still the latest official figure.
- **Tool calling is a server problem as much as a model problem.** [Kimi K2.6 on vLLM 0.28 gave schema-valid tool calls only 92% of the time, fixed by `--tool-strict-level parameter`](https://dev.to/alanwest/vllm-tool-calling-not-working-1-in-12-calls-fail-by-default-heres-the-fix-hnp). A parser and chat-template mismatch yields `finish_reason=stop` with tool text in content ([Qwen3.5 needs qwen3_xml](https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451); [Qwen2.5-Coder fails with both parsers](https://github.com/vllm-project/vllm/issues/29192)). [gpt-oss-120b parallel tool calling is broken](https://www.morphllm.com/best-open-source-llm). onpar scores that case as `ToolLeak`, a serving-config fault.
- **Benchmarks.** [tau2-bench v2.0.0, 19 Oct 2026](https://github.com/sierra-research/tau2-bench/releases/tag/v2.0.0): multilingual, voice, intake domain, post-hoc judges. [BFCL v4 Agentic (Jul 2025)](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html): 40% weight on agentic, 26 prompt-format variations. [Terminal-Bench 2.0](https://www.morphllm.com/best-open-source-llm): Kimi K2.6 66.7, DeepSeek-V4-Pro 67.9, GPT-5.5 82.7, self-reported, mixed scaffolds. [Artificial Analysis, 30 Apr 2026](https://ecorpit.com/open-model-production-gap-tooling-decision-2026/): best open 54 against GPT-5.5 at 60. The gap is real but task-dependent. That is onpar's thesis.
- **Judging.** [An outcome-only judge catches 45% of silent faults and flags 33% of correct trajectories. A step-rubric judge reaches 77% silent recall at 3x cost](https://arxiv.org/abs/2609.00038). [Counsel: a per-span judge sees only what the agent saw](https://arxiv.org/pdf/2606.21627). [AgentJudgeBench](https://arxiv.org/abs/2608.26623) calibrates judges on tool calls.
- **Prefix caching.** vLLM APC is default in V1 and SGLang Radix is default. [Reported hit rates of 75 to 95% on shared-scaffold agents](https://www.spheron.network/blog/context-engineering-production-ai-agents-kv-cache-long-context/) are vendor numbers. Cost per million tokens for agent loops depends on hit rate, so a cost model without it is wrong.
- **Replay.** [LangSmith offers checkpoint replay via LangGraph](https://www.digitalapplied.com/blog/agent-observability-platforms-langsmith-langfuse-arize-2026). Braintrust, Langfuse and Arize go from trace to dataset to experiment, not tool-history replay. A [Hugging Face post from Sep 2026](https://huggingface.co/blog/TechforHumans/evaluating-llms-under-production-parity) describes reference-response injection on matched steps. Kitaru is covered above.

---

## Where onpar stands

Features none of them ship, and onpar does:

1. A serve command that carries the tool-calling config (parser, strict level, chat template per family) next to the memory fit. The planner sets it as one of thirteen knobs.
2. A statistical verdict with a confidence interval and a content-addressed, local-only receipt.
3. Teacher-forced step replay and whole-session replay from the recording. A miss stops the run. It never makes a live tool call. See [ADR-0020](adr/0020-agentic-evals-replay-the-step-not-the-prompt.md).
4. Capture of Responses and Messages traffic, including tool_use and tool_result blocks, in one record.
5. A per-span judge disclosed with a human-agreement rate measured through `onpar label`.

Not built yet, and listed in [the roadmap](50-roadmap.md) as candidates:

- Tool-call schema-validity rate as a headline metric beside quality (the 92% problem).
- Prefix-overlap measurement on the captured corpus, turned into a predicted hit rate and cost per million tokens per runtime. Any hit-rate figure would be labelled an estimate.
- A protocol-translation diff that reports fields dropped between Responses or Messages and Chat Completions.
- Format-sensitivity tests on your own traffic, BFCL-style.
- Kitaru and LangSmith trace import. `onpar distill --import` reads a Langfuse generations export or OpenAI-compatible request/response pairs today.
- tau2 and BFCL subsets as an optional sanity check, never the gate.
