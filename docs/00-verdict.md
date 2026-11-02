# Verdict: build vs. buy vs. don't

**Date:** 2026-07-27 · **Status:** decided, and built (refreshed 2026-11-01 against 1.3.4) · **Author:** research pass 1

You asked: *do I need this, or does it already exist?* Honest answer in three parts.

---

## 1. What you described mostly exists. Don't build it.

| What you asked for | Who already owns it | Verdict |
|---|---|---|
| Deploy open LLMs on k8s | [KServe](https://kserve.github.io) (CNCF incubating), [llm-d](https://llm-d.ai), vLLM production-stack, [KAITO](https://github.com/kaito-project/kaito), BentoML/Yatai | **Don't build.** Four funded projects, one CNCF-blessed. |
| High-perf inference engine | vLLM, SGLang, TensorRT-LLM, LMDeploy | **Don't build.** You will never beat these. |
| Speculative decoding / EAGLE-3 | Already shipped *inside* vLLM (`--speculative-config`), SGLang (`--speculative-algorithm EAGLE3`), and mlx-lm 0.21 | **Don't build. Configure it.** |
| KV-cache-aware routing | [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) + llm-d Endpoint Picker. 3× output tok/s, 2× TTFT reduction, measured. | **Don't build. Adopt it.** |
| Local one-click LLM | Ollama, LM Studio, Jan, GPT4All | **Don't build.** Ollama has 4,500+ models and a daemon. |
| Multi-provider gateway | LiteLLM (self-host), OpenRouter, Portkey | **Don't build the multi-provider transport.** Solved. The gateway we ship is a thin capture datapath, not a router. |
| Eval harness | lm-eval-harness, Inspect AI, promptfoo, DeepEval | **Don't build the benchmark runner.** We own the part they lack: grading on your own captured traffic. |
| Chat UI / RAG | Open WebUI, AnythingLLM, Dify (140K stars) | **Don't build.** |

Building any one of these is a losing race. Building all of them is how a startup dies.

---

## 2. The gap nobody owns

Your clarification is the whole product:

> *"help builders leverage this to pivot from using closed source models and make their life easy"*

The blocker for that pivot **is not deployment**. Ollama solved deployment in 2023. The blocker is **evidence**.

A team running a closed model in production will not switch because a leaderboard says so. They switch when someone shows them, with a confidence interval: *on your actual traffic, this open model matches your closed model at or above your bar, costs a fraction as much, runs on hardware you already have, and here is the rollback button.* (Illustrative. The real numbers come from `onpar prove` and are printed with their interval, or as `?` when the evidence is thin.)

Nobody produces that artifact. Look at the seams:

- **LiteLLM** sees all your traffic and does nothing with it. No eval, no equivalence, no migration.
- **promptfoo** evaluates outputs but has no idea what your production traffic looks like, and can't deploy. OpenAI has owned it since 9 Mar 2026.
- **lm-eval-harness** measures MMLU. Your users don't send MMLU.
- **Ollama** deploys the model but can't tell you if it's good enough for *your* job.
- **OpenRouter** is structurally disincentivized from helping you self-host.
- **llm-d / KServe** assume you already decided which model, and already believe.

Every tool covers one step. **The migration itself is unowned.** That's the wedge.

Re-checked on 31 Oct 2026 ([landscape](10-landscape.md)): the closest newcomer, Kitaru, replays recorded agent sessions and swaps the model. It sizes no hardware, emits no serving config and signs no receipt. The wedge held.

---

## 3. The reframe

**Not:** "a next-gen platform to deploy open LLMs on Kubernetes."
**But:** *"Point it at your OpenAI/Anthropic traffic. It proves which open model can replace it, on your hardware, and moves you there without a risky cutover."*

Deployment becomes an **output**, not the product. You still ship all the k8s/vLLM/SGLang/EAGLE-3 machinery you wanted — but as generated artifacts on the far side of a decision you *earned*, not as another orchestrator competing with CNCF.

This is defensible because the moat is data you generate and nobody else holds: a customer's private eval set distilled from their own traffic, plus the equivalence history behind it. That compounds. A Helm chart does not.

**Verdict: build it — but build the migration, not the platform.** Full spec in [`20-prd.md`](20-prd.md).

---

## 4. Hard constraint you need to know now

**Your machine is an Apple M4 Max (128 GB unified, 16 cores, no NVIDIA).**

Half the stack you named is CUDA-first and will not run locally:

| Component | On M4 Max |
|---|---|
| vLLM (CUDA) | ❌ |
| SGLang | ❌ |
| llm-d | ❌ (needs multi-GPU CUDA nodes) |
| TensorRT-LLM | ❌ |
| EAGLE-3 as documented in vLLM | ❌ (mlx-lm 0.21 has its own spec-decode since May 2026) |
| **MLX / mlx-lm** | ✅ highest throughput, continuous batching, spec-decode since 0.21, ~525 tok/s small models on M4 Max, ~1,150 tok/s aggregate @ 32 concurrency |
| **llama.cpp Metal** | ✅ best single-stream, broadest model support |
| **Ollama** | ✅ zero-config daemon, configured by environment rather than flags |

onpar now carries adapters for exactly seven engines: vLLM, vLLM-TPU, SGLang, llm-d, llama.cpp, MLX and Ollama. `onpar build` picks the one that fits the silicon and emits its native config. `onpar fit` and `onpar doctor` run on the M4 Max with no CUDA.

This is not a footnote — it's an architectural forcing function. It means the product **must** treat the runtime as a swappable backend from commit one, because your dev machine and your customers' prod clusters run *different engines*. Which is exactly the abstraction that makes the product portable. See [ADR-0002](adr/0002-runtime-abstraction.md).

It also means: **you can dogfood the whole loop locally**, with 128 GB of unified memory, which is more than most GPUs. That's a genuine advantage.

---

## 5. What the verdict became

The reframe shipped as the seven-stage loop: observe, distill, fit, prove, deploy, cut over, guard.

- **Size it.** `onpar fit` sizes every catalogue model on the host and gets MoE totals, GQA `kv_heads` and MLA `kv_lora_rank` right. Throughput is a roofline estimate, not measured.
- **Tune it.** `onpar build` sets thirteen knobs per engine and gives each one a reason.
- **Prove it.** `onpar prove` grades candidates on your captured traffic, with Wilson intervals, a disclosed judge and measured judge bias. Agentic traffic is replayed per step and per session ([ADR-0020](adr/0020-agentic-evals-replay-the-step-not-the-prompt.md)). `onpar label` has a human grade the judge, blind.
- **Move it.** `onpar migrate` runs shadow, canary and rollback. Nothing but shadow mode authorises a cutover. A judge alone never promotes.
- **Learn it.** `onpar learn` teaches 36 lessons in five tracks on the host's own numbers. `onpar doctor` says what is installed and what to run next.

The datapath is in the request path only while migrating ([ADR-0015](adr/0015-in-the-path-only-while-migrating.md)).

---

## Sources

[vLLM vs SGLang 2026](https://www.yottalabs.ai/post/vllm-vs-sglang-which-inference-engine-should-you-use-in-2026) ·
[AI/ML on Kubernetes 2026 stack](https://kubernetesguru.com/ai-ml-on-kubernetes-2026-stack-guide/) ·
[KServe production serving](https://medium.com/@nonickedgr/kserve-production-ml-serving-on-kubernetes-from-sklearn-to-llms-6a0bbc923fd5) ·
[llm-d KV-cache aware routing](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference) ·
[Gateway API Inference Extension](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/) ·
[EAGLE-3 / P-EAGLE in vLLM](https://vllm.ai/blog/2026-03-13-p-eagle) ·
[llama.cpp vs MLX vs Ollama vs vLLM on Apple Silicon](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026) ·
[LLM gateway comparison](https://wavect.io/blog/llm-gateway-router-comparison-2026/) ·
[Eval framework comparison](https://machinelearningmastery.com/llm-evaluation-frameworks-compared-how-to-actually-measure-what-your-model-does/) ·
[Open-source LLM leaderboard July 2026](https://www.vellum.ai/open-llm-leaderboard) ·
[Open WebUI alternatives](https://onyx.app/insights/openwebui-alternatives) ·
[Self-hosted inference platforms](https://www.digitalocean.com/resources/articles/ai-inference-platforms)
