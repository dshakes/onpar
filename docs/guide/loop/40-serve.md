# Serve: the engine of seven, configured

Which engine should serve this model for your workload, and with which flags?

`onpar run` answers by planning before it starts anything. It sizes the model
with the fit solver, picks one of seven engines, decides each setting from your
workload, and translates those settings into that engine's own flags. The
engines are vLLM, vLLM on TPU, SGLang, llm-d, llama.cpp, MLX and Ollama. Then
it starts the engine. It is the only command in onpar that starts anything.

The plan is written in settings, not flags. There are thirteen: context length,
maximum concurrency, prefill chunk, prefix reuse, speculative decoding, memory
fraction, tensor parallelism, structured output, quantisation, KV cache dtype, a
LoRA fleet, tool calling and thinking. Thinking is set only for models whose
chat template reasons before it answers. Each setting carries the sentence that produced it, so a value you disagree with
can be argued with.

![Autotune: requirements on the left, settings in the middle, native config on the right](../../assets/autotune.svg)

Follow a requirement from the left to the setting it decides, and note which
fork in the middle sends it there.

## Three forks decide almost everything

1. **Prefix sharing picks the engine.** When 55% or more of prompt tokens are
   shared across requests, SGLang's RadixAttention reuses that prefix's KV
   rather than recomputing it. On NVIDIA that wins. SGLang ships CUDA-only
   builds, so on an AMD GPU the plan names vLLM and says the better stack is
   unavailable, not unconsidered.
2. **Concurrency decides speculative decoding.** Above 16 concurrent requests
   there is no spare compute for a draft pass, so drafting is off. Batch work
   turns it off too.
3. **A latency budget decides scheduling.** Interactive work gets chunked
   prefill at 2,048 tokens, so one long prompt cannot stall every response
   behind it. Batch work gets large chunks, because no token is waiting.

The thresholds are `PREFIX_SHARING_FOR_RADIX`, `MAX_SPEC_DECODE_CONCURRENCY` and
`PREFILL_CHUNK_TOKENS` in `src/onpar/plan.py`.

## The run

<!-- run: run-dry -->

The `RUN` line is the engine's own command. It runs with onpar uninstalled:
there is no wrapper, no sidecar and nothing to keep onpar around for. Copy it
into a unit file or a container spec and it starts the same engine.

`NOT EXPRESSED` is the honest list. When an engine has no verified flag for a
setting, the plan says so and why, rather than inventing a flag the engine would
reject or silently ignore.

## Describe the workload instead of tuning it

`onpar build` reads a sentence and plans from it. It quotes back each value it
inferred and the words that produced it:

<!-- run: build -->

`--tool-calling` asks for the engine's tool-call parser for this model family.
Without the right parser, tool calls come back as plain text with
`finish_reason: stop`, which looks like an answer and is a failure. The parser
tables cover vLLM and SGLang and cite their sources. A family outside them, or an
engine without a verified flag, lands in `NOT EXPRESSED`, as MLX does above.

## Reasoning models answer with thinking off

A model whose chat template thinks first, such as Qwen3, spends a small token
budget on reasoning and has nothing left for the answer. `onpar run` and
`onpar build` turn thinking off by default for every catalogue model marked
`reasoning`. Each engine gets its own spelling:

| Engine | Flag that turns thinking off |
|---|---|
| vLLM, SGLang, llm-d | `--default-chat-template-kwargs '{"enable_thinking":false}'` |
| MLX | `--chat-template-args '{"enable_thinking":false}'` |
| llama.cpp | `--reasoning-budget 0` |
| Ollama | none server-wide: send `"think": false` on each request |

Pass `--thinking` to keep the reasoning phase on.

## When it says no

A budget the hardware cannot meet is not configured around. The plan records it
in `Plan.warnings`, prints it under `CANNOT MEET`, and still shows the flags so
you can see what was tried. Here a 2 ms inter-token budget meets a roofline
estimate of about 17 ms per token:

<!-- run: build-cannot-meet -->

The warning also leads the summary line at the top. A model that does not fit gets a warning as well: every setting is
moot until that is resolved. On hardware whose memory bandwidth is unknown, a
stated budget is reported as unchecked, never as met.

A speed measurement can say no as well. `onpar measure` times decode on a running
endpoint and refuses to call the result a measurement when the samples disagree
by more than 20% of their median, or the load average is above 0.80 per core.
Both limits are set in `src/onpar/measure.py`. This run was taken while the
machine was busy:

<!-- run: measure -->

The refusal names the processes that were busy and exits 1. The roofline
estimate stands, because a number taken under load would carry more authority
than the estimate and less truth.

## What the engine itself will not do

A plan can only promise what the engine enforces. `tools/engine_probe.py` sends
three real requests to a running endpoint and reports which behaviours hold.
This run is against the MLX server that `onpar run` started, with thinking off:

<!-- run: engine-probe -->

Two findings from the proof run on this machine on 2026-11-02 explain the
lines above:

- **Thinking on, no answer.** Served by `mlx_lm.server` with thinking on,
  Qwen3-30B-A3B spent a small token budget on reasoning and returned empty
  content. The reasoning arrived in a separate `reasoning` field. Served with
  `--chat-template-args '{"enable_thinking":false}'`, it answered `billing` in
  2 tokens, as the `answers` line shows. This is why `onpar run` turns thinking
  off by default.
- **`response_format` is not enforced.** `mlx_lm.server` has no grammar or
  structured-output flag, so the plan lists `structured_output` under
  `NOT EXPRESSED` for MLX. In the proof, all 40 items in the eval set's
  structured-JSON cluster asked for `json_object`. The incumbent's baselines
  are 40 JSON objects. The candidate's replies, recorded in the resume ledger
  `~/.onpar/story.candidate.jsonl`, hold none: 13 are valid JSON arrays and 27
  are not valid JSON. It scored 0% on that cluster in the
  [receipt](../reference/20-receipt.md), which kept it on the incumbent. The
  `json` probe line shows the same failure in one request.

Run the probe before you prove against an endpoint. A cluster that fails
because the server ignores a request field is a server finding, not a model
finding:

```bash
python3 tools/engine_probe.py --endpoint http://127.0.0.1:8000/v1 --model mlx-community/Qwen3-30B-A3B-8bit
```

It exits 0 when all three probes pass and 1 otherwise.

## Measure on a quiet machine

Every speed above is a roofline estimate. `onpar measure` replaces one with an
observation, when the machine lets it:

```bash
onpar measure --endpoint http://127.0.0.1:8000/v1 --served-model mlx-community/Qwen3-30B-A3B-8bit --model qwen3-30b-a3b
```

`--served-model` is the name the endpoint knows the model by. Pass it whenever
that differs from the catalogue id: `onpar run` serves the MLX repo
`mlx-community/Qwen3-30B-A3B-8bit`, not `qwen3-30b-a3b`. `--model` keeps the
catalogue id for the roofline comparison. `--warmup N` discards the first decodes
on hardware that throttles, and `--quant` names what the endpoint actually
serves. A usable run still says it is one machine, one moment and one prompt
shape.

## The knobs that matter

- `--context` and `--concurrency` size and tune the server. The defaults are 32k
  and 4.
- `--engine` forces an engine. `--engine ollama` also needs `--tag`, because
  Ollama's registry names are not catalogue ids.
- `--quant` forces a quantisation. The default is the best that fits.
- `--host` and `--port` set the bind address: loopback and 8000 by default.
- `--dry-run` prints the plan and the exact command, and starts nothing.
- On `build`: `--ttft-ms` and `--itl-ms` state latency budgets,
  `--prefix-sharing` states the shared fraction, and `--on` plans for a
  hardware profile you do not have.

Next: [prove](50-prove.md) the candidate on your own traffic with
`onpar prove evalset.json --candidate-endpoint http://127.0.0.1:8000/v1`.
