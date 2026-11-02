# CLI reference

Generated from the parser by `tools/cli_reference.py`; `onpar --help` is the
authority and a test fails when this page lags it. Verbs are in the order the
loop runs. Exit codes: `0` done; `1` the call was well-formed and either the world
did not cooperate (retry or report) or the answer is no (a model that does not fit, a
receipt that no longer holds); `2` the invocation itself was wrong (fix it).

## Start here

### `onpar doctor`

what is installed, what each stage needs, what to run next

| flag | what it does |
|---|---|
| `--json` | machine-readable |

### `onpar learn`

LLM inference curriculum grounded in this machine's numbers

| flag | what it does |
|---|---|
| `TRACK` | track to walk (basics, intermediate, hardware, serving, hosting) |
| `LESSON` | lesson number (1-indexed); walks the whole track if omitted |
| `--model MODEL` | catalogue model id to use as the example |
| `--no-quiz` | show answers without prompting |
| `--json` | machine-readable output |

### `onpar version`

print the installed version

| flag | what it does |
|---|---|
| `--json` | machine-readable |

### `onpar upgrade`

how to upgrade this install

## Observe

### `onpar observe`

record real traffic through a capture gateway

| flag | what it does |
|---|---|
| `--upstream UPSTREAM` | the provider you use today |
| `--port PORT` | local port to listen on; point your base_url here |
| `--candidate CANDIDATE` | a second backend to shadow against |
| `--capture CAPTURE` | where the log goes (default: ~/.onpar/captures.log) |
| `--no-capture` | proxy only; write nothing to disk |

## Distill

### `onpar distill`

turn a capture log into an eval set

| flag | what it does |
|---|---|
| `--name-endpoint NAME_ENDPOINT` | OpenAI-compatible endpoint used to give clusters readable names (e.g. 'Refund requests' instead of 'free text · <=1k context'). Opt-in per run: naming is the only step that sends captured prompts anywhere, so it never happens without this flag. |
| `--name-model NAME_MODEL` | model id for --name-endpoint |
| `--capture CAPTURE` | the log to read (default: ~/.onpar/captures.log) |
| `--key KEY` | the capture key (default: ~/.onpar/capture.key) |
| `--import FILE` | trace export to import as capture rows (repeatable; JSONL or JSON). Supported formats: Langfuse generations export, OpenAI-compatible request/response pairs ({"request":{...},"response":{...}}). May be combined with --capture. Text is redacted before storage (requires onpar-core; see --unredacted). |
| `--unredacted` | skip redaction of imported traces. The eval set will contain unredacted text and must not leave this machine. |
| `--out OUT` | where to write the eval set |
| `--budget BUDGET` | total eval items to sample |
| `--min-per-cluster MIN_PER_CLUSTER` | fewest items any cluster gets |
| `--json` | machine-readable output |

## Fit and plan

### `onpar fit`

what runs on this machine

| flag | what it does |
|---|---|
| `--context CONTEXT` | context length, e.g. 8k, 32k, 128000 |
| `--concurrency CONCURRENCY` | simultaneous requests |
| `--explain MODEL_ID` | show the arithmetic for one model |
| `--json` | machine-readable output |
| `--quiet` | hide the NOT FEASIBLE section |

### `onpar where`

which hardware can run a given model

| flag | what it does |
|---|---|
| `MODEL` | catalogue model id, e.g. glm-5.2 |
| `--context CONTEXT` | context length, e.g. 8k, 32k |
| `--concurrency CONCURRENCY` | simultaneous requests |
| `--json` | machine-readable output |
| `--quiet` | hide the WILL NOT RUN section |

### `onpar models`

list the catalog

| flag | what it does |
|---|---|
| `--json` | machine-readable |

### `onpar advise`

what to change, and where production diverged

| flag | what it does |
|---|---|
| `--context CONTEXT` | planned context length, e.g. 32k |
| `--concurrency CONCURRENCY` | planned requests in flight |
| `--workload WORKLOAD` | what the latency budget is for |
| `--ttft-ms TTFT_MS` | time-to-first-token budget, in ms |
| `--itl-ms ITL_MS` | inter-token latency budget, in ms |
| `--prefix-sharing PREFIX_SHARING` | share of prompt tokens that repeat a prefix, 0 to 1, as planned |
| `--structured-output` | the plan under review serves requests that carry response_format |
| `--tool-calling` | the plan under review serves requests that offer tools |
| `--seen-concurrency SEEN_CONCURRENCY` | peak concurrency production showed |
| `--seen-prefix-sharing SEEN_PREFIX_SHARING` | prefix sharing production showed, 0 to 1 |
| `--seen-ttft-ms SEEN_TTFT_MS` | time to first token production showed, in ms |
| `--seen-peak-context SEEN_PEAK_CONTEXT` | longest context production showed, e.g. 48k |
| `--seen-kv-utilisation SEEN_KV_UTILISATION` | peak KV cache fill production showed, 0 to 1; above 0.9 is flagged |
| `--json` | machine-readable output |

### `onpar build`

say what you're building; get a deployment and a proof plan

| flag | what it does |
|---|---|
| `DESCRIPTION` | e.g. "coding assistant for 20 engineers" |
| `--on ON` | hardware profile id; omit to detect this machine |
| `--concurrency CONCURRENCY` | requests in flight at once; overrides the description |
| `--context CONTEXT` | context length to size for, e.g. 32k; overrides the description |
| `--ttft-ms TTFT_MS` | time-to-first-token budget, in ms |
| `--itl-ms ITL_MS` | inter-token latency budget, in ms; a plan that cannot meet it refuses |
| `--prefix-sharing PREFIX_SHARING` | share of prompt tokens that repeat a prefix, 0 to 1 (`onpar distill` measures it) |
| `--tool-calling` | the workload calls tools; emit the engine's tool-call parser for this model family |
| `--thinking` | keep a reasoning model's thinking phase on (default: off for models whose chat template thinks first, e.g. Qwen3) |
| `--structured-output` | requests carry response_format; the server must enforce it (constrained decoding) |
| `--save SAVE` | write the session state here |
| `--resume RESUME` | continue a saved session |
| `--json` | machine-readable output |

### `onpar catalog`

look up one entry, or verify entries against published configs

| flag | what it does |
|---|---|
| `MODEL_ID` | show this catalogue entry (the README trains you to expect this; it used to be a usage error) |
| `--json` | machine-readable output |
| `--model MODEL` | check one entry instead of all |
| `--network` | allow network access (required) |
| `--apply` | write the proposed changes |

### `onpar catalog-add`

add a model from its published config, no reinstall

| flag | what it does |
|---|---|
| `REPO` | Hugging Face repo, e.g. meta-llama/Llama-3.1-8B |
| `--params-b PARAMS_B` | total parameters in billions — not in config.json, so you supply it |
| `--active-b ACTIVE_B` | active parameters for MoE; defaults to total (dense) |
| `--id ID` | catalogue id; defaults to the repo name, lowercased |
| `--name NAME` | display name; defaults to the repo name |
| `--license LICENSE` | licence name to record, e.g. Apache-2.0 |
| `--license-ok` | mark the licence as cleanly commercial (read it first) |
| `--quants QUANTS` | comma-separated, default q4,q8 |
| `--out OUT` | write here instead of the drop-in directory |
| `--network` | allow network access (required) |

### `onpar catalog-sources`

where the catalogue's models come from

| flag | what it does |
|---|---|
| `--json` | machine-readable |

### `onpar discover`

trending models not yet in the catalogue

| flag | what it does |
|---|---|
| `--network` | allow network access (required) |
| `--limit LIMIT` | how many index entries to consider |
| `--json` | machine-readable |

### `onpar watch`

discover new models on a schedule; stages, never publishes

| flag | what it does |
|---|---|
| `--list` | show what is staged and waiting |
| `--install` | print the scheduler fragment |
| `--every-hours EVERY_HOURS` | interval for --install (default 24) |
| `--limit LIMIT` | how many index entries to consider |
| `--max-add MAX_ADD` | cap per run |
| `--network` | allow network access (required) |
| `--json` | machine-readable output |

## Serve

### `onpar run`

serve a model here: resolve weights, launch, wait for healthy

| flag | what it does |
|---|---|
| `MODEL` | catalogue model id, e.g. qwen3-30b-a3b |
| `--quant QUANT` | force a quantisation; default is the best that fits |
| `--context CONTEXT` | context length, e.g. 8k, 32k |
| `--concurrency CONCURRENCY` | simultaneous requests to tune for (a server is not a single-stream benchmark) |
| `--host HOST` | bind address (loopback by default) |
| `--port PORT` | port the engine listens on (default 8000) |
| `--engine ENGINE` | force this engine rather than letting the planner pick one (e.g. ollama, which has no hardware signature to be auto-selected by) |
| `--tag TAG` | the Ollama tag to pull, e.g. llama3.1:8b-instruct-q8_0 — required with --engine ollama, since its registry names are not the catalogue's ids |
| `--thinking` | keep a reasoning model's thinking phase on (default: off for models whose chat template thinks first, e.g. Qwen3) |
| `--structured-output` | requests carry response_format; the server must enforce it (constrained decoding) |
| `--dry-run` | print the plan and the exact command; start nothing |

### `onpar host`

where to run a model this machine cannot; free tiers first

| flag | what it does |
|---|---|
| `MODEL` | catalogue model id, e.g. deepseek-v3 |
| `--context CONTEXT` | context length, e.g. 8k, 32k |
| `--concurrency CONCURRENCY` | simultaneous requests |
| `--quant QUANT` | force a quantisation; default is the best that fits each shape |
| `--free-only` | only shapes that cost nothing |
| `--provider PROVIDER` | pick this provider instead of the cheapest that fits |
| `--write DIR` | emit the deploy artifact here |
| `--json` | machine-readable output |

### `onpar box`

build the OCI box: one artifact, every platform

| flag | what it does |
|---|---|
| `MODEL` | catalogue model id, e.g. qwen3-30b-a3b |
| `--write DIR` | emit the box here |
| `--context CONTEXT` | context length to tune for, e.g. 8k, 32k |
| `--concurrency CONCURRENCY` | simultaneous requests to tune for (a box is published for a fleet) |
| `--quant QUANT` | force a quantisation; default is the best that fits each host |
| `--receipt RECEIPT` | attach this proof; without it the box claims nothing |
| `--ref REF` | registry reference the printed push commands use |
| `--json` | the plan as JSON; nothing is written |

### `onpar cache`

what the engines downloaded, and what to evict

| flag | what it does |
|---|---|
| `ACTION` | omit to report; evict needs --yes before anything is deleted |
| `REPO` | repo to pin or unpin, e.g. org/Model-4bit |
| `--budget BUDGET` | cache budget, e.g. 200G; remembered once set |
| `--yes` | actually delete (evict prints otherwise) |
| `--json` | machine-readable |

### `onpar kernel`

scaffold a vLLM plugin and its proof plan

| flag | what it does |
|---|---|
| `NAME` | plugin name, e.g. fused-rmsnorm |
| `--kind KIND` | vLLM entry-point group |
| `--speedup SPEEDUP` | the speedup you claim |
| `--bit-identical` | claims byte-for-byte output |
| `--drift DRIFT` | what you expect to change, if not bit-identical |
| `--out OUT` | output directory (default: the plugin name) |

### `onpar measure`

measure real decode throughput, or refuse to

| flag | what it does |
|---|---|
| `--endpoint ENDPOINT` | an OpenAI-compatible base URL |
| `--model MODEL` | catalogue id, for the roofline comparison |
| `--served-model SERVED_MODEL` | the name the endpoint knows it by, when it differs from the catalogue id |
| `--samples SAMPLES` | timed requests; the median is reported |
| `--max-tokens MAX_TOKENS` | tokens to generate per request |
| `--context CONTEXT` | context for the roofline comparison |
| `--warmup N` | decodes to run and discard before sampling, to reach thermal equilibrium. On hardware that throttles the first samples are the fastest, so a cold run measures the burst rate rather than what the machine sustains |
| `--quant QUANT` | quantisation the endpoint actually serves (q4, q8, ...); without it the roofline assumes the best fit, which may be a different model |
| `--concurrency CONCURRENCY` | concurrency for the roofline estimate; requests are still sent one at a time |
| `--out OUT` | write the measurement as JSON |
| `--json` | machine-readable output |

## Prove

### `onpar prove`

run the eval suite and print the verdict

| flag | what it does |
|---|---|
| `EVALSET` | path to an eval-set JSON file |
| `--candidate CANDIDATE` | the open model under test |
| `--incumbent INCUMBENT` | the model being replaced |
| `--issued ISSUED` | ISO date stamped on the receipt |
| `--bar BAR` | equivalence bar; a cluster moves only when the whole interval clears it |
| `--out OUT` | write the receipt to this path |
| `--json` | print the receipt as JSON |
| `--candidate-endpoint CANDIDATE_ENDPOINT` | OpenAI-compatible base URL to collect the candidate's replies from, e.g. http://127.0.0.1:8000/v1 (from `onpar run`) |
| `--candidate-model CANDIDATE_MODEL` | model id to ask that endpoint for; defaults to --candidate |
| `--incumbent-endpoint INCUMBENT_ENDPOINT` | same, for the model being replaced |
| `--incumbent-model INCUMBENT_MODEL` | model id to ask that endpoint for; defaults to --incumbent |
| `--judge-endpoint JUDGE_ENDPOINT` | OpenAI-compatible base URL for the tier-3 judge. Only consulted where the deterministic graders could not decide, and never enough on its own to move traffic |
| `--judge-model JUDGE_MODEL` | the judge's id. Required with --judge-endpoint and stamped on the receipt — an undisclosed judge makes every score it touched unauditable |
| `--judge-agreed JUDGE_AGREED` | how many of the sampled comparisons a human agreed with |
| `--judge-samples JUDGE_SAMPLES` | how many comparisons a human checked. Zero means the judge is UNMEASURED, which the report says out loud and the gate refuses to advance on |
| `--judge-sample LEDGER` | append every comparison the judge is shown, and its answer, to this JSONL ledger — then `onpar label LEDGER` to grade them blind |
| `--judge-labels LEDGER` | a ledger `onpar label` has worked through; the human-agreement rate is measured from it instead of typed with --judge-agreed/--judge-samples |
| `--replay-sessions` | also replay each multi-step agent session end to end against --candidate-endpoint, answering tool calls from the recording; a miss stops the session (never a live tool call) |
| `--collect-workers COLLECT_WORKERS` | parallel requests while collecting |
| `--collect-timeout COLLECT_TIMEOUT` | seconds to wait for one reply |
| `--resume PREFIX` | record each reply to PREFIX.<side>.jsonl as it lands, and reuse what is already there. Collection is the only thing here that spends real money, and a run killed at 380 of 400 otherwise re-buys all 380. Nothing is written without this flag. |
| `--incumbent-cost INCUMBENT_COST` | what you spend on the incumbent per month, in dollars. Without it the report says the saving is unknown rather than guessing one — a fabricated saving is the single most damaging number here. |
| `--candidate-cost CANDIDATE_COST` | what the candidate would cost per month. `onpar fit` and `onpar host` both estimate this for self-hosting; pass the figure you believe. |
| `--traffic-window PERIOD` | how long the traffic was captured over, e.g. '14 days'. Captures carry no timestamps, so this is not derivable — and without it a monthly saving would be an extrapolation from an unknown period, which the receipt refuses to make. |
| `--fingerprints PROVE_FINGERPRINTS` | JSON file mapping model name to whatever identifies the weights you served — a digest, a revision, a build id. Recorded in the receipt so `onpar guard` can later detect a silent provider-side swap. Without it that check has nothing to compare against and cannot run. |

### `onpar label`

grade judged comparisons blind, to measure the judge

| flag | what it does |
|---|---|
| `LEDGER` | the JSONL written by `onpar prove --judge-sample` |
| `--limit LIMIT` | stop after this many labels |

### `onpar receipt`

render or verify a migration receipt

| flag | what it does |
|---|---|
| `--json` | machine-readable |
| `RECEIPT` | path to a receipt JSON file |
| `--against AGAINST` | a second receipt to verify this one against |

### `onpar brief`

render a receipt as a page you can send to someone who did not run it

| flag | what it does |
|---|---|
| `RECEIPT` | path to a receipt JSON file |
| `--out OUT` | write to this file instead of stdout |
| `--title TITLE` | heading for the page |

## Migrate and guard

### `onpar migrate`

walk the loop: distill, prove, and what the gate permits

| flag | what it does |
|---|---|
| `--candidate CANDIDATE` | the model being evaluated |
| `--incumbent INCUMBENT` | name of the model being replaced |
| `--candidate-endpoint CANDIDATE_ENDPOINT` | OpenAI-compatible base URL to collect the candidate's replies from |
| `--candidate-model CANDIDATE_MODEL` | model id to ask that endpoint for; defaults to --candidate |
| `--incumbent-endpoint INCUMBENT_ENDPOINT` | same, for the model being replaced |
| `--incumbent-model INCUMBENT_MODEL` | model id to ask that endpoint for; defaults to --incumbent |
| `--capture CAPTURE` | capture log (default: ~/.onpar/captures.log) |
| `--key KEY` | capture key (default: ~/.onpar/capture.key) |
| `--state STATE` | migration state file (default: ~/.onpar/migration.json) |
| `--budget BUDGET` | total eval items to sample per pass |
| `--min-per-cluster MIN_PER_CLUSTER` | fewest items any cluster gets |
| `--bar BAR` | equivalence bar; a cluster moves only when the whole interval clears it |
| `--collect-workers COLLECT_WORKERS` | parallel requests while collecting |
| `--collect-timeout COLLECT_TIMEOUT` | seconds to wait for one reply |
| `--resume PREFIX` | record each reply to PREFIX.<side>.jsonl as it lands, and reuse what is already there. Collection is the only thing here that spends real money, and a run killed at 380 of 400 otherwise re-buys all 380. Nothing is written without this flag. |
| `--status` | where the migration is; runs nothing |
| `--step` | run one pass (the default) |
| `--install` | print a crontab fragment |
| `--every-hours EVERY_HOURS` | interval for --install (default 24) |
| `--json` | machine-readable |

### `onpar guard`

check whether a receipt still holds

| flag | what it does |
|---|---|
| `RECEIPT` | path to a receipt JSON file |
| `--traffic TRAFFIC` | JSON file of current cluster shares |
| `--fingerprints FINGERPRINTS` | JSON file of current model fingerprints |
| `--available AVAILABLE` | a candidate model that exists now (repeatable) |
| `--today TODAY` | ISO date to evaluate against (default: today) |
| `--fail-on FAIL_ON` | which findings exit nonzero. 'invalidating' (default): only what voids the proof — the model changed, or traffic moved outside the eval set. 'any': also age and new releases, for a release gate that should not deploy on an unreviewed proof. |
| `--json` | machine-readable output |

## Interfaces

### `onpar ui`

launch the local workbench

| flag | what it does |
|---|---|
| `--host HOST` | bind address (loopback by default) |
| `--port PORT` | local port for the page (default 7171) |
| `--no-open` | don't open a browser |

### `onpar desktop`

install a double-clickable launcher

| flag | what it does |
|---|---|
| `--port PORT` | port the launcher opens `onpar ui` on (default 7171) |

