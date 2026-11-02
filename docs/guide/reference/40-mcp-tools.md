# MCP tools

Every tool `onpar-mcp` exposes, with its inputs, from the `TOOLS` registry in
`src/onpar/mcp.py`. What the tools are for, and the wall around them, is on
[for agents](../agents/10-mcp.md). Inputs are optional unless marked required.

## onpar_fit

Which open models run on this machine.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `context` | string | `32k` | Context length, such as `8k`, `32k` or `128000`. |
| `concurrency` | integer, at least 1 | 1 | Simultaneous requests to plan for. |

## onpar_explain

The arithmetic behind one model's verdict.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `model_id` | string, required | | Catalogue model id. |
| `context` | string | `32k` | Context length. |
| `concurrency` | integer, at least 1 | 1 | Simultaneous requests. |

## onpar_where

Which hardware would run one model.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `model` | string, required | | Catalogue model id. |
| `context` | string | `32k` | Context length. |
| `concurrency` | integer, at least 1 | 1 | Simultaneous requests. |

## onpar_catalog

The model catalogue. It takes no inputs.

## onpar_build

The whole flow as a conversation. Pass `state` back to continue it.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `description` | string | | What you are building, in plain language. |
| `machine` | string | this machine | A hardware profile id. |
| `state` | object | | The `state` object from the previous response. |
| `concurrency` | integer, at least 1 | | Requests in flight. |
| `context` | integer | | Context length in tokens. |
| `ttft_ms` | integer | | Time-to-first-token budget. |
| `itl_ms` | integer | | Inter-token latency budget. |
| `workload` | `interactive`, `realtime` or `batch` | | The kind of work. |
| `prefix_sharing` | number, 0 to 1 | | Fraction of prompt tokens shared across requests. |

## onpar_advise

What a careful reviewer would raise about a deployment.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `context` | string | `32k` | Context length. |
| `concurrency` | integer, at least 1 | 1 | Requests in flight. |
| `workload` | `interactive`, `realtime` or `batch` | `interactive` | The kind of work. |
| `ttft_ms` | integer | | Time-to-first-token budget. |
| `itl_ms` | integer | | Inter-token latency budget. |
| `prefix_sharing` | number, 0 to 1 | 0.0 | Fraction of prompt tokens shared. |
| `structured_output` | boolean | false | Whether requests carry `response_format`. |
| `observed` | object | | Measured reality, for drift: any of `concurrency`, `prefix_sharing`, `ttft_ms`, `itl_ms`, `peak_context`, `kv_utilisation`. |

## onpar_distill

Captured traffic turned into an eval set.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `captures` | string | the standard log | Capture log path, inside the capture root `ONPAR_HOME`. |
| `min_per_cluster` | integer, at least 1 | 3 | The smallest task shape worth scoring separately. |

## onpar_prove

The equivalence verdict and a receipt, over an eval set.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `eval_set` | string, required | | Eval-set path, inside the eval root `ONPAR_EVAL_ROOT`. |
| `candidate` | string | `candidate` | The open model under test. |
| `incumbent` | string | `incumbent` | The model being replaced. |
| `issued` | string | today | ISO date for the receipt. |
| `bar` | number, strictly between 0 and 1 | 0.9 | Equivalence bar. A shape moves only when its whole interval clears it. |

This tool grades the file as it is and collects no replies. Fill the
`candidate` answers first, with `onpar prove --candidate-endpoint` on the
command line.

## onpar_receipt

What a receipt proves, what must stay, and what is unproven.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `path` | string, required | | Receipt path, inside the eval root. |

## onpar_guard

Whether a receipt still holds.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `path` | string, required | | Receipt path, inside the eval root. |
| `today` | string | today | ISO date to evaluate against. |
| `available` | array of strings | | Candidate models that exist now. |

## onpar_learn

Inference lessons computed for this machine.

| Input | Type | Default | Meaning |
|---|---|---|---|
| `track` | `basics`, `intermediate`, `hardware`, `serving`, `hosting` or empty | all | The track to return. |
| `lesson` | integer, at least 1 | | Lesson number within the track. |
| `model` | string | the largest that fits at 32k and 1 | Catalogue model id to use as the example. |

Next: [the CLI reference](cli.md), every command and flag, generated from
`--help`.
