# The eval set

The eval set is the JSON file `onpar distill` writes from your captures and
`onpar prove` reads. Each item is one real request, the answer your incumbent
gave, and a slot for the candidate's answer. One reader parses it on every
surface: `EvalItem.from_row` in `src/onpar/prove/graders.py`.

## The run

<!-- run: distill -->

The candidate column is blank on purpose. `onpar prove --candidate-endpoint`
fills it by replaying each item against the model you are testing.

## The file

```json
{
  "items": [
    {
      "item_id": "req-1",
      "cluster": "refunds",
      "prompt": "Can I return an order after 30 days?",
      "baseline": "Returns are accepted within 30 days of delivery.",
      "candidate": "",
      "baseline_tool_calls": [],
      "response_format": null,
      "messages": [
        {"role": "system", "content": "You answer customer support questions."},
        {"role": "user", "content": "Can I return an order after 30 days?"}
      ],
      "tools": [],
      "session": "",
      "step": 0
    }
  ],
  "shares": {"refunds": 1.0},
  "names": {"refunds": "Refund requests"},
  "provenance": {"captures": 1, "clusters": 1}
}
```

That is a hand-written example of the shape, not a capture.

| Top-level key | What it holds |
|---|---|
| `items` | The eval items. Required, and refused when empty. |
| `shares` | Each cluster's fraction of captured traffic, keyed by cluster. Share of traffic, not of the sample. |
| `names` | A readable name for each cluster. |
| `provenance` | How the set was drawn: captures, clusters, budget, per-cluster floor, clusters that contributed nothing, agent sessions, and a note. `prove` does not read it. |

`prove` also accepts a bare JSON list of items. Every cluster then gets equal
weight, which is rarely what your traffic looks like.

## One item

The reader gives every field a default, so a hand-written item can stay short.
An item still needs a `baseline`, or there is nothing to match.

| Field | Default | What it holds |
|---|---|---|
| `item_id` | its position | The request id from capture, so a result traces back to a request. |
| `cluster` | `all` | The task shape this item belongs to. Keys into `shares` and `names`. |
| `prompt` | `""` | The user's text from the request, redacted. |
| `baseline` | `""` | The incumbent's reply. It is the bar to match, not ground truth. |
| `candidate` | `""` | The candidate's reply. Filled by `--candidate-endpoint`. |
| `baseline_tool_calls` | `[]` | Tool calls the incumbent made, as `{"name", "arguments"}` objects or bare names. |
| `candidate_tool_calls` | `[]` | The same, for the candidate. |
| `response_format` | `null` | The structured-output request: a name such as `json_schema`, or the OpenAI object. |
| `messages` | `[]` | The whole redacted conversation the incumbent answered. This is what is replayed. |
| `tools` | `[]` | Tool schemas the request offered, replayed with the messages. |
| `session` | `""` | The agent session this step belongs to. Empty for a lone request. |
| `step` | `0` | This step's position in that session. |

An item without `messages` replays `prompt` alone. That loses the system prompt
and every earlier turn, so it is a different task for any system-prompted or
tool-using workload.

## When it says no

- A file with no items is refused.
- A file whose every `candidate` is blank, run without `--candidate-endpoint`, is
  refused before grading. Grading it would report 0% and read as a verdict.
- `distill` leaves out any capture that was an upstream error, with an empty reply
  and no usage, and says how many it left out. The recording above shows one.

## What `--import` accepts

`onpar distill --import FILE` turns traces from elsewhere into capture rows, so
you can build an eval set without running the gateway. The flag repeats, and it
combines with `--capture`.

- **OpenAI request and response pairs.** Each record is
  `{"request": {...}, "response": {...}}`, where the request is a chat
  completions body. Use this shape to bring traces from any platform without a
  native importer.
- **Langfuse generations export.** Records with `"type": "GENERATION"`. An input
  in Anthropic Messages shape is normalised to the chat shape.

The file may be JSONL or a JSON array. The format is detected from each record's
shape, never from the filename. A record in neither shape is skipped, and the
skip is counted with its reason.

Imported text is redacted before it becomes a row, which needs `onpar-core`.
`--unredacted` skips redaction and says so. The eval set then holds unredacted
text and must not leave the machine.

Next: [the MCP tools](40-mcp-tools.md), every tool an agent may call, with its
inputs.
