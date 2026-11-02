# Distill: your traffic becomes an eval set

Which tasks does your traffic actually contain, and how much of each?

`onpar distill` decrypts the capture log, groups the requests into task shapes,
and samples each shape into an eval set. A task shape is structural, not
semantic. Two requests with the same system prompt, the same tools offered, the
same output format and the same context size are the same task, even when their
text shares almost no words. Structure is also deterministic: the same log
produces the same eval set every time, which is what lets a verdict be re-run
and audited.

Each shape is sampled across its response-length range, shortest to longest,
and the two extremes are always kept. Those tails are where a candidate
diverges. A small shape is never dropped to save budget, because a cluster of
12 requests may be the one that blocks the migration. Every item keeps the share
of traffic its shape holds, not its share of the sample, so a small shape
sampled up to its floor does not count for more than it is.

Agent loops are put back together on the way. An agent sends the conversation,
gets a tool call back, runs the tool and sends the conversation again, so each
step is a request that extends the one before. Two captures belong to one
session when the shorter one's messages are a prefix of the longer one's. No
timestamps are involved, because the gateway records none.

## The run

The traffic came from `tools/story_traffic.py`, which sends four task shapes
through the gateway to gpt-4o-mini: ticket classification, invoice line items
as JSON, one-line summaries and arithmetic checks, 40 of each. It also runs 12
refund-agent sessions that call `lookup_order` and then `refund`.

<!-- run: distill -->

Read it from the top:

- **197 captures, 7 task shapes.** The ticket, summary and arithmetic prompts
  all return free text at under 1k context, so their names differ only by
  number. Their system prompts tell them apart. A fourth free-text shape holds
  a single capture, and it stays in the set with its 0.5% share.
- **12 agent sessions, 36 steps.** Each refund session is three requests: two
  tool calls and the final answer. 24 of those steps called a tool.
- **About 56% of prompt tokens were a re-sent prefix.** That is what a prefix
  cache would not recompute. It is an estimate from the prompt sizes the
  provider reported, and a lower bound, because the model's own replies are
  re-sent too and not counted.
- **The plan hint** turns those numbers into `onpar build` flags, so the
  serving plan is sized for the traffic you have.

The names are structural on purpose. `--name-endpoint` asks a model of your
choice for readable names such as "Refund requests". It is the only step in the
loop that sends captured prompts anywhere, so it never runs without that flag.
A name is a label: it cannot change a share, a score or a verdict.

## When it says no

An upstream error is never an eval item. The gateway already refuses to capture
a non-2xx reply. A capture with an empty reply and no usage is still treated as
an error, left out of the set, and counted on its own line, as the run above
shows. An error is not the incumbent's answer, and an empty baseline would
quietly fail every candidate that did answer.

`distill` also says when the budget is too small to cover every shape. A shape
that got no items is listed under `uncovered` in the file's provenance, and the
verdict says nothing about that traffic.

Repeated prompts are kept here and collapsed later. `onpar prove` treats a
prompt asked twice in one shape as one observation and says how many it merged.

## The knobs that matter

- `--out PATH` names the eval set. The default is `evalset.json`.
- `--budget N` caps the total items. The default is 200.
- `--min-per-cluster N` is the floor for each shape. The default is 3.
- `--import FILE` reads a Langfuse generations export or OpenAI request and
  response pairs instead of, or as well as, the capture log. Imported text is
  redacted first.
- `--name-endpoint URL` with `--name-model NAME` gives shapes readable names.

The file format is in [the eval set](../reference/30-eval-set.md).

Next: [check what runs on your hardware](30-fit.md), or go straight to
[prove](50-prove.md) if you already serve a candidate.
