# Your first proof, in ten minutes

Can an open model on your own machine take over the work you send to a closed
one?

This page follows one real run from start to finish, on an M4 Max with 128 GB,
on 2026-11-02. The incumbent is gpt-4o-mini. The candidate is Qwen3 30B-A3B,
served locally. Every block below is that run's real output. Each stage has its
own page with the mechanism and the knobs.

## 1. See what fits

`onpar fit --context 32k --concurrency 8` sizes every catalogue model on this
machine. Qwen3 30B-A3B fits at q8 with 39.8 GB to spare, and it is the only
mixture-of-experts model that fits. See [ten seconds to a fit](20-ten-seconds.md) for the
run and how to read it.

## 2. Capture real traffic

`onpar observe` puts the gateway between the client and the provider. The only
change on the client side is the base URL.

<!-- run: observe -->

The traffic came from `tools/story_traffic.py`, an ordinary OpenAI-style client
that sends four kinds of work a small support product sends: ticket
classification, invoice line items as JSON, one-line summaries and arithmetic
checks. It also runs a refund agent that calls `lookup_order` and then
`refund`. Prompts are redacted before they touch disk, and nothing leaves the
machine. See [observe](../loop/10-observe.md).

## 3. Turn it into an eval set

`onpar distill` groups the captures into task shapes and samples each one.

<!-- run: distill -->

Seven shapes, 12 agent sessions, and an estimate that about 56% of prompt
tokens were a re-sent prefix. One capture was an upstream error and is not in
the set. See [distill](../loop/20-distill.md).

## 4. Serve the candidate

`onpar run qwen3-30b-a3b` picks MLX on this machine, writes the native
`mlx_lm.server` command and starts it on port 8000. Thinking is off by default
for Qwen3, so short answers do not run out of tokens mid-thought. See
[serve](../loop/40-serve.md) for the dry run and the settings.

## 5. Prove it, per task

`onpar prove` replays every item against the candidate and grades each reply
against what gpt-4o-mini said.

<!-- run: prove -->

The verdict is `Move 0% of traffic`, and it is the right one. The agent work
looks perfect: 100% decision agreement over 36 steps and 12 of 12 sessions to
the end. Perfect over 12 items is still a lower bound of 76%. Structured JSON
failed outright because this server does not enforce `response_format`. The
other shapes are not proven yet. See [prove](../loop/50-prove.md).

## 6. Keep the receipt, and the gate

`onpar prove --out receipt.json` wrote a receipt that prints its bad news
first and verifies by digest. `onpar guard receipt.json` says it still holds.
`onpar migrate --step` ran one pass and held in shadow, waiting on a shape with
one item. See [the receipt](../reference/20-receipt.md),
[migrate](../loop/60-migrate.md) and [guard](../loop/70-guard.md).

## What the owner does next

The proof says what evidence is missing, so the next steps are specific:

- **Collect more items where the rate is already good.** Three shapes scored
  100% with 12, 12 and 29 items, and a fourth with one. Each needs 35 to clear
  the bar.
- **Add a judge for free text.** Free text #1 scored 92% and needs 525 items to
  clear the bar by deterministic grading alone. A judge, checked by a human with
  `onpar label`, can decide the items exact matching cannot.
- **Serve structured output on an engine that enforces it.** vLLM has a
  structured-output flag that MLX lacks, and it needs a CUDA machine.
  `onpar run qwen3-30b-a3b --structured-output --dry-run` lists
  `structured_output` under `NOT EXPRESSED` when the engine cannot enforce it, so
  you can check before you prove again.

Next: [observe](../loop/10-observe.md), the first stage of the loop on your own
traffic.
