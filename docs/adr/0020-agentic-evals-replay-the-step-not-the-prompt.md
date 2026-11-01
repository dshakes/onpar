# ADR-0020 — Agentic evals: replay the step, not the prompt

**Status:** accepted · 2026-10-31

## Context

The eval set `onpar distill` wrote carried one field for the question:
`prompt`, the concatenated user turns. The collector sent that to the candidate
as a single user message:

```python
"messages": [{"role": "user", "content": prompt}]
```

The incumbent had answered something else. It had seen a system prompt, the
prior turns of the conversation, the tool results the harness fed back, the
tool schemas it was allowed to call, and the output format demanded. The
distiller *clustered* on exactly those things — the system-prompt hash, the
tool names, the response format — and then threw them away before the
candidate was asked. Every system-prompted workload was replayed as a
different task; every tool-using cluster graded as a regression by
construction, because a candidate offered no tools cannot call one; and the
pairwise judge, shown text only, compared two empty strings on a tool-call step
and called them a tie. `collect.py`'s own docstring said so: *"Tool-using items
cannot be reproduced faithfully … Collect those from your own harness."*

In 2026 that is not a corner. Production traffic to a closed-model API is
increasingly an agent loop: one request per step, each a strict extension of
the one before, most of them tool calls. A proof that cannot score those steps
cannot say whether an open model is on par for the traffic that actually costs
money. And the gateway recorded tool-call **names only**, on the argument that
"arguments are where the user's data lives" — which is true, and which in a
multi-turn loop protected nothing, because step *k*'s request already contains
step *k-1*'s assistant tool calls, arguments and all, redacted by the same walk
that redacts every other message.

## Decision

**The unit of agentic evaluation is one step, replayed with everything the
incumbent saw, and graded on the move it made.** Concretely:

1. **Faithful replay (teacher-forced).** An eval item carries `messages` (the
   whole redacted conversation prefix), `tools` (the schemas offered) and
   `response_format`. The collector sends all three. The candidate is asked to
   take one step given the exact prefix the incumbent had, and is graded on that
   step — the same per-turn protocol agent benchmarks score, and the only
   honest replay a tool that does not run your tools can perform. An item with
   no `messages` (an older eval set) falls back to the bare prompt and the
   collector counts and prints how many did.
2. **Arguments are captured, redacted.** The gateway records each tool call's
   name *and* argument string, run through `redact::redact` before persistence,
   failing closed like every other field. Old logs with bare names still read.
3. **Argument validity is graded deterministically.** `ToolArgsSchema` validates
   the candidate's arguments against the `parameters` schema the request itself
   offered — type, required, enum, additionalProperties, items — and fails a
   call to a tool that was never offered. Malformed arguments are the commonest
   way an open model fails an agentic workload, and no judge is needed to see it.
4. **The judge sees the step.** Tool calls are rendered into both answers
   (`→ refund({"order_id": 42})`), and the request shown to the judge is the
   system prompt (truncated), the tools on offer by name, and the last turns.
   Position swapping is unchanged.
5. **Decision agreement is reported chance-corrected.** Over every step in a
   tool-using context, the candidate's action (tool name, or `respond`) is
   compared with the incumbent's: Cohen's κ with its Landis–Koch band, a Wilson
   interval on raw agreement, the two asymmetric failures counted separately
   (wrote text where a tool was called; called a tool where text was expected),
   and a per-tool table with intervals so a tool seen three times reads as
   unknown rather than 0% or 100%.
6. **Sessions are recovered structurally.** Captures whose messages are strict
   prefixes of one another are one agent session; the step is the depth in that
   tree. No timestamps, no wording heuristics — the same log links the same way
   every time, so a receipt can be re-derived. The share of prompt tokens that
   were a re-sent prefix is reported as the measured input to the planner's
   prefix-reuse decision (`onpar build --prefix-sharing`), labelled as an
   estimate and a lower bound.
7. **The human-agreement rate becomes a measurement.** `prove --judge-sample`
   logs every comparison the judge is shown; `onpar label` walks a human
   through them blind, in the same anonymous A/B order; `prove --judge-labels`
   computes `Agreement` from the ledger. Typing `--judge-agreed 17` still works
   and is still a claim; the ledger is the evidence. Judge **length bias** is
   measured and disclosed on the same footing as position bias, because the swap
   cannot catch a preference for the longer answer.

## Consequences

- The eval set file now contains the system prompt in clear, where before it
  held only its hash. It was always in the capture log (encrypted at rest); the
  eval set is plaintext JSON. The provenance note says what the file contains.
  The alternative — replaying every system-prompted task without its system
  prompt — made the proof wrong for all of them.
- The eval-set digest is unchanged in definition (id, cluster, prompt), so old
  receipts still verify against old sets. A regenerated set has different
  content and a different digest, as it should.
- The resume cache key now includes the messages and tools, so a reply bought
  against one system prompt is never reused against another.
- Tool-call steps now reach the judge with content. Expect judged agentic
  clusters to change verdict — in both directions.
- The agentic block is in the rendered report and the MCP `onpar_prove`
  output. It is **not** yet a field on the receipt, whose claim set remains
  per-cluster pass/fail. Adding it is a separate, small change to the receipt's
  declared-type sweep.
- Whole sessions can be replayed **from the recording** (`prove
  --replay-sessions`, `prove/replay.py`): the candidate's own turns are kept
  and its tool calls are answered with the tool results the incumbent received
  at that step; a call the recording cannot answer stops the session. No tool
  is ever executed. A candidate that reaches the same outcome by a different
  path reads as diverged, and the report says so — that is the known limit of
  replay from a recording, and the wording is chosen not to overclaim.

## What would falsify this

- A corpus where per-step agreement is high and end-to-end task success, run in
  the customer's own harness, is low. That would mean step fidelity is not
  predictive and the unit is wrong.
- A captured tool schema that uses `$ref`/`allOf` heavily enough that the
  minimal validator accepts invalid arguments. The validator accepts unknown
  keywords on purpose (refusing them would fail every call); the fix is to
  extend it, not to drop the grader.
- A redaction gap in argument strings that the message walk would have caught.
  Both go through the same `redact`; a divergence would be a bug, not a design
  question.

## Alternatives rejected

- **Keep user-text replay, document the limit.** Rejected: the limit covers the
  traffic that matters most, and a proof labelled "unfaithful for agents" is a
  proof nobody can act on.
- **Replay whole trajectories with a tool sandbox.** Rejected for now: it needs
  the customer's tools, is a harness (which has incumbents — invariant 1), and
  per-step replay is the standard it would be compared against anyway.
- **Semantic clustering of steps with embeddings.** Rejected, as the distiller
  already was: structural linking is deterministic and reproducible; embeddings
  are neither and add a dependency to a zero-dependency tool.
- **Recover arguments from the next step's messages instead of capturing them.**
  Clever, dependency-free, and wrong for the last step of every session and for
  every single-step tool call. Capturing them, redacted, is simpler and complete.
