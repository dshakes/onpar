# Say what you are building

Can you describe the workload in a sentence instead of filling in a form?

Type `onpar` with no verb in a terminal and it asks one thing: what you are
building. It reads your answer, works out what it can, and shows a full plan
straight away. Then it asks at most one question per turn, and only a question
whose answer would change the deployment.

That last rule is computed, not curated. There are three questions it can ask:
how many requests are in flight, how long the prompts are, and whether requests
share a long system prompt. Before asking one, the session re-plans under an
alternative answer. If both lead to the same
deployment, it stays silent and moves on. Your attention is
the thing a form wastes, and this spends it only where it buys a different
result.

Questions refine the answer. They never gate it. At every turn there is a usable
plan built from what you said plus assumptions it names, and you can stop and
take it.

## The run

`onpar build` is the same conversation in one command, for a script or a
pasted description:

<!-- run: build -->

Every inference shows its evidence. `concurrency = 4` arrives with the words it
came from and the reasoning that turned 20 agents into 4 in flight. A misreading
is visible now, not in production. When a value is guessed rather than read, it
appears under `assuming:` instead, and the one question worth asking follows it,
marked with `?`.

The last two lines are the boundary. The session produces a command and a proof
plan. It does not run either, and nothing you say to it changes that.

## When it says no

A bare `onpar` only converses when standard input is a terminal. In a script, a
Dockerfile or a CI step it prints the usage error and exits 2, exactly as a
missing verb always has. A command that waits forever for input in CI is worse
than one that fails at once.

Inside the conversation, `q` stops with the answer so far. Stopping before you
have described anything prints that there is nothing to answer, rather than a
plan built on nothing. Closing input with Ctrl-D prints the answer so far, and
Ctrl-C does the same and exits 130.

## The knobs that matter

- A flag states a value outright, and the session never asks about it again:
  `--concurrency`, `--context`, `--ttft-ms`, `--itl-ms`, `--prefix-sharing`,
  `--tool-calling`, `--structured-output`, `--thinking`.
- `--on PROFILE` plans for a hardware profile from `onpar where` rather than
  this machine.
- `--save PATH` writes the session state, and `--resume PATH` continues it in a
  later process. The whole state is JSON.
- `--json` returns the plan and the session state for a program. Over MCP,
  `onpar_build` does the same: pass back the `state` from one response to
  continue in the next.

Next: read [the receipt](../reference/20-receipt.md), the file the proof plan
ends in.
