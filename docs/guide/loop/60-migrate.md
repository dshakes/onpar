# Migrate: shadow first, always

How do you move real users to the open model without betting them on one eval?

You climb a ladder, and the evidence decides each rung. A migration starts in
shadow: the gateway mirrors each request to the candidate, scores its reply, and
serves the incumbent's. Real users see the candidate only after shadow evidence
supports it, and then a little at a time. The ladder in `src/onpar/prove/gate.py`
is shadow, then canary at 5%, 25% and 50%, then the cut to 100%. It never jumps
from shadow to a cut.

`onpar migrate` keeps that evidence fresh. Each pass distils the captures,
proves the candidate, and asks the gate what the evidence permits. It writes
what it found to a state file, so the next pass continues rather than restarts.
The gate gives one of three answers: hold, advance or roll back. An advance is a
proposal for a human. The loop prints it and stops. Nothing in it moves traffic,
touches the gateway's control surface, or promotes itself.

## The run

One pass, against the candidate served by `onpar run`:

<!-- run: migrate-step -->

And where the migration stands, across every pass so far:

<!-- run: migrate -->

## What advances it, and what never does

The two directions follow different rules, because they cost different amounts
when they are wrong.

- **Rollback is the easy direction.** The gateway applies it from its own health
  signals without asking `migrate`. Its control surface accepts any change that
  lowers or keeps the candidate's share, so it works at 3am with nobody watching.
- **Advancing needs a human.** The control surface refuses any increase in the
  candidate's share unless the request says a human confirmed it. When
  `ONPAR_ADMIN_TOKEN` is set, that token is required too.
- **A judge alone never advances.** A shape whose whole case rests on LLM judge
  scores cannot support an advance.
- **Underpowered never advances.** A shape with fewer than 8 graded items is not
  evidence yet, and the gate says how many more it needs.
- **Health beats quality.** A candidate that errors or crawls trips a rollback
  before its grades are read.
- **Startup cannot move traffic.** The gateway refuses a share above zero on its
  command line. A candidate starts in shadow, and escalation goes through the
  control surface, which records a reason.

A shape proven below the bar is not a failure in shadow. Nothing is at risk
there, so the shape is pinned to the incumbent and the migration continues
around it. Under canary or a cut the same shape is live, and the same evidence
means stop.

## When it says no

`HOLD` is the answer above. Free text #4 has one graded item, and the gate wants
7 more before its interval means anything. Every pass so far has held for the
same reason, and `--status` shows each one. A hold is not a failure of the run:
`migrate --step` exits 0 whatever the gate decided, so a nightly job does not
page anyone for a correct "not yet".

## The knobs that matter

- `--step` runs one pass. It is the default.
- `--status` shows where the migration is and runs nothing.
- `--install` prints a crontab fragment. It does not install it: a recurring job
  over your traffic is your decision. `--every-hours` sets the interval, 24 by
  default.
- `--candidate-endpoint`, `--candidate-model` and `--resume` work as they do for
  `onpar prove`.
- `--state PATH` moves the state file from `~/.onpar/migration.json`.
- For the shadow itself, start the gateway with
  `onpar observe --upstream URL --candidate URL`.

Next: [guard](70-guard.md) the receipt once traffic moves.
