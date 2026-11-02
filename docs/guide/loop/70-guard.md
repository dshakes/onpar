# Guard: three kinds of stale

Does the proof you migrated on still describe what production is doing?

A receipt is a statement about a moment. Three things can happen after it is
issued, and `onpar guard` refuses to treat them as one event:

- **The model moved.** The thing you proved was replaced behind its name. The
  receipt is void. Urgency: now.
- **Your traffic moved.** The eval set was drawn from work nobody does any more.
  The receipt answers questions that stopped being asked. Urgency: soon.
- **The field moved.** Something new was released. Your receipt is still true,
  and there may be something better. Urgency: whenever.

Only the first two mean you no longer know whether production is adequate. A
tool that pages you for a new release as loudly as for a silent provider-side
swap gets muted, and then it does not page you for the swap either.

Traffic drift is the one a generic eval harness cannot see. It needs a stable
notion of which tasks your workload consists of, which is what distill's task
shapes are. If a fifth of your volume moved into a shape your eval set never
sampled, a high equivalence figure is a true statement about a workload you no
longer run.

## The run

The receipt from [prove](50-prove.md):

<!-- run: receipt -->

And the guard over it:

<!-- run: guard -->

It holds because nothing was given to compare. The receipt was issued today,
no current traffic or fingerprints were passed, and no new release was named.
Guard reports only what it checked.

## When it says no

Each finding prints with a mark. `[void]` means the receipt's claim about
production no longer holds. `[note]` means it is still true and worth knowing.
The thresholds are calibration knobs in `src/onpar/guard.py`:

| Finding | Trigger | Mark |
|---|---|---|
| Model changed | a fingerprint differs from the one the receipt recorded | `[void]` |
| Traffic uncovered | more than 10% of current traffic sits in shapes the receipt never scored | `[void]` |
| Traffic moved | total-variation distance between the receipt's mix and today's above 0.2 | `[void]` |
| Aged | the receipt is older than 90 days | `[note]` |
| New candidate | a model passed with `--available` exists now | `[note]` |
| Uncheckable | you passed fingerprints and the receipt recorded none | `[note]` |

The receipt above recorded no fingerprints, because `prove` ran without
`--fingerprints`. Passing current fingerprints to `guard` would therefore print
the uncheckable note, not a pass. Record them at prove time.

Guard never moves traffic. Every output is a proposal, and the strongest verb it
has is "re-prove". That matters more here than anywhere, because guard is built
to run unattended on a timer.

## The knobs that matter

- `--fingerprints FILE` is a JSON object of model name to whatever identifies
  the weights you serve today: a digest, a revision, a build id.
- `--traffic FILE` is a JSON object of shape key to current share, the same
  shape as the `shares` object a fresh `onpar distill` writes.
- `--available MODEL` names a candidate that exists now. It repeats.
- `--fail-on any` exits 1 on notes too, for a release gate that should not
  deploy on an unreviewed proof. The default, `invalidating`, exits 1 only on
  `[void]`.
- `--today DATE` evaluates as of another day.

The full receipt format is in [the receipt](../reference/20-receipt.md).

Next: when guard says re-prove, run [distill](20-distill.md) on fresh captures
and [prove](50-prove.md) again.
