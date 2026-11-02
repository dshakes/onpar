# The receipt

A receipt is the file `onpar prove` writes: what was proven about each task in
your traffic, with its confidence, and what was not. You can hand it to someone
who does not trust you. The format lives in `src/onpar/prove/receipt.py`.

## It is content-addressed, not signed

The file is `{"receipt": {...}, "digest": "..."}`. The digest is a SHA-256 over
the receipt body as canonical JSON, with sorted keys and no whitespace.

- A file whose digest does not match its content is refused as altered since it
  was issued.
- A file with no digest is refused, because nothing about it can be verified.
- A digest says the file is unchanged. It does not say who wrote it.
- The stronger claim is reproduction: re-run the same eval set and the evidence
  must match. `onpar receipt --against` checks exactly that.

The current format is `onpar.receipt/v3`. Readers accept v1, v2 and v3, and
refuse any other format rather than guess at its fields.

## Bad news is printed first

There is no field that can hold a bare `PASSED`. `regret` and `unproven` are
required, and the rendered receipt shows them above the proven clusters: what
must stay on the incumbent, then what is not proven either way, then what moved.

<!-- run: receipt -->

This receipt moves nothing. One cluster is proven below the bar and six lack the
evidence to decide, so `Movable` reads 0%. Each unproven line says how many
graded items it would need. The saving reads unknown because no cost was
passed, rather than showing an invented figure.

![Anatomy of a receipt: the clusters that must stay are printed above the ones that passed](../../assets/receipt-anatomy.svg)

Read the receipt in the diagram top to bottom: the regret cluster comes before
the proven one, and each line carries its interval.

## The fields

| Field | What it holds |
|---|---|
| `format` | `onpar.receipt/v3`. |
| `incumbent` | The model being replaced. |
| `candidate` | The open model under test. |
| `issued` | ISO date, passed in, so the same evidence issues the same receipt. |
| `eval_set` | Digest of the eval set: exactly which questions were asked. |
| `bar` | The equivalence threshold every claim was measured against. |
| `proven` | Clusters whose whole interval clears the bar. |
| `regret` | Clusters proven below the bar. They stay on the incumbent. |
| `unproven` | Clusters with too little evidence to say either way. |
| `traffic_captures` | How many captured requests the eval set was drawn from. |
| `traffic_window` | The capture period you stated with `--traffic-window`. |
| `judge_model` | The LLM judge, if one was used. `null` means deterministic graders only. |
| `judge_agreement` | How often a human agreed with that judge. |
| `judge_trustworthy` | Whether onpar treats that agreement as enough to trust the judge. |
| `judge_calibration` | How often the judge reproduced the deterministic graders. |
| `redacted` | What redaction removed, by kind. |
| `incumbent_cost` | Monthly dollars on the incumbent. `0.0` means not supplied. |
| `candidate_cost` | Monthly dollars the candidate would cost. `0.0` means not supplied. |
| `fingerprints` | What identified each model's weights, so a silent swap is detectable. |
| `tool_version` | The onpar version that issued it. |
| `agentic` | Decision agreement over tool-using steps. Empty when there were none. |

Each entry in `proven`, `regret` and `unproven` is a claim:

| Field | What it holds |
|---|---|
| `cluster` | The task shape's stable key. |
| `name` | Its readable name. |
| `share` | Fraction of captured traffic in this cluster. |
| `passed`, `total` | Graded items that passed, out of all graded items. |
| `low`, `high` | The confidence interval on the pass rate. |
| `ungraded` | Items that could not be graded, disclosed rather than dropped. |
| `duplicates` | Items merged because an identical prompt already appeared. |
| `needed` | Graded items this cluster would need to clear the bar at its observed rate. |

## What `onpar receipt` re-verifies

```bash
onpar receipt receipt.json
onpar receipt receipt.json --against rerun.json
```

- On read, it checks the digest and refuses an altered file.
- On read, it refuses a claim that is the right shape and an impossible
  measurement, such as a share above 1.
- With `--against`, it compares evidence: the eval set, the bar, both model
  names, every fingerprint both files carry, and every cluster's claim.
- It leaves out `issued` and `tool_version`, because an honest rerun happens on
  a later day and often a newer build.
- It prints `verified` or `DOES NOT VERIFY` with each disagreement, and exits 1
  when they disagree.
- `--json` prints the receipt and its digest for a program.

## What `onpar guard` voids

A receipt describes a moment. `onpar guard` checks whether that moment still
describes production, and separates three kinds of stale:

- **The model moved.** A fingerprint changed behind the model's name. The receipt
  is void. Found with `--fingerprints`.
- **Your traffic moved.** More than a fifth of volume moved between task shapes,
  or more than 10% of current traffic sits in clusters the receipt never scored. The eval
  set answers questions nobody asks anymore. Found with `--traffic`.
- **The field moved.** A new candidate was released. The receipt is still true.
  Found with `--available`.

The first two print as `[void]`. The third prints as `[note]`, as does a receipt
older than 90 days and a receipt with no fingerprints to compare against. The
thresholds are `TRAFFIC_DRIFT_LIMIT`, `UNCOVERED_SHARE_LIMIT` and
`MAX_RECEIPT_AGE_DAYS` in `src/onpar/guard.py`.

Run with no current fingerprints, traffic or releases to compare, `guard` has
nothing that moved to report:

<!-- run: guard -->

`guard` exits 1 when the receipt no longer holds. `--fail-on any` also exits 1
on notes, which suits a release gate that should not deploy on an unreviewed
proof. The strongest thing it can propose is a re-prove. It never moves
traffic.

Next: [the eval set](30-eval-set.md), the file a receipt's `eval_set` digest
names.
