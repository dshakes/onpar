"""The human-agreement loop: the number every report discloses, finally produced.

Invariant 6 says every report names the judge *and* its human-agreement rate.
The rate was a flag — ``--judge-agreed 17 --judge-samples 20`` — which is a
number the user typed, and a typed number is a claim. This module makes it a
measurement:

1. ``onpar prove --judge-endpoint … --judge-sample judged.jsonl`` wraps the
   judge so every comparison it is shown, and what it answered, is appended to
   a ledger. The two answers are the same anonymous A/B the judge saw.
2. ``onpar label judged.jsonl`` walks a human through those comparisons —
   blind, in the same A/B order, with no sight of the judge's answer — and
   records their verdict beside it.
3. ``onpar prove … --judge-labels judged.jsonl`` reads the ledger back and
   computes :class:`~.judge.Agreement` from the rows a human labelled.

Blind on purpose: a labeller who can see the judge's answer agrees with it, and
the rate measures suggestibility rather than the judge. The ledger holds the
same redacted prompts and answers the eval set already does, so no new class of
data reaches the disk (NFR-2), and nothing in it ever leaves the machine.

The judge is shown each item twice, positions swapped. Both passes are logged
and both are labelled when a human sits down, because a human who disagrees
with themself across the swap has just measured their own position bias — the
same disclosure the judge gets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .judge import Agreement, Comparison, JudgeFn, Reply

__all__ = ["agreement_from", "label", "load", "recording"]

#: What a labeller may type. Case-insensitive; anything else is asked again.
ANSWERS = {"a": "a", "b": "b", "t": "tie", "tie": "tie", "s": "skip", "skip": "skip", "q": "quit"}


def _key(c: Comparison) -> str:
    return hashlib.sha256(
        json.dumps([c.prompt, c.a, c.b], ensure_ascii=False).encode()
    ).hexdigest()[:16]


def recording(judge: JudgeFn, path: Path) -> JudgeFn:
    """Wrap a judge so every comparison and its reply is appended to ``path``.

    One line per call, written as it happens and flushed: a run that dies at
    item 300 has 300 judged comparisons a human can still label. The judge's
    own failures are recorded too (``winner`` empty), so a labeller sees the
    items the instrument could not decide — those are the interesting ones.
    """

    def ask(c: Comparison) -> Reply:
        r = judge(c)
        row = {
            "key": _key(c),
            "prompt": c.prompt,
            "a": c.a,
            "b": c.b,
            "judge": r.winner.strip().lower(),
            "reasoning": r.reasoning[:200],
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return r

    return ask


def load(path: Path) -> list[dict[str, Any]]:
    """Every well-formed row, the latest state of each key winning.

    A label is appended as a new row carrying the same key plus ``human``, so
    the file stays append-only (no rewrite can lose a judged comparison) and
    reading it is a fold.
    """
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn tail from a killed run costs one line, not the file
        if not isinstance(row, dict) or "key" not in row:
            continue
        prior = rows.get(row["key"], {})
        rows[row["key"]] = {**prior, **row}
    return list(rows.values())


def agreement_from(path: Path, model: str) -> Agreement:
    """The judge's human-agreement rate over the rows a human has labelled.

    A row counts when the human gave a verdict and the judge gave one; an item
    the judge could not decide is excluded — that is a coverage fact, disclosed
    by the calibration line, not an agreement fact. ``skip`` is not a verdict.
    """
    agreed = total = 0
    for row in load(path):
        human, judge = row.get("human"), row.get("judge")
        if human in ("a", "b", "tie") and judge in ("a", "b", "tie"):
            total += 1
            agreed += human == judge
    return Agreement(agreed=agreed, total=total, model=model)


def label(
    path: Path,
    *,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    limit: int | None = None,
) -> int:
    """Walk a human through the unlabelled comparisons, blind. Returns how many
    were labelled this sitting.

    The judge's verdict is never shown. The prompt and both answers are the
    captured text the judge saw, printed between the same markers, because the
    labeller is being asked precisely the question the judge was.
    """
    todo = [r for r in load(path) if "human" not in r]
    if not todo:
        show("  nothing left to label")
        return 0
    done = 0
    show(f"  {len(todo)} comparisons to label. Answer A, B, T(ie), S(kip) or Q(uit).\n")
    for n, row in enumerate(todo, start=1):
        if limit is not None and done >= limit:
            break
        show(f"── {n}/{len(todo)} ─────────────────────────────────────────")
        show("<<<REQUEST>>>\n" + str(row.get("prompt", "")))
        show("<<<ANSWER A>>>\n" + str(row.get("a", "")))
        show("<<<ANSWER B>>>\n" + str(row.get("b", "")))
        show("<<<END>>>")
        while True:
            try:
                raw = ask("  which better satisfies the request? [A/B/T/S/Q] ").strip().lower()
            except EOFError:
                raw = "q"
            answer = ANSWERS.get(raw)
            if answer:
                break
            show("  A, B, T, S or Q.")
        if answer == "quit":
            break
        if answer == "skip":
            continue
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": row["key"], "human": answer}) + "\n")
        done += 1
    ag = agreement_from(path, "judge")
    rate = "" if ag.rate is None else f" · human agreement so far {ag.rate:.2f} (n={ag.total})"
    show(f"\n  labelled {done}{rate}")
    return done


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "judged.jsonl"

        def judge(c: Comparison) -> Reply:
            return Reply("a" if "good" in c.a else "b")

        rec = recording(judge, path)
        rec(Comparison("q1", a="good answer", b="bad answer"))
        rec(Comparison("q1", a="bad answer", b="good answer"))  # the swap
        rec(Comparison("q2", a="meh", b="meh"))  # judge says b; human will say tie
        assert len(load(path)) == 3 and all("human" not in r for r in load(path))

        answers = iter(["x", "a", "b", "t"])  # one bad keypress first
        shown: list[str] = []
        n = label(path, ask=lambda _: next(answers), show=shown.append)
        assert n == 3
        assert not any("judge" in s.lower() and "says" in s.lower() for s in shown), "blind"

        ag = agreement_from(path, "j")
        assert (ag.agreed, ag.total) == (2, 3), (ag.agreed, ag.total)
        assert ag.rate is not None and abs(ag.rate - 2 / 3) < 1e-9

        # Append-only: the three originals are still there, labels folded in.
        assert path.read_text().count("\n") == 6
        assert label(path, ask=lambda _: "a", show=shown.append) == 0, "nothing left"

        # Quit leaves the rest unlabelled and loses nothing.
        rec(Comparison("q3", a="x", b="y"))
        assert label(path, ask=lambda _: "q", show=shown.append) == 0
        assert sum("human" not in r for r in load(path)) == 1

    print(ag.render())
    print("ok")


if __name__ == "__main__":
    demo()
