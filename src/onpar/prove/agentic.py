"""The agentic verdict: did the candidate make the same *moves*?

The equivalence matrix scores items pass/fail and that is the right gate. But
for an agent workload a reader needs a second, more specific question answered:
at each step, did the candidate decide the same thing the incumbent did —
answer in text, or call a tool, and which one? A model that writes beautiful
prose where the incumbent called ``refund()`` fails the task however good the
prose is, and a model that calls ``refund()`` where the incumbent asked a
clarifying question has just refunded someone.

Two measurements, both chance-corrected or interval-bearing, never a bare rate:

* **Decision agreement**, as Cohen's kappa over the per-step action — the tool
  name called, or ``respond`` for a text-only turn. Raw agreement flatters a
  workload where one action dominates; kappa subtracts what two coin-flippers
  biased the same way would agree on. Landis & Koch's bands label it.
* **Per-tool regret**: for each tool the incumbent called, how often the
  candidate also called it — with a Wilson interval, so a tool seen three times
  reads as unknown rather than as 0% or 100%.

Everything here is derived from the same :class:`~.graders.EvalItem` rows the
matrix scored, so it cannot disagree with the matrix about what happened.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .graders import EvalItem, _call_names
from .stats import Interval, wilson

__all__ = ["AgenticReport", "ToolAgreement", "kappa", "report"]

#: Action label for a step that produced text and called nothing.
RESPOND = "respond"

#: Landis & Koch (1977) bands. A convention, not a truth — stated so a reader
#: can disagree with the label while trusting the number.
_BANDS = ((0.80, "almost perfect"), (0.60, "substantial"), (0.40, "moderate"), (0.20, "fair"))


def action(calls: tuple[Any, ...]) -> str:
    """The move a step made: the tool names called in order, or ``respond``."""
    names = _call_names(calls)
    return " → ".join(names) if names else RESPOND


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa between two labellers over paired categorical decisions.

    ``None`` when there is nothing to compute: no pairs, or every label on both
    sides identical (kappa is 0/0 there, and the raw agreement already says 100%).
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[k] * right.get(k, 0) for k in left) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def band(k: float) -> str:
    for floor, label in _BANDS:
        if k >= floor:
            return label
    return "slight" if k > 0 else "none"


@dataclass(frozen=True, slots=True)
class ToolAgreement:
    """For one tool the incumbent called: how often the candidate did too."""

    tool: str
    #: Steps where the incumbent called this tool.
    incumbent: int
    #: Of those, steps where the candidate called it as well.
    matched: int
    #: Steps where the candidate called it and the incumbent did not.
    spurious: int

    @property
    def interval(self) -> Interval:
        return wilson(self.matched, self.incumbent)

    def render(self) -> str:
        extra = f" · candidate called it {self.spurious}× unprompted" if self.spurious else ""
        return f"    {self.tool:<28}{self.interval.render():>16} of {self.incumbent}{extra}"


@dataclass(frozen=True, slots=True)
class AgenticReport:
    """Decision agreement over the agentic steps of an eval set."""

    #: Steps where either side was in a tool-using context.
    steps: int
    agreed: int
    kappa: float | None
    tools: tuple[ToolAgreement, ...]
    #: Steps the incumbent answered in text and the candidate called a tool.
    acted_unasked: int
    #: Steps the incumbent called a tool and the candidate only wrote text.
    talked_instead: int

    @property
    def interval(self) -> Interval:
        return wilson(self.agreed, self.steps)

    def render(self) -> str:
        if not self.steps:
            return ""
        k = "n/a" if self.kappa is None else f"κ={self.kappa:.2f} ({band(self.kappa)})"
        out = [
            "AGENTIC — did the candidate make the same moves?",
            f"  decision agreement {self.interval.render()} over {self.steps} tool-context steps · "
            f"{k}, chance-corrected",
        ]
        if self.talked_instead or self.acted_unasked:
            out.append(
                f"  {self.talked_instead} steps the candidate wrote text where the incumbent "
                f"called a tool · {self.acted_unasked} steps it called a tool where the "
                f"incumbent answered"
            )
        if self.tools:
            out.append("  per tool, how often the candidate also called it:")
            out += [t.render() for t in self.tools]
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "agreed": self.agreed,
            "agreement": [round(self.interval.low, 4), round(self.interval.high, 4)],
            "kappa": None if self.kappa is None else round(self.kappa, 4),
            "talked_instead": self.talked_instead,
            "acted_unasked": self.acted_unasked,
            "tools": [
                {
                    "tool": t.tool,
                    "incumbent": t.incumbent,
                    "matched": t.matched,
                    "spurious": t.spurious,
                    "interval": [round(t.interval.low, 4), round(t.interval.high, 4)],
                }
                for t in self.tools
            ],
        }


def report(items: list[EvalItem]) -> AgenticReport:
    """Decision agreement over every item that is in a tool-using context.

    An item counts when tools were offered or either side called one. An empty
    candidate answer is a decision too — the model said nothing where the
    incumbent acted — because by the time items reach the suite every one of
    them was collected; the ones that never answered were dropped upstream.
    """
    pairs: list[tuple[str, str]] = []
    acted_unasked = talked_instead = 0
    incumbent_calls: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    spurious: Counter[str] = Counter()
    for it in items:
        if not it.agentic and not it.candidate_tool_calls:
            continue
        a, b = action(it.baseline_tool_calls), action(it.candidate_tool_calls)
        pairs.append((a, b))
        if a == RESPOND and b != RESPOND:
            acted_unasked += 1
        if a != RESPOND and b == RESPOND:
            talked_instead += 1
        want, got = (
            set(_call_names(it.baseline_tool_calls)),
            set(_call_names(it.candidate_tool_calls)),
        )
        for t in want:
            incumbent_calls[t] += 1
            if t in got:
                matched[t] += 1
        for t in got - want:
            spurious[t] += 1
    tools = tuple(
        ToolAgreement(t, incumbent_calls[t], matched[t], spurious[t])
        for t in sorted(incumbent_calls | spurious, key=lambda t: (-incumbent_calls[t], t))
    )
    return AgenticReport(
        steps=len(pairs),
        agreed=sum(1 for a, b in pairs if a == b),
        kappa=kappa(pairs),
        tools=tools,
        acted_unasked=acted_unasked,
        talked_instead=talked_instead,
    )


def demo() -> None:
    def call(name: str) -> dict[str, Any]:
        return {"function": {"name": name, "arguments": "{}"}}

    tools = ({"function": {"name": "refund", "parameters": {}}},)

    def it(n: int, base: tuple = (), cand: tuple = (), text: str = "x") -> EvalItem:
        return EvalItem(
            f"i{n}",
            "c",
            "p",
            "b",
            text,
            baseline_tool_calls=base,
            candidate_tool_calls=cand,
            tools=tools,
        )

    items = [
        it(1, (call("refund"),), (call("refund"),)),
        it(2, (call("refund"),), (call("refund"),)),
        it(3, (call("refund"),), (), "I'd be happy to help"),  # talked instead
        it(4, (), (call("refund"),)),  # acted unasked
        it(5, (), ()),
        it(6, (), ()),
        it(7, (call("lookup"),), (call("lookup"),)),
        EvalItem("plain", "c", "p", "b", "c"),  # not agentic, not counted
    ]
    r = report(items)
    assert r.steps == 7 and r.agreed == 5
    assert r.talked_instead == 1 and r.acted_unasked == 1
    assert r.kappa is not None and 0.5 < r.kappa < 0.7, r.kappa
    by = {t.tool: t for t in r.tools}
    assert by["refund"].incumbent == 3 and by["refund"].matched == 2 and by["refund"].spurious == 1
    assert by["lookup"].interval.low < 1.0, "one observation must not read as certainty"
    assert r.tools[0].tool == "refund", "most-called tool first"

    # kappa: perfect agreement on a degenerate corpus is undefined, not 1.0.
    assert kappa([("respond", "respond")] * 5) is None
    # ...and two sides agreeing exactly as often as chance would read as zero.
    assert abs(kappa([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) or 0.0) < 1e-9
    assert kappa([]) is None
    assert band(0.85) == "almost perfect" and band(0.1) == "slight" and band(-0.2) == "none"

    assert not report([EvalItem("plain", "c", "p", "b", "c")]).render()
    text = r.render()
    assert "κ=" in text and "refund" in text and "unprompted" in text
    print(text)
    print("ok")


if __name__ == "__main__":
    demo()
