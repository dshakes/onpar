"""Session replay — the whole agent loop, with tool results answered from the recording.

Per-step replay (ADR-0020) asks the candidate for one move given exactly the
prefix the incumbent saw. It cannot say whether the candidate would have *got
there*: a model that takes a different first step is scored against a prefix it
would never have produced. This module answers the other question, without
running anybody's tools:

1. Start from the session's first step, exactly as captured.
2. Ask the candidate. If it calls the same tools the incumbent called at that
   step, append *its* assistant message and then the incumbent's recorded tool
   results for those calls, and ask again with the next recorded step's shape.
3. If it answers in text where the incumbent called a tool, calls a tool the
   incumbent did not, or the recording has no result for what it asked, the
   session **diverges** there. A miss is a stop, never a live call: the tool
   results in the recording are the only ones that exist, and inventing one
   would be a tool execution with extra steps.

The score is how far the candidate got — steps completed before divergence,
out of the steps the incumbent took — reported with a Wilson interval over
sessions that reached the end. It is deliberately a *path* metric: a
candidate that reaches the same outcome by a different route reads as
diverged, and the report says so, because that is the known limit of replay
from a recording and overclaiming it is how a migration goes wrong.

Nothing here sends anything. The ``ask`` callable is injected; the CLI builds it
from :mod:`.collect`'s one egress path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .graders import EvalItem, _call_names
from .stats import Interval, wilson

__all__ = ["Ask", "SessionReplay", "ReplayReport", "replay", "replay_all", "sessions"]

#: Candidate transport: (messages, tools) -> (assistant text, tool calls).
Ask = Callable[[list[dict[str, Any]], list[dict[str, Any]]], tuple[str, tuple[dict[str, Any], ...]]]


@dataclass(frozen=True, slots=True)
class SessionReplay:
    session: str
    #: Steps the incumbent took.
    steps: int
    #: Steps the candidate completed in agreement before stopping.
    completed: int
    #: Why it stopped short, empty when it reached the end.
    diverged: str = ""

    @property
    def finished(self) -> bool:
        return self.completed == self.steps


@dataclass(frozen=True, slots=True)
class ReplayReport:
    sessions: tuple[SessionReplay, ...]

    @property
    def finished(self) -> int:
        return sum(1 for s in self.sessions if s.finished)

    @property
    def interval(self) -> Interval:
        return wilson(self.finished, len(self.sessions))

    def render(self) -> str:
        if not self.sessions:
            return ""
        n = len(self.sessions)
        steps = sum(s.steps for s in self.sessions)
        done = sum(s.completed for s in self.sessions)
        out = [
            "SESSION REPLAY — the whole loop, tool results answered from the recording",
            f"  {self.finished}/{n} sessions followed the incumbent's path to the end · "
            f"{self.interval.render()} · {done}/{steps} steps completed",
        ]
        reasons: dict[str, int] = {}
        for s in self.sessions:
            if s.diverged:
                reasons[s.diverged] = reasons.get(s.diverged, 0) + 1
        for why, k in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
            out.append(f"    {k} × diverged: {why}")
        out.append(
            "  A different route to the same outcome counts as diverged — this measures "
            "whether the candidate takes the incumbent's path, not whether it would succeed."
        )
        return "\n".join(out)


def sessions(items: Sequence[EvalItem]) -> dict[str, list[EvalItem]]:
    """Multi-step sessions, steps in order. Lone requests are not sessions."""
    by: dict[str, list[EvalItem]] = {}
    for it in items:
        if it.session and it.step:
            by.setdefault(it.session, []).append(it)
    return {k: sorted(v, key=lambda i: i.step) for k, v in by.items() if len(v) > 1}


def _recorded_results(next_step: EvalItem, after: int) -> list[dict[str, Any]]:
    """The tool-result messages the harness appended between two recorded steps.

    Step k+1's messages are step k's messages, then the incumbent's assistant
    turn, then the tool results. Everything past ``after`` that is not the
    assistant turn is what the harness fed back.
    """
    tail = list(next_step.messages[after:])
    return [m for m in tail if m.get("role") != "assistant"]


def replay(steps: Sequence[EvalItem], ask: Ask) -> SessionReplay:
    """Walk one session against the candidate until it diverges or finishes."""
    sid = steps[0].session
    messages = [dict(m) for m in steps[0].messages]
    tools = [dict(t) for t in steps[0].tools]
    for k, step in enumerate(steps):
        want = _call_names(step.baseline_tool_calls)
        text, calls = ask(messages, tools)
        got = _call_names(calls)
        if not want:
            # The incumbent answered in text here: the last step of a loop,
            # usually. Any text completes it; a tool call is a divergence.
            if got:
                return SessionReplay(
                    sid, len(steps), k, f"called {got} where the incumbent answered"
                )
            continue
        if got != want:
            why = (
                "answered in text where the incumbent called a tool"
                if not got
                else f"called {got}, incumbent called {want}"
            )
            return SessionReplay(sid, len(steps), k, why)
        if k + 1 >= len(steps):
            # Agreed on the final recorded step; nothing further was recorded.
            break
        results = _recorded_results(steps[k + 1], after=len(step.messages) + 1)
        if len([m for m in results if m.get("role") == "tool"]) < len(want):
            return SessionReplay(sid, len(steps), k, "recording holds no result for that call")
        # The candidate's own call ids must be what the results point at, or a
        # strict server rejects the conversation. Rewrite ids positionally.
        ids = [c.get("id") for c in calls if isinstance(c, dict)]
        tool_msgs = [m for m in results if m.get("role") == "tool"]
        for m, cid in zip(tool_msgs, ids, strict=False):
            if cid:
                m["tool_call_id"] = cid
        messages.append({"role": "assistant", "content": text or None, "tool_calls": list(calls)})
        messages.extend(results)
        tools = [dict(t) for t in steps[k + 1].tools] or tools
    return SessionReplay(sid, len(steps), len(steps))


def replay_all(items: Sequence[EvalItem], ask: Ask) -> ReplayReport:
    return ReplayReport(tuple(replay(steps, ask) for steps in sessions(items).values()))


def demo() -> None:
    def call(name: str, cid: str = "c1") -> dict[str, Any]:
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}

    sysm = {"role": "system", "content": "refunds desk"}
    u = {"role": "user", "content": "refund 42"}
    a1 = {"role": "assistant", "content": None, "tool_calls": [call("refund")]}
    t1 = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    tools = ({"type": "function", "function": {"name": "refund", "parameters": {}}},)
    s1 = EvalItem(
        "r1",
        "c",
        "refund 42",
        "",
        "",
        baseline_tool_calls=(call("refund"),),
        messages=(sysm, u),
        tools=tools,
        session="s",
        step=1,
    )
    s2 = EvalItem(
        "r2",
        "c",
        "refund 42",
        "Refunded.",
        "",
        messages=(sysm, u, a1, t1),
        tools=tools,
        session="s",
        step=2,
    )
    lone = EvalItem(
        "r3",
        "c",
        "hi",
        "hello",
        "",
        messages=(sysm, {"role": "user", "content": "hi"}),
        session="t",
        step=1,
    )

    seen: list[list[dict[str, Any]]] = []

    def agrees(messages, tools_):
        seen.append(list(messages))
        if any(m.get("role") == "tool" for m in messages):
            return "Refunded.", ()
        return "", (call("refund", "cand-1"),)

    r = replay_all([s1, s2, lone], agrees)
    assert len(r.sessions) == 1 and r.sessions[0].finished, r
    # The second ask carried the candidate's own assistant turn and the
    # recorded tool result, re-pointed at the candidate's call id.
    assert seen[1][-2]["tool_calls"][0]["id"] == "cand-1"
    assert seen[1][-1] == {"role": "tool", "tool_call_id": "cand-1", "content": "ok"}

    def talks(messages, tools_):
        return "Happy to help!", ()

    r2 = replay_all([s1, s2], talks)
    assert not r2.sessions[0].finished and r2.sessions[0].completed == 0
    assert "answered in text" in r2.sessions[0].diverged

    def wrong_tool(messages, tools_):
        return "", (call("lookup"),)

    assert "incumbent called ['refund']" in replay_all([s1, s2], wrong_tool).sessions[0].diverged
    assert r2.interval.low < r2.interval.high
    assert "SESSION REPLAY" in r.render() and not replay_all([lone], agrees).render()
    print(r.render())
    print("ok")


if __name__ == "__main__":
    demo()
