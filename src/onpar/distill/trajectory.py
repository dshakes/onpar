"""Agent sessions, recovered from a flat capture log.

An agent loop is not one request. It is a chain: the harness sends the
conversation, the model calls a tool, the harness runs it, appends the result
and sends the conversation again — one request per step, each a strict
extension of the one before. The gateway records each request on its own,
because that is what crossed the wire. This module puts them back together.

**Why it matters for a proof.** A verdict per request is a verdict per *step*,
and that is the right unit for a model that cannot run your tools: at each step
the candidate is handed the exact prefix the incumbent saw and asked for one
move. But a reader needs to know which steps belong to one task, how long the
tasks are, and how much of every request is prefix it has already seen —
because that last number is the whole economics of self-hosting an agent.

**Linking is structural and deterministic.** Two captures belong to one session
when the shorter one's messages are a prefix of the longer one's. No timestamps
(the gateway records none, deliberately), no heuristics about wording. The same
log links the same way every time, which is what lets a receipt be re-derived.

ponytail: prefix matching is O(n²) over captures sharing an opener. The opener
(system prompt + first user turn) keeps groups small in practice; switch to a
trie if a corpus has thousands of steps under one opener.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .shape import Capture

__all__ = ["Session", "SessionReport", "link", "report"]


def _canonical(m: dict[str, Any]) -> str:
    return json.dumps(m, sort_keys=True, separators=(",", ":"), default=str)


def _opener(cap: Capture) -> str:
    """Session key: the system prompt and the first user turn.

    Everything an agent loop does grows *after* these; they are the part that
    identifies which task was being worked on.
    """
    sys_ = cap.system_prompt.strip()
    first = next((m for m in cap.messages if m.get("role") == "user"), None)
    h = hashlib.sha256()
    for part in (sys_, _canonical(first) if first else ""):
        b = part.encode()
        h.update(f"{len(b)}:".encode())
        h.update(b)
    return h.hexdigest()[:16]


def _is_prefix(short: Capture, long: Capture) -> bool:
    a, b = short.messages, long.messages
    return len(a) < len(b) and all(
        _canonical(x) == _canonical(y) for x, y in zip(a, b, strict=False)
    )


@dataclass
class Session:
    """One agent task: its steps, each with its depth and the step it extends."""

    key: str
    #: (depth, capture, parent capture or None for the root).
    steps: list[tuple[int, Capture, Capture | None]] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return max((d for d, _, _ in self.steps), default=0)

    @property
    def tool_steps(self) -> int:
        return sum(1 for _, c, _ in self.steps if c.tool_calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.approx_context_tokens for _, c, _ in self.steps)

    @property
    def shared_prefix_tokens(self) -> int:
        """Tokens of prompt that were already sent at the step before.

        Every step re-sends the whole prior conversation, so a step's shared
        prefix is at least the prompt size of the step it extends. An estimate,
        and labelled one in the report — a lower bound, since the model's own
        reply is also re-sent and not counted here.
        """
        return sum(p.approx_context_tokens for _, _, p in self.steps if p is not None)


def _tree(captures: list[Capture]) -> list[tuple[str, int, Capture | None]]:
    """Per capture, in input order: (session id, depth, parent capture).

    Positional throughout — never keyed on `request_id`. The gateway records a
    client's `x-request-id` and most clients send none, so a real log has every
    row sharing one empty id; a map keyed on it collapses the corpus into one
    session and a depth walk over it can follow a pointer to itself forever.
    """
    groups: dict[str, list[int]] = {}
    for n, c in enumerate(captures):
        groups.setdefault(_opener(c), []).append(n)

    out: list[tuple[str, int, Capture | None] | None] = [None] * len(captures)
    for key, members in groups.items():
        ordered = sorted(members, key=lambda n: (len(captures[n].messages), n))
        parent: list[int | None] = []
        for pos, n in enumerate(ordered):
            # Longest prefix wins; among equals, the earliest in the log, so the
            # choice is the same on every run. Only earlier, strictly shorter
            # captures qualify, so a parent position is always below its child's.
            parent.append(
                min(
                    (q for q in range(pos) if _is_prefix(captures[ordered[q]], captures[n])),
                    key=lambda q: (-len(captures[ordered[q]].messages), ordered[q]),
                    default=None,
                )
            )
        roots = [q for q, up in enumerate(parent) if up is None]
        root_ix = {q: k for k, q in enumerate(roots)}
        for pos, n in enumerate(ordered):
            depth, node = 1, pos
            while (up := parent[node]) is not None:
                depth, node = depth + 1, up
            sid = f"s-{key}" if len(roots) == 1 else f"s-{key}-{root_ix[node] + 1}"
            up = parent[pos]
            out[n] = (sid, depth, None if up is None else captures[ordered[up]])
    return [row for row in out if row is not None]


def link(captures: list[Capture]) -> list[tuple[str, int]]:
    """Per capture, in input order: (session id, step number, 1-based).

    Every capture lands in exactly one session. A lone request is a session of
    one step, so callers never special-case it — but :func:`report` counts only
    multi-step sessions as agentic.

    Each capture's parent is the longest already-seen capture whose messages are
    a prefix of its own. A session is one tree of those; the step is the depth.
    A fork — two different continuations of the same step, as when a harness
    retries — stays in one session, because it is one task being worked on.
    """
    return [(sid, depth) for sid, depth, _ in _tree(captures)]


@dataclass(frozen=True, slots=True)
class SessionReport:
    """How agentic a corpus is, and what that implies for serving it."""

    captures: int
    sessions: int
    #: Sessions of two or more steps — the ones that are actually loops.
    multi_step: int
    longest: int
    #: Captures that called at least one tool.
    tool_captures: int
    prompt_tokens: int
    shared_prefix_tokens: int
    multi_step_captures: int = 0
    #: Captures that asked for a response format (JSON). An engine without
    #: constrained decoding cannot honour it, and the plan should say so first.
    structured_captures: int = 0

    @property
    def agentic_share(self) -> float:
        """Fraction of captures that belong to a multi-step session."""
        return self.multi_step_captures / self.captures if self.captures else 0.0

    @property
    def prefix_share(self) -> float:
        """Fraction of all prompt tokens that were a re-sent prefix.

        This is what a prefix cache (SGLang RadixAttention, vLLM automatic
        prefix caching) would not recompute — the reason agent loops are cheaper
        self-hosted than per-token billing suggests. An estimate from reported
        prompt sizes, and a lower bound; it is labelled as both.
        """
        return self.shared_prefix_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def render(self) -> list[str]:
        if not self.multi_step and not self.structured_captures:
            return []
        structured = " --structured-output" if self.structured_captures else ""
        if not self.multi_step:
            return [self._structured_line(structured)]
        avg = self.multi_step_captures / self.multi_step
        lines = [
            f"  {self.multi_step} agent sessions · {self.multi_step_captures} steps "
            f"({self.agentic_share:.0%} of captures) · avg {avg:.1f} steps, longest {self.longest}",
            f"  {self.tool_captures} steps called a tool",
        ]
        if self.prompt_tokens:
            lines.append(
                f"  ~{self.prefix_share:.0%} of prompt tokens were a re-sent prefix — "
                f"a prefix cache would skip at least that (estimate from reported prompt "
                f"sizes, lower bound)"
            )
            # The planner's own threshold, so this line and the plan agree on
            # when prefix reuse is worth turning on. Below it, say nothing: an
            # agent loop too short to share much is not a reason to tune for it.
            from ..plan import PREFIX_CACHING_FLOOR

            if self.prefix_share >= PREFIX_CACHING_FLOOR:
                tools = " --tool-calling" if self.tool_captures else ""
                lines.append(
                    f"  plan for it: onpar build --prefix-sharing {self.prefix_share:.2f}"
                    f"{tools}{structured} "
                    f"(turns on prefix reuse — vLLM --enable-prefix-caching, SGLang's radix "
                    f"cache — and sizes KV for it"
                    + (
                        "; --tool-calling emits the engine's tool-call parser for the model, "
                        "so calls come back as calls rather than text"
                        if tools
                        else ""
                    )
                    + (
                        "; --structured-output asks the engine to enforce response_format, "
                        "and the plan lists it under NOT EXPRESSED where the engine cannot"
                        if structured
                        else ""
                    )
                    + f"; onpar advise --seen-prefix-sharing {self.prefix_share:.2f} checks a "
                    f"running plan against it)"
                )
            elif structured:
                lines.append(self._structured_line(structured))
        elif structured:
            lines.append(self._structured_line(structured))
        return lines

    def _structured_line(self, flag: str) -> str:
        return (
            f"  {self.structured_captures} captures asked for a response format · plan for it: "
            f"onpar build{flag} (asks the engine to enforce response_format; the plan lists it "
            "under NOT EXPRESSED where the engine cannot, so a JSON cluster that fails is a "
            "server limit you knew about)"
        )


def report(captures: list[Capture]) -> SessionReport:
    by_session: dict[str, Session] = {}
    for c, (sid, depth, parent) in zip(captures, _tree(captures), strict=True):
        by_session.setdefault(sid, Session(sid)).steps.append((depth, c, parent))
    sessions = list(by_session.values())
    multi = [s for s in sessions if len(s.steps) > 1]
    return SessionReport(
        captures=len(captures),
        sessions=len(sessions),
        multi_step=len(multi),
        longest=max((s.depth for s in sessions), default=0),
        tool_captures=sum(1 for c in captures if c.tool_calls),
        prompt_tokens=sum(s.prompt_tokens for s in multi),
        shared_prefix_tokens=sum(s.shared_prefix_tokens for s in multi),
        multi_step_captures=sum(len(s.steps) for s in multi),
        structured_captures=sum(1 for c in captures if getattr(c, "response_format", None)),
    )


def demo() -> None:
    def cap(rid: str, msgs: list[dict[str, Any]], tokens: int, calls: tuple = ()) -> Capture:
        return Capture(
            request_id=rid,
            model="gpt-5",
            messages=tuple(msgs),
            response="",
            prompt_tokens=tokens,
            tool_calls=calls,
        )

    sysm = {"role": "system", "content": "You are a coding agent."}
    u1 = {"role": "user", "content": "fix the failing test"}
    a1 = {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]}
    t1 = {"role": "tool", "content": "def f(): ..."}
    a2 = {"role": "assistant", "tool_calls": [{"function": {"name": "write_file"}}]}
    t2 = {"role": "tool", "content": "ok"}

    s1 = cap("r1", [sysm, u1], 100, ({"name": "read_file"},))
    s2 = cap("r2", [sysm, u1, a1, t1], 300, ({"name": "write_file"},))
    s3 = cap("r3", [sysm, u1, a1, t1, a2, t2], 500)
    lone = cap("r4", [sysm, {"role": "user", "content": "unrelated"}], 50)
    # Same opener, a different second step: a fork of the same task.
    fork = cap("r5", [sysm, u1, {"role": "assistant", "content": "I'll look."}], 120)

    # Order of the log must not matter.
    log = [s3, lone, s1, fork, s2]
    by = dict(zip(["r3", "r4", "r1", "r5", "r2"], link(log), strict=True))
    assert by["r1"][1] == 1 and by["r2"][1] == 2 and by["r3"][1] == 3
    assert by["r1"][0] == by["r2"][0] == by["r3"][0]
    assert by["r4"][0] != by["r1"][0], "a different task is a different session"
    assert by["r5"][0] == by["r1"][0], "a fork is the same task, same session"
    assert by["r5"][1] == 2, "and sits at the depth it forked from"

    r = report([s1, s2, s3, lone, fork])
    assert r.sessions == 2 and r.multi_step == 1 and r.longest == 3, r
    assert r.tool_captures == 2
    # r5 and r2 re-send r1's 100 tokens; r3 re-sends r2's 300. 1020 sent in all.
    assert r.prompt_tokens == 1020 and r.shared_prefix_tokens == 500, r
    assert abs(r.prefix_share - 500 / 1020) < 1e-9
    assert any("agent sessions" in line for line in r.render())
    assert not report([lone]).render(), "no loops, no agentic block"

    # Two identical openers captured twice are two roots, two sessions.
    twin = cap("r0", [sysm, u1], 100)
    two = link([s1, twin])
    assert two[0][0] != two[1][0] and two[0][1] == two[1][1] == 1

    # Every request id identical — what a real log looks like when clients send
    # no x-request-id — links exactly as distinct ids do, and never hangs.
    from dataclasses import replace

    blank = [replace(c, request_id="") for c in log]
    assert link(blank) == link(log)

    print("\n".join(r.render()))
    print("ok")


if __name__ == "__main__":
    demo()
