"""The grader stack — tiers 1 and 2: deterministic, cheap, and unarguable.

Three tiers, escalating only when the cheaper one cannot decide:

    1. assert   structure — valid JSON, same keys, same tool called, non-empty
    2. task     task semantics — exact match, numbers within tolerance, same files
    3. judge    pairwise LLM comparison, last resort (see :mod:`.judge`)

**A grader that does not apply returns NOT_APPLICABLE, never PASS.** That is the
single most important rule here. If "no assertion fired" silently counted as
success, every model would score near 100% on any cluster our graders don't
understand, and the resulting matrix would be confidently, uselessly green.

Tiers 1 and 2 are worth more than they look. Most real regressions are
structural — the candidate emits prose where JSON was required, calls the wrong
tool, or drops a file from a diff — and those are caught here, deterministically,
for free, without asking a language model to have an opinion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Tier(StrEnum):
    """Which stage of the stack a grader belongs to."""

    ASSERT = "assert"
    TASK = "task"
    JUDGE = "judge"


class Outcome(StrEnum):
    """What a grader concluded."""

    PASS = "pass"
    FAIL = "fail"
    #: The grader had nothing to say. Never counted as a pass.
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True, slots=True)
class Score:
    """One grader's verdict on one item."""

    grader: str
    tier: Tier
    outcome: Outcome
    detail: str = ""

    @property
    def applicable(self) -> bool:
        return self.outcome is not Outcome.NOT_APPLICABLE

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One prompt, the incumbent's answer, and the candidate's answer.

    ``baseline`` is what the model being replaced actually produced. It is not
    ground truth — it is the bar to match, which is a weaker and more honest claim.
    """

    item_id: str
    cluster: str
    prompt: str
    baseline: str
    candidate: str
    #: Each call is either the OpenAI request shape (a dict) or a bare tool
    #: name (a string) — see `_call_names`. Same unvalidated-file caveat as
    #: `response_format`.
    baseline_tool_calls: tuple[dict[str, Any] | str, ...] = ()
    candidate_tool_calls: tuple[dict[str, Any] | str, ...] = ()
    #: Either a bare name ("enum", "json_schema") or the OpenAI request shape
    #: `{"type": "json_object"}` — the CLI reads this straight out of an eval-set
    #: file, so whatever a user wrote arrives here unvalidated.
    response_format: str | dict | None = None
    #: The whole conversation the incumbent answered — system prompt, prior
    #: turns, tool results — redacted at capture. This is what gets replayed to
    #: the candidate. Empty for eval sets written before it existed; those
    #: replay ``prompt`` alone, which is unfaithful for any system-prompted or
    #: tool-using workload and the collector says so.
    messages: tuple[dict[str, Any], ...] = ()
    #: Tool schemas the request offered, as sent. Replayed with the messages,
    #: and the standard ``ToolArgsSchema`` validates candidate arguments against
    #: them.
    tools: tuple[dict[str, Any], ...] = ()
    #: Agent session this step belongs to and its position in it, from
    #: :mod:`onpar.distill.trajectory`. Empty / 0 for a lone request.
    session: str = ""
    step: int = 0

    @property
    def agentic(self) -> bool:
        """Whether this step is part of a tool-using workload at all."""
        return bool(self.tools) or bool(self.baseline_tool_calls)

    @classmethod
    def from_row(cls, r: dict[str, Any], n: int = 0) -> EvalItem:
        """One row of an eval-set file, as `onpar distill` writes it or a hand edits it.

        One reader for every surface. The CLI and the MCP server each had their
        own copy, which is how a field added to the file reached one and not the
        other — and a loader that drops `messages` silently reverts the replay
        to the unfaithful one.
        """
        return cls(
            item_id=str(r.get("item_id", n)),
            cluster=str(r.get("cluster", "all")),
            prompt=str(r.get("prompt", "")),
            baseline=str(r.get("baseline", "")),
            candidate=str(r.get("candidate", "")),
            baseline_tool_calls=tuple(r.get("baseline_tool_calls", ()) or ()),
            candidate_tool_calls=tuple(r.get("candidate_tool_calls", ()) or ()),
            response_format=r.get("response_format"),
            messages=tuple(m for m in (r.get("messages") or ()) if isinstance(m, dict)),
            tools=tuple(t for t in (r.get("tools") or ()) if isinstance(t, dict)),
            session=str(r.get("session") or ""),
            step=int(r.get("step") or 0),
        )


@runtime_checkable
class Grader(Protocol):
    """Anything that can judge one item."""

    name: str
    tier: Tier

    def grade(self, item: EvalItem) -> Score: ...


#: Returned by `_parse_json` for text that is not JSON at all. A sentinel rather
#: than `None`, because `null` is perfectly valid JSON that parses *to* `None` —
#: conflating the two graded a candidate that correctly answered `null` as
#: unparseable, and excused the baseline check as "not JSON" on a baseline that
#: was.
_NOT_JSON = object()


def _parse_json(text: str) -> Any:
    """Parsed JSON, or `_NOT_JSON`. Never `None` to mean failure."""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return _NOT_JSON


# --------------------------------------------------------------------------- #
# Tier 1 — structure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NonEmpty:
    """The candidate must answer when the incumbent did."""

    name: str = "non-empty"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.baseline.strip():
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline was empty")
        if item.candidate.strip():
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(self.name, self.tier, Outcome.FAIL, "candidate returned nothing")


@dataclass(frozen=True, slots=True)
class JsonValid:
    """If the incumbent produced JSON, the candidate must produce valid JSON."""

    name: str = "json-valid"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if _parse_json(item.baseline) is _NOT_JSON:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not JSON")
        if _parse_json(item.candidate) is _NOT_JSON:
            return Score(self.name, self.tier, Outcome.FAIL, "candidate is not parseable JSON")
        return Score(self.name, self.tier, Outcome.PASS)


@dataclass(frozen=True, slots=True)
class JsonShape:
    """Same top-level keys. Values may differ; the contract is the shape.

    Downstream code indexes by key. A response with the right values under the
    wrong keys breaks the caller just as hard as garbage.
    """

    name: str = "json-shape"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        b, c = _parse_json(item.baseline), _parse_json(item.candidate)
        if not isinstance(b, dict):
            return Score(
                self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not a JSON object"
            )
        if not isinstance(c, dict):
            return Score(self.name, self.tier, Outcome.FAIL, "candidate is not a JSON object")
        missing, extra = set(b) - set(c), set(c) - set(b)
        if missing or extra:
            bits = []
            if missing:
                bits.append(f"missing {sorted(missing)}")
            if extra:
                bits.append(f"unexpected {sorted(extra)}")
            return Score(self.name, self.tier, Outcome.FAIL, "; ".join(bits))
        return Score(self.name, self.tier, Outcome.PASS)


def _call_names(calls: tuple[object, ...]) -> list[str]:
    """Tool names, from any of the three shapes a call arrives in.

    An eval set is a file. It comes from `onpar distill`, from a hand edit,
    or from another tool entirely, and invariant 7 applies to it as much as to
    the corpus: it is data, not a contract. A bare `"refund"` is an obvious way
    to write a tool call by hand, and raising `AttributeError` from four frames
    inside a grader is the wrong answer to it — the whole run dies over one
    malformed row.
    """
    out = []
    for c in calls:
        if isinstance(c, str):
            n: object = c
        elif isinstance(c, dict):
            fn = c.get("function")
            n = c.get("name") or (fn.get("name") if isinstance(fn, dict) else None)
        else:
            continue
        if n:
            out.append(str(n))
    return out


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """The candidate must call the same tools, in the same order.

    Order matters: a plan that reads before writing is not the same plan as one
    that writes before reading, even with identical tool names.
    """

    name: str = "tool-choice"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.baseline_tool_calls:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline called no tools")
        want, got = _call_names(item.baseline_tool_calls), _call_names(item.candidate_tool_calls)
        if want == got:
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(self.name, self.tier, Outcome.FAIL, f"expected {want}, got {got}")


@dataclass(frozen=True, slots=True)
class ToolArgs:
    """Required argument *keys* must match on every shared tool call.

    Values are not compared — a different file path can be a correct answer to a
    different prompt. A missing argument cannot.
    """

    name: str = "tool-args"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.baseline_tool_calls or not item.candidate_tool_calls:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "no comparable tool calls")
        problems = []
        # Length mismatches are ToolChoice's job; here we only compare the pairs.
        for i, (b, c) in enumerate(
            zip(item.baseline_tool_calls, item.candidate_tool_calls, strict=False)
        ):
            bk, ck = set(_args(b)), set(_args(c))
            if bk - ck:
                problems.append(f"call {i} missing {sorted(bk - ck)}")
        if problems:
            return Score(self.name, self.tier, Outcome.FAIL, "; ".join(problems))
        return Score(self.name, self.tier, Outcome.PASS)


def _args(call: object) -> dict[str, Any]:
    if not isinstance(call, dict):
        # A bare name like "refund" — the same shape `_call_names` accepts.
        # No arguments to compare, so there is nothing to report missing.
        return {}
    raw = call.get("arguments") or (call.get("function") or {}).get("arguments") or {}
    if isinstance(raw, str):
        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _parameter_schemas(tools: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    """Tool name -> its ``parameters`` JSON schema, for both spellings in the wild."""
    out: dict[str, dict[str, Any]] = {}
    for t in tools:
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        params = fn.get("parameters") or fn.get("input_schema")
        if name and isinstance(params, dict):
            out[str(name)] = params
    return out


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def schema_problems(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Where ``value`` violates ``schema``, as readable paths. Empty means valid.

    The subset of JSON Schema that tool definitions actually use: ``type``,
    ``required``, ``properties``, ``additionalProperties: false``, ``enum``,
    ``items``. Anything else is accepted, not refused — a validator that
    rejected a schema keyword it did not know would fail every call that used
    it, which reads exactly like a model that cannot call tools.

    ponytail: no $ref, no allOf/anyOf, no formats. Add when a captured schema
    uses one; today the whole corpus is flat argument objects.
    """
    problems: list[str] = []
    want = schema.get("type")
    if want:
        names = want if isinstance(want, list) else [want]
        allowed = tuple(t for n in names for t in _JSON_TYPES.get(str(n), ()))
        # bool is an int to Python; JSON Schema disagrees.
        ok = isinstance(value, allowed) and not (isinstance(value, bool) and bool not in allowed)
        if allowed and not ok:
            got = "null" if value is None else type(value).__name__
            problems.append(f"{path}: expected {'/'.join(map(str, names))}, got {got}")
            return problems  # the checks below assume the right type
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        problems.append(f"{path}: {value!r} not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key in schema.get("required") or ():
            if isinstance(key, str) and key not in value:
                problems.append(f"{path}.{key}: required")
        for key, sub in value.items():
            if key in props and isinstance(props[key], dict):
                problems += schema_problems(sub, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                problems.append(f"{path}.{key}: not in schema")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, sub in enumerate(value):
            problems += schema_problems(sub, schema["items"], f"{path}[{i}]")
    return problems


@dataclass(frozen=True, slots=True)
class ToolArgsSchema:
    """Every argument object the candidate emits must satisfy the tool's schema.

    The commonest way an open model fails an agentic workload is not choosing
    the wrong tool — it is calling the right tool with arguments the tool will
    reject: a string where an integer was declared, a required field missing,
    a value outside the enum. The incumbent's harness would have thrown; this
    grader throws first, deterministically, against the schema the request
    itself offered.

    Not applicable when the candidate called nothing (``ToolChoice`` owns that)
    or when no schema for the called tools was captured.
    """

    name: str = "tool-args-schema"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.candidate_tool_calls:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "candidate called no tools")
        schemas = _parameter_schemas(item.tools)
        if not schemas:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "no tool schemas on item")
        problems: list[str] = []
        checked = 0
        for i, call in enumerate(item.candidate_tool_calls):
            names = _call_names((call,))
            if not names or names[0] not in schemas:
                # A tool the request never offered. ToolChoice misses this when
                # the baseline called nothing, and a hallucinated tool is the
                # one argument error no harness can recover from — so it is a
                # problem on this item, not a call to skip.
                problems.append(f"call {i} ({names[0] if names else '?'}): tool never offered")
                continue
            checked += 1
            # A call whose arguments are not even JSON is the worst case and
            # `_args` hides it as `{}` — surface it instead.
            raw = (
                call.get("arguments") or (call.get("function") or {}).get("arguments")
                if isinstance(call, dict)
                else None
            )
            if isinstance(raw, str) and _parse_json(raw) is _NOT_JSON:
                problems.append(f"call {i} ({names[0]}): arguments are not JSON")
                continue
            problems += [
                f"call {i} ({names[0]}) {p}"
                for p in schema_problems(_args(call), schemas[names[0]])
            ]
        if problems:
            return Score(self.name, self.tier, Outcome.FAIL, "; ".join(problems)[:400])
        return Score(self.name, self.tier, Outcome.PASS)


#: Markers a serving stack leaves in `content` when the model emitted a tool
#: call and the server's parser did not recognise it. Each is a chat-template
#: dialect in the wild; the list is a calibration knob, extended when a new one
#: is seen, never a guess.
TOOL_LEAK_MARKERS: tuple[str, ...] = (
    "<tool_call>",
    "<function_call>",
    "<|tool_call|>",
    "<|tool_calls_section_begin|>",
    "<function=",
    "[TOOL_CALLS]",
    '"tool_calls":',
    "<｜tool▁calls▁begin｜>",
)


@dataclass(frozen=True, slots=True)
class ToolLeak:
    """A tool call that arrived as text is a failed call, not an answer.

    When the serving engine's `--tool-call-parser` or chat template does not
    match the model, the model still emits the call — and the server returns it
    as `content` with `finish_reason: stop`, no `tool_calls`. Graded as prose it
    reads as a verbose answer; graded by ToolChoice it reads as "chose not to
    call", which is a different failure with a different fix. This is the
    commonest reason an open model *looks* worse at tool use than it is, and
    the fix is a server flag, so the detail says so.
    """

    name: str = "tool-leak"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.tools or item.candidate_tool_calls:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "no leak possible")
        hit = next((m for m in TOOL_LEAK_MARKERS if m in item.candidate), None)
        if hit is None:
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(
            self.name,
            self.tier,
            Outcome.FAIL,
            f"tool call leaked into text ({hit!r}) — the server did not parse it; "
            f"check the engine's tool-call parser / chat template for this model",
        )


# --------------------------------------------------------------------------- #
# Tier 2 — task semantics
# --------------------------------------------------------------------------- #

_CLASSIFICATION_MAX_WORDS = 3


@dataclass(frozen=True, slots=True)
class ExactMatch:
    """For classification-shaped answers, the label must match.

    Applies only when the incumbent's answer is a short label — for prose, exact
    match is the wrong question and would fail every correct paraphrase.
    """

    name: str = "exact-match"
    tier: Tier = Tier.TASK

    def grade(self, item: EvalItem) -> Score:
        b = item.baseline.strip()
        if not _looks_like_a_label(b, item.response_format):
            return Score(
                self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not a short label"
            )
        if _label_key(b) == _label_key(item.candidate):
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(
            self.name,
            self.tier,
            Outcome.FAIL,
            f"expected {b!r}, got {item.candidate.strip()[:60]!r}",
        )


def _label_key(text: str) -> str:
    """A label reduced to what actually distinguishes it from another label.

    Measured against a real model: asked "capital of France? Reply with only the
    city name", Llama 3.1 answers `Paris.` — correct, and scored FAIL against a
    baseline of `Paris` for the full stop. That ran across all ten extraction
    items and the suite reported REGRET, keep the incumbent, for a model that
    got every question right.

    Only wrapping punctuation is removed: surrounding quotes and backticks, and
    trailing sentence marks. Internal punctuation is untouched, because it
    carries meaning — `3.14` and `314` are different answers, and so are
    `no, refund` and `no refund`. If stripping would empty the string the raw
    text is kept, so a label that IS punctuation still compares as itself.
    """
    t = text.strip()
    stripped = t.strip("\"'`\u201c\u201d\u2018\u2019").rstrip(".!?").strip()
    return (stripped or t).casefold()


def _format_name(response_format: str | dict | None) -> str | None:
    """The format's name, from either spelling people actually write.

    `{"type": "json_object"}` is the OpenAI request shape and the one the
    `prove` help text points at, so it is what ends up in an eval-set file. It
    reached `x in {"enum", "json_schema"}` as a dict and raised TypeError —
    unhashable — which killed the entire suite AFTER every reply had been
    collected and paid for. A malformed field must cost one grader's opinion,
    never the run.
    """
    if isinstance(response_format, str):
        return response_format
    if isinstance(response_format, dict):
        kind = response_format.get("type")
        return kind if isinstance(kind, str) else None
    return None


def _looks_like_a_label(text: str, response_format: str | dict | None) -> bool:
    """Whether an answer is a classification label rather than a short sentence.

    Word count alone is not enough: "Revenue was 42.5m." is three words and a
    sentence, and exact-matching it would fail every correct paraphrase. Sentence
    punctuation is the discriminator. When the request declared an enum or
    json_schema format, that is authoritative and the heuristic is skipped.
    """
    # `json_object` is deliberately NOT here: it constrains the answer to JSON,
    # which is the opposite of a short label — exact-matching a serialised object
    # would fail on key order and whitespace. Only formats that pin the answer to
    # a fixed set are authoritative.
    if _format_name(response_format) in {"enum", "json_schema"}:
        return True
    if not text or "\n" in text:
        return False
    if len(text.split()) > _CLASSIFICATION_MAX_WORDS:
        return False
    # A label does not end a sentence and does not contain one.
    return not text.endswith((".", "!", "?", ":", ";")) and "," not in text


#: Numbers, including thousands-grouped ones. The grouped alternative comes
#: FIRST because Python's `|` is first-match-wins: with the plain form first,
#: `1,234,567` matched as `1` and the rest was rescanned.
#:
#: That is not cosmetic. `-?\d+(?:\.\d+)?` split `$1,234,567` into `1`, `234`,
#: `567`, and `grade` only checks each value appears *somewhere* in the
#: candidate — so "1 prior conviction … served 234 days … 567 dollars in fines"
#: scored PASS against a baseline awarding $1,234,567. A deterministic grader,
#: the tier trusted to advance traffic, passing a completely wrong money figure
#: — and grouping is exactly what money figures have.
#:
#: `\d{1,3}(?:,\d{3})+` requires full three-digit groups, so a genuine list like
#: "1, 234" (with the space) still reads as two numbers. "1,234" with no space
#: is read as one, which is the right call when the alternative is the above.
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _as_float(token: str) -> float:
    """A matched token as a number, with any thousands separators removed."""
    return float(token.replace(",", ""))


@dataclass(frozen=True, slots=True)
class NumericAgreement:
    """Numbers in the answer must agree within a relative tolerance.

    Catches the failure mode where a candidate writes fluent, plausible prose
    around a wrong figure — which an LLM judge frequently waves through.
    """

    tolerance: float = 0.01
    name: str = "numeric-agreement"
    tier: Tier = Tier.TASK

    def grade(self, item: EvalItem) -> Score:
        want = [_as_float(x) for x in _NUM.findall(item.baseline)]
        if not want:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline has no numbers")
        got = {_as_float(x) for x in _NUM.findall(item.candidate)}
        missing = [w for w in want if not any(_close(w, g, self.tolerance) for g in got)]
        if missing:
            return Score(self.name, self.tier, Outcome.FAIL, f"missing or altered: {missing[:5]}")
        return Score(self.name, self.tier, Outcome.PASS)


def _close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= tol


_DIFF_FILE = re.compile(r"^\+\+\+ [ab]/(.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class DiffFiles:
    """A code diff must touch the same files.

    The long-context refactor failure in practice: the candidate produces a
    perfectly valid diff that silently omits one of the files that needed changing.
    """

    name: str = "diff-files"
    tier: Tier = Tier.TASK

    def grade(self, item: EvalItem) -> Score:
        want = set(_DIFF_FILE.findall(item.baseline))
        if not want:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not a diff")
        got = set(_DIFF_FILE.findall(item.candidate))
        if missing := want - got:
            return Score(
                self.name, self.tier, Outcome.FAIL, f"files not modified: {sorted(missing)}"
            )
        return Score(self.name, self.tier, Outcome.PASS)


#: The deterministic stack, cheapest first. Run in order; each is independent.
DEFAULT_GRADERS: tuple[Grader, ...] = (
    NonEmpty(),
    JsonValid(),
    JsonShape(),
    ToolChoice(),
    ToolArgs(),
    ToolArgsSchema(),
    ToolLeak(),
    ExactMatch(),
    NumericAgreement(),
    DiffFiles(),
)


@dataclass(frozen=True, slots=True)
class ItemResult:
    """Every grader's verdict on one item."""

    item_id: str
    cluster: str
    scores: tuple[Score, ...]

    @property
    def applicable(self) -> tuple[Score, ...]:
        return tuple(s for s in self.scores if s.applicable)

    @property
    def graded(self) -> bool:
        """Whether any grader had an opinion at all."""
        return bool(self.applicable)

    @property
    def passed(self) -> bool:
        """Every applicable grader passed.

        Conjunctive on purpose: a response that is valid JSON with the wrong keys
        has not "mostly passed".
        """
        return self.graded and all(s.passed for s in self.applicable)

    @property
    def failures(self) -> tuple[Score, ...]:
        return tuple(s for s in self.applicable if not s.passed)


def grade(item: EvalItem, graders: tuple[Grader, ...] = DEFAULT_GRADERS) -> ItemResult:
    """Run the deterministic stack over one item."""
    return ItemResult(
        item_id=item.item_id,
        cluster=item.cluster,
        scores=tuple(g.grade(item) for g in graders),
    )


def demo() -> None:
    def item(base: str, cand: str, **kw) -> EvalItem:
        return EvalItem("i1", "c1", "prompt", base, cand, **kw)

    # An identical answer passes everything that applies.
    same = grade(item('{"a": 1, "b": 2}', '{"a": 1, "b": 2}'))
    assert same.passed and same.graded

    # Right values, wrong keys — callers index by key, so this is a failure.
    shape = grade(item('{"a": 1}', '{"b": 1}'))
    assert not shape.passed
    assert any(s.grader == "json-shape" for s in shape.failures)

    # Prose where JSON was required.
    broke = grade(item('{"a": 1}', "Sure! Here's the answer: a is 1."))
    assert not broke.passed
    assert any(s.grader == "json-valid" for s in broke.failures)

    # THE critical invariant: nothing applicable must not read as success.
    nothing = grade(item("", ""))
    assert not nothing.graded, "empty baseline should leave nothing to grade"
    assert not nothing.passed, "ungraded must never count as passed"

    # Fluent prose around a wrong number is caught deterministically.
    num = grade(item("Revenue was 42.5 million.", "Revenue came to about 51.9 million."))
    assert not num.passed
    assert any(s.grader == "numeric-agreement" for s in num.failures)
    assert grade(item("Revenue was 42.5m.", "It was 42.5m, roughly.")).passed

    # Tool choice and order.
    def calls(*ns):
        return tuple({"name": n, "arguments": {"path": "x"}} for n in ns)

    assert grade(
        item(
            "ok",
            "ok",
            baseline_tool_calls=calls("read", "write"),
            candidate_tool_calls=calls("read", "write"),
        )
    ).passed
    wrong_order = grade(
        item(
            "ok",
            "ok",
            baseline_tool_calls=calls("read", "write"),
            candidate_tool_calls=calls("write", "read"),
        )
    )
    assert not wrong_order.passed, "tool order is part of the plan"

    # A dropped argument fails; a different value does not.
    dropped = grade(
        item(
            "ok",
            "ok",
            baseline_tool_calls=({"name": "w", "arguments": {"path": "a", "mode": "x"}},),
            candidate_tool_calls=({"name": "w", "arguments": {"path": "b"}},),
        )
    )
    assert any(s.grader == "tool-args" for s in dropped.failures)

    # A diff that silently drops a file.
    d1 = "--- a/x.py\n+++ b/x.py\n@@\n-1\n+2\n--- a/y.py\n+++ b/y.py\n@@\n-3\n+4\n"
    d2 = "--- a/x.py\n+++ b/x.py\n@@\n-1\n+2\n"
    dropped_file = grade(item(d1, d2))
    assert not dropped_file.passed
    assert any("y.py" in s.detail for s in dropped_file.failures)

    # Classification uses exact match; prose does not.
    assert grade(item("spam", "spam")).passed
    assert not grade(item("spam", "ham")).passed
    prose = grade(item("The meeting is on Tuesday at three.", "It's Tuesday at 3."))
    assert all(s.grader != "exact-match" for s in prose.applicable), (
        "exact match must not fire on prose — it would fail every valid paraphrase"
    )

    print(f"graders: {len(DEFAULT_GRADERS)} deterministic, tiers 1-2")
    print("ok")


if __name__ == "__main__":
    demo()
