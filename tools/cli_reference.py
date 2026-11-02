#!/usr/bin/env python3
"""Write the CLI reference page from the parser itself, so it cannot drift.

    python3 tools/cli_reference.py            # rewrite docs/guide/reference/cli.md
    python3 tools/cli_reference.py --check    # exit 1 if the page is stale

Every verb, every flag, every help string comes from `onpar.cli.build_parser()`.
The page is grouped the way the loop runs, and a verb the grouping does not
name lands under "Everything else" rather than being dropped: a reference that
omits a real command is the defect this script exists to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guide" / "reference" / "cli.md"

#: The loop's order, which is the order a reader meets the verbs.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Start here", ("doctor", "learn", "version", "upgrade")),
    ("Observe", ("observe",)),
    ("Distill", ("distill",)),
    (
        "Fit and plan",
        (
            "fit",
            "where",
            "models",
            "advise",
            "build",
            "catalog",
            "catalog-add",
            "catalog-sources",
            "discover",
            "watch",
        ),
    ),
    ("Serve", ("run", "host", "box", "cache", "kernel", "measure")),
    ("Prove", ("prove", "label", "receipt", "brief")),
    ("Migrate and guard", ("migrate", "guard")),
    ("Interfaces", ("ui", "desktop")),
)


def _parser() -> argparse.ArgumentParser:
    sys.path.insert(0, str(ROOT / "src"))
    from onpar import cli

    return cli.build_parser()


def _flags(sub: argparse.ArgumentParser) -> list[tuple[str, str]]:
    rows = []
    for a in sub._actions:  # noqa: SLF001 - the parser is the source of truth
        if isinstance(a, argparse._HelpAction):  # noqa: SLF001
            continue
        takes_value = a.nargs != 0 and not isinstance(
            a,
            (argparse._StoreConstAction, argparse._AppendConstAction, argparse._CountAction),  # noqa: SLF001
        )
        if a.option_strings:
            name = ", ".join(a.option_strings)
            if takes_value:
                name += f" {a.metavar or a.dest.upper()}"
        else:
            name = a.metavar or a.dest.upper()
        rows.append((name, " ".join((a.help or "").split())))
    return rows


def render() -> str:
    p = _parser()
    subs = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
    verbs = dict(subs.choices)
    helps = {a.dest: " ".join((a.help or "").split()) for a in subs._choices_actions}  # noqa: SLF001
    placed: set[str] = set()
    out = [
        "# CLI reference",
        "",
        "Generated from the parser by `tools/cli_reference.py`; `onpar --help` is the",
        "authority and a test fails when this page lags it. Verbs are in the order the",
        "loop runs. Exit codes: `0` done; `1` the call was well-formed and either the world",
        "did not cooperate (retry or report) or the answer is no (a model that does not fit, a",
        "receipt that no longer holds); `2` the invocation itself was wrong (fix it).",
        "",
    ]
    groups = list(GROUPS) + [
        (
            "Everything else",
            tuple(v for v in sorted(verbs) if v not in {x for _, vs in GROUPS for x in vs}),
        )
    ]
    for title, names in groups:
        names = tuple(n for n in names if n in verbs)
        if not names:
            continue
        out += [f"## {title}", ""]
        for name in names:
            placed.add(name)
            sub = verbs[name]
            out += [f"### `onpar {name}`", ""]
            if helps.get(name):
                out += [helps[name], ""]
            rows = _flags(sub)
            if rows:
                out += ["| flag | what it does |", "|---|---|"]
                out += [f"| `{n}` | {h} |" for n, h in rows]
                out += [""]
    missing = sorted(set(verbs) - placed)
    if missing:  # cannot happen: the catch-all group takes the rest
        raise RuntimeError(f"unplaced verbs: {missing}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if the page is stale instead of writing it"
    )
    args = ap.parse_args()
    text = render()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(
                f"{OUT.relative_to(ROOT)} is stale; run python3 tools/cli_reference.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUT.relative_to(ROOT)} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
