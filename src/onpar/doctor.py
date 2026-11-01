"""`onpar doctor` — what is installed, what each stage needs, and what to do next.

The product is three distributions and seven stages, and the first-run audit
found that a new user discovers the missing pieces one failure at a time:
`observe` says "no gateway binary", then `distill` says "no extension", then
`prove` says "no eval set". Each message is good. Meeting them in sequence is
not. This runs every check up front, in the order the loop runs, and ends with
the one command that moves the user forward.

Read-only. It touches nothing, installs nothing, and reaches no network — it
reports what is here. The fix lines are the same sentences the stages print
when they refuse, so a user who skips the doctor meets the same words later.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import __version__

__all__ = ["Check", "Report", "run"]

#: Serving engines this machine might have, as a Python package or a binary.
#: `onpar run` picks among these; the doctor only says which are present.
ENGINES: tuple[tuple[str, str, str], ...] = (
    ("vllm", "module", "pip install vllm            (CUDA / ROCm)"),
    ("sglang", "module", "pip install sglang          (CUDA)"),
    ("mlx_lm", "module", "pip install mlx-lm          (Apple silicon)"),
    ("llama-server", "binary", "brew install llama.cpp      (any)"),
    ("ollama", "binary", "https://ollama.com/download (any)"),
)


@dataclass(frozen=True, slots=True)
class Check:
    """One thing looked at. `ok` is tri-state: None means 'optional and absent'."""

    name: str
    ok: bool | None
    detail: str
    fix: str = ""
    #: Which stage needs it. Empty for things every stage shares.
    stage: str = ""

    def render(self) -> str:
        mark = {True: "✓", False: "✗", None: "·"}[self.ok]
        line = f"  {mark} {self.name:<22}{self.detail}"
        if self.fix and self.ok is not True:
            line += f"\n    {'':<24}fix: {self.fix}"
        return line


@dataclass(frozen=True, slots=True)
class Report:
    checks: tuple[Check, ...]
    #: The one thing to run now.
    next_step: str

    @property
    def blocking(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.ok is False)

    def render(self) -> str:
        out = [f"  onpar {__version__} · what this machine can do", ""]
        out += [c.render() for c in self.checks]
        out += ["", f"  next: {self.next_step}"]
        return "\n".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "checks": [asdict(c) for c in self.checks],
            "blocking": [c.name for c in self.blocking],
            "next": self.next_step,
        }


def _present(name: str, kind: str) -> bool:
    if kind == "binary":
        return shutil.which(name) is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run(home: Path | None = None) -> Report:
    """Every check, in loop order, against this machine and `home`."""
    from . import core, hardware, observe

    home = home or observe.state_dir()
    checks: list[Check] = []

    hw = hardware.detect()
    checks.append(
        Check(
            "hardware",
            True,
            f"{hw.name} · {hw.total_gb:.0f} GB, {hw.usable_gb:.0f} GB usable",
            stage="fit",
        )
    )

    # The three distributions, in the order a user meets them.
    gateway = _gateway(observe)
    checks.append(
        Check(
            "capture gateway",
            # Optional until the loop reaches `observe`: absent is a gap with a
            # fix, not a broken machine, so it never makes `doctor` exit 1.
            True if gateway is not None else None,
            str(gateway) if gateway else "onpar-gateway binary not found",
            fix="pip install onpar-gateway",
            stage="observe",
        )
    )
    ext = core.available()
    checks.append(
        Check(
            "compiled extension",
            True if ext else None,
            f"_onpar_core {core.version()}" if ext else (core.why_unavailable() or "absent"),
            fix=core.INSTALL_HINT,
            stage="distill",
        )
    )

    engines = [n for n, kind, _ in ENGINES if _present(n, kind)]
    checks.append(
        Check(
            "serving engines",
            bool(engines) or None,
            ", ".join(engines)
            if engines
            else "none found — `onpar fit` still works, `onpar run` needs one",
            fix="one of: " + "; ".join(" ".join(fix.split()) for _, _, fix in ENGINES),
            stage="run",
        )
    )

    # State on disk: what the loop has produced so far.
    log, key = home / "captures.log", home / "capture.key"
    captured = log.exists() and log.stat().st_size > 0
    checks.append(
        Check(
            "captured traffic",
            True if captured else None,
            f"{log} · {log.stat().st_size:,} bytes" if captured else f"nothing at {log}",
            fix="onpar observe --upstream https://api.openai.com/v1",
            stage="observe",
        )
    )
    if captured and not key.exists():
        checks.append(
            Check(
                "capture key",
                False,
                f"{log} exists but {key} does not — the log cannot be read",
                fix="restore the key the gateway wrote; without it the captures are unreadable",
                stage="distill",
            )
        )
    evalset = Path("evalset.json")
    checks.append(
        Check(
            "eval set",
            True if evalset.exists() else None,
            str(evalset.resolve()) if evalset.exists() else "no evalset.json here",
            fix="onpar distill",
            stage="prove",
        )
    )

    return Report(tuple(checks), _next(checks, captured=captured, evalset=evalset.exists()))


def _gateway(observe: Any) -> Path | None:
    try:
        return observe.find_gateway()
    except FileNotFoundError:
        # An ONPAR_GATEWAY_BIN that points nowhere. Report absence; `observe`
        # itself refuses loudly, which is the right place for the error.
        return None


def _next(checks: list[Check], *, captured: bool, evalset: bool) -> str:
    by = {c.name: c for c in checks}
    if evalset:
        return "onpar prove evalset.json --candidate-endpoint http://127.0.0.1:8000/v1"
    if captured and by["compiled extension"].ok is True:
        return "onpar distill"
    if captured:
        return f"{by['compiled extension'].fix}  then  onpar distill"
    if by["capture gateway"].ok is not True:
        return "onpar fit  — then `pip install onpar-gateway` to start capturing traffic"
    return "onpar observe --upstream https://api.openai.com/v1   (then point your base_url at it)"


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        r = run(Path(d))
        names = [c.name for c in r.checks]
        assert names[:3] == ["hardware", "capture gateway", "compiled extension"], names
        assert "captured traffic" in names and "eval set" in names
        assert r.next_step.startswith("onpar ")
        assert "next:" in r.render()
        d2 = r.to_dict()
        assert d2["version"] == __version__ and isinstance(d2["blocking"], list)

        # With a capture log and the extension, the next step is distill; with
        # a log and no key, that is a blocking check, not a silent empty read.
        (Path(d) / "captures.log").write_bytes(b"x" * 10)
        r2 = run(Path(d))
        assert any(c.name == "capture key" and c.ok is False for c in r2.checks)
        assert "distill" in r2.next_step

    print(r.render())
    print("ok")


if __name__ == "__main__":
    demo()
