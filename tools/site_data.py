#!/usr/bin/env python3
"""Generate site/assets/fit-explorer.json from the real fit solver.

Every number written here is onpar's solver output.  The site explorer
fetches this file at page-load and renders it; it never runs the solver
itself, so the page cannot drift from what ``onpar fit`` would say.

Run: ``python3 tools/site_data.py``
     or via ``python3 tools/diagrams.py`` (called from its __main__).
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import re
import sys
from datetime import UTC, datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "assets" / "fit-explorer.json"

CONTEXTS = [8_192, 32_768, 131_072]
CONCURRENCIES = [1, 8, 32]
GB = 1024**3


def _version() -> str:
    m = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (ROOT / "pyproject.toml").read_text(),
        re.M,
    )
    if m:
        return m.group(1)
    # Fallback: try importlib.metadata (only works when the package is installed).
    try:
        import importlib.metadata

        return importlib.metadata.version("onpar")
    except Exception:
        return "unknown"


def generate() -> dict:
    """Return the full payload as a plain dict."""
    sys.path.insert(0, str(ROOT / "src"))

    from onpar import catalog as cat
    from onpar.catalog import QUANT_BITS
    from onpar.fit import rank, recommend_runtime, solve
    from onpar.hardware_catalog import PROFILES

    models = cat.load()
    model_idx: dict[str, int] = {m.id: i for i, m in enumerate(models)}

    profiles_out = [
        {
            "id": p.id,
            "name": p.name,
            "mem": p.total_memory_gb,
            # usable_gb: what the solver treats as the ceiling, so JS can
            # compute headroom without re-implementing the reservation maths.
            "usable": round(p.to_hardware().usable_bytes / GB, 1),
            "kind": p.kind,
        }
        for p in PROFILES
    ]

    models_out = [
        {
            "id": m.id,
            "name": m.name,
            "params": m.params_b,
            "lic": m.license,
            "ok": m.license_ok,
        }
        for m in models
    ]

    data: dict[str, dict] = {}
    for p in PROFILES:
        hw = p.to_hardware()
        for ctx in CONTEXTS:
            for conc in CONCURRENCIES:
                key = f"{p.id}/{ctx}/{conc}"

                # Runtime recommendation: what onpar run would actually start.
                rt_engine: str | None = None

                with contextlib.suppress(Exception):
                    rt_engine, _ = recommend_runtime(hw, ctx, conc)

                # Use rank() so the rejected-reason strings match exactly what
                # ``onpar fit`` prints.  For the bytes columns we need a second
                # solve pass on rejected models; that is one extra dict lookup,
                # not a re-implementation of the arithmetic.
                feasible, rejected = rank(hw, ctx, conc)

                rows: list[list] = []

                for f in feasible:
                    tps = round(f.tokens_per_sec) if f.tokens_per_sec else None
                    rows.append(
                        [
                            model_idx[f.model.id],
                            f.quant,
                            round(f.weight_bytes / GB, 1),
                            round(f.kv_bytes / GB, 1),
                            round(f.total_bytes / GB, 1),
                            True,
                            None,
                            tps,
                        ]
                    )

                for m, why in rejected:
                    smallest = min(m.quants, key=lambda q: QUANT_BITS[q])
                    probe = solve(m, smallest, hw, ctx, conc)
                    rows.append(
                        [
                            model_idx[m.id],
                            smallest,
                            round(probe.weight_bytes / GB, 1),
                            round(probe.kv_bytes / GB, 1),
                            round(probe.total_bytes / GB, 1),
                            False,
                            why,
                            None,
                        ]
                    )

                entry: dict = {"rows": rows}
                if rt_engine is not None:
                    entry["rt"] = rt_engine
                data[key] = entry

    return {
        "meta": {
            "version": _version(),
            "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
            # Invariant 6: every throughput figure carries its basis.
            "tps_note": "roofline estimate, not measured",
        },
        "profiles": profiles_out,
        "models": models_out,
        "contexts": CONTEXTS,
        "concurrencies": CONCURRENCIES,
        "data": data,
    }


def write(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    OUT.write_text(text)
    kb = len(text) / 1024
    print(f"  fit-explorer.json  {kb:.0f} KB  →  site/assets/")


if __name__ == "__main__":
    print("generating fit-explorer.json:")
    write(generate())
    print("done")
