#!/usr/bin/env bash
# Record the real run the docs embed. Every transcript under docs/guide/_runs/
# comes from here, from this machine, on this date; the docs sync refuses to
# build a page whose recording is missing, and hand edits are overwritten by the
# next run. Stages that need inputs you have not produced yet are skipped and
# named, so the next command is always printed.
#
#   tools/story_run.sh                  # record what can be recorded
#   tools/story_run.sh --candidate http://127.0.0.1:8000/v1   # also prove against a served model
#
# Paths are rewritten to the reader's (~), never the author's.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=docs/guide/_runs
mkdir -p "$OUT"
CANDIDATE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --candidate) CANDIDATE="$2"; shift 2 ;;
    *) echo "usage: $0 [--candidate URL]" >&2; exit 2 ;;
  esac
done

machine="$(onpar fit --json 2>/dev/null | python3 -c 'import sys,json; h=json.load(sys.stdin)["hardware"]; print(h.get("name","?"), "·", round(h.get("total_bytes",0)/2**30), "GB")' 2>/dev/null || echo "this machine")"
stamp="recorded $(date -u +%Y-%m-%d) on $machine · onpar $(onpar version | head -1 | awk '{print $2}')"

record() {  # record <name> <command...>
  local name="$1"; shift
  local tmp; tmp="$(mktemp)"
  if "$@" >"$tmp" 2>&1; then :; else
    # Non-zero is part of some stories (a refusal is an answer); keep the output.
    :
  fi
  {
    # The command line too: $HOME expands before it is printed, and the reader's path is ~.
    printf '$ %s\n\n' "$*" | sed -E -e "s#/Users/[^/ ]+#~#g" -e "s#/home/[^/ ]+#~#g"
    # A progress line rewrites itself with carriage returns; keep what it last said.
    python3 -c 'import sys; print("\n".join(l.split("\r")[-1] for l in sys.stdin.read().split("\n")))' <"$tmp" \
      | sed -E -e "s#/Users/[^/ ]+#~#g" -e "s#/home/[^/ ]+#~#g"
    printf '\n# %s\n' "$stamp"
  } >"$OUT/$name.txt"
  rm -f "$tmp"
  echo "  recorded $name"
}

skipped=()
need() {  # need <name> <what to run first> <file that must exist>
  if [ ! -e "$3" ]; then skipped+=("$1: run $2 first"); return 1; fi
  return 0
}

echo "recording into $OUT ($stamp)"
record doctor        onpar doctor
record fit           onpar fit --context 32k --concurrency 8
record fit-explain   onpar fit --explain qwen3-30b-a3b --context 32k --concurrency 8
record models        onpar models
record learn-1       onpar learn basics 1 --no-quiz
record run-dry       onpar run qwen3-30b-a3b --dry-run
record build         onpar build "support assistant for 20 agents" --context 32k --concurrency 8 --tool-calling
record catalog       onpar catalog qwen3-30b-a3b
record where         onpar where qwen3-30b-a3b --context 32k
# Two refusals the loop makes before any model is served.
record build-cannot-meet onpar build "chat assistant" --context 32k --concurrency 8 --itl-ms 2
if [ -n "$CANDIDATE" ]; then
  record engine-probe python3 tools/engine_probe.py --endpoint "$CANDIDATE" --model mlx-community/Qwen3-30B-A3B-8bit
  # On a machine doing anything else this refuses to call its number a measurement.
  record measure     onpar measure --endpoint "$CANDIDATE" --served-model mlx-community/Qwen3-30B-A3B-8bit --model qwen3-30b-a3b --samples 3
fi

if need distill "onpar observe (and send traffic through it)" "$HOME/.onpar/captures.log"; then
  record distill     onpar distill --out evalset.json
fi
if need prove "onpar distill" evalset.json; then
  if [ -n "$CANDIDATE" ]; then
    # Deterministic graders only: no judge endpoint means nothing here spends money.
    # Costs are left out on purpose; the receipt then says the saving is unknown,
    # which is the true state until the owner passes figures they believe.
    record prove     onpar prove evalset.json --candidate-endpoint "$CANDIDATE" \
                       --candidate qwen3-30b-a3b --candidate-model mlx-community/Qwen3-30B-A3B-8bit \
                       --incumbent gpt-4o-mini --replay-sessions --collect-workers 4 \
                       --resume "$HOME/.onpar/story" --out receipt.json
  else
    skipped+=("prove: pass --candidate URL of a served model (onpar run <model>)")
  fi
fi
if need receipt "onpar prove --out receipt.json" receipt.json; then
  record receipt     onpar receipt receipt.json
  record guard       onpar guard receipt.json
  if [ -n "$CANDIDATE" ]; then
    # One pass of the migration loop on the captured traffic: shadow first, always.
    record migrate-step onpar migrate --candidate qwen3-30b-a3b --candidate-model mlx-community/Qwen3-30B-A3B-8bit \
                       --candidate-endpoint "$CANDIDATE" --incumbent gpt-4o-mini \
                       --resume "$HOME/.onpar/story" --step
  fi
  record migrate     onpar migrate --status
fi

# The figures the generated diagrams quote, from the run's own artifacts (never
# from the recordings' text): docs/guide/_runs/figures.json. tools/diagrams.py
# refuses to draw the loop without it.
STAMP="$stamp" python3 - <<'PY'
import json, os, re
from pathlib import Path
out = {"recorded": os.environ["STAMP"]}
fit = Path("docs/guide/_runs/fit.txt").read_text()
feasible, rest = fit.split("NOT FEASIBLE", 1) if "NOT FEASIBLE" in fit else (fit, "")
row = re.compile(r"^\s{2}[A-Z][\w .()\-]+?\s{2,}(?:q\d|fp\d+|bf16)\s", re.M)
out["feasible_models"] = len(row.findall(feasible))
out["infeasible_models"] = len([l for l in rest.splitlines() if re.match(r"^\s{2}[A-Z][\w .,()\-]+?\s{2,}weights", l)])
out["catalogue_models"] = out["feasible_models"] + out["infeasible_models"]
if Path("evalset.json").exists():
    ev = json.load(open("evalset.json"))
    items = ev["items"]
    sess = {}
    for i in items:
        if i.get("session"):
            sess[i["session"]] = sess.get(i["session"], 0) + 1
    out.update(captures=len(items), shapes=len({i["cluster"] for i in items}), items=len(items),
               sessions=sum(1 for n in sess.values() if n > 1), steps=sum(n for n in sess.values() if n > 1))
if Path("receipt.json").exists():
    doc = json.load(open("receipt.json")); body = doc.get("receipt", {})
    share = lambda xs: round(100 * sum(float(x.get("share", 0)) for x in xs)) if isinstance(xs, list) else None
    out.update(candidate=body.get("candidate"), incumbent=body.get("incumbent"),
               proven=len(body.get("proven") or []), unproven=len(body.get("unproven") or []), regret=len(body.get("regret") or []),
               movable_pct=share(body.get("proven") or []), digest=str(doc.get("digest") or body.get("digest") or "")[:12])
prove = Path("docs/guide/_runs/prove.txt").read_text() if Path("docs/guide/_runs/prove.txt").exists() else ""
if m := re.search(r"weighted verdict (\d+)% \[(\d+)[–-](\d+)\]", prove):
    out["weighted"] = {"point": int(m[1]), "low": int(m[2]), "high": int(m[3])}
if m := re.search(r"decision agreement (\d+)% \[(\d+)[–-](\d+)\] over (\d+) tool-context steps · κ=([\d.]+)", prove):
    out["agentic"] = {"agreement": int(m[1]), "low": int(m[2]), "high": int(m[3]), "steps": int(m[4]), "kappa": float(m[5])}
if m := re.search(r"^Move (\d+)% of traffic", prove, re.M):
    out["move_pct"] = int(m[1])
Path("docs/guide/_runs/figures.json").write_text(json.dumps(out, indent=2) + "\n")
print("  wrote figures.json:", json.dumps({k: v for k, v in out.items() if k != "recorded"}))
PY

if [ ${#skipped[@]} -gt 0 ]; then
  echo "not recorded yet:"; printf '  %s\n' "${skipped[@]}"
fi
echo "done: $(ls "$OUT" | wc -l | tr -d ' ') recordings"
