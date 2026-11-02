#!/usr/bin/env python3
"""Send a realistic week of traffic through the capture gateway, so the docs'
walkthrough is a real run and `onpar prove` has enough evidence per task.

Four task shapes plus a tool-calling agent session, the mix a small support
product sends: ticket classification, invoice line-item extraction (JSON),
one-line summaries, and arithmetic checks, with a refund agent that calls
`lookup_order` then `refund`. Prompts are generated deterministically from a
seed, so two runs produce the same corpus and the same eval set.

Counts are chosen for the proof, not for show: at a perfect score a cluster
needs 35 flawless items before its Wilson lower bound clears 90%, so each
shape sends 40 and the agent sends 12 sessions of 2 steps.

    export OPENAI_API_KEY=...            # never passed on the command line
    onpar observe --upstream https://api.openai.com/v1 --port 8788 &
    python3 tools/story_traffic.py --gateway http://127.0.0.1:8788/v1 --model gpt-4o-mini

Nothing here is onpar-specific: it is an ordinary OpenAI-SDK-shaped client
talking to the gateway's base URL, which is the whole point of the capture
design. Zero dependencies. Cost at gpt-4o-mini prices is a few cents.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CATEGORIES = ["billing", "login", "bug", "feature request", "cancellation", "shipping"]
PRODUCTS = [
    "Pro plan",
    "mobile app",
    "invoice export",
    "team workspace",
    "API key",
    "annual renewal",
]
NAMES = ["Priya", "Tomás", "Wei", "Amara", "Jonas", "Yuki", "Fatima", "Leo"]
ITEMS = [
    ("Widget A", 3, 12.50),
    ("Service hours", 2, 95.00),
    ("Shipping", 1, 8.99),
    ("Widget B", 5, 4.25),
    ("Support plan", 1, 49.00),
    ("Cable pack", 4, 6.75),
]


def ticket(rng: random.Random) -> str:
    cat = rng.choice(CATEGORIES)
    prod = rng.choice(PRODUCTS)
    name = rng.choice(NAMES)
    body = {
        "billing": f"Hi, I'm {name}. I was charged twice for the {prod} this month, can you check?",
        "login": (
            f"{name} here. The {prod} won't let me log in since yesterday, says token expired."
        ),
        "bug": f"When I open the {prod} the page goes blank after a second. Happens every time.",
        "feature request": f"It would help if the {prod} could export to CSV. Any plans?",
        "cancellation": f"Please cancel my {prod}. I no longer need it after this cycle.",
        "shipping": f"Order with the {prod} says delivered but nothing arrived. I'm {name}.",
    }[cat]
    return body


def invoice(rng: random.Random) -> str:
    lines = rng.sample(ITEMS, k=3)
    text = "\n".join(f"{n} x{q} @ ${p:.2f}" for n, q, p in lines)
    total = sum(q * p for _, q, p in lines)
    return f"INVOICE #{rng.randint(1000, 9999)}\n{text}\nTotal: ${total:.2f}"


def paragraph(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    prod = rng.choice(PRODUCTS)
    n = rng.randint(2, 9)
    return (
        f"{name} from the {prod} team reported that after the latest release {n} customers saw "
        f"slower load times on the dashboard. The team traced it to a missing database index, "
        f"added the index on Tuesday, and confirmed with the affected customers that the pages "
        f"now load in under a second. A follow-up task tracks adding a regression check."
    )


def arithmetic(rng: random.Random) -> str:
    a, b = rng.randint(12, 980), rng.randint(3, 97)
    op = rng.choice(["+", "-", "*"])
    return f"What is {a} {op} {b}? Reply with the number only."


SHAPES = {
    "ticket": (
        "You classify support tickets. Reply with exactly one label from: billing, login, bug, "
        "feature request, cancellation, shipping.",
        ticket,
    ),
    "invoice": (
        "Extract line items from the invoice as JSON: "
        '{"items":[{"name":str,"qty":int,"unit_price":number}],'
        '"total":number}. Reply with JSON only.',
        invoice,
    ),
    "summary": ("Summarise the text in one sentence of at most 20 words.", paragraph),
    "arithmetic": ("You are a calculator. Reply with the number only.", arithmetic),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund",
            "description": "Refund an order, fully or partially.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string", "enum": ["damaged", "late", "duplicate", "other"]},
                },
                "required": ["order_id", "amount", "reason"],
                "additionalProperties": False,
            },
        },
    },
]


def post(base: str, key: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def agent_session(base: str, key: str, model: str, rng: random.Random) -> int:
    """Two steps: the model looks the order up, then refunds it. Returns steps sent."""
    order = f"A{rng.randint(10000, 99999)}"
    amount = round(rng.uniform(5, 120), 2)
    reason = rng.choice(["damaged", "late", "duplicate"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a refund agent. Use the tools. Refund only after looking the order up."
            ),
        },
        {
            "role": "user",
            "content": f"Order {order} arrived {reason}. Please refund ${amount:.2f}.",
        },
    ]
    sent = 0
    for _ in range(3):
        resp = post(
            base,
            key,
            {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0,
            },
        )
        sent += 1
        msg = resp["choices"][0]["message"]
        messages.append(
            {
                k: v
                for k, v in msg.items()
                if k in ("role", "content", "tool_calls") and v is not None
            }
        )
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            if name == "lookup_order":
                result = {
                    "order_id": args.get("order_id"),
                    "status": "delivered",
                    "total": amount,
                    "customer": rng.choice(NAMES),
                }
            else:
                result = {
                    "ok": True,
                    "refunded": args.get("amount"),
                    "order_id": args.get("order_id"),
                }
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
            )
    return sent


def _dotenv_key() -> str:
    """OPENAI_API_KEY from ./.env, read in this process only and never printed.

    Keeps the key off the command line and out of shell history; the gateway
    forwards it to the upstream and does not store it.
    """
    try:
        for raw in Path(".env").read_text().splitlines():
            line = raw.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--gateway", default="http://127.0.0.1:8788/v1", help="the capture gateway's base URL"
    )
    ap.add_argument(
        "--model", default="gpt-4o-mini", help="the incumbent model the traffic goes to"
    )
    ap.add_argument(
        "--per-shape",
        type=int,
        default=40,
        help="requests per task shape (35 clears a 90%% bar at a perfect score)",
    )
    ap.add_argument(
        "--sessions", type=int, default=12, help="refund-agent sessions, two tool calls each"
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY", "") or _dotenv_key()
    if not key:
        print(
            "error: OPENAI_API_KEY is not set; export it in this shell or put it in ./.env",
            file=sys.stderr,
        )
        return 2
    rng = random.Random(args.seed)
    sent = failed = 0
    t0 = time.time()
    for shape, (system, make) in SHAPES.items():
        for i in range(args.per_shape):
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": make(rng)},
                ],
                "temperature": 0,
                "max_tokens": 200,
            }
            if shape == "invoice":
                payload["response_format"] = {"type": "json_object"}
            try:
                post(args.gateway, key, payload)
                sent += 1
            except urllib.error.HTTPError as e:
                failed += 1
                print(f"{shape} #{i}: HTTP {e.code} from the gateway; stopping", file=sys.stderr)
                return 1
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"error: cannot reach {args.gateway}: {e}", file=sys.stderr)
                return 1
        print(f"{shape:<11} {args.per_shape} sent")
    steps = 0
    for s in range(args.sessions):
        try:
            steps += agent_session(args.gateway, key, args.model, rng)
        except urllib.error.HTTPError as e:
            print(f"session {s}: HTTP {e.code}; stopping", file=sys.stderr)
            return 1
    sent += steps
    print(f"agent       {args.sessions} sessions, {steps} steps")
    print(
        f"\n{sent} requests through {args.gateway} in {time.time() - t0:.0f}s "
        f"({failed} failed). Next: onpar distill"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
