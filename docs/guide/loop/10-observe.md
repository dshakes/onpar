# Observe: your traffic becomes the benchmark

What do your requests actually look like, and can you keep them without
risking them?

`onpar observe` puts a small Rust gateway between your application and the
provider you pay today. You change one setting, the base URL. Every request goes
upstream unchanged, every reply comes back unchanged, and each successful
exchange is written to a local capture log. That log is the raw material for
everything after this stage: the eval set, the proof, the receipt.

The write path runs in a fixed order: redact, then encrypt, then write.
Redaction runs inside the gateway before anything touches disk, so unredacted
prompt text never reaches the log. If redaction fails, the record is dropped
rather than stored. What does reach disk is encrypted with XChaCha20-Poly1305.
The key sits beside the log, so this protects a stolen laptop or a copied
backup, not a host someone is already on. Point `ONPAR_CAPTURE_KEY` at a key
your own KMS issues if you need more.

The gateway speaks the three API shapes teams actually use: OpenAI chat
completions, OpenAI Responses and Anthropic Messages. All three land as one
record shape, so the stages after this one never ask which provider you came
from. The measured cost is +0.07 ms added p95 against a 15 ms budget. That
figure comes from `onpar-gateway/tests/latency.rs`, which also proves it can detect
delay that is really there. Nothing is sent anywhere else, and the gateway is in
your request path only until you stop it ([ADR-0015](../../adr/0015-in-the-path-only-while-migrating.md)).

## The run

<!-- run: observe -->

The banner says where the prompts will land and how long the gateway stays in
your path. Point your client's base URL at the `listening` address and leave
everything else alone.

You need two packages beyond `onpar`: `onpar-gateway` to capture, and
`onpar-core` to read captures back. If `onpar-core` is missing, `observe` warns
before the first request rather than letting you collect a log that `distill`
cannot open.

## When it says no

An upstream error is not your incumbent's answer, so it is never captured as
evidence. A 401 from the provider, a 429 or a 500 is forwarded to your
application with its status intact. The gateway logs
`upstream error: not captured as evidence` and writes nothing. Only a 2xx reply
becomes a record. Without that rule, a request sent before the API key was set
would become a cluster in your eval set whose baseline answer is empty.

## The knobs that matter

- `--upstream URL` is required: the provider you use today, for example
  `https://api.openai.com/v1`.
- `--port N` sets the listening port. The default is 8787.
- `--candidate URL` mirrors each request to a second backend, such as an open
  model you started with `onpar run`. Its replies are scored, never served.
  This is shadow mode, and it is the only thing that can later authorise moving
  traffic.
- `--capture PATH` moves the log from `~/.onpar/captures.log`.
- `--no-capture` runs the gateway as a plain proxy that records nothing.

Leave it in your path for as long as it takes to see your real mix of work.
The receipt later asks you to state that window with `--traffic-window`,
because captures carry no timestamps.

Next: [distill](20-distill.md) the log into an eval set with `onpar distill`.
