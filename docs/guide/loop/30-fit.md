# Fit: what runs here, and the arithmetic

Which open model can you actually serve on this hardware, at your context and
your concurrency?

`onpar fit` answers with a memory budget, not a guess. For each catalogue model
it adds three terms and compares the sum with the memory the accelerator may
use. A model that fits is ranked by size, largest first, because the biggest
model you can run is usually the best one you can run. A model that does not
fit is listed with the term that broke the budget.

Every number is one flag away from its arithmetic. `--explain` prints the
formula with this machine's values in it, so a sizing you disagree with can be
argued line by line rather than taken on trust.

## The run

<!-- run: fit -->

## The arithmetic, for one model

<!-- run: fit-explain -->

Read it top to bottom. The three memory terms are:

- **Weights**: parameters times bits per weight, divided by 8. For an MoE model
  that is **every** expert, which the line marks as `ALL experts resident`.
- **KV cache**: bytes per token, times context, times concurrency. Bytes per
  token is `2 × kv_heads × head_dim × layers × 2` for grouped-query attention.
  For MLA models it is `kv_lora_rank × layers × 2` instead.
- **Overhead**: 8% of the weights plus a 1.5 GB floor, for activations,
  workspace and allocator fragmentation.

The catalogue entry supplies every input. Check the KV line against it:
4 KV heads, a head dimension of 128 and 48 layers give the 98,304 bytes per
token shown above.

The MLA formula is why DeepSeek-family models are not oversized. DeepSeek V3
has 61 layers, 128 heads of dimension 128, and a 512-rank latent. The
grouped-query formula gives 61 × 2 × 128 × 128 × 2 = 3,997,696 bytes per token.
The latent gives 61 × 512 × 2 = 62,464. That is 64 times less cache, and a
sizing that used the first figure would buy hardware for memory the model
never touches.

<!-- run: catalog -->

A GB on these pages is 2^30 bytes, the unit `src/onpar/fit.py` computes in.

## Speed is a roofline estimate, not a measurement

Decode is bandwidth-bound. Each generated token reads the active weights once
and the whole KV cache once, so single-stream speed is effective bandwidth
divided by bytes read per token. Effective bandwidth is peak bandwidth times
`BANDWIDTH_EFFICIENCY`. Every `~tok/s` figure onpar prints is that estimate,
and every surface labels it as one.

The constants behind these figures are calibration knobs, not physics. They
live at the top of `src/onpar/fit.py`:

| Constant | Value | What it stands for |
|---|---|---|
| `OVERHEAD_FRACTION` | 0.08 | non-weight, non-KV memory, as a share of weights |
| `OVERHEAD_FLOOR` | 1.5 GB | the fixed runtime floor under that share |
| `BANDWIDTH_EFFICIENCY` | 0.72 | the share of peak bandwidth a real decode loop reaches |

`BANDWIDTH_EFFICIENCY` scales every throughput figure, and it has not yet been
checked against a measurement. `onpar measure` is the upgrade path: it times a
running endpoint, and with `--model` it prints the measured rate beside this
estimate.

The `runtime ->` line at the bottom of `fit` names the engine family this
hardware can start. On Apple silicon the CUDA engines cannot run at all, so it
names a Metal option. [Serve](40-serve.md) makes the full choice.

## When it says no

A `NOT FEASIBLE` row always says why, in one of two shapes:

- `weights alone need N GB` means no context or concurrency setting helps. You
  need a heavier quantisation, a smaller model or more memory. For MoE models
  the row adds that sparsity cuts compute, not memory.
- `weights fit (...) but KV ... puts it N GB over` means the weights fit and the
  cache does not. A shorter context or fewer concurrent requests may close it.

`--explain` on a model that does not fit prints the same breakdown and ends with
`verdict: DOES NOT FIT`, exiting 1.

## Which machine would run it

<!-- run: where -->

`where` sizes one model against every hardware profile onpar knows. `$/Mtok`
assumes the machine is busy at the requested concurrency, and the output says
how far that moves at low utilisation.

## The knobs that matter

- `--context` and `--concurrency` set the workload. The defaults are 32k and 1.
- `--explain MODEL_ID` prints the arithmetic and a verdict.
- `--json` returns the same numbers with `estimate_basis` on every throughput
  figure, for a script or an agent.
- A catalogue file in `~/.config/onpar/models.d/` adds or corrects a model
  without a release. A malformed file is an error, never a silent skip. An MLA
  entry without `kv_lora_rank` is refused rather than sized with the wrong
  formula.

Next: [serve the model you picked](40-serve.md) with `onpar run`.
