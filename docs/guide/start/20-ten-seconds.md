# Ten seconds to a fit

What can this machine run, before you install anything?

`onpar fit` reads the hardware it is on and sizes every model in the catalogue
at the context and concurrency you name. A model fits when its weights, its KV
cache and a runtime overhead together stay inside the memory the accelerator may
use. Everything else is listed with the reason it does not.

Nothing is downloaded and nothing is started. The answer is arithmetic over the
catalogue's published architectures, so it comes back before a model could
finish loading.

## The run

<!-- run: fit -->

## How to read the table

The first three lines describe the machine. On Apple silicon, macOS lets the GPU
wire down 75% of unified memory by default, which is the `usable for inference`
figure. The `raise to ~118 GB` line is the 92% ceiling and the `sysctl` that
reaches it. Both fractions are named constants in `src/onpar/hardware.py`. Once
you raise the limit, the next run reads it back and sizes against it.

Each row is one model at one quantisation:

- **quant** is the highest precision that fits, capped at 8-bit.
- **weights** is parameters times bits per weight.
- **kv** is the cache for every token of context, for every concurrent request.
- **total** adds a runtime overhead to those two, and **free** is what is left.
- **~tok/s** is single-stream decode speed, a roofline estimate, not a
  measurement. The footer of every run says so.
- **license** reads `OK` when the licence is clean for commercial use and `!`
  when it carries caps or restrictions you need to read first.

Two marks can follow a row. `slow` means the model fits but the roofline
estimate is under 15 tok/s, the threshold in `Fit.slow`. A `?` after the name
means the catalogue has not verified that model's architecture, so its KV figure
is an estimate. The run above has no `?` rows, because every model in it is
verified.

## Three ways to get the sizing wrong

The solver exists because the common shortcuts are wrong in ways that cost
money or crash a server:

- **Mixture of experts.** All experts must be resident, so weights count
  **total** parameters. Sparsity cuts compute per token, not memory. Look at the
  `NOT FEASIBLE` rows above: each MoE line says exactly that.
- **Grouped-query attention.** KV grows with `kv_heads`, not with attention
  heads. Counting attention heads overstates the cache.
- **Multi-head latent attention.** DeepSeek-family models store one compressed
  latent of `kv_lora_rank` per layer. For DeepSeek V3, the grouped-query formula
  overstates the cache 64 times. [Fit](../loop/30-fit.md) shows the arithmetic.

![Three ways KV cache sizing goes wrong](../../assets/sizing-three-ways.svg)

Look at the direction of each bar: the MoE error says a model fits when it does
not, while the other two make you buy hardware you did not need.

## When it says no

Every model that does not fit is printed with its reason, never dropped. A
reason names the term that broke the budget. `weights alone need 123 GB` is a
different purchase from `puts it 25 GB over`, which a shorter context or lower
concurrency may fix.

`--explain` shows the arithmetic for one model, line by line:

<!-- run: fit-explain -->

The MoE warning at the bottom is the honest part. At batch 1 the weight read is
exact. At batch 8 each sequence routes to its own experts, so the real read sits
in a range, and the throughput figures take the optimistic end.

## The inverse: what box would run it?

`onpar where` asks the question from the other side. It sizes one model against
every hardware profile in the catalogue, cheapest capable first:

<!-- run: where -->

The `WILL NOT RUN` section keeps the near misses. A machine `short by 160 MB` is
a config change away, not a different purchase.

## What is in the catalogue

<!-- run: models -->

`onpar catalog <model-id>` prints one entry in full, including the fields the
KV arithmetic reads.

## The knobs that matter

- `--context 32k` and `--concurrency 8` set the workload. The defaults are 32k
  and 1, and one request at a time is rarely what you serve.
- `--explain MODEL_ID` prints the arithmetic for one model.
- `--json` emits the same answer for a script, including the basis of every
  throughput estimate.
- `--quiet` hides the `NOT FEASIBLE` section.

Next: [your first proof, in ten minutes](30-walkthrough.md).
