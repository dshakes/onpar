# Learn inference on your own numbers

Why does a model fit or not, and what limits its speed, on the machine you are
sitting at?

`onpar learn` teaches LLM inference with figures computed for your hardware, not
for a textbook machine. Each figure comes from onpar's own code: hardware
detection, the catalogue, the fit solver, the planner, the engine adapters and
the cutover gate. Where a lesson compares machines, it solves on named hardware
profiles. No number is hard-coded, and a roofline figure is labelled a roofline
estimate, as it is in `onpar fit`.

## The run

<!-- run: learn-1 -->

Every number in that lesson is this machine's: 128 GB installed, 96 GB counted
as usable, and the weight arithmetic for a real catalogue model. Run it on
another machine and the lesson changes with it.

## Five tracks, 36 lessons

`onpar learn` with no track prints this list.

**basics**, for anyone: how a model fits and what limits it.

1. What a model is in bytes (weights)
2. What context costs (KV cache)
3. Weights + KV + overhead vs usable memory
4. Why tok/s is memory bandwidth (roofline estimate)
5. What quantisation buys and costs
6. Why MoE is sized on total params (the dangerous error)
7. What 'proven on par' means (Wilson interval)

**intermediate**, for engineers: GQA, MLA, concurrency, speculative decoding.

1. GQA vs MHA and kv_heads
2. MLA and kv_lora_rank (the 64× overestimate)
3. Concurrency and the KV budget
4. Prefix caching and agent loops
5. Speculative decoding and high-batch limits
6. Chunked prefill / TTFT vs ITL budgets
7. Tool calling as a server problem
8. Teacher-forced step replay and judge intervals

**hardware**, for anyone choosing a machine: memory, bandwidth, what each box
runs.

1. Unified memory and VRAM are different budgets
2. Bandwidth is the ceiling on tokens per second
3. Usable memory is less than installed memory
4. What a 24 GB to 1 TB box can run
5. A TPU is sized per host, not per chip
6. A bigger model at q4 or a smaller one at q8?
7. What CPU-only really gets you

**serving**, for engineers choosing a stack: engines, settings, forks, native
config.

1. The engines onpar can configure, and what each is for
2. What onpar picks on this machine, and why
3. The settings, and the forks between engines
4. The config onpar writes runs without onpar
5. Tool calling is decided by the server
6. Continuous batching spends KV, not weights
7. What an engine cannot express

**hosting**, for teams deciding where it runs: rented, box, Kubernetes, cost,
cutover.

1. Run it here, rented, or managed
2. One box, tuned per host class, re-solved on arrival
3. On Kubernetes the operator sizes before it applies
4. The saving is a range, never a point
5. The gateway is in your path only while you migrate
6. Nothing leaves your machine, and nothing raw hits disk
7. The cutover ladder, and who may climb it

## The knobs that matter

- `onpar learn basics` walks one track. `onpar learn basics 3` opens one lesson.
- `--model MODEL_ID` uses another catalogue model as the worked example.
- `--no-quiz` shows each answer without asking the question first.
- `--json` returns lessons, figures, questions and answers for a program.
- Over MCP, `onpar_learn` returns the same lessons to an agent. It is read-only:
  nothing is configured or started.

The [interactive lab](/onpar/lab/#edu-why) has a memory lab that sizes weights,
KV cache and overhead as you move the sliders, and the written chapters with
diagrams.

Next: [ten seconds to a fit](../start/20-ten-seconds.md), the same arithmetic as
a single command.
