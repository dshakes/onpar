import type { FlowDef } from "./FlowDiagram";

// The seven-stage loop on the docs home. Every `p:` path must exist and every
// command named in the text must be a registered CLI verb: tests/test_docs_lab.py
// checks both, so a renamed module or verb fails the Python gate, not a reader.
//
// Layout: traffic on the top lane, hardware on the bottom lane, both converge on
// Prove; the receipt then feeds Migrate and Guard, and Guard points back to Prove.

export const LOOP: FlowDef = {
  id: "loop",
  title: "From captured traffic to a receipt you can re-check",
  caption: "Seven stages. Each one can say no.",
  brand: "rt",
  vb: [1190, 400],
  nodes: [
    {
      id: "app", label: "Your app", sub: "new base URL", tone: "client", x: 20, y: 40,
      detail: "Your existing client, with its base URL pointed at the gateway. The provider still answers every request.",
    },
    {
      id: "gateway", label: "Gateway", sub: "onpar-gateway · Rust", tone: "rt", x: 220, y: 40,
      detail: "Captures OpenAI chat, OpenAI Responses and Anthropic Messages as one record. Redaction runs before persistence and fails closed; XChaCha20-Poly1305 at rest; +0.07 ms added p95 against a 15 ms budget.",
      src: [{ p: "onpar-gateway/src/proxy.rs" }, { p: "onpar-gateway/src/capture/store.rs" }, { p: "onpar-gateway/tests/latency.rs" }],
    },
    {
      id: "capture", label: "Capture log", sub: "encrypted on disk", tone: "store", x: 420, y: 40,
      detail: "Redacted, encrypted records on your own disk. Zero telemetry and zero egress by default: captured traffic never leaves the machine on its own.",
      src: [{ p: "onpar-gateway/src/capture/store.rs" }],
    },
    {
      id: "distill", label: "Distill", sub: "onpar distill", tone: "rt", x: 620, y: 40,
      detail: "Clusters the log by task shape, deterministically, and samples each cluster across its response-length range with the extremes kept. Agent steps are linked into sessions when one request's messages are a prefix of the next, never by timestamps.",
      src: [{ p: "src/onpar/distill/shape.py" }, { p: "src/onpar/distill/cluster.py" }, { p: "src/onpar/distill/trajectory.py" }],
    },
    {
      id: "evalset", label: "Eval set", sub: "evalset.json", tone: "store", x: 820, y: 40,
      detail: "Items sampled from your real traffic, each cluster carrying its share of that traffic, so the verdict is weighted the way your workload is.",
      src: [{ p: "src/onpar/distill/cluster.py" }],
    },
    {
      id: "fit", label: "Fit", sub: "onpar fit", tone: "rt", x: 20, y: 300,
      detail: "Sizes every catalog model against this machine. MoE weights count total params, because every expert stays resident; KV cache uses kv_heads for GQA and kv_lora_rank for MLA.",
      src: [{ p: "src/onpar/fit.py" }, { p: "src/onpar/catalog.py" }],
    },
    {
      id: "plan", label: "Plan", sub: "onpar build", tone: "rt", x: 220, y: 300,
      detail: "Picks one of seven engines (vLLM, vLLM-TPU, SGLang, llm-d, llama.cpp, MLX, Ollama) and sets twelve knobs, each with its reason in Knob.why. It emits the engine's own command or config, which runs standalone.",
      src: [{ p: "src/onpar/plan.py" }, { p: "src/onpar/engines.py" }],
    },
    {
      id: "cand", label: "Candidate endpoint", sub: "onpar run", tone: "plain", x: 420, y: 300, w: 170,
      detail: "The open model, served by the engine the plan chose. Prove sends it the same prompts the incumbent answered.",
    },
    {
      id: "prove", label: "Prove", sub: "onpar prove", tone: "rt", x: 820, y: 170,
      detail: "Per-cluster Wilson intervals against a 90% bar, and a cell with no evidence shows ? instead of a score. Any judge is named, with its position and length bias measured; agent steps are replayed teacher-forced, tool arguments are schema-checked, tool leaks are caught and decision agreement is reported as Cohen's κ.",
      src: [
        { p: "src/onpar/prove/stats.py" }, { p: "src/onpar/prove/equivalence.py" }, { p: "src/onpar/prove/graders.py" },
        { p: "src/onpar/prove/judge.py" }, { p: "src/onpar/prove/agentic.py" }, { p: "src/onpar/prove/collect.py" },
        { p: "src/onpar/prove/replay.py" },
      ],
    },
    {
      id: "receipt", label: "Receipt", sub: "receipt.json · v3", tone: "ok", x: 1020, y: 170,
      detail: "A content-addressed local file: re-running the same eval set must reproduce its digest. What must stay on the incumbent prints first, then what is unproven, then what is proven.",
      src: [{ p: "src/onpar/prove/receipt.py" }],
    },
    {
      id: "migrate", label: "Migrate", sub: "onpar migrate", tone: "rt", x: 1020, y: 300,
      detail: "Shadow, then canary at 5%, 25% and 50%, then cut. Advancing is a proposal a human acts on; rollback is automatic; judge-only evidence never advances.",
      src: [{ p: "src/onpar/migrate.py" }, { p: "src/onpar/prove/gate.py" }],
    },
    {
      id: "guard", label: "Guard", sub: "onpar guard", tone: "rt", x: 820, y: 300,
      detail: "Three kinds of stale, kept apart: the model moved behind its name (void, now), your traffic moved into shapes never scored (void, soon), the field moved (still true). Its strongest verb is re-prove.",
      src: [{ p: "src/onpar/guard.py" }],
    },
  ],
  edges: [
    { from: "app", to: "gateway" },
    { from: "gateway", to: "capture" },
    { from: "capture", to: "distill" },
    { from: "distill", to: "evalset" },
    { from: "evalset", to: "prove" },
    { from: "fit", to: "plan" },
    { from: "plan", to: "cand" },
    { from: "cand", to: "prove", label: "candidate answers", via: [[700, 327], [700, 197]], at: 0.2 },
    { from: "prove", to: "receipt" },
    { from: "receipt", to: "migrate" },
    { from: "migrate", to: "guard" },
    { from: "guard", to: "prove", label: "re-prove", dashed: true },
  ],
  legend: [["client", "your app"], ["rt", "gateway and solver"], ["store", "stores"], ["ok", "proven"], ["deny", "refused"]],
  scenarios: [
    {
      id: "clean",
      label: "A clean migration",
      steps: [
        {
          title: "Capture your real traffic",
          note: "Run onpar observe --upstream https://api.openai.com/v1 and point your app at it. Prompts are redacted before they touch disk, and a redaction failure drops the record.",
          nodes: ["app", "gateway", "capture"], edges: ["app>gateway", "gateway>capture"],
          src: [{ p: "src/onpar/observe.py" }, { p: "onpar-gateway/src/capture/store.rs" }],
        },
        {
          title: "Distill it into an eval set",
          note: "onpar distill clusters the log by task shape and writes evalset.json. Each cluster is sampled across its response lengths, extremes kept.",
          nodes: ["capture", "distill", "evalset"], edges: ["capture>distill", "distill>evalset"],
          src: [{ p: "src/onpar/distill/cluster.py" }],
        },
        {
          title: "Fit models to your machine",
          note: "onpar fit sizes every model against your memory: MoE on total params, GQA on kv_heads, MLA on kv_lora_rank. onpar fit --explain prints the arithmetic.",
          nodes: ["fit"],
          src: [{ p: "src/onpar/fit.py" }, { p: "src/onpar/catalog.py" }],
        },
        {
          title: "Plan and serve the candidate",
          note: "onpar build picks one of seven engines and sets twelve knobs, each with its reason. onpar run serves it as the candidate endpoint.",
          nodes: ["fit", "plan", "cand"], edges: ["fit>plan", "plan>cand"],
          src: [{ p: "src/onpar/plan.py" }, { p: "src/onpar/engines.py" }],
        },
        {
          title: "Prove it on your traffic",
          note: "onpar prove evalset.json --candidate-endpoint http://localhost:8000/v1 scores every cluster with a Wilson interval against a 90% bar. The receipt lists what must stay on the incumbent first.",
          nodes: ["evalset", "cand", "prove", "receipt"], edges: ["evalset>prove", "cand>prove", "prove>receipt"],
          src: [{ p: "src/onpar/prove/stats.py" }, { p: "src/onpar/prove/receipt.py" }],
        },
        {
          title: "Migrate in shadow first",
          note: "onpar migrate walks shadow, canary 5%, 25%, 50%, then cut, and stops at each gate for a human. A judge's score alone never advances a cluster.",
          nodes: ["receipt", "migrate"], edges: ["receipt>migrate"],
          src: [{ p: "src/onpar/migrate.py" }, { p: "src/onpar/prove/gate.py" }],
        },
        {
          title: "Guard the receipt",
          note: "onpar guard receipt.json --traffic mix.json checks whether the proof still holds. A model swapped behind its name, or traffic in shapes never scored, voids it.",
          nodes: ["migrate", "guard"], edges: ["migrate>guard"],
          src: [{ p: "src/onpar/guard.py" }],
        },
      ],
    },
    {
      id: "thin",
      label: "Prove refuses: too few items",
      steps: [
        {
          title: "Fifteen real requests",
          note: "Fifteen requests pass the gateway in one real run on a laptop. The log on disk holds no readable prompt text.",
          nodes: ["app", "gateway", "capture"], edges: ["app>gateway", "gateway>capture"],
        },
        {
          title: "Two shapes, fifteen items",
          note: "onpar distill finds two task shapes and writes fifteen eval items. Their traffic shares are 80% and 20%.",
          nodes: ["capture", "distill", "evalset"], edges: ["capture>distill", "distill>evalset"],
        },
        {
          title: "A perfect score",
          note: "onpar prove grades all fifteen items, and every one matches the incumbent. On the point estimate alone, this would ship.",
          nodes: ["evalset", "cand", "prove"], edges: ["evalset>prove", "cand>prove"],
        },
        {
          title: "Movable: 0%",
          note: "Fifteen flawless items cannot clear a 90% bar: the Wilson lower bound for 15 of 15 is 15/(15+3.84), about 0.80. You need 35 flawless items in a cluster, 35/(35+3.84) ≈ 0.901, before a perfect score counts.",
          nodes: ["prove", "receipt"], edges: ["prove>receipt"], deny: true,
          src: [{ p: "src/onpar/prove/stats.py" }, { p: "src/onpar/prove/equivalence.py" }],
        },
        {
          title: "More traffic, not a lower bar",
          note: "Keep capturing until each cluster has enough items, then prove again. The receipt records the bar it was measured against, so lowering it changes the claim.",
          nodes: ["gateway", "capture", "prove"], edges: ["gateway>capture"],
          src: [{ p: "src/onpar/prove/receipt.py" }],
        },
      ],
    },
    {
      id: "void",
      label: "Guard voids the receipt",
      steps: [
        {
          title: "A receipt, a migration under way",
          note: "The receipt records the exact model fingerprints it was proven against. onpar migrate advances only as far as a human approves.",
          nodes: ["prove", "receipt", "migrate"], edges: ["prove>receipt", "receipt>migrate"],
          src: [{ p: "src/onpar/prove/receipt.py" }, { p: "src/onpar/migrate.py" }],
        },
        {
          title: "Same name, different model",
          note: "The provider updates the model behind the name you proved. onpar guard receipt.json --fingerprints now.json compares today's fingerprints with the receipt's.",
          nodes: ["migrate", "guard"], edges: ["migrate>guard"],
          src: [{ p: "src/onpar/guard.py" }],
        },
        {
          title: "Receipt VOID",
          note: "A changed fingerprint voids the receipt: not wrong, void, and urgent now. Age and new releases only warn; they never void a receipt.",
          nodes: ["guard"], edges: ["migrate>guard"], deny: true,
          src: [{ p: "src/onpar/guard.py" }],
        },
        {
          title: "Back to Prove, never to cutover",
          note: "The strongest verb the guard has is re-prove, and it never moves traffic. The next move is onpar prove on fresh traffic for a new receipt.",
          nodes: ["guard", "prove"], edges: ["guard>prove"],
          src: [{ p: "src/onpar/guard.py" }],
        },
      ],
    },
  ],
};
