# mutalisk

Mutalisk is the standalone OpenAgents **Python DSPy/GEPA offline-optimization
lane** — sibling to `hydralisk` (the Python/NVIDIA inference lane).

It exists because of the DSPy/RLM backend audit
(`openagents:docs/research/2026-06-28-dspy-rlm-python-backend-vs-effect-audit.md`),
which recommended a **hybrid**: adopt the real, fast-moving Python optimizers
(DSPy + GEPA) for **offline optimize/compile**, while the **online serving and
governance** path stays native in Effect/TypeScript on Cloudflare Workers.

## What mutalisk is

- A **non-Worker batch service** that runs DSPy programs and the GEPA optimizer
  over executed traces/evals and emits **candidate artifacts only** — improved
  prompts/modules/policies plus the evidence behind them.
- Python can't run in Workers and must never sit on the hot path. Mutalisk is
  offline/leaf compute that **produces untrusted candidates + evidence**; the
  Effect online authority (Khala/Artanis) **selects, gates, and admits** them
  through the existing Blueprint signature / evidence / receipt model.

## What mutalisk is NOT

- Not the online serving path (that's Khala on Workers/Effect).
- Not the governance authority (selection/admission/promotion stays in the
  product surface; mutalisk output is candidate evidence, never a write).
- Not Psionic (Rust ML substrate) and not hydralisk (NVIDIA inference serving).
- It does not hold product authority: pricing, credits, payout, customer
  routing, and public promises live in the product repos.

## The candidate contract (the seam)

Mutalisk's only output is a **candidate artifact**: a structured, public-safe
record `{ signature, base_module, optimized_module, metric, eval_evidence_refs,
trace_provenance }` written to a shared store (R2/object). The Effect side reads
candidates, runs its own acceptance gate, and only then promotes. Mutalisk never
mutates production state.

See `docs/ARCHITECTURE.md`.
