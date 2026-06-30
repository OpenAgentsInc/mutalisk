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

Mutalisk's only output is a **candidate artifact** plus an Effect-ingestible
manifest summary. The detailed artifact carries the public-safe optimizer payload
(`signature`, `base_module`, `optimized_module`, `metric`,
`eval_evidence_refs`, and `trace_provenance`). The manifest summary is
field-for-field with the GD-3 Khala admission seam:
`psionic.probe_gepa_candidate_manifest.v1` with `candidateManifestRef`,
`candidateRef`, `baseModuleRef`, `optimizedModuleRef`, `metricName`,
`metricValueBps`, `evalEvidenceRefs`, and `traceProvenanceRefs`.

The local emitter writes the detailed artifact envelope into the gitignored
`candidates/` directory. The R2 emitter writes two JSON objects: the exact
manifest summary and the detailed artifact, then upserts the summary row into a
D1-compatible index. The Effect side reads the manifest refs, runs its own
acceptance gate, and only then promotes. Mutalisk never mutates production state.

The current offline runner can use the in-repo synthetic fixture or an explicit
sanitized trace/eval JSONL file:

```bash
python -m mutalisk.optimize --optimizer local --trace-evals trace-evals.jsonl
python -m mutalisk.optimize --optimizer gepa --max-metric-calls 60 --trace-evals trace-evals.jsonl
python -m mutalisk.optimize --target khala-fleet-delegation --optimizer gepa --max-metric-calls 60
```

Each JSONL record must already be public-safe and contain only optimizer input
plus refs:

```json
{"public_text":"great green build","label":"positive","split":"train","trace_ref":"trace://run/1","eval_ref":"eval://run/1"}
```

Candidate JSON records the eval and trace refs, dataset hash, metric, base
module, optimized module, optimizer version, stable content hash, manifest hash,
and the GD-3 manifest summary. It does not serialize raw trace/eval records.

## Khala fleet-delegation program mirror

Mutalisk also contains the first GD-2 program target from
`OpenAgentsInc/openagents#7730`: an offline mirror of the fixed
`khala.fleet.delegate` pipeline:

```text
ensure_pylon -> advertise_capacity -> select_account -> prepare_work -> dispatch -> verify_closeout
```

The control flow is fixed. The optimizable candidate is only the public-safe
parameter dict:

```python
{
    "objective_template": "... {objective} ... {issue} ... {repo} ... {verify} ...",
    "verifier_selection": "...",
    "dispatch_policy": "...",
    "merge_resolution_template": "...",
}
```

`mutalisk.delegation` exposes the seed candidate, synthetic GD-0-style
delegation examples, and deterministic offline scoring helpers. The
`MutaliskOfflineAdapter` in `mutalisk.optimizer` evaluates those examples for
GEPA and emits reflective records with `Inputs`, `Generated Outputs`, and
`Feedback(ASI)` containing the concrete blocker refs (for example
`blocker.public.pylon_dispatch.no_available_codex_capacity`). It does not call
an LM or network API. The CLI target above runs upstream `gepa.optimize` over
that adapter and emits a frozen `khala.fleet.delegation` candidate artifact with
the base seed, optimized policy text, held-out metric gain, evidence refs, and
trace provenance refs.

See `docs/ARCHITECTURE.md`.
