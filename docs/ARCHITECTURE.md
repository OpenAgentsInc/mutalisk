# Mutalisk Architecture

Mutalisk is the **offline DSPy/GEPA optimizer**. It turns executed traces + eval
results into **candidate artifacts** that the Effect online authority gates.

## Tier boundary (from the DSPy/RLM audit)

```
                OFFLINE / LEAF (Python — this repo)        ONLINE AUTHORITY (Effect/Workers)
   traces+evals ─► DSPy programs ─► GEPA/teleprompt ─► Candidate ──► [Action Submission /
   (D1/R2)          (signatures)     optimizer          artifact       Blueprint signature gate]
                                                          │                    │
                                                          └── evidence ────────┘ selects/admits/promotes
```

- **Python = untrusted candidate + evidence producer.** Runs as a batch job
  (not in Workers). No production writes.
- **Effect = the authority.** Reads candidates, runs acceptance gates, admits/
  promotes via the existing evidence/receipt/Blueprint-signature model.

## Components (build-out)

1. **Program registry** — DSPy signatures/modules mirroring the Blueprint
   signatures we want to optimize. The initial registry includes the import-light
   sentiment fixture plus the GD-2 `khala.fleet.delegate` mirror: a fixed
   `ensure_pylon -> advertise_capacity -> select_account -> prepare_work ->
   dispatch -> verify_closeout` pipeline whose optimizable candidate is limited
   to `objective_template`, `verifier_selection`, `dispatch_policy`, and
   `merge_resolution_template`.
2. **Trace + eval ingestion** — load public-safe executed traces and eval
   results as sanitized JSONL (`public_text`, `label`, `split`, `trace_ref`,
   `eval_ref`). The optimizer receives the sanitized input text/labels; emitted
   candidates carry only evidence/provenance refs, dataset hash, source label,
   and record count. Raw prompts, customer data, private source, credentials,
   and wallet material are rejected before optimization.
3. **Optimizer runners** — GEPA (reflective) and DSPy teleprompters
   (MIPRO/BootstrapFewShot). Each run records `optimizer@version`, base module,
   metric, dataset, and provenance.
4. **Candidate emitter** — validates (`Candidate.validate`, fail-closed) and
   writes candidates under the public-safe schema
   `{ signature, base_module, optimized_module, metric, eval_evidence_refs,
   trace_provenance }`. The local file emitter writes the exact schema to a
   gitignored directory; an R2/object-store emitter can later reuse the same
   interface.
5. **Reproducibility** — pinned deps, recorded versions, deterministic seeds
   where possible.

## Non-goals

- No online serving, no routing, no governance/admission, no production writes,
  no payment/credit/promise logic. Those are product-surface authority.

## Open seam to coordinate with the Effect side

The shared **candidate schema** (`Candidate` here ↔ the Action Submission /
candidate-manifest shape on the Effect side, e.g. the GEPA candidate manifest
`psionic.probe_gepa_candidate_manifest.v1`) must be agreed so the online
authority can read and gate mutalisk output. That schema agreement is the first
build-out task.

The Khala fleet-delegation target is deliberately offline-only in Mutalisk. The
Python mirror can score public-safe GD-0-style examples and expose the seed
parameter dict for GEPA, but the Effect/TypeScript implementation remains the
only runtime that selects accounts, dispatches assignments, admits candidate
parameters, or writes OpenAgents state.
