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
   signatures we want to optimize (start with the Khala/Artanis operator
   programs).
2. **Trace + eval ingestion** — pull public-safe executed traces and eval
   results (provenance refs only; no raw prompts/customer data) as the
   optimizer's training/validation sets.
3. **Optimizer runners** — GEPA (reflective) and DSPy teleprompters
   (MIPRO/BootstrapFewShot). Each run records `optimizer@version`, base module,
   metric, dataset, and provenance.
4. **Candidate emitter** — validates (`Candidate.validate`, fail-closed) and
   writes candidates to the shared store (R2/object) under a public-safe schema
   agreed with the Effect side.
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
