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
   metric, dataset, and provenance. The fleet-delegation `MutaliskOfflineAdapter`
   scores GD-0-style examples through the deterministic mirror and produces GEPA
   reflective records with `Inputs`, `Generated Outputs`, and `Feedback(ASI)`
   sections carrying the GD-1 blocker refs.
4. **Candidate emitter** — validates (`Candidate.validate`, fail-closed) and
   writes candidates under the public-safe seam. `Candidate.to_manifest_summary`
   mirrors the Effect-side GD-3
   `KhalaFleetDelegationCandidateManifestSummary` field-for-field:
   `schemaVersion: psionic.probe_gepa_candidate_manifest.v1`,
   `candidateManifestRef`, `candidateRef`, `baseModuleRef`,
   `optimizedModuleRef`, `signature`, `metricName`, `metricValueBps`,
   `evalEvidenceRefs`, and `traceProvenanceRefs`. The local file emitter writes
   a detailed artifact envelope containing that summary plus the optimized
   module text; `R2CandidateEmitter` writes both the exact manifest summary and
   the detailed artifact to an R2-compatible object store and upserts the summary
   into a D1-compatible index. The `khala-fleet-delegation` CLI target emits a
   `khala.fleet.delegation` candidate with measured held-out gain over the seed
   policy.
5. **Reproducibility** — pinned deps, recorded versions, deterministic seeds
   where possible.

## Part 2 demo command

The recording path is:

```bash
uv run mutalisk-optimize demo khala-fleet-delegation \
  --dataset fixtures/khala_fleet_delegation_demo.json \
  --max-metric-calls 8 \
  --emit-openagents-summary out/khala-fleet-delegation-summary.json
```

This command is intentionally bounded. It uses only
`fixtures/khala_fleet_delegation_demo.json`, the offline
`MutaliskOfflineAdapter`, and local file outputs. It does not call an LM,
network, Pylon, OpenAgents, or a production service by default. It emits:

- a detailed candidate artifact under `out/candidates/`, with the optimized
  policy text, metric evidence, provenance, and public GD-1-style feedback
  dimensions;
- an exact `psionic.probe_gepa_candidate_manifest.v1` JSON summary at
  `out/khala-fleet-delegation-summary.json`, ready for the OpenAgents no-UI
  bridge path.

The manifest summary stays compact and authority-free:
`signature: "khala.fleet.delegation"`,
`metricName: "khala.fleet.delegation"`, integer `metricValueBps`,
`evalEvidenceRefs`, and `traceProvenanceRefs`. It does not carry raw prompts,
raw traces, local paths, private source, or credentials. The CLI prints the
bridge command that OpenAgents issue #7754 implements:

```bash
bun clients/khala-code-desktop/scripts/part2-gepa-manifest-bridge.ts \
  --summary out/khala-fleet-delegation-summary.json \
  --out out/khala-gepa-bridge-proof.json
```

## Non-goals

- No online serving, no routing, no governance/admission, no production writes,
  no payment/credit/promise logic. Those are product-surface authority.

## Effect seam

The shared **candidate schema** is now explicit on the Mutalisk side. The
manifest summary matches the Effect admission loop's
`psionic.probe_gepa_candidate_manifest.v1` Khala summary fields, while the
detailed artifact remains the offline optimizer payload. R2 and D1 are sinks
only: they store refs and public-safe proposal text. The online authority still
performs Blueprint lookup, evidence checks, Action Submission, review, and any
later admission.

The Khala fleet-delegation target is deliberately offline-only in Mutalisk. The
Python mirror can score public-safe GD-0-style examples and expose the seed
parameter dict for GEPA, but the Effect/TypeScript implementation remains the
only runtime that selects accounts, dispatches assignments, admits candidate
parameters, or writes OpenAgents state.
