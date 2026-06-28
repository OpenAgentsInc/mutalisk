# AGENTS

Mutalisk is the standalone OpenAgents Python DSPy/GEPA offline-optimization lane.

## Scope

- Use this repo for Python optimizer work: DSPy programs/signatures/modules,
  the GEPA reflective optimizer, teleprompters/compilers, eval harnesses, and
  the batch jobs that turn executed traces + evals into **candidate artifacts**.
- Keep the **online serving + governance** path out of this repo — it stays
  native in Effect/TypeScript on Cloudflare Workers (Khala/Artanis, the
  Blueprint Signature Lookup Service, the evidence-only Action Submission
  boundary). Mutalisk produces candidates + evidence; it never selects, admits,
  promotes, or writes production state.
- Keep Psionic as the Rust-native ML substrate and hydralisk as the NVIDIA
  inference-serving lane. Mutalisk may consume their evidence and produce
  optimization targets, but is neither.
- Keep OpenAgents/Khala product authority outside this repo: pricing, credits,
  payout, referral, customer routing, and public promises live in the product
  repos.

## Invariants

- Never commit secrets, raw prompts, private source, hidden reasoning traces,
  model-provider credentials, or large model artifacts/weights/checkpoints.
- Candidate artifacts emitted by mutalisk are **untrusted proposals carrying
  evidence**, never authorized writes. The Effect online authority gates them.
- Make the optimizer (DSPy/GEPA) version, base module, metric, eval dataset,
  and trace provenance explicit for every emitted candidate. Fail closed when
  provenance or metric evidence is missing.
- Do not put production data, customer prompts, or wallet material into traces
  or candidate artifacts. Keep everything public-safe.
- Pin dependencies (DSPy, GEPA) explicitly; record the optimizer version in
  every candidate.

## Workspace rules

- This is a normal standalone Git clone at the workspace root (sibling to
  `hydralisk`), intentionally not a submodule.
- Work on `main` by default; commit and push scoped changes here only.
- Read the umbrella `~/work/CLAUDE.md` for cross-repo guidance.

## Canonical context

- The decision audit that created this lane:
  `openagents:docs/research/2026-06-28-dspy-rlm-python-backend-vs-effect-audit.md`
- The RLM direction (separate but related): openagents issue #6654 + the
  backroom FRLM crate.
- The governance model candidates must satisfy: the Blueprint signature /
  evidence / receipt model in the openagents product surface.
