"""Optimizer runners.

Two optimizers share one `Optimizer` protocol:

- `LocalSearchOptimizer`: a zero-dependency, deterministic offline hill-climb.
  It is the import-light default so the loop runs green with no heavy deps.
- `GepaOptimizer`: the *real* upstream GEPA optimizer
  (`projects/repos/gepa`, pinned `gepa>=0.1.1`) driven by a custom offline
  adapter (no LM, no network). This is the production-intent path.

Every optimizer records `optimizer@version` so it can be stamped onto the
candidate (AGENTS.md invariant: make the optimizer version explicit).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from . import __version__ as _mutalisk_version
from .delegation import (
    DISPATCH_POLICY,
    MERGE_RESOLUTION_TEMPLATE,
    OBJECTIVE_TEMPLATE,
    VERIFIER_SELECTION,
    DelegationExample,
    DelegationProgramRun,
    run_delegation_program,
)
from .eval_set import Example
from .program import (
    NEGATIVE,
    NEGATIVE_COMPONENT,
    POSITIVE,
    POSITIVE_COMPONENT,
    accuracy,
    classify,
    parse_cues,
    score_example,
)

METRIC_NAME = "val_accuracy"
DELEGATION_METRIC_NAME = "khala.fleet.delegation"

_WORD_RE = re.compile(r"[a-z]{3,}")
# Trivial stopwords excluded from mined cues so candidates stay meaningful.
_STOPWORDS = frozenset(
    {"the", "and", "with", "are", "this", "that", "was", "has", "have", "for"}
)


@dataclass(frozen=True)
class OptimizeResult:
    """The output of an optimizer run."""

    optimized_candidate: dict[str, str]
    metric_name: str
    metric_value: float  # optimized candidate score on the valset
    base_metric_value: float  # seed candidate score on the valset
    optimizer_id: str  # e.g. "gepa@0.1.1" or "mutalisk.local_search@0.0.1"


class Optimizer(Protocol):
    @property
    def optimizer_id(self) -> str: ...

    def optimize(
        self,
        seed_candidate: dict[str, str],
        trainset: list[Example],
        valset: list[Example],
    ) -> OptimizeResult: ...


def _mine_words(texts: list[str]) -> list[str]:
    """Deterministically rank content words by frequency then alphabetically."""
    counts: Counter[str] = Counter()
    for text in texts:
        for tok in set(_WORD_RE.findall(text.lower())):
            if tok not in _STOPWORDS:
                counts[tok] += 1
    return sorted(counts, key=lambda w: (-counts[w], w))


class LocalSearchOptimizer:
    """Deterministic, offline greedy cue-word search. Zero dependencies."""

    @property
    def optimizer_id(self) -> str:
        return f"mutalisk.local_search@{_mutalisk_version}"

    def optimize(
        self,
        seed_candidate: dict[str, str],
        trainset: list[Example],
        valset: list[Example],
    ) -> OptimizeResult:
        base = accuracy(seed_candidate, valset)
        cand = dict(seed_candidate)
        best = base

        pos_words = _mine_words([e.text for e in trainset if e.label == POSITIVE])
        neg_words = _mine_words([e.text for e in trainset if e.label == NEGATIVE])

        for component, words in (
            (POSITIVE_COMPONENT, pos_words),
            (NEGATIVE_COMPONENT, neg_words),
        ):
            for word in words:
                existing = parse_cues(cand.get(component, ""))
                if word in existing:
                    continue
                trial = dict(cand)
                trial[component] = ", ".join(existing + [word])
                acc = accuracy(trial, valset)
                if acc > best + 1e-9:  # strict improvement keeps candidates tight
                    cand = trial
                    best = acc

        return OptimizeResult(
            optimized_candidate=cand,
            metric_name=METRIC_NAME,
            metric_value=best,
            base_metric_value=base,
            optimizer_id=self.optimizer_id,
        )


def _delegation_step_dict(step) -> dict:
    return {
        "fallback_module": step.fallback_module,
        "module": step.module,
        "precondition": step.precondition,
        "refs": list(step.refs),
        "status": step.status,
    }


def _delegation_run_output(run: DelegationProgramRun) -> dict:
    return {
        "blocker_refs": list(run.blocker_refs),
        "missing_features": list(run.missing_features),
        "module_trace": [_delegation_step_dict(step) for step in run.trace],
        "rendered_objective": run.rendered_objective,
        "satisfied_features": list(run.satisfied_features),
        "score": run.score,
        "status": run.status,
    }


def _delegation_asi_feedback(example: DelegationExample, run: DelegationProgramRun) -> str:
    blockers = ", ".join(run.blocker_refs) if run.blocker_refs else "none"
    missing = ", ".join(run.missing_features) if run.missing_features else "none"
    modules = " -> ".join(step.module for step in run.trace)
    return "\n".join(
        [
            "Inputs:",
            f"- example_ref: {example.example_ref}",
            f"- issue: {example.issue}",
            f"- objective: {example.objective}",
            f"- repo: {example.repo}",
            f"- verify: {example.verify}",
            f"- capacity_context: {example.capacity_context}",
            "Generated Outputs:",
            f"- status: {run.status}",
            f"- score: {run.score:.3f}",
            f"- rendered_objective: {run.rendered_objective}",
            f"- module_trace: {modules}",
            "Feedback(ASI):",
            f"- failure_refs: {blockers}",
            f"- missing_features: {missing}",
            "- direction: mutate only delegation policy text; keep the module pipeline fixed.",
        ]
    )


class MutaliskOfflineAdapter:
    """GEPA adapter for Khala fleet-delegation examples.

    The adapter evaluates the deterministic offline `khala.fleet.delegate`
    mirror. It does not call an LM, network, browser, Pylon, or OpenAgents API.
    """

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        runs = [run_delegation_program(candidate, example) for example in batch]
        outputs = [_delegation_run_output(run) for run in runs]
        scores = [run.score for run in runs]
        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "Feedback(ASI)": _delegation_asi_feedback(example, run),
                    "Generated Outputs": output,
                    "Inputs": {
                        "capacity_context": example.capacity_context,
                        "eval_ref": example.eval_ref,
                        "example_ref": example.example_ref,
                        "issue": example.issue,
                        "objective": example.objective,
                        "repo": example.repo,
                        "trace_ref": example.trace_ref,
                        "verify": example.verify,
                    },
                }
                for example, run, output in zip(batch, runs, outputs)
            ]
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        trajectories = eval_batch.trajectories or []
        dataset: dict[str, list[dict]] = {}
        for component in components_to_update:
            dataset[component] = [
                {
                    "Feedback(ASI)": trajectory.get("Feedback(ASI)", ""),
                    "Generated Outputs": trajectory.get("Generated Outputs", {}),
                    "Inputs": trajectory.get("Inputs", {}),
                }
                for trajectory in trajectories
            ]
        return dataset

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        proposals: dict[str, str] = {}
        for component in components_to_update:
            current = str(candidate.get(component, "")).strip()
            feedback_text = "\n".join(
                str(record.get("Feedback(ASI)", ""))
                for record in reflective_dataset.get(component, [])
            )
            proposals[component] = _delegation_policy_proposal(
                component,
                current,
                feedback_text,
            )
        return proposals


def _append_policy_sentence(current: str, sentence: str) -> str:
    if sentence.lower() in current.lower():
        return current
    return f"{current.rstrip()} {sentence}".strip()


def _delegation_policy_proposal(component: str, current: str, feedback_text: str) -> str:
    proposed = current
    if component == OBJECTIVE_TEMPLATE:
        proposed = _append_policy_sentence(
            proposed,
            "Always cite {issue}, {repo}, {verify}, and the concrete {objective} in the worker prompt.",
        )
    if component == VERIFIER_SELECTION:
        proposed = _append_policy_sentence(
            proposed,
            "For repo work, require an explicit public verifier or fall back only to an admitted default verifier.",
        )
    if component == DISPATCH_POLICY:
        if "no_available_codex_capacity" in feedback_text:
            proposed = _append_policy_sentence(
                proposed,
                "On no_available_codex_capacity, return to advertise_capacity and republish per-account Codex capacity before retrying dispatch.",
            )
        if "stale_heartbeat" in feedback_text:
            proposed = _append_policy_sentence(
                proposed,
                "On stale_heartbeat, publish a fresh heartbeat and retry once before blocking.",
            )
        if "duplicate_active_assignment" in feedback_text:
            proposed = _append_policy_sentence(
                proposed,
                "On duplicate_active_assignment, backoff before retrying and keep the assignment count within advertised free slots.",
            )
        if "load_gated" in feedback_text:
            proposed = _append_policy_sentence(
                proposed,
                "Load gate dispatch by skipping new assignments when local machine load is too high.",
            )
    if component == MERGE_RESOLUTION_TEMPLATE:
        if "pr_conflicted" in feedback_text:
            proposed = _append_policy_sentence(
                proposed,
                "For pr_conflicted outcomes, rebase on main, preserve every sibling lane addition, run {verify}, and report unresolved blockers.",
            )
    return proposed


def _build_gepa_adapter():
    """Construct an offline GEPAAdapter (no LM). Imported lazily."""
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: F401

    class SentimentOfflineAdapter:
        """Evaluates and mutates candidates with no LM (fully offline)."""

        def evaluate(self, batch, candidate, capture_traces=False):
            outputs = [classify(candidate, ex.text) for ex in batch]
            scores = [score_example(candidate, ex) for ex in batch]
            trajectories = None
            if capture_traces:
                trajectories = [
                    {
                        "text": ex.text,
                        "true_label": ex.label,
                        "predicted": out,
                        "correct": out == ex.label,
                    }
                    for ex, out in zip(batch, outputs)
                ]
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            traj = eval_batch.trajectories or []
            dataset: dict[str, list[dict]] = {}
            for component in components_to_update:
                dataset[component] = [dict(t) for t in traj]
            return dataset

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            proposals: dict[str, str] = {}
            for component in components_to_update:
                examples = reflective_dataset.get(component, [])
                want = POSITIVE if component == POSITIVE_COMPONENT else NEGATIVE
                # Mine words from examples of this component's class that were missed.
                missed = [
                    e["text"]
                    for e in examples
                    if e.get("true_label") == want and not e.get("correct", False)
                ]
                existing = parse_cues(candidate.get(component, ""))
                new_words = [w for w in _mine_words(missed) if w not in existing]
                proposals[component] = ", ".join(existing + new_words)
            return proposals

    return SentimentOfflineAdapter()


class GepaOptimizer:
    """Real upstream GEPA, driven by an offline adapter (no LM, no network)."""

    def __init__(self, max_metric_calls: int = 60, seed: int = 0) -> None:
        self.max_metric_calls = max_metric_calls
        self.seed = seed

    @property
    def optimizer_id(self) -> str:
        from importlib.metadata import version

        return f"gepa@{version('gepa')}"

    def optimize(
        self,
        seed_candidate: dict[str, str],
        trainset: list[Example],
        valset: list[Example],
    ) -> OptimizeResult:
        import gepa

        base = accuracy(seed_candidate, valset)
        result = gepa.optimize(
            seed_candidate=dict(seed_candidate),
            trainset=list(trainset),
            valset=list(valset),
            adapter=_build_gepa_adapter(),
            max_metric_calls=self.max_metric_calls,
            seed=self.seed,
            display_progress_bar=False,
        )
        best_candidate = dict(result.best_candidate)
        # Report the metric on our own valset for an apples-to-apples comparison
        # with base; GEPA's val_aggregate_scores use the same per-example metric.
        metric_value = accuracy(best_candidate, valset)
        return OptimizeResult(
            optimized_candidate=best_candidate,
            metric_name=METRIC_NAME,
            metric_value=metric_value,
            base_metric_value=base,
            optimizer_id=self.optimizer_id,
        )
