"""Offline mirror of the ``khala.fleet.delegate`` deterministic program.

The online OpenAgents implementation owns execution and authority. This module
only models the fixed module pipeline and the textual/policy parameters GEPA is
allowed to optimize. It is intentionally deterministic and offline: no LM, no
network, no production writes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
import re

KHALA_FLEET_DELEGATE_PROGRAM_ID = "khala.fleet.delegate"
KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID = "khala.fleet.delegation"
KHALA_FLEET_DELEGATION_PARAMETER_SCHEMA = (
    "openagents.khala.fleet_delegation.parameters.v0"
)

OBJECTIVE_TEMPLATE = "objective_template"
VERIFIER_SELECTION = "verifier_selection"
DISPATCH_POLICY = "dispatch_policy"
MERGE_RESOLUTION_TEMPLATE = "merge_resolution_template"

DELEGATION_CANDIDATE_COMPONENTS = (
    OBJECTIVE_TEMPLATE,
    VERIFIER_SELECTION,
    DISPATCH_POLICY,
    MERGE_RESOLUTION_TEMPLATE,
)

DELEGATION_MODULE_PIPELINE = (
    "ensure_pylon",
    "advertise_capacity",
    "select_account",
    "prepare_work",
    "dispatch",
    "verify_closeout",
)

DELEGATION_PRECONDITIONS = {
    "ensure_pylon": "pylon_online",
    "advertise_capacity": "advertised_codex_capacity",
    "select_account": "ready_account_free_slot",
    "prepare_work": "work_prepared",
    "dispatch": "dispatch_accepted",
    "verify_closeout": "closeout_verified",
}

DELEGATION_SEED_CANDIDATE: dict[str, str] = {
    OBJECTIVE_TEMPLATE: (
        "Implement public issue {issue} in {repo}. Keep changes scoped, run "
        "{verify}, commit, push to main, and report public evidence for: "
        "{objective}"
    ),
    VERIFIER_SELECTION: (
        "Prefer the issue-provided verifier. Require an explicit verifier for "
        "repo work; use the public fixture verifier only for fixture/no-repo work."
    ),
    DISPATCH_POLICY: (
        "Before dispatch, advertise per-account Codex capacity with "
        "OPENAGENTS_PYLON_CODEX_ACCOUNT_CONCURRENCY, publish a fresh heartbeat, "
        "select a ready account with a free slot, and retry stale heartbeat or "
        "no_available_codex_capacity by returning to advertise_capacity."
    ),
    MERGE_RESOLUTION_TEMPLATE: (
        "If a merge conflict appears, rerun {verify} and report unresolved blockers."
    ),
}

FeatureName = Literal[
    "dispatch_duplicate_backoff",
    "dispatch_load_gate",
    "dispatch_no_capacity_advertise",
    "dispatch_per_account_capacity",
    "dispatch_stale_heartbeat_retry",
    "merge_rebase_preserve_verify",
    "objective_issue_repo_verify",
    "verifier_required_for_repo",
]

FEATURE_BLOCKERS: dict[FeatureName, str] = {
    "dispatch_duplicate_backoff": "blocker.public.pylon_dispatch.duplicate_active_assignment",
    "dispatch_load_gate": "blocker.public.khala_fleet_delegate.load_gated",
    "dispatch_no_capacity_advertise": (
        "blocker.public.pylon_dispatch.no_available_codex_capacity"
    ),
    "dispatch_per_account_capacity": (
        "blocker.public.pylon_dispatch.no_available_codex_capacity"
    ),
    "dispatch_stale_heartbeat_retry": "blocker.public.pylon_dispatch.stale_heartbeat",
    "merge_rebase_preserve_verify": "blocker.public.khala_delegation.pr_conflicted",
    "objective_issue_repo_verify": "blocker.public.khala_delegation.objective_too_vague",
    "verifier_required_for_repo": "blocker.public.pylon_assignment.verify_failed",
}

FEATURE_MODULES: dict[FeatureName, str] = {
    "dispatch_duplicate_backoff": "dispatch",
    "dispatch_load_gate": "dispatch",
    "dispatch_no_capacity_advertise": "advertise_capacity",
    "dispatch_per_account_capacity": "advertise_capacity",
    "dispatch_stale_heartbeat_retry": "dispatch",
    "merge_rebase_preserve_verify": "verify_closeout",
    "objective_issue_repo_verify": "prepare_work",
    "verifier_required_for_repo": "prepare_work",
}

_UNSAFE_POLICY_TEXT = re.compile(
    r"(/Users/|/home/|auth\.json|bearer|credential|gh[pousr]_[A-Za-z0-9_]+|"
    r"mnemonic|private[_-]?key|provider[_-]?(grant|payload|secret|token)|"
    r"raw[_-]?(prompt|runner|trace)|secret|sk-[A-Za-z0-9_-]+|token|wallet)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DelegationExample:
    """A public-safe GD-0-style example reduced for offline optimization."""

    example_ref: str
    issue: str
    objective: str
    repo: str
    verify: str
    capacity_context: str
    required_features: tuple[FeatureName, ...]
    trace_ref: str
    eval_ref: str


@dataclass(frozen=True)
class DelegationStep:
    module: str
    precondition: str
    status: Literal["blocked", "recovered", "satisfied"]
    refs: tuple[str, ...] = ()
    fallback_module: str | None = None


@dataclass(frozen=True)
class DelegationProgramRun:
    """Deterministic offline evaluation result for one delegation example."""

    example_ref: str
    signature: str
    status: Literal["blocked", "completed"]
    score: float
    rendered_objective: str
    trace: tuple[DelegationStep, ...]
    blocker_refs: tuple[str, ...]
    satisfied_features: tuple[FeatureName, ...]
    missing_features: tuple[FeatureName, ...]


def seed_delegation_candidate() -> dict[str, str]:
    """Return a fresh copy of the optimizable delegation parameter dict."""
    return dict(DELEGATION_SEED_CANDIDATE)


def validate_delegation_candidate(candidate: dict[str, str]) -> None:
    """Fail closed on missing components or private-looking policy text."""
    missing = [key for key in DELEGATION_CANDIDATE_COMPONENTS if not candidate.get(key)]
    if missing:
        raise ValueError(f"delegation candidate missing components: {', '.join(missing)}")
    for key in DELEGATION_CANDIDATE_COMPONENTS:
        value = candidate[key].strip()
        if not value or len(value) > 8_000 or _UNSAFE_POLICY_TEXT.search(value):
            raise ValueError(f"{key} must be bounded public-safe policy text")


def render_delegation_objective(
    candidate: dict[str, str],
    example: DelegationExample,
) -> str:
    """Render the objective template using only public-safe example fields."""
    validate_delegation_candidate(candidate)
    rendered = (
        candidate[OBJECTIVE_TEMPLATE]
        .replace("{objective}", example.objective)
        .replace("{issue}", example.issue)
        .replace("{repo}", example.repo)
        .replace("{verify}", example.verify)
    )
    return " ".join(rendered.split())


def delegation_policy_features(candidate: dict[str, str]) -> frozenset[FeatureName]:
    """Detect bounded policy capabilities from candidate text.

    This is not intent routing. It is an offline audit predicate over a candidate
    artifact's own policy text, used only by Mutalisk's deterministic eval.
    """
    validate_delegation_candidate(candidate)
    objective = candidate[OBJECTIVE_TEMPLATE].lower()
    verifier = candidate[VERIFIER_SELECTION].lower()
    dispatch = candidate[DISPATCH_POLICY].lower()
    merge = candidate[MERGE_RESOLUTION_TEMPLATE].lower()

    features: set[FeatureName] = set()
    if all(token in objective for token in ("{objective}", "{issue}", "{repo}", "{verify}")):
        features.add("objective_issue_repo_verify")
    if "verify" in verifier and "repo" in verifier and (
        "require" in verifier or "explicit" in verifier
    ):
        features.add("verifier_required_for_repo")
    if "per-account" in dispatch or "account_concurrency" in dispatch:
        features.add("dispatch_per_account_capacity")
    if "no_available_codex_capacity" in dispatch and "advertise" in dispatch:
        features.add("dispatch_no_capacity_advertise")
    if "stale" in dispatch and "heartbeat" in dispatch and (
        "retry" in dispatch or "refresh" in dispatch or "fresh" in dispatch
    ):
        features.add("dispatch_stale_heartbeat_retry")
    if "duplicate" in dispatch and ("backoff" in dispatch or "back-off" in dispatch):
        features.add("dispatch_duplicate_backoff")
    if "load" in dispatch and ("gate" in dispatch or "skip" in dispatch):
        features.add("dispatch_load_gate")
    if (
        ("rebase" in merge or "merge main" in merge)
        and ("preserve" in merge or "keep" in merge or "union" in merge)
        and "verify" in merge
    ):
        features.add("merge_rebase_preserve_verify")
    return frozenset(features)


def run_delegation_program(
    candidate: dict[str, str],
    example: DelegationExample,
) -> DelegationProgramRun:
    """Run the fixed delegation pipeline over one public-safe example."""
    features = delegation_policy_features(candidate)
    missing = tuple(
        feature for feature in example.required_features if feature not in features
    )
    satisfied = tuple(
        feature for feature in example.required_features if feature in features
    )
    blocker_refs = tuple(dict.fromkeys(FEATURE_BLOCKERS[feature] for feature in missing))
    missing_modules = {FEATURE_MODULES[feature] for feature in missing}
    trace: list[DelegationStep] = []

    for module in DELEGATION_MODULE_PIPELINE:
        module_missing = tuple(
            feature for feature in missing if FEATURE_MODULES[feature] == module
        )
        if module_missing:
            fallback_module = (
                "advertise_capacity"
                if module == "dispatch"
                and any(
                    feature
                    in ("dispatch_no_capacity_advertise", "dispatch_stale_heartbeat_retry")
                    for feature in module_missing
                )
                else None
            )
            trace.append(
                DelegationStep(
                    module=module,
                    precondition=DELEGATION_PRECONDITIONS[module],
                    status="blocked",
                    refs=tuple(FEATURE_BLOCKERS[feature] for feature in module_missing),
                    fallback_module=fallback_module,
                )
            )
            continue
        status: Literal["blocked", "recovered", "satisfied"] = (
            "recovered"
            if module == "advertise_capacity"
            and (
                "0/1" in example.capacity_context
                or "no_available_codex_capacity" in example.capacity_context
            )
            and module not in missing_modules
            else "satisfied"
        )
        trace.append(
            DelegationStep(
                module=module,
                precondition=DELEGATION_PRECONDITIONS[module],
                status=status,
                refs=(),
            )
        )

    required_count = len(example.required_features)
    score = 1.0 if required_count == 0 else len(satisfied) / required_count
    return DelegationProgramRun(
        blocker_refs=blocker_refs,
        example_ref=example.example_ref,
        missing_features=missing,
        rendered_objective=render_delegation_objective(candidate, example),
        satisfied_features=satisfied,
        score=score,
        signature=KHALA_FLEET_DELEGATE_PROGRAM_ID,
        status="completed" if not missing else "blocked",
        trace=tuple(trace),
    )


def score_delegation_example(
    candidate: dict[str, str],
    example: DelegationExample,
) -> float:
    """Per-example score in [0, 1] for GEPA-style optimization."""
    return run_delegation_program(candidate, example).score


def delegation_score(
    candidate: dict[str, str],
    examples: Iterable[DelegationExample],
) -> float:
    """Mean delegation score across examples."""
    items = list(examples)
    if not items:
        return 0.0
    return sum(score_delegation_example(candidate, item) for item in items) / len(items)


_TRAIN: tuple[DelegationExample, ...] = (
    DelegationExample(
        capacity_context="cold start projected 0/1 Codex capacity",
        eval_ref="eval://openagents/khala-delegation/no-capacity-recovery",
        example_ref="delegation_example.synthetic.no_capacity_recovery",
        issue="7730",
        objective="recover from a cold no-capacity delegation start",
        repo="OpenAgentsInc/openagents",
        required_features=(
            "dispatch_per_account_capacity",
            "dispatch_no_capacity_advertise",
        ),
        trace_ref="trace://openagents/khala-delegation/no-capacity-recovery",
        verify="bun test clients/khala-code-desktop/tests/khala-codex-fleet-tools.test.ts",
    ),
    DelegationExample(
        capacity_context="heartbeat may be stale between capacity probe and dispatch",
        eval_ref="eval://openagents/khala-delegation/stale-heartbeat",
        example_ref="delegation_example.synthetic.stale_heartbeat",
        issue="7730",
        objective="refresh stale heartbeat and retry dispatch once",
        repo="OpenAgentsInc/openagents",
        required_features=("dispatch_stale_heartbeat_retry",),
        trace_ref="trace://openagents/khala-delegation/stale-heartbeat",
        verify="bun run --cwd apps/openagents.com check:deploy",
    ),
    DelegationExample(
        capacity_context="repo work with a public issue and explicit verifier",
        eval_ref="eval://openagents/khala-delegation/repo-verifier",
        example_ref="delegation_example.synthetic.repo_verifier",
        issue="7736",
        objective="implement the public issue and close it with evidence",
        repo="OpenAgentsInc/openagents",
        required_features=(
            "objective_issue_repo_verify",
            "verifier_required_for_repo",
        ),
        trace_ref="trace://openagents/khala-delegation/repo-verifier",
        verify="bun test packages/khala-tools",
    ),
    DelegationExample(
        capacity_context="duplicate active assignment can appear on sibling work",
        eval_ref="eval://openagents/khala-delegation/duplicate-assignment",
        example_ref="delegation_example.synthetic.duplicate_assignment",
        issue="7730",
        objective="retry safely when a duplicate active assignment is observed",
        repo="OpenAgentsInc/openagents",
        required_features=("dispatch_duplicate_backoff",),
        trace_ref="trace://openagents/khala-delegation/duplicate-assignment",
        verify="bun test clients/khala-code-desktop/tests/khala-codex-fleet-tools.test.ts",
    ),
    DelegationExample(
        capacity_context="machine load crosses the operator load gate",
        eval_ref="eval://openagents/khala-delegation/train-load-gate",
        example_ref="delegation_example.synthetic.train_load_gate",
        issue="7730",
        objective="avoid dispatching new work while local machine load is too high",
        repo="OpenAgentsInc/openagents",
        required_features=("dispatch_load_gate",),
        trace_ref="trace://openagents/khala-delegation/train-load-gate",
        verify="bun run --cwd apps/openagents.com check:deploy",
    ),
    DelegationExample(
        capacity_context="sibling PRs conflict on shared package exports",
        eval_ref="eval://openagents/khala-delegation/train-conflict-churn",
        example_ref="delegation_example.synthetic.train_conflict_churn",
        issue="7730",
        objective="resolve sibling lane conflicts without dropping public-safe additions",
        repo="OpenAgentsInc/openagents",
        required_features=("merge_rebase_preserve_verify",),
        trace_ref="trace://openagents/khala-delegation/train-conflict-churn",
        verify="bun run --cwd apps/openagents.com check:deploy",
    ),
)

_VAL: tuple[DelegationExample, ...] = (
    DelegationExample(
        capacity_context="machine load crosses the operator load gate",
        eval_ref="eval://openagents/khala-delegation/load-gate",
        example_ref="delegation_example.synthetic.load_gate",
        issue="7730",
        objective="avoid dispatching new work while local machine load is too high",
        repo="OpenAgentsInc/openagents",
        required_features=("dispatch_load_gate",),
        trace_ref="trace://openagents/khala-delegation/load-gate",
        verify="bun run --cwd apps/openagents.com check:deploy",
    ),
    DelegationExample(
        capacity_context="sibling PRs conflict on shared package exports",
        eval_ref="eval://openagents/khala-delegation/conflict-churn",
        example_ref="delegation_example.synthetic.conflict_churn",
        issue="7730",
        objective="resolve sibling lane conflicts without dropping public-safe additions",
        repo="OpenAgentsInc/openagents",
        required_features=("merge_rebase_preserve_verify",),
        trace_ref="trace://openagents/khala-delegation/conflict-churn",
        verify="bun run --cwd apps/openagents.com check:deploy",
    ),
    DelegationExample(
        capacity_context="fixture no-repo smoke with available ready account",
        eval_ref="eval://openagents/khala-delegation/fixture-clean",
        example_ref="delegation_example.synthetic.fixture_clean",
        issue="fixture",
        objective="run the no-spend public fixture",
        repo="OpenAgentsInc/openagents",
        required_features=(),
        trace_ref="trace://openagents/khala-delegation/fixture-clean",
        verify="fixture:codex_agent_task",
    ),
)


def delegation_trainset() -> list[DelegationExample]:
    """Return public-safe synthetic GD-0-style training examples."""
    return list(_TRAIN)


def delegation_valset() -> list[DelegationExample]:
    """Return public-safe synthetic GD-0-style validation examples."""
    return list(_VAL)
