from mutalisk.delegation import (
    DISPATCH_POLICY,
    MERGE_RESOLUTION_TEMPLATE,
    OBJECTIVE_TEMPLATE,
    delegation_trainset,
    seed_delegation_candidate,
)
from mutalisk.optimizer import MutaliskOfflineAdapter


def _weak_candidate():
    seed = seed_delegation_candidate()
    return {
        **seed,
        DISPATCH_POLICY: "Select a ready account and dispatch once.",
        MERGE_RESOLUTION_TEMPLATE: "Fix conflicts if any.",
        OBJECTIVE_TEMPLATE: "Do the work.",
    }


def test_delegation_offline_adapter_scores_gd0_examples_with_traces():
    adapter = MutaliskOfflineAdapter()
    batch = delegation_trainset()
    evaluation = adapter.evaluate(batch, _weak_candidate(), capture_traces=True)

    assert len(evaluation.outputs) == len(batch)
    assert len(evaluation.scores) == len(batch)
    assert evaluation.scores[0] == 0.0
    assert evaluation.trajectories is not None
    first = evaluation.trajectories[0]
    assert first["Inputs"]["example_ref"] == "delegation_example.synthetic.no_capacity_recovery"
    assert first["Generated Outputs"]["status"] == "blocked"
    assert "Feedback(ASI):" in first["Feedback(ASI)"]
    assert "blocker.public.pylon_dispatch.no_available_codex_capacity" in first["Feedback(ASI)"]


def test_delegation_reflective_dataset_contains_inputs_outputs_and_asi():
    adapter = MutaliskOfflineAdapter()
    evaluation = adapter.evaluate(delegation_trainset(), _weak_candidate(), capture_traces=True)
    reflective = adapter.make_reflective_dataset(
        _weak_candidate(),
        evaluation,
        [DISPATCH_POLICY, OBJECTIVE_TEMPLATE],
    )

    assert set(reflective) == {DISPATCH_POLICY, OBJECTIVE_TEMPLATE}
    dispatch_records = reflective[DISPATCH_POLICY]
    assert dispatch_records
    assert "Inputs" in dispatch_records[0]
    assert "Generated Outputs" in dispatch_records[0]
    assert "Feedback(ASI)" in dispatch_records[0]
    assert "no_available_codex_capacity" in dispatch_records[0]["Feedback(ASI)"]


def test_delegation_adapter_proposes_policy_text_from_failure_refs():
    adapter = MutaliskOfflineAdapter()
    weak = _weak_candidate()
    evaluation = adapter.evaluate(delegation_trainset(), weak, capture_traces=True)
    reflective = adapter.make_reflective_dataset(
        weak,
        evaluation,
        [DISPATCH_POLICY, MERGE_RESOLUTION_TEMPLATE],
    )
    proposals = adapter.propose_new_texts(
        weak,
        reflective,
        [DISPATCH_POLICY, MERGE_RESOLUTION_TEMPLATE],
    )

    assert "no_available_codex_capacity" in proposals[DISPATCH_POLICY]
    assert "advertise_capacity" in proposals[DISPATCH_POLICY]
    assert "duplicate_active_assignment" in proposals[DISPATCH_POLICY]
    assert "pr_conflicted" in proposals[MERGE_RESOLUTION_TEMPLATE]
    assert "rebase on main" in proposals[MERGE_RESOLUTION_TEMPLATE]
