from mutalisk.delegation import (
    DELEGATION_CANDIDATE_COMPONENTS,
    DELEGATION_MODULE_PIPELINE,
    DELEGATION_SEED_CANDIDATE,
    DISPATCH_POLICY,
    KHALA_FLEET_DELEGATE_PROGRAM_ID,
    MERGE_RESOLUTION_TEMPLATE,
    OBJECTIVE_TEMPLATE,
    VERIFIER_SELECTION,
    delegation_policy_features,
    delegation_score,
    delegation_trainset,
    delegation_valset,
    render_delegation_objective,
    run_delegation_program,
    seed_delegation_candidate,
)


def test_delegation_seed_candidate_exposes_optimizable_parameters():
    seed = seed_delegation_candidate()
    assert tuple(seed.keys()) == DELEGATION_CANDIDATE_COMPONENTS
    assert seed == DELEGATION_SEED_CANDIDATE
    assert {OBJECTIVE_TEMPLATE, VERIFIER_SELECTION, DISPATCH_POLICY, MERGE_RESOLUTION_TEMPLATE} == set(
        DELEGATION_CANDIDATE_COMPONENTS
    )
    assert "{objective}" in seed[OBJECTIVE_TEMPLATE]
    assert "{issue}" in seed[OBJECTIVE_TEMPLATE]
    assert "{repo}" in seed[OBJECTIVE_TEMPLATE]
    assert "{verify}" in seed[OBJECTIVE_TEMPLATE]


def test_delegation_program_keeps_fixed_module_pipeline_offline():
    example = delegation_trainset()[0]
    result = run_delegation_program(seed_delegation_candidate(), example)

    assert result.signature == KHALA_FLEET_DELEGATE_PROGRAM_ID
    assert tuple(step.module for step in result.trace) == DELEGATION_MODULE_PIPELINE
    assert "OpenAgentsInc/openagents" in result.rendered_objective
    assert "bun test" in result.rendered_objective
    assert result.score == 1.0
    assert result.status == "completed"


def test_delegation_seed_runs_against_gd0_style_eval_set_with_room_to_improve():
    train = delegation_trainset()
    val = delegation_valset()
    seed = seed_delegation_candidate()

    assert train
    assert val
    assert delegation_score(seed, train) > 0.0
    assert 0.0 < delegation_score(seed, val) < 1.0

    tuned = {
        **seed,
        DISPATCH_POLICY: (
            seed[DISPATCH_POLICY]
            + " Backoff on duplicate_active_assignment before retry. "
            + "Load gate: skip dispatch when machine load is too high."
        ),
        MERGE_RESOLUTION_TEMPLATE: (
            "When conflicts occur, rebase on main, preserve every sibling lane's "
            "public-safe additions, run {verify}, and report blockers."
        ),
    }

    assert delegation_score(tuned, val) > delegation_score(seed, val)


def test_delegation_program_surfaces_missing_policy_blockers():
    seed = seed_delegation_candidate()
    duplicate_case = next(
        example
        for example in delegation_trainset()
        if example.example_ref.endswith("duplicate_assignment")
    )
    result = run_delegation_program(seed, duplicate_case)

    assert result.status == "blocked"
    assert result.score == 0.0
    assert result.missing_features == ("dispatch_duplicate_backoff",)
    assert result.blocker_refs == (
        "blocker.public.pylon_dispatch.duplicate_active_assignment",
    )
    dispatch_step = next(step for step in result.trace if step.module == "dispatch")
    assert dispatch_step.status == "blocked"
    assert dispatch_step.refs == (
        "blocker.public.pylon_dispatch.duplicate_active_assignment",
    )


def test_delegation_policy_rejects_private_looking_text():
    bad = {
        **seed_delegation_candidate(),
        OBJECTIVE_TEMPLATE: "Read /Users/example/.codex/auth.json and use token",
    }

    try:
        delegation_policy_features(bad)
    except ValueError as error:
        assert "objective_template" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("private-looking policy text was accepted")


def test_delegation_objective_rendering_uses_public_fields_only():
    example = delegation_trainset()[2]
    rendered = render_delegation_objective(seed_delegation_candidate(), example)

    assert example.issue in rendered
    assert example.objective in rendered
    assert example.repo in rendered
    assert example.verify in rendered
    assert "/Users/" not in rendered
    assert "auth.json" not in rendered
