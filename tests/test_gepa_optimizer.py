"""Heavy optimizer test, gated behind the gepa dependency.

Runs the REAL upstream GEPA optimizer fully offline (custom adapter, no LM, no
network). Skipped automatically when gepa is not installed so the rest of the
suite stays green in an import-light environment.
"""

import pytest

pytest.importorskip("gepa")

from mutalisk import eval_set  # noqa: E402
from mutalisk.delegation import (  # noqa: E402
    KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID,
    delegation_trainset,
    delegation_valset,
    seed_delegation_candidate,
)
from mutalisk.optimizer import DelegationGepaOptimizer, GepaOptimizer  # noqa: E402
from mutalisk.program import SEED_CANDIDATE  # noqa: E402


def test_gepa_runs_offline_and_emits_result():
    opt = GepaOptimizer(max_metric_calls=40, seed=0)
    result = opt.optimize(dict(SEED_CANDIDATE), eval_set.trainset(), eval_set.valset())
    assert opt.optimizer_id.startswith("gepa@")
    assert isinstance(result.optimized_candidate, dict)
    assert "positive_cues" in result.optimized_candidate
    # GEPA should not do worse than the seed on the held-out val set.
    assert result.metric_value >= result.base_metric_value


def test_delegation_gepa_improves_policy_on_held_out_examples():
    opt = DelegationGepaOptimizer(max_metric_calls=40, seed=0)
    result = opt.optimize(
        seed_delegation_candidate(),
        delegation_trainset(),
        delegation_valset(),
    )
    assert KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID == "khala.fleet.delegation"
    assert opt.optimizer_id.startswith("gepa@")
    assert result.metric_name == "khala.fleet.delegation"
    assert result.metric_value > result.base_metric_value
    assert "load" in result.optimized_candidate["dispatch_policy"].lower()
