from mutalisk import eval_set
from mutalisk.optimizer import METRIC_NAME, LocalSearchOptimizer
from mutalisk.program import SEED_CANDIDATE


def test_local_search_improves_val_accuracy():
    opt = LocalSearchOptimizer()
    result = opt.optimize(dict(SEED_CANDIDATE), eval_set.trainset(), eval_set.valset())
    assert result.metric_name == METRIC_NAME
    assert result.metric_value > result.base_metric_value
    assert 0.0 <= result.metric_value <= 1.0


def test_local_search_optimizer_id_records_version():
    opt = LocalSearchOptimizer()
    assert opt.optimizer_id.startswith("mutalisk.local_search@")


def test_local_search_is_deterministic():
    a = LocalSearchOptimizer().optimize(
        dict(SEED_CANDIDATE), eval_set.trainset(), eval_set.valset()
    )
    b = LocalSearchOptimizer().optimize(
        dict(SEED_CANDIDATE), eval_set.trainset(), eval_set.valset()
    )
    assert a == b
