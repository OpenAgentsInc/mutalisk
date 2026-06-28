from mutalisk import eval_set
from mutalisk.program import (
    NEGATIVE,
    POSITIVE,
    SEED_CANDIDATE,
    accuracy,
    classify,
    parse_cues,
)


def test_parse_cues_normalizes():
    assert parse_cues("Good, Great ,  ,bad") == ["good", "great", "bad"]


def test_classify_uses_candidate_components():
    cand = {"positive_cues": "great", "negative_cues": "broken"}
    assert classify(cand, "this is great work") == POSITIVE
    assert classify(cand, "this is broken") == NEGATIVE


def test_classify_ties_fail_closed_to_negative():
    cand = {"positive_cues": "great", "negative_cues": "broken"}
    # both cues present -> tie -> negative; no signal -> negative
    assert classify(cand, "great but broken") == NEGATIVE
    assert classify(cand, "nothing notable here") == NEGATIVE


def test_weak_seed_is_beatable():
    # The deliberately weak seed should not already be perfect on the val set,
    # so there is room for the optimizer to demonstrate improvement.
    assert accuracy(SEED_CANDIDATE, eval_set.valset()) < 1.0
