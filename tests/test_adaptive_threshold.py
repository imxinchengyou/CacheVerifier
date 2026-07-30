import random

import numpy as np
import pytest

from cacheverifier.cache.adaptive_threshold import _VARIANCE_MAP, AdaptiveThresholdPolicy
from cacheverifier.cache.store import CacheEntry, NeighborMatch
from cacheverifier.data.schema import QueryRecord

QUERY = QueryRecord(query_id="q", query="hello", answer="a", equivalence_id="c1")
ENTRY_A = CacheEntry(query_id="entry-a", query="hi", answer="a", equivalence_id="c1")


def test_rejects_invalid_target_error_rate():
    with pytest.raises(ValueError):
        AdaptiveThresholdPolicy(target_error_rate=0.0)
    with pytest.raises(ValueError):
        AdaptiveThresholdPolicy(target_error_rate=1.0)


def test_decide_on_empty_cache_is_a_miss():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1)
    decision = policy.decide(QUERY, np.zeros(1), match=None)
    assert decision.action == "miss"
    assert decision.branch == "empty_cache"


def test_decide_before_min_observations_is_a_cold_start_miss():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1, min_observations=6)
    match = NeighborMatch(entry=ENTRY_A, similarity=0.95)
    for _ in range(5):
        policy.observe(QUERY, match, would_be_correct=True, action="miss")

    decision = policy.decide(QUERY, np.zeros(1), match)
    assert decision.action == "miss"
    assert decision.branch == "cold_start"


def test_observe_only_records_on_miss():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1)
    match = NeighborMatch(entry=ENTRY_A, similarity=0.9)

    policy.observe(QUERY, match, would_be_correct=True, action="hit")
    assert policy._get_stats(ENTRY_A.query_id).observations == []

    policy.observe(QUERY, match, would_be_correct=True, action="miss")
    assert policy._get_stats(ENTRY_A.query_id).observations == [(0.9, 1)]


def test_observe_ignores_missing_match_or_unknown_correctness():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1)
    policy.observe(QUERY, None, would_be_correct=None, action="miss")
    policy.observe(QUERY, NeighborMatch(entry=ENTRY_A, similarity=0.9), would_be_correct=None, action="miss")
    assert policy._get_stats(ENTRY_A.query_id).observations == []


def test_estimate_parameters_recovers_a_sensible_threshold_from_separable_data():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1)
    similarities = np.array([0.5, 0.55, 0.6, 0.65, 0.9, 0.92, 0.95, 0.98])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    fit = policy._estimate_parameters(similarities, labels)
    assert fit is not None
    t_hat, gamma, var_t = fit
    assert 0.65 < t_hat < 0.9  # threshold should land between the two clusters
    assert gamma > 0  # higher similarity -> more likely correct
    assert var_t == _VARIANCE_MAP[len(similarities)]  # perfectly separable -> lookup table


def test_estimate_parameters_returns_none_for_single_class_labels():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1)
    similarities = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    labels = np.array([1, 1, 1, 1, 1, 1])

    assert policy._estimate_parameters(similarities, labels) is None


def test_get_var_t_falls_back_to_max_variance_map_key_beyond_table_range():
    policy = AdaptiveThresholdPolicy(target_error_rate=0.1)
    var_t = policy._get_var_t(
        perfect_separation=True, n_observations=1000, design=np.zeros((1, 2)), gamma=1.0, intercept=0.0, model=None
    )
    assert var_t == _VARIANCE_MAP[max(_VARIANCE_MAP)]


def test_decide_is_randomized_and_biased_toward_exploiting_reliable_entries():
    random.seed(0)
    policy = AdaptiveThresholdPolicy(target_error_rate=0.3, min_observations=6)
    match = NeighborMatch(entry=ENTRY_A, similarity=0.95)

    # Feed a strongly separable, reliable history: this exact entry has
    # always been correct whenever similarity was high.
    for s, correct in [(0.5, False), (0.55, False), (0.6, False), (0.9, True), (0.92, True), (0.95, True)]:
        policy.observe(QUERY, NeighborMatch(entry=ENTRY_A, similarity=s), would_be_correct=correct, action="miss")

    hits = sum(policy.decide(QUERY, np.zeros(1), match).action == "hit" for _ in range(200))
    assert hits > 100  # should exploit most of the time at a generous 0.3 error budget


def test_decide_favors_exploring_when_history_is_unreliable():
    random.seed(0)
    policy = AdaptiveThresholdPolicy(target_error_rate=0.01, min_observations=6)
    match = NeighborMatch(entry=ENTRY_A, similarity=0.7)

    # Noisy, weakly-separated history at a tight error budget.
    for s, correct in [(0.5, True), (0.55, False), (0.6, True), (0.65, False), (0.7, True), (0.75, False)]:
        policy.observe(QUERY, NeighborMatch(entry=ENTRY_A, similarity=s), would_be_correct=correct, action="miss")

    hits = sum(policy.decide(QUERY, np.zeros(1), match).action == "hit" for _ in range(200))
    assert hits < 100  # tight budget + noisy history should favor exploring
