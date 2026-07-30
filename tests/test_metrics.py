import pytest

from cacheverifier.metrics.bootstrap import bootstrap_ci
from cacheverifier.metrics.core import (
    RequestOutcome,
    confusion_counts,
    error_rate,
    false_accept_rate,
    hit_rate,
    mean_verifier_latency_ms,
)
from cacheverifier.metrics.pareto import ParetoPoint, pareto_frontier

OUTCOMES = [
    RequestOutcome(action="hit", correct=True),
    RequestOutcome(action="hit", correct=False),
    RequestOutcome(action="miss", correct=True),
    RequestOutcome(action="miss", correct=True),
]


def test_hit_rate():
    assert hit_rate(OUTCOMES) == pytest.approx(0.5)


def test_error_rate_is_global_not_conditional_on_hit():
    assert error_rate(OUTCOMES) == pytest.approx(0.25)


def test_false_accept_rate_is_conditional_on_hit():
    assert false_accept_rate(OUTCOMES) == pytest.approx(0.5)


def test_empty_outcomes_are_zero():
    assert hit_rate([]) == 0.0
    assert error_rate([]) == 0.0
    assert false_accept_rate([]) == 0.0


def test_mean_verifier_latency_only_counts_invoked():
    outcomes = [
        RequestOutcome(action="hit", correct=True, verifier_invoked=True, verifier_latency_ms=10.0),
        RequestOutcome(action="hit", correct=True, verifier_invoked=True, verifier_latency_ms=30.0),
        RequestOutcome(action="miss", correct=True, verifier_invoked=False, verifier_latency_ms=0.0),
    ]
    assert mean_verifier_latency_ms(outcomes) == pytest.approx(20.0)


def test_bootstrap_ci_contains_point_estimate():
    result = bootstrap_ci(OUTCOMES, hit_rate, n_resamples=200, seed=0)
    assert result.ci_low <= result.point_estimate <= result.ci_high


def test_confusion_counts_match_vcache_style_categories():
    outcomes = [
        RequestOutcome(action="hit", correct=True, would_be_correct=True),  # TP
        RequestOutcome(action="hit", correct=False, would_be_correct=False),  # FP
        RequestOutcome(action="miss", correct=True, would_be_correct=True),  # FN: missed a valid opportunity
        RequestOutcome(action="miss", correct=True, would_be_correct=False),  # TN: correctly rejected
        RequestOutcome(action="miss", correct=True, would_be_correct=None),  # empty cache, no neighbor
    ]
    counts = confusion_counts(outcomes)
    assert (counts.tp, counts.fp, counts.tn, counts.fn, counts.no_neighbor) == (1, 1, 1, 1, 1)
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)


def test_confusion_counts_precision_recall_are_zero_with_no_data():
    counts = confusion_counts([])
    assert counts.precision == 0.0
    assert counts.recall == 0.0


def test_pareto_frontier_drops_dominated_points():
    points = [
        ParetoPoint("A", hit_rate=0.5, error_rate=0.05),
        ParetoPoint("B", hit_rate=0.6, error_rate=0.05),  # dominates A (same error, more hits)
        ParetoPoint("C", hit_rate=0.4, error_rate=0.01),  # non-dominated tradeoff
    ]
    frontier_labels = {p.label for p in pareto_frontier(points)}
    assert frontier_labels == {"B", "C"}
