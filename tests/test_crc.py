"""Step 1 synthetic sanity tests for the CRC selector (locked protocol:
CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md). All four tests use synthetic,
hand-verifiable or i.i.d.-by-construction data -- deliberately independent of
any real dataset's state-dependence questions, which are a separate concern
(see the protocol doc's Protocol R / Protocol T split).
"""

import numpy as np
import pytest

from cacheverifier.metrics.core import crc_select_threshold, gray_zone_risk_curve

# Shared fixture for tests A/C: scores 1..10, incorrect (label=0) only for the
# three lowest scores. Hand-computed risk/crc_value table (see
# CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md Step 1 derivation):
#   threshold:   1      2      3    4..10
#   risk:        0.2    0.1    0.0  0.0
#   crc_value:   3/11   2/11   1/11 1/11   (= 10/11*risk + 1/11)
SCORES_A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
LABELS_A = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 1=correct, 0=incorrect


def test_a_continuous_scores_match_hand_computed_threshold():
    # alpha = 0.1: crc_value(1)=3/11≈0.273>0.1, crc_value(2)=2/11≈0.182>0.1,
    # crc_value(3)=1/11≈0.091<=0.1 -- first feasible threshold is 3.
    assert crc_select_threshold(SCORES_A, LABELS_A, alpha=0.10) == pytest.approx(3.0)

    # alpha = 0.2: crc_value(2)=2/11≈0.182<=0.2 -- first feasible is 2.
    assert crc_select_threshold(SCORES_A, LABELS_A, alpha=0.20) == pytest.approx(2.0)

    # alpha = 0.28: crc_value(1)=3/11≈0.273<=0.28 -- first feasible is 1 (loosest candidate).
    assert crc_select_threshold(SCORES_A, LABELS_A, alpha=0.28) == pytest.approx(1.0)


def test_b_tied_scores_use_strict_inequality_not_inclusive():
    # Two candidates tied at score=2, both incorrect. Under the correct
    # STRICT accept rule (score > lambda), threshold=2 excludes both tied
    # points -- only score=3 (correct) is accepted, risk=0. Under the old
    # (buggy) inclusive `>=` rule, threshold=2 would accept both tied points
    # too, giving risk=2/4=0.5. This test fails loudly if right-continuity
    # regresses.
    scores = np.array([1.0, 2.0, 2.0, 3.0])
    labels = np.array([1, 0, 0, 1])  # both score=2 points are incorrect

    curve = gray_zone_risk_curve(scores, labels)
    idx = np.where(curve.thresholds == 2.0)[0]
    assert len(idx) == 2  # both tied candidates appear as their own threshold row
    for i in idx:
        assert curve.risk[i] == pytest.approx(0.0), "tied score must NOT count itself as accepted (right-continuity)"


def test_c_alpha_below_nontrivial_floor_forces_lambda_max_fallback():
    # n=10 -> non-trivial floor is alpha >= 1/(n+1) = 1/11 ≈ 0.0909. At
    # alpha=0.05 (below the floor), even lambda_max (risk=0 there) fails the
    # selector's own criterion (B/(n+1) = 1/11 > 0.05), so the algorithm must
    # fall back to lambda_max = max(scores) = 10 -- not a crash, not None.
    lam = crc_select_threshold(SCORES_A, LABELS_A, alpha=0.05)
    assert lam == pytest.approx(10.0)
    assert lam == pytest.approx(float(SCORES_A.max()))


def test_d_known_monotone_distribution_repeated_trials_respect_risk_bound():
    # P(correct=1 | score=s) = s for score ~ Uniform(0,1) (label = Bernoulli(score)).
    # Closed form: R(lambda) = integral_{lambda}^{1} (1-s) ds = 0.5 - lambda + lambda^2/2.
    # Repeated calibration/test draws from this i.i.d. process (fresh draw each
    # trial, so exchangeability holds by construction -- this test is about
    # validating the SELECTOR's statistical guarantee, not about any real
    # dataset's state-dependence) should give mean realized test risk <= alpha,
    # per Theorem 1's E[L_{n+1}(lambda_hat)] <= alpha guarantee.
    rng = np.random.default_rng(0)
    alpha = 0.05
    n_cal = 300
    n_test_per_trial = 500
    n_trials = 300

    realized_losses = []
    for _ in range(n_trials):
        cal_scores = rng.uniform(0, 1, n_cal)
        cal_labels = rng.binomial(1, cal_scores)
        lam = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)

        test_scores = rng.uniform(0, 1, n_test_per_trial)
        test_labels = rng.binomial(1, test_scores)
        accepted = test_scores > lam
        incorrect = test_labels == 0
        realized_losses.extend((accepted & incorrect).astype(float))

    mean_realized_risk = float(np.mean(realized_losses))
    # Guarantee is on E[L], not a per-run bound -- pooling ~150k test points
    # across 300 independent trials keeps Monte Carlo noise well under 1%,
    # so a modest tolerance above alpha is a meaningful implementation check,
    # not just absorbing noise.
    assert mean_realized_risk <= alpha + 0.01, (
        f"mean realized risk {mean_realized_risk:.4f} exceeds target alpha={alpha} "
        f"by more than Monte Carlo tolerance -- possible selector bug"
    )
