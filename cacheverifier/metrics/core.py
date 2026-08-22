from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RequestOutcome:
    """One eval-stream request's outcome, as logged by
    `cacheverifier.experiments.runner.ExperimentRunner`.

    `correct` is True for every miss (falling through to the source always
    yields the ground-truth answer in this simulation) and, for a hit,
    whether the served cached answer's `equivalence_id` actually matched the
    query's.

    `would_be_correct` is the counterfactual signal vCache's own benchmark
    harness tracks even on a miss (`benchmarks/benchmark.py::update_stats`):
    whether the nearest neighbor at decide-time — regardless of whether it
    was actually served — shared the query's `equivalence_id`. `None` iff
    there was no neighbor at all (empty cache). This is what turns
    hit/miss + correctness into the same TP/FP/TN/FN confusion matrix vCache
    reports, and it's the signal Group B's online threshold learner trains
    on via `CachePolicy.observe`.

    `verifier_calls` defaults to 0/1 for every existing (K=1) policy, which
    only ever calls a verifier once per request. `cacheverifier.experiments.
    topk_sweep`'s cascade policies (Top-K retrieval direction, see memory)
    can call the verifier more than once per request (rank 1 rejected, try
    rank 2, ...), so `verifier_latency_ms` there is the SUM of every call in
    the cascade for that request, not a single call's latency —
    `mean_verifier_latency_ms` below still averages correctly over
    `verifier_invoked` requests either way.
    """

    action: Literal["hit", "miss"]
    correct: bool
    would_be_correct: bool | None = None
    verifier_invoked: bool = False
    verifier_latency_ms: float = 0.0
    verifier_calls: int = 0


def hit_rate(outcomes: list[RequestOutcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(o.action == "hit" for o in outcomes) / len(outcomes)


def error_rate(outcomes: list[RequestOutcome]) -> float:
    """Fraction of ALL requests that were served an incorrect cached answer.

    This is the headline metric plotted against `hit_rate` for the
    hit-rate/error-rate Pareto frontier (proposal Section 5.3).
    """
    if not outcomes:
        return 0.0
    return sum(o.action == "hit" and not o.correct for o in outcomes) / len(outcomes)


def false_accept_rate(outcomes: list[RequestOutcome]) -> float:
    """Fraction of HITS that were incorrect, i.e. error rate conditional on
    having served from cache at all. Undefined (returns 0.0) if there were
    no hits."""
    hits = [o for o in outcomes if o.action == "hit"]
    if not hits:
        return 0.0
    return sum(not o.correct for o in hits) / len(hits)


def mean_verifier_latency_ms(outcomes: list[RequestOutcome]) -> float:
    invoked = [o.verifier_latency_ms for o in outcomes if o.verifier_invoked]
    if not invoked:
        return 0.0
    return float(np.mean(invoked))


def mean_verifier_calls(outcomes: list[RequestOutcome]) -> float:
    """Mean number of verifier calls per verified request -- always 1.0 for
    every K=1 policy (Group C/D); only differs from 1.0 for a cascade policy
    (`cacheverifier.experiments.topk_sweep`) that may try more than one
    candidate per request."""
    invoked = [o.verifier_calls for o in outcomes if o.verifier_invoked]
    if not invoked:
        return 0.0
    return float(np.mean(invoked))


@dataclass(frozen=True)
class ConfusionCounts:
    """Same confusion-matrix categories vCache's benchmark harness computes
    (`tp_list`/`fp_list`/`tn_list`/`fn_list` in `update_stats`), treating
    "would this request have had a correct cache hit" as the thing being
    predicted by the hit/miss decision:

      tp: hit and correct                       — accepted a good candidate
      fp: hit and NOT correct                   — accepted a bad candidate
      fn: miss but the neighbor WOULD have been correct — a missed opportunity
      tn: miss and the neighbor would NOT have been correct — correctly rejected

    Requests with no neighbor at all (empty cache) contribute to neither fn
    nor tn and are reported separately in `no_neighbor`.
    """

    tp: int
    fp: int
    tn: int
    fn: int
    no_neighbor: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0


def confusion_counts(outcomes: list[RequestOutcome]) -> ConfusionCounts:
    tp = fp = tn = fn = no_neighbor = 0
    for o in outcomes:
        if o.would_be_correct is None:
            no_neighbor += 1
            continue
        if o.action == "hit":
            if o.correct:
                tp += 1
            else:
                fp += 1
        else:
            if o.would_be_correct:
                fn += 1
            else:
                tn += 1
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn, no_neighbor=no_neighbor)


@dataclass(frozen=True)
class VerifierFidelity:
    """Real false-approve / false-reject rates on the GRAY-ZONE SUBSET only
    (outcomes where `verifier_invoked` is True) — proposal Section 4.3's
    "统计其真实假通过率(false approve)与假拒绝率(false reject)". This is
    deliberately NOT `confusion_counts` over all outcomes: that mixes in
    the tau_high/tau_low shortcut branches, which never called a verifier at
    all, and would understate or overstate the verifier's own error rate
    depending on how wide the gray zone happens to be.

    false_approve_rate: of everything the verifier approved, fraction that
        was actually wrong (fp / (tp + fp)).
    false_reject_rate: of everything the verifier rejected, fraction that
        was actually right — i.e. a missed opportunity (fn / (tn + fn)).
    """

    n_verified: int
    false_approve_rate: float
    false_reject_rate: float
    counts: ConfusionCounts


@dataclass(frozen=True)
class GrayZoneRiskCurve:
    """R_GZ(lambda) = P(accept_lambda AND incorrect | gray zone), swept over
    every candidate score as a threshold -- the unconditional per-query risk
    a CRC-style calibration procedure would control, as opposed to
    `VerifierFidelity.false_approve_rate` (conditional on accept) or
    `error_rate` (unconditional but scoped to the whole pipeline, not just
    the gray zone).

    Monotone non-increasing in lambda BY CONSTRUCTION on any fixed sample:
    accept_lambda(x) = score(x) > lambda gives nested acceptance sets
    (A_lambda2 subseteq A_lambda1 for lambda2 > lambda1), so the cumulative
    false-accept count can only shrink or stay flat as lambda increases. This
    is exact arithmetic on the observed sample, not an empirical property
    that can fail from sampling noise -- there is nothing to "check" here
    the way there would be for a non-threshold decision rule.

    Accept rule is STRICT (`score > lambda`, not `>=`), deliberately, to make
    L_i(lambda) right-continuous -- the condition Conformal Risk Control's
    Theorem 1 (Angelopoulos et al., arXiv:2208.02814v4) requires for its
    selector (eq. 4) to carry the stated guarantee. With `score >= lambda`,
    a tied observation sits on the *left*-continuous side of its own jump
    (the loss's value AT lambda=score(x) equals the pre-jump/accepted state,
    while the right-hand limit is the post-jump/rejected state) -- verified
    against the paper's own multilab example, which is right-continuous only
    because ITS acceptance direction (larger lambda = more inclusive) is the
    mirror image of this one (larger lambda = stricter = less inclusive).

    `thresholds[i]` and `risk[i]` are aligned: `risk[i]` is R_GZ evaluated at
    `thresholds[i]` (accepting every gray-zone candidate whose score is
    STRICTLY GREATER than thresholds[i] -- a candidate whose score exactly
    equals thresholds[i] is NOT accepted at that threshold). `thresholds` is
    sorted ascending, and since a higher threshold is stricter (accepts a
    subset of what a lower threshold accepts), `risk` is non-increasing as
    `thresholds` increases -- i.e. `risk[0]` (loosest threshold, accept
    almost everything) is the largest value and `risk[-1]` (strictest,
    accept nothing since no score exceeds the maximum observed score) is 0.
    """

    thresholds: np.ndarray
    risk: np.ndarray
    n_gray_zone: int


def gray_zone_risk_curve(scores: np.ndarray, labels: np.ndarray) -> GrayZoneRiskCurve:
    """Compute R_GZ(lambda) for every candidate threshold in one pass.

    `scores`: verifier score per gray-zone candidate.
    `labels`: 1 if that candidate would actually be a correct cache hit, 0
    otherwise (same convention as `select_threshold`'s calibration labels).
    """
    n = len(labels)
    if n == 0:
        return GrayZoneRiskCurve(thresholds=np.array([]), risk=np.array([]), n_gray_zone=0)

    order = np.argsort(scores)  # ascending: thresholds[0] is the loosest candidate threshold, thresholds[-1] the strictest.
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    incorrect = (sorted_labels == 0).astype(np.int64)
    # accept_lambda(x) = score(x) > lambda (STRICT -- see class docstring for
    # why). A threshold value is a single number: EVERY point tied at that
    # exact score must be excluded together, not just "this array position"
    # (subtracting only the current index's own label was wrong for groups of
    # ties -- two candidates sharing a score got different risk values
    # depending on array position, caught by test_b in tests/test_crc.py).
    # `searchsorted(..., side="right")` gives, for each threshold candidate,
    # the index just past the LAST occurrence of that value -- identical for
    # every tied copy, so ties resolve to the same risk.
    suffix_fp_inclusive = np.concatenate([np.cumsum(incorrect[::-1])[::-1], [0]])  # length n+1; index n -> 0
    idx_right = np.searchsorted(sorted_scores, sorted_scores, side="right")
    suffix_fp_strict = suffix_fp_inclusive[idx_right]
    risk = suffix_fp_strict / n
    return GrayZoneRiskCurve(thresholds=sorted_scores, risk=risk, n_gray_zone=n)


def crc_select_threshold(scores: np.ndarray, labels: np.ndarray, alpha: float, loss_bound: float = 1.0) -> float | None:
    """Conformal Risk Control's selector (Angelopoulos et al., Theorem 1 / eq.
    4, arXiv:2208.02814v4):

        lambda_hat = inf{ lambda : (n/(n+1)) * R_hat_n(lambda) + B/(n+1) <= alpha }

    where R_hat_n is the empirical risk over the n calibration points and B
    bounds the per-sample loss (1.0 for our 0/1 false-reuse loss). When the
    candidate set is empty, the paper's own algorithm defines
    lambda_hat := lambda_max (the strictest candidate threshold) -- NOT a
    failure state. In this application L_i(lambda_max) = 0 for every i by
    construction (nothing can have score > max(scores)), so the guarantee
    E[L_{n+1}(lambda_hat)] <= alpha holds even along that fallback path. See
    CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md for the full derivation and the
    n_nontrivial(alpha) sample-size threshold this implies.

    Returns None only if there are zero calibration points (undefined).
    """
    n = len(labels)
    if n == 0:
        return None

    curve = gray_zone_risk_curve(scores, labels)
    crc_value = (n / (n + 1)) * curve.risk + loss_bound / (n + 1)
    feasible = np.where(crc_value <= alpha)[0]
    if len(feasible) == 0:
        return float(curve.thresholds[-1])  # lambda_max fallback, per the paper's own convention
    return float(curve.thresholds[feasible[0]])


def verifier_fidelity(outcomes: list[RequestOutcome]) -> VerifierFidelity:
    verified = [o for o in outcomes if o.verifier_invoked]
    counts = confusion_counts(verified)
    n_approved = counts.tp + counts.fp
    n_rejected = counts.tn + counts.fn
    return VerifierFidelity(
        n_verified=len(verified),
        false_approve_rate=(counts.fp / n_approved) if n_approved else 0.0,
        false_reject_rate=(counts.fn / n_rejected) if n_rejected else 0.0,
        counts=counts,
    )
