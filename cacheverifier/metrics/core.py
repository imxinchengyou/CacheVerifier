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
    """

    action: Literal["hit", "miss"]
    correct: bool
    would_be_correct: bool | None = None
    verifier_invoked: bool = False
    verifier_latency_ms: float = 0.0


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
