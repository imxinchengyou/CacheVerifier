from dataclasses import dataclass
from typing import Callable

import numpy as np

from cacheverifier.metrics.core import RequestOutcome


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float


def bootstrap_ci(
    outcomes: list[RequestOutcome],
    metric_fn: Callable[[list[RequestOutcome]], float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for a metric over the eval-stream outcomes.

    Used to back the Go/No-Go criterion (Section 6): a hit-rate or
    error-rate improvement only counts if its CI does not overlap the
    baseline's.
    """
    if not outcomes:
        return BootstrapResult(0.0, 0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(outcomes)
    point_estimate = metric_fn(outcomes)

    samples = np.empty(n_resamples, dtype=np.float64)
    outcomes_arr = np.array(outcomes, dtype=object)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = outcomes_arr[idx].tolist()
        samples[i] = metric_fn(resample)

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(samples, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapResult(point_estimate=point_estimate, ci_low=float(ci_low), ci_high=float(ci_high))
