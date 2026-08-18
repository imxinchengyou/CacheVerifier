import random
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from cacheverifier.cache.policy import CachePolicy, Decision
from cacheverifier.cache.store import NeighborMatch
from cacheverifier.data.schema import QueryRecord

# penalty=None matches vCache's own source exactly (see class docstring);
# requirements.txt pins scikit-learn<1.10 to keep it supported, so this
# specific deprecation warning is expected and would otherwise fire on every
# single decide() call once an entry clears min_observations.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")

# Empirical variance-of-t_hat lookup table for the "perfectly separable"
# case (every correct-similarity observed so far > every incorrect one),
# keyed by number of observations. Copied verbatim from vCache's own
# `vcache/vcache_policy/strategies/verified.py::_Algorithm.variance_map` —
# this is part of the published algorithm's fixed definition (presumably
# from their own offline calibration), not something to refit per dataset.
_VARIANCE_MAP: dict[int, float] = {
    6: 0.035445, 7: 0.028285, 8: 0.026436, 9: 0.021349, 10: 0.019371,
    11: 0.012615, 12: 0.011433, 13: 0.010228, 14: 0.009963, 15: 0.009253,
    16: 0.011674, 17: 0.013015, 18: 0.010897, 19: 0.011841, 20: 0.013081,
    21: 0.010585, 22: 0.014255, 23: 0.012058, 24: 0.013002, 25: 0.011715,
    26: 0.00839, 27: 0.008839, 28: 0.010628, 29: 0.009899, 30: 0.008033,
    31: 0.00457, 32: 0.007335, 33: 0.008932, 34: 0.00729, 35: 0.007445,
    36: 0.00761, 37: 0.011423, 38: 0.011233, 39: 0.006783, 40: 0.005233,
    41: 0.00872, 42: 0.010005, 43: 0.01199, 44: 0.00977, 45: 0.01891,
    46: 0.01513, 47: 0.02109, 48: 0.01531,
}


@dataclass
class _EmbeddingStats:
    """Per-cache-entry online state, mirroring vCache's `EmbeddingMetadataObj`."""

    observations: list[tuple[float, int]] = field(default_factory=lambda: [(0.0, 0), (1.0, 1)])
    gamma: float | None = None
    t_hat: float | None = None
    var_t: float | None = None


class AdaptiveThresholdPolicy(CachePolicy):
    """Group B baseline: a port of vCache's `VerifiedDecisionPolicy`
    (arXiv:2502.03771; `vcache-project/vCache`,
    `vcache/vcache_policy/strategies/verified.py::_Algorithm`) — the
    "vCacheLocal" baseline in their own `benchmarks/benchmark.py`.

    Per cache entry, once it has accumulated >= `min_observations` verified
    outcomes, fits a 1D logistic regression of correctness on similarity to
    get a threshold estimate `t_hat` and its variance `var_t` (via the delta
    method, or the hardcoded `_VARIANCE_MAP` when the observations are
    perfectly separable), then computes a *randomized* explore probability
    `tau` such that exploiting is only allowed when doing so provably keeps
    the long-run error rate below `target_error_rate` at some confidence
    level swept over an internal epsilon grid. The hit/miss decision is a
    coin flip against that probability — not a deterministic threshold
    comparison.

    Critically, only MISSES ever get their correctness checked and folded
    into an entry's observation history (`observe` below) — hits are trusted
    without verification, exactly matching vCache's own update rule
    (`VerifiedDecisionPolicy.__update_cache` only runs on the EXPLORE
    branch). That's what makes this an *asynchronous*-verification baseline:
    verification, when it happens at all, never sits on the hit path.
    """

    name = "adaptive_threshold"

    def __init__(self, target_error_rate: float, min_observations: int = 6) -> None:
        if not 0.0 < target_error_rate < 1.0:
            raise ValueError("target_error_rate must be in (0, 1)")
        self.delta = target_error_rate
        self.p_c = 1.0 - self.delta
        self.min_observations = min_observations
        self.epsilon_grid = np.linspace(1e-6, 1 - 1e-6, 50)
        self._stats: dict[str, _EmbeddingStats] = {}

    def _get_stats(self, entry_id: str) -> _EmbeddingStats:
        return self._stats.setdefault(entry_id, _EmbeddingStats())

    def decide(
        self,
        query: QueryRecord,
        query_embedding: np.ndarray,
        match: NeighborMatch | None,
    ) -> Decision:
        if match is None:
            return Decision(action="miss", branch="empty_cache")

        similarity = round(match.similarity, 3)
        stats = self._get_stats(match.entry.query_id)

        if len(stats.observations) < self.min_observations:
            return Decision(action="miss", branch="cold_start", metadata={"n_obs": len(stats.observations)})

        similarities = np.array([s for s, _ in stats.observations])
        labels = np.array([label for _, label in stats.observations])

        fit = self._estimate_parameters(similarities, labels)
        if fit is None:
            return Decision(action="miss", branch="fit_failed")
        t_hat, gamma, var_t = fit
        stats.t_hat, stats.gamma, stats.var_t = t_hat, gamma, var_t

        tau = self._get_tau(var_t=var_t, s=similarity, t_hat=t_hat, gamma=gamma)
        action: Literal["hit", "miss"] = "miss" if random.uniform(0, 1) <= tau else "hit"
        return Decision(
            action=action,
            branch=f"tau_{'explore' if action == 'miss' else 'exploit'}",
            metadata={"tau": tau, "t_hat": t_hat, "gamma": gamma, "var_t": var_t},
        )

    def observe(
        self,
        query: QueryRecord,
        match: NeighborMatch | None,
        would_be_correct: bool | None,
        action: Literal["hit", "miss"],
    ) -> None:
        if match is None or would_be_correct is None or action != "miss":
            return
        stats = self._get_stats(match.entry.query_id)
        stats.observations.append((round(match.similarity, 3), 1 if would_be_correct else 0))

    def _estimate_parameters(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float, float] | None:
        design = np.column_stack([np.ones_like(similarities), similarities])
        try:
            model = LogisticRegression(
                penalty=None, solver="lbfgs", tol=1e-8, max_iter=1000, fit_intercept=False
            )
            model.fit(design, labels)
            intercept, gamma = model.coef_[0]
            gamma = max(gamma, 1e-6)
            t_hat = float(np.clip(-intercept / gamma, 0.0, 1.0))

            perfect_separation = np.min(similarities[labels == 1]) > np.max(similarities[labels == 0])
            var_t = self._get_var_t(perfect_separation, len(similarities), design, gamma, intercept, model)
            return round(t_hat, 3), round(gamma, 3), var_t
        except Exception:
            return None

    def _get_var_t(
        self,
        perfect_separation: bool,
        n_observations: int,
        design: np.ndarray,
        gamma: float,
        intercept: float,
        model: LogisticRegression,
    ) -> float:
        if perfect_separation:
            return _VARIANCE_MAP.get(n_observations, _VARIANCE_MAP[max(_VARIANCE_MAP)])

        p = model.predict_proba(design)[:, 1]
        w = p * (1 - p)
        hessian = design.T @ (w[:, None] * design)
        cov_beta = np.linalg.inv(hessian)
        grad = np.array([-1.0 / gamma, intercept / (gamma**2)])
        return max(0.0, float(grad @ cov_beta @ grad))

    def _get_tau(self, var_t: float, s: float, t_hat: float, gamma: float) -> float:
        quantiles = 1 - self.epsilon_grid
        t_primes = np.clip(t_hat + norm.ppf(quantiles) * np.sqrt(var_t), 0.0, 1.0)
        likelihoods = expit(gamma * (s - t_primes))
        alpha_lower_bounds = (1 - self.epsilon_grid) * likelihoods
        taus = 1 - (1 - self.p_c) / (1 - alpha_lower_bounds)
        return float(np.min(taus))
