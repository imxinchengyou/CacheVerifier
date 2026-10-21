"""Direction A, step 2 (AUDITED_ACI_PROOF_DRAFT.md): numerical sanity check of
the high-probability bound for audited ACI with a predictable control variate.

Synthetic gray-zone stream, NOT a benchmark: verifier score s_t ~ U(0,1), the
cached answer is wrong with probability g(s) = 0.6 (1 - s)^2. Accept iff
s_t >= theta_t. A rejected request calls the LLM, so its label is free. An
accepted request is audited (forced miss + label) with probability p(s_t).

Estimator (unbiased given the past and s_t, w_t):
    e_hat_t = accept_t * [ f(s_t) + (Z_t / p_t) (w_t - f(s_t)) ]
Update (projected only from below):
    theta_{t+1} = max(theta_lo, theta_t + gamma (e_hat_t - alpha))

Theorem (draft), with prob >= 1 - delta:
    (1/T) sum e_t <= alpha + (b - theta_1 + gamma/p_min)/(gamma T)
                     + [2 sqrt(max(V_T, 1) L) + (2/3)(1/p_min) L] / T
    V_T = sum accept_t (1 - p_t)/p_t (w_t - f_t)^2,  L = log((K+1)/delta),
    K = ceil(log2(T / p_min)),  e_t = accept_t * w_t.
"""
import json
from pathlib import Path

import numpy as np

ALPHA = 0.05
GAMMA = 0.01
DELTA = 0.05
T = 20_000
N_SEEDS = 1000
P_MIN = 0.02
BUDGET = 0.10  # mean audit probability among accepted requests
THETA_1 = 0.5
THETA_LO = 0.0
B = 1.0


def g(s):
    return 0.6 * (1.0 - s) ** 2


# Static full-information threshold: integral_theta^1 g = alpha -> 0.2 (1-theta)^3 = alpha
THETA_STAR = 1.0 - (ALPHA / 0.2) ** (1.0 / 3.0)


def neyman_p(v_fn):
    """p(s) = clip(lam * sqrt(v(s)), P_MIN, 1), lam set so mean p over the
    static accept region [THETA_STAR, 1] equals BUDGET."""
    grid = np.linspace(THETA_STAR, 1.0, 20001)
    root_v = np.sqrt(np.maximum(v_fn(grid), 0.0))
    lo, hi = 0.0, 1e4
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        if np.clip(lam * root_v, P_MIN, 1.0).mean() > BUDGET:
            hi = lam
        else:
            lo = lam
    return lambda s: np.clip(lam * np.sqrt(np.maximum(v_fn(s), 0.0)), P_MIN, 1.0)


def run(f_fn, p_fn, rng):
    theta = np.full(N_SEEDS, THETA_1)
    e_sum = np.zeros(N_SEEDS)
    V = np.zeros(N_SEEDS)
    n_acc = np.zeros(N_SEEDS)
    n_audit = np.zeros(N_SEEDS)
    served_err = np.zeros(N_SEEDS)
    for _ in range(T):
        s = rng.random(N_SEEDS)
        w = (rng.random(N_SEEDS) < g(s)).astype(float)
        acc = (s >= theta).astype(float)
        p = p_fn(s)
        Z = (rng.random(N_SEEDS) < p).astype(float)
        f = f_fn(s)
        e_hat = acc * (f + Z / p * (w - f))
        e_sum += acc * w
        served_err += acc * (1 - Z) * w
        V += acc * (1 - p) / p * (w - f) ** 2
        n_acc += acc
        n_audit += acc * Z
        theta = np.maximum(THETA_LO, theta + GAMMA * (e_hat - ALPHA))
    K = int(np.ceil(np.log2(T / P_MIN)))
    L = np.log((K + 1) / DELTA)
    det = (B - THETA_1 + GAMMA / P_MIN) / (GAMMA * T)
    mart = (2 * np.sqrt(np.maximum(V, 1.0) * L) + (2 / 3) * (1 / P_MIN) * L) / T
    risk = e_sum / T
    bound = ALPHA + det + mart
    return {
        "mean_risk": float(risk.mean()),
        "sd_risk": float(risk.std()),
        "sd_risk_times_sqrtT": float(risk.std() * np.sqrt(T)),
        "p95_risk": float(np.quantile(risk, 0.95)),
        "mean_bound": float(bound.mean()),
        "violation_rate": float((risk > bound).mean()),
        "mean_V_over_T": float((V / T).mean()),
        "mean_hit_rate": float(((n_acc - n_audit) / T).mean()),
        "mean_audit_rate_among_accepted": float((n_audit / np.maximum(n_acc, 1)).mean()),
        "mean_served_risk": float((served_err / T).mean()),
    }


def main():
    rng = np.random.default_rng(0)
    zero = lambda s: np.zeros_like(s)
    uniform = lambda s: np.full_like(s, BUDGET)
    variants = {
        "IPW f=0, uniform p": (zero, uniform),
        "CV f=g (calibrated), uniform p": (g, uniform),
        "CV f=g/3 (miscalibrated), uniform p": (lambda s: g(s) / 3, uniform),
        "IPW f=0, Neyman p~sqrt(g)": (zero, neyman_p(g)),
        "CV f=g, Neyman p~sqrt(g(1-g))": (g, neyman_p(lambda s: g(s) * (1 - g(s)))),
    }
    # Analytic check of the variance-reduction ceiling: over the static accept
    # region, E[g(1-g)] / E[g] = 1 - E[g^2]/E[g] >= 1 - max g.
    grid = np.linspace(THETA_STAR, 1.0, 20001)
    ratio = float((g(grid) * (1 - g(grid))).mean() / g(grid).mean())
    print(f"theta*={THETA_STAR:.4f}  max g on accept region={g(grid).max():.4f}  "
          f"E[g(1-g)]/E[g]={ratio:.4f}  (CV variance reduction ceiling = {1-ratio:.1%})")
    rows = {}
    for name, (f_fn, p_fn) in variants.items():
        r = run(f_fn, p_fn, rng)
        rows[name] = r
        print(f"{name:38s} risk={r['mean_risk']:.4f}±{r['sd_risk']:.4f} p95={r['p95_risk']:.4f} "
              f"bound={r['mean_bound']:.4f} viol={r['violation_rate']:.1%} V/T={r['mean_V_over_T']:.4f} "
              f"hit={r['mean_hit_rate']:.4f} audit|acc={r['mean_audit_rate_among_accepted']:.3f}")
    out = Path(__file__).resolve().parent.parent / "results" / "audited_aci_bound_check.json"
    out.write_text(json.dumps({
        "config": {"alpha": ALPHA, "gamma": GAMMA, "delta": DELTA, "T": T, "n_seeds": N_SEEDS,
                   "p_min": P_MIN, "budget": BUDGET, "theta_1": THETA_1, "g": "0.6(1-s)^2",
                   "theta_star": THETA_STAR, "cv_variance_ratio_E[g(1-g)]/E[g]": ratio},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
