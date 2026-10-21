"""Direction A, step 1 (RESEARCH_PROPOSAL.md): check I-ACI (Zhao et al., ICLR 2025, arXiv:2410.08852) Prop 1 / Prop 2.

Prop 1 (IQT, constant step gamma): q_{t+1} = q_t + (gamma/p_t)(err_t - alpha) obs_t,
err_t = 1{s_t > q_t}, s_t in [0, B]. Claimed for ANY sequence (no expectation):
    |(1/T) sum err_t - alpha| <= (B + gamma/p_min) / (T gamma)

Prop 2 (IACI, "with probability one"): alpha_{t+1} = alpha_t + (gamma/p_t)(alpha - err_t) obs_t,
err_t = 1{beta_t < alpha_t}. Claimed:
    |(1/T) sum err_t - alpha| <= (max(alpha_1, 1-alpha_1) + gamma/M_T) / (T gamma)

Telescoping only controls the IPW sum sum (obs/p)(err - alpha); the residual
sum (1 - obs/p)(err - alpha) is a martingale of order sqrt(T). We check whether
the realized error rate violates the claimed bound.
"""
import json
from pathlib import Path

import numpy as np

ALPHA = 0.1
GAMMA = 0.01
N_SEEDS = 1000


def run_iqt(T, p, rng):
    # vectorized over seeds: shape (N_SEEDS,)
    q = np.full(N_SEEDS, 0.5)  # q_1 in [0, B], B = 1
    err_sum = np.zeros(N_SEEDS)
    ipw_sum = np.zeros(N_SEEDS)
    for _ in range(T):
        s = rng.random(N_SEEDS)
        err = (s > q).astype(float)
        obs = (rng.random(N_SEEDS) < p).astype(float)
        err_sum += err
        ipw_sum += obs / p * (err - ALPHA)
        q += GAMMA / p * (err - ALPHA) * obs
    dev = err_sum / T - ALPHA
    bound = (1.0 + GAMMA / p) / (T * GAMMA)
    return dev, ipw_sum / T, bound


def run_iaci(T, p, rng):
    a = np.full(N_SEEDS, ALPHA)  # alpha_1
    err_sum = np.zeros(N_SEEDS)
    for _ in range(T):
        beta = rng.random(N_SEEDS)
        err = (beta < a).astype(float)
        obs = (rng.random(N_SEEDS) < p).astype(float)
        err_sum += err
        a += GAMMA / p * (ALPHA - err) * obs
    dev = err_sum / T - ALPHA
    bound = (max(ALPHA, 1 - ALPHA) + GAMMA / p) / (T * GAMMA)
    return dev, bound


def report(name, dev, bound, T, rows, p):
    viol = np.mean(np.abs(dev) > bound)
    rows.append({
        "variant": name.split()[0], "p": p, "T": T, "bound": bound,
        "mean_dev": float(dev.mean()), "sd_dev": float(dev.std()),
        "violation_rate": float(viol), "max_abs_dev_over_bound": float(np.abs(dev).max() / bound),
        "sd_times_sqrtT": float(dev.std() * np.sqrt(T)), "sd_times_T": float(dev.std() * T),
        "predicted_sd_times_sqrtT": float(np.sqrt((1 - p) / p * ALPHA * (1 - ALPHA))),
    })
    print(f"{name:26s} T={T:>7d} bound={bound:.5f} "
          f"mean_dev={dev.mean():+.5f} sd_dev={dev.std():.5f} "
          f"P(|dev|>bound)={viol:6.1%} max|dev|/bound={np.abs(dev).max()/bound:6.2f} "
          f"sd*sqrtT={dev.std()*np.sqrt(T):.3f} sd*T={dev.std()*T:9.1f}")


def main():
    rng = np.random.default_rng(0)
    rows = []
    for T in (10_000, 40_000, 160_000):
        for p in (1.0, 0.5, 0.1):
            dev, ipw, bound = run_iqt(T, p, rng)
            report(f"IQT(Prop1) p={p}", dev, bound, T, rows, p)
            # sanity: the IPW average itself must always satisfy the bound
            assert np.all(np.abs(ipw) <= bound + 1e-12), "IPW telescoping bound broken -> code bug"
            dev2, bound2 = run_iaci(T, p, rng)
            report(f"IACI(Prop2) p={p}", dev2, bound2, T, rows, p)
        print()
    out = Path(__file__).resolve().parent.parent / "results" / "iaci_bound_counterexample.json"
    out.write_text(json.dumps({
        "config": {"alpha": ALPHA, "gamma": GAMMA, "n_seeds": N_SEEDS, "rng_seed": 0,
                   "scores": "iid Uniform(0,1)", "paper": "arXiv:2410.08852 Prop 1 (IQT) / Prop 2 (IACI)"},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
