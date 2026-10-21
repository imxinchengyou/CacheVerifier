"""RESEARCH_PROPOSAL.md 方向 26.9 analysis: gamma-sensitivity and statefulness
of the write-on-miss gray-zone distribution, from the runs written by
scripts/cache_sensitivity_closed_loop.py. Criteria are the preregistered ones
in 26.9."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.02
DELTA_LAMBDA = {"quora": 0.26, "lmarena": 0.97, "search_queries_corrected": 0.10}


def second_half(run, n):
    o = np.array(run["obs"], dtype=float)
    o = o[o[:, 0] >= n // 2]
    return o[:, 1], o[:, 2].astype(bool), o[:, 3].astype(bool)


def risk(v, correct, lam0):
    x = (v > lam0) & ~correct
    return x.mean(), np.sqrt(max(x.mean() * (1 - x.mean()), 1e-12) / len(x))


def main(ds):
    d = json.loads((ROOT / f"results/cache_sensitivity_{ds}.json").read_text())
    n = d["n_records"]
    lam = {float(q): v for q, v in d["lambda"].items()}
    runs = {r["label"]: r for r in d["runs"]}
    grid = sorted(lam.items(), key=lambda kv: kv[1])
    lam0s = [x for _, x in grid]
    print(f"\n== {ds}  lambda grid {[round(x, 2) for x in lam0s]}")
    R, SE, V = {}, {}, {}
    for q, x in grid:
        v, c, a = second_half(runs[f"fixed_q{q}"], n)
        V[x] = v
        R[x] = [risk(v, c, l0)[0] for l0 in lam0s]
        SE[x] = [risk(v, c, l0)[1] for l0 in lam0s]
        own = ((v > x) & ~c).mean()
        print(f"  q{q} lambda {x:6.2f}: gray(2nd half) {len(v)}  pos {c.mean():.3f}  accept {a.mean():.3f}  "
              f"risk@own {own:.4f}  store {runs[f'fixed_q{q}']['final_store']}")
    gam, gam_sig, w1g = 0.0, 0.0, 0.0
    for (_, x1), (_, x2) in zip(grid[:-1], grid[1:]):
        dl = x2 - x1
        for k in range(len(lam0s)):
            diff = abs(R[x1][k] - R[x2][k])
            se = np.hypot(SE[x1][k], SE[x2][k])
            gam = max(gam, diff / dl)
            gam_sig = max(gam_sig, max(diff - 2 * se, 0.0) / dl)
        w1g = max(w1g, wasserstein_distance(V[x1], V[x2]) / dl)
    delta = gam * DELTA_LAMBDA[ds]
    delta_sig = gam_sig * DELTA_LAMBDA[ds]
    verdict = "usable" if delta <= ALPHA / 4 else ("unusable" if delta >= ALPHA else "marginal")
    print(f"  gamma_R {gam:.4f} per V unit (noise-corrected {gam_sig:.4f}); W1(V)/dlambda max {w1g:.3f}")
    print(f"  Delta = gamma_R x dlambda({DELTA_LAMBDA[ds]}) = {delta:.4f} (noise-corrected {delta_sig:.4f}) -> {verdict}")

    lamA, lamB = lam[d["A_q"]], lam[d["B_q"]]
    hyst = {}
    stateless = True
    for switch, ref, name in (("A_then_B", lamB, "A->B vs B"), ("B_then_A", lamA, "B->A vs A")):
        v1, c1, _ = second_half(runs[switch], n)
        v2, c2, _ = second_half(runs[f"fixed_q{d['B_q'] if ref == lamB else d['A_q']}"], n)
        worst = 0.0
        rows = []
        for l0 in lam0s:
            r1, s1 = risk(v1, c1, l0)
            r2, s2 = risk(v2, c2, l0)
            tol = max(0.005, 2 * np.hypot(s1, s2))
            rows.append((round(l0, 2), round(r1 - r2, 4), round(tol, 4)))
            if abs(r1 - r2) > tol:
                stateless = False
            worst = max(worst, abs(r1 - r2))
        hyst[name] = {"max_abs_risk_diff": worst, "rows": rows,
                      "gray_n": [int(len(v1)), int(len(v2))], "pos_rate": [float(c1.mean()), float(c2.mean())]}
        print(f"  hysteresis {name}: max |risk diff| {worst:.4f}; gray n {len(v1)} vs {len(v2)}; "
              f"pos rate {c1.mean():.3f} vs {c2.mean():.3f}; per lambda0 (diff, tol) {rows}")
    print(f"  stateless approximation: {'holds' if stateless else 'FAILS'}")
    return {"gamma_R": gam, "gamma_R_noise_corrected": gam_sig, "w1_per_lambda": w1g, "delta_lambda": DELTA_LAMBDA[ds],
            "Delta": delta, "Delta_noise_corrected": delta_sig, "verdict": verdict, "hysteresis": hyst,
            "stateless_holds": stateless}


if __name__ == "__main__":
    dss = sys.argv[1:] or [p.stem.replace("cache_sensitivity_", "") for p in (ROOT / "results").glob("cache_sensitivity_*.json")]
    out = {ds: main(ds) for ds in dss}
    (ROOT / "results/cache_sensitivity_summary.json").write_text(json.dumps(out, indent=1))
