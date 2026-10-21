"""RESEARCH_PROPOSAL.md 方向 26.10: how fast does the influence of a past
threshold fade in a write-on-miss cache? Uses 26.9's switch runs (threshold A
for the first half, B for the second, and the reverse) against the constant-B
/ constant-A runs: after the switch, the risk gap between the two at the same
evaluation threshold, window by window, is the lingering effect of history."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
N_WIN = 10


def windows(run, n, lam0):
    o = np.array(run["obs"], dtype=float)
    half = n // 2
    edges = np.linspace(half, n, N_WIN + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = o[(o[:, 0] >= lo) & (o[:, 0] < hi)]
        x = (w[:, 1] > lam0) & ~w[:, 2].astype(bool)
        out.append((x.mean(), x.sum(), len(w), w[:, 2].mean()))
    return np.array(out), (edges[:-1] + edges[1:]) / 2 - half


def fit(t, d, se):
    res = {}
    for name, f, p0 in (
        ("exponential", lambda t, d0, tau: d0 * np.exp(-t / tau), (d[0], t[-1] / 2)),
        ("power", lambda t, d0, t0: d0 / (1 + t / t0), (d[0], t[-1] / 4)),
    ):
        try:
            p, _ = curve_fit(f, t, d, p0=p0, sigma=se, absolute_sigma=True, maxfev=20000,
                             bounds=([-1, 1.0], [1, 1e8]))
            chi2 = float(np.sum(((d - f(t, *p)) / se) ** 2))
            half_life = p[1] * np.log(2) if name == "exponential" else p[1]
            res[name] = {"d0": float(p[0]), "scale": float(p[1]), "chi2": chi2, "half_life_requests": float(half_life)}
        except RuntimeError:
            res[name] = None
    return res


def main(ds):
    d = json.loads((ROOT / f"results/cache_sensitivity_{ds}.json").read_text())
    n = d["n_records"]
    lam = {float(q): v for q, v in d["lambda"].items()}
    runs = {r["label"]: r for r in d["runs"]}
    lam95 = lam[0.95]
    print(f"\n== {ds} (second half = {n - n // 2} requests, windows of {(n - n // 2) // N_WIN})")
    out = {}
    for switch, ref_q, name in (("A_then_B", d["B_q"], "A->B vs B"), ("B_then_A", d["A_q"], "B->A vs A")):
        out[name] = {}
        for lab, lam0 in (("lambda_q0.95", lam95), ("own_lambda", lam[ref_q])):
            ws, t = windows(runs[switch], n, lam0)
            wr, _ = windows(runs[f"fixed_q{ref_q}"], n, lam0)
            diff = ws[:, 0] - wr[:, 0]
            se = np.sqrt(ws[:, 0] * (1 - ws[:, 0]) / ws[:, 2] + wr[:, 0] * (1 - wr[:, 0]) / wr[:, 2])
            se = np.maximum(se, 1e-4)
            sig0 = abs(diff[0]) > 2 * se[0]
            fits = fit(t, diff, se) if sig0 else None
            out[name][lab] = {"t": t.tolist(), "diff": diff.tolist(), "se": se.tolist(),
                              "pos_rate_diff": (ws[:, 3] - wr[:, 3]).tolist(),
                              "gray_n": [ws[:, 2].tolist(), wr[:, 2].tolist()], "first_window_significant": bool(sig0),
                              "fits": fits}
            print(f"  {name} @ {lab} ({lam0:.2f}): diff by window {np.round(diff, 4).tolist()}")
            print(f"      2*se {np.round(2 * se, 4).tolist()}  first window significant: {sig0}")
            print(f"      gray n switch {ws[:, 2].astype(int).tolist()} vs ref {wr[:, 2].astype(int).tolist()}")
            if fits:
                for k, v in fits.items():
                    if v:
                        print(f"      fit {k}: d0 {v['d0']:+.4f} half-life {v['half_life_requests']:.0f} requests chi2 {v['chi2']:.1f}")
    return out


if __name__ == "__main__":
    dss = sys.argv[1:] or ["quora", "search_queries_corrected", "lmarena"]
    res = {ds: main(ds) for ds in dss}
    (ROOT / "results/history_decay_summary.json").write_text(json.dumps(res, indent=1))
