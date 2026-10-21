"""Percent-cost-reduction curves for the product site's cost-ratio chart
(cacheverifier-service Research.tsx / SemanticCacheThresholds.tsx), on the
current published basis: same-population comparison (2026-09-24 erratum,
scripts/cost_sensitive_reanalysis_fair.py, grid9 Group A) with the direct-hit
label-noise correction at the point estimate of pi (2026-09-27 erratum,
scripts/denoised_cost_sensitive.py).

value(r) = max over operating points p of 100 * margin_p(r) / costA_p(r),
margin_p(r) = min_t costA(t, r | pop_p) - cost_p(r). Positive = the verifier
is the cheaper policy. The sign pattern is checked against the published
winning windows on the fine grid."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_sensitive_reanalysis_fair as cf  # noqa: E402
from denoised_cost_sensitive import a_grid9_w  # noqa: E402
from denoised_rescore import TAU_HIGH, pis  # noqa: E402

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITE_R = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]


def curves(key, grp, pi, rs):
    cfg = cf.DATASETS[key]
    trace = load_match_trace(cf.CACHE / cfg["trace"])
    scored_idx = set(load_scored(cf.CACHE / cfg["scored"]))
    has = np.array([t.similarity is not None for t in trace])
    sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
    correct = np.array([bool(t.would_be_correct) for t in trace])
    wrong_direct = has & (sim >= TAU_HIGH) & ~correct
    w = (~correct).astype(float)
    w[wrong_direct] = 1 - pi
    best = np.full(len(rs), -np.inf)
    for p in json.loads((ROOT / "results" / cfg[grp]).read_text()):
        keep, n_test = cf.population(trace, scored_idx, p["tau_low"], p["tau_high"])
        assert n_test == p["n_test"]
        err_p = p["error_rate"] * (1 - pi) if p["tau_low"] >= TAU_HIGH else p["error_rate"] - pi * wrong_direct[keep].sum() / len(keep)
        err_a, hit_a = a_grid9_w(sim[keep], w[keep], has[keep])
        for i, r in enumerate(rs):
            ca = np.min(r * err_a + (1 - hit_a))
            best[i] = max(best[i], 100 * (ca - (r * err_p + (1 - p["hit_rate"]))) / ca)
    return best


def main():
    pub = json.loads((ROOT / "results" / "denoised_cost_sensitive.json").read_text())
    out = {}
    for key in cf.DATASETS:
        pi = pis("search_queries_corrected" if key == "search_queries" else key)["point"]
        out[key] = {}
        for grp in ("D", "E"):
            fine = curves(key, grp, pi, cf.FINE_GRID)
            assert [tuple(x) for x in cf.windows(fine > 0)] == [tuple(x) for x in pub[key][grp]["point"]], (key, grp)
            site = curves(key, grp, pi, SITE_R)
            out[key][grp] = [round(float(v), 1) for v in site]
            print(key, grp, out[key][grp])
    (ROOT / "results" / "denoised_cost_curve_for_site.json").write_text(json.dumps({"r": SITE_R, **out}, indent=1))


if __name__ == "__main__":
    main()
