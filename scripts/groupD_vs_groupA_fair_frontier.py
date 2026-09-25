"""Audit of the paper's §5.4 (2026-09-24 erratum in PAPER.md) "Group D honest
calibration beats the Group A static-threshold frontier" result.

Triggered by scripts/gray_zone_high_precision_compare.py: inside the default
gray zone, ranking by raw similarity beats the off-the-shelf cross-encoder at
every error budget on SearchQueries and Quora -- which should make Group D
(cross-encoder in the gray zone) LOSE to Group A (similarity threshold) there,
yet the published comparison has D winning 6/6 on SearchQueries.

Two suspected artifacts in the published comparison
(scripts/compare_groupE_honest_to_groupA.py / separability_payoff.py):
  1. Group A's frontier has only 9 thresholds (0.80, 0.83, ...), compared by
     LINEAR interpolation. A hit/error frontier is concave, so the chord
     between grid points lies below it and understates Group A.
  2. Population mismatch: Group D is scored on (every non-gray-zone request +
     the chronological second half of the gray zone), Group A on all
     requests. Different request mixes, different denominators.

For each tau_low this script reproduces the published D honest point exactly
(same Youden threshold on the calibration half, same replay, same outcome
filter), then compares its hit rate against three versions of Group A:
  (a) published: 9-point grid on all requests, linear interpolation
  (b) dense grid on all requests (every distinct similarity value)
  (c) dense grid on EXACTLY the same request set Group D was scored on
(c) is the fair comparison.
"""

import json
from pathlib import Path

import numpy as np

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored, replay
from cacheverifier.metrics.core import error_rate, hit_rate

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
CE = "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2"
TAU_HIGH = 0.97
TAU_LOW_GRID = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95]

SETS = {
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000.trace.json",
                                 f"search_queries_corrected__precomputed__n150000__{CE}__lo0.8__hi0.97.scored.json",
                                 "search_queries_groupA.json"),
    "lmarena": ("lmarena__precomputed__n60000.trace.json",
                f"lmarena__precomputed__n60000__{CE}__lo0.8__hi0.97.scored.json",
                "lmarena_groupA.json"),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
              f"quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__{CE}__lo0.8__hi0.97.scored.json",
              "quora_groupA.json"),
}


def _load_published_helpers():
    """select_threshold / interpolate straight from the scripts that produced
    the published numbers (threshold_calibration_ablation.py for the Youden's J
    threshold, compare_groupE_honest_to_groupA.py for the Group A
    interpolation) -- no re-implementation to drift from."""
    import importlib.util

    def load(name):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return load("threshold_calibration_ablation").select_threshold, load("compare_groupE_honest_to_groupA").interpolate


select_threshold, interpolate = _load_published_helpers()


def dense_frontier(sim, correct, has_match):
    """Group A at every distinct similarity threshold over a request set:
    hit = share with sim >= t, error = share with sim >= t and wrong."""
    s = np.where(has_match, sim, -np.inf)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    wrong_sorted = (~correct[order]) & np.isfinite(s_sorted)
    n = len(s)
    cum_hit = np.arange(1, n + 1) / n
    cum_err = np.cumsum(wrong_sorted) / n
    # keep only the last index of each tie group (a threshold accepts all ties)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True] & np.isfinite(s_sorted)
    return cum_err[last], cum_hit[last]


def hit_at_error(err, hit, target):
    """Best Group A hit rate with error <= target (a step frontier -- no
    interpolation between thresholds, which is what a deployer can actually
    pick)."""
    ok = err <= target
    return float(hit[ok].max()) if ok.any() else 0.0


def main():
    out = {}
    for d, (trace_f, ce_f, a_f) in SETS.items():
        trace = load_match_trace(CACHE / trace_f)
        scored = load_scored(CACHE / ce_f)
        group_a = json.loads((ROOT / "results" / a_f).read_text())
        pub_frontier = [(r["error_rate"], r["hit_rate"]) for r in group_a]
        has = np.array([t.similarity is not None for t in trace])
        sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
        correct = np.array([bool(t.would_be_correct) for t in trace])
        all_err, all_hit = dense_frontier(sim, correct, has)
        rows = []
        print(f"\n== {d}  (published Group A grid: {len(group_a)} thresholds)")
        for tau_low in TAU_LOW_GRID:
            gz = [i for i, t in enumerate(trace) if t.similarity is not None and tau_low <= t.similarity < TAU_HIGH and i in scored]
            split = len(gz) // 2
            calib, test = gz[:split], gz[split:]
            thr = select_threshold(np.array([scored[i].score for i in calib]),
                                   np.array([1 if trace[i].would_be_correct else 0 for i in calib]))
            if thr is None:
                continue
            test_set = set(test)
            outcomes = replay(trace, scored, tau_low=tau_low, tau_high=TAU_HIGH, threshold=thr)
            keep = [idx for idx in range(len(trace))
                    if trace[idx].similarity is None or not (tau_low <= trace[idx].similarity < TAU_HIGH) or idx in test_set]
            d_out = [outcomes[i] for i in keep]
            d_hr, d_er = hit_rate(d_out), error_rate(d_out)
            keep_arr = np.array(keep)
            pop_err, pop_hit = dense_frontier(sim[keep_arr], correct[keep_arr], has[keep_arr])
            a_pub = interpolate(pub_frontier, d_er)
            a_all = hit_at_error(all_err, all_hit, d_er)
            a_pop = hit_at_error(pop_err, pop_hit, d_er)
            row = {"tau_low": tau_low, "threshold": thr, "D_hit": d_hr, "D_err": d_er,
                   "A_published_interp": a_pub, "A_dense_all": a_all, "A_dense_same_population": a_pop,
                   "delta_published": None if a_pub is None else d_hr - a_pub,
                   "delta_dense_all": d_hr - a_all, "delta_dense_same_population": d_hr - a_pop}
            rows.append(row)
            fmt = lambda v: "   n/a " if v is None else f"{v:+.4f}"
            print(f"  tau_low={tau_low:.2f}  D hit {d_hr:.4f} err {d_er:.4f} | delta vs A: published {fmt(row['delta_published'])}"
                  f"  dense-all {fmt(row['delta_dense_all'])}  dense-same-pop {fmt(row['delta_dense_same_population'])}")
        out[d] = rows
    path = ROOT / "results" / "groupD_vs_groupA_fair_frontier.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
