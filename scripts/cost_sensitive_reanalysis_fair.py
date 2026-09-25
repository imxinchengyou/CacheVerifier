"""Same-population version of scripts/cost_sensitive_reanalysis.py (PAPER.md
§5.17), after the 2026-09-24 audit (see the 2026-09-24 erratum at the top
of PAPER.md).

The original compares each Group D/E honest-calibration operating point --
scored on (every request outside that point's gray zone + the chronological
TEST half of its gray zone) -- against Group A's 9 static thresholds scored on
ALL requests. Different request sets, so the cost comparison mixes "the
verifier is better" with "the verifier's evaluation set is easier".

Here every D/E point p is compared against Group A evaluated on p's OWN
request set. Two versions of Group A:
  - "grid9" (PRIMARY): the paper's own 9 thresholds {0.80..0.99}, recomputed on
    p's request set. Only the request-set mismatch is fixed; A keeps exactly
    the options it had, so the comparison stays like-for-like with D/E's own
    9 honest operating points.
  - "dense" (sensitivity only): every distinct similarity threshold, chosen on
    the evaluation data itself. At high r the best policy accepts only a sliver
    of the safest requests, and an in-sample choice among thousands of
    thresholds is an advantage D/E never get -- so this version is biased
    toward A there and is reported only to show how much the verdict depends
    on A's grid.
    margin_p(r) = min_t costA(t, r | population_p) - cost_p(r)
    cost(r)     = r * error_rate + (1 - hit_rate)          [units of C_miss]
The verifier "wins at r" if some operating point has margin_p(r) > 0 -- the
same "best of group vs best of group" semantics as the original script, with
each comparison made on a single request set.

D/E hit/error rates are the published ones (results/*merged*.json). Each
point's request set is rebuilt from the match trace; the rebuilt test-half
size is asserted equal to the published n_test.
"""

import json
from pathlib import Path

import numpy as np

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
CE = "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2"
FINE_GRID = [10 ** (x / 20) for x in range(-40, 61)]  # r in [0.01, 1000], same as the original

DATASETS = {
    "lmarena": {
        "trace": "lmarena__precomputed__n60000.trace.json",
        "scored": f"lmarena__precomputed__n60000__{CE}__lo0.8__hi0.97.scored.json",
        "D": "threshold_calibration_lmarena_60k_merged.json",
        "E": "lmarena_groupE_honest_calibration_merged.json",
    },
    "quora": {
        "trace": "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "scored": f"quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__{CE}__lo0.8__hi0.97.scored.json",
        "D": "threshold_calibration_quora_merged.json",
        "E": "quora_groupE_honest_calibration_merged.json",
    },
    "search_queries": {
        "trace": "search_queries_corrected__precomputed__n150000.trace.json",
        "scored": f"search_queries_corrected__precomputed__n150000__{CE}__lo0.8__hi0.97.scored.json",
        "D": "threshold_calibration_sq_corrected_merged.json",
        "E": "search_queries_corrected_groupE_honest_calibration_merged.json",
    },
}


def population(trace, scored_idx, tau_low, tau_high):
    """(request indices, n_test) for one honest-calibration operating point.
    The original [0.80, 0.97) runs only count gray-zone requests that were
    scored; the extended tau_high=0.999 runs scored every gray-zone request
    that had a candidate."""
    if tau_high <= 0.97 + 1e-9:
        gz = [i for i, t in enumerate(trace) if t.similarity is not None and tau_low <= t.similarity < tau_high and i in scored_idx]
    else:
        gz = [i for i, t in enumerate(trace) if t.similarity is not None and tau_low <= t.similarity < tau_high]
    test = set(gz[len(gz) // 2:])
    in_gz = set(gz)
    keep = np.array([i for i in range(len(trace)) if i not in in_gz or i in test])
    return keep, len(test)


GRID9 = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]


def a_grid9(sim, correct, has):
    """(error, hit) for Group A at the paper's 9 thresholds on one request set."""
    n = len(sim)
    err, hit = [], []
    for t in GRID9:
        acc = has & (sim >= t)
        hit.append(acc.sum() / n)
        err.append((acc & ~correct).sum() / n)
    return np.array(err), np.array(hit)


def a_cost_curve(sim, correct, has):
    """(error, hit) for Group A at every distinct similarity threshold on one
    request set, plus the accept-nothing point."""
    s = np.where(has, sim, -np.inf)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    wrong = (~correct[order]) & np.isfinite(s_sorted)
    n = len(s)
    hit = np.arange(1, n + 1) / n
    err = np.cumsum(wrong) / n
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True] & np.isfinite(s_sorted)
    return np.r_[0.0, err[last]], np.r_[0.0, hit[last]]


def windows(flags):
    out, start = [], None
    for r, ok in zip(FINE_GRID, flags):
        if ok and start is None:
            start = r
        if not ok and start is not None:
            out.append((round(start, 3), round(prev, 3)))
            start = None
        prev = r
    if start is not None:
        out.append((round(start, 3), round(FINE_GRID[-1], 3)))
    return out


def main():
    results = {}
    for key, cfg in DATASETS.items():
        trace = load_match_trace(CACHE / cfg["trace"])
        scored_idx = set(load_scored(CACHE / cfg["scored"]))
        has = np.array([t.similarity is not None for t in trace])
        sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
        correct = np.array([bool(t.would_be_correct) for t in trace])
        results[key] = {}
        print(f"\n== {key}")
        for grp in ("D", "E"):
            points = json.loads((ROOT / "results" / cfg[grp]).read_text())
            margins = {"grid9": [], "dense": []}
            for p in points:
                keep, n_test = population(trace, scored_idx, p["tau_low"], p["tau_high"])
                assert n_test == p["n_test"], (key, grp, p["tau_low"], n_test, p["n_test"])
                curves = {"grid9": a_grid9(sim[keep], correct[keep], has[keep]),
                          "dense": a_cost_curve(sim[keep], correct[keep], has[keep])}
                for name, (err_a, hit_a) in curves.items():
                    margins[name].append(np.array([
                        np.min(r * err_a + (1 - hit_a)) - (r * p["error_rate"] + (1 - p["hit_rate"])) for r in FINE_GRID
                    ]))
            results[key][grp] = {}
            for name in ("grid9", "dense"):
                best = np.max(np.vstack(margins[name]), axis=0)
                win = windows(best > 0)
                results[key][grp][name] = {"winning_r_windows": win,
                                           "best_margin_at_r": {str(round(r, 3)): float(m) for r, m in zip(FINE_GRID, best)}}
                print(f"  {grp} vs A[{name}]: " + (", ".join(f"[{lo}, {hi}]" for lo, hi in win) if win else "never beats A in [0.01, 1000]"))
    out = ROOT / "results" / "cost_sensitive_reanalysis_fair.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
