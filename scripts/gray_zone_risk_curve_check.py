"""Phase 1 diagnostic for the CRC-based risk-controlled calibration direction
(RESEARCH_PROPOSAL.md, not yet written up -- see conversation notes): plot
R_GZ(lambda) = P(accept AND incorrect | gray zone) across the full verifier
threshold range, on each of the 3 datasets, using the off-the-shelf
cross-encoder scores already cached by `cacheverifier.experiments.run_verified`
/ `scripts/threshold_calibration_ablation.py` -- no new model inference.

Also reports, for a few illustrative target risk levels alpha, the loosest
threshold whose EMPIRICAL risk on the full gray-zone population is <= alpha
(a naive point estimate, NOT yet the real CRC finite-sample-corrected
selection rule -- that's the next step, this script only checks curve shape
and feasibility at the point-estimate level).

Usage:
    python scripts/gray_zone_risk_curve_check.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import gray_zone_risk_curve

CACHE_DIR = Path("results/.cache")

DATASETS = {
    "lmarena": (
        "lmarena__precomputed__n60000.trace.json",
        "lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
    "quora": (
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
    "search_queries_corrected": (
        "search_queries_corrected__precomputed__n150000.trace.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
}

TARGET_ALPHAS = [0.05, 0.02, 0.01, 0.005]


def main() -> None:
    report = {}
    for name, (trace_file, scored_file) in DATASETS.items():
        trace = load_match_trace(CACHE_DIR / trace_file)
        scored = load_scored(CACHE_DIR / scored_file)

        gz_indices = sorted(scored.keys())
        scores = np.array([scored[i].score for i in gz_indices])
        labels = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])

        curve = gray_zone_risk_curve(scores, labels)

        # Sanity check: monotonicity must hold EXACTLY (this is arithmetic,
        # not a statistical property) -- assert rather than eyeball.
        diffs = np.diff(curve.risk)
        assert np.all(diffs <= 1e-12), f"{name}: R_GZ(lambda) is not monotone non-increasing -- implementation bug"

        n = curve.n_gray_zone
        risk_loosest = float(curve.risk[0])  # accept everything in the gray zone
        risk_strictest = float(curve.risk[-1])  # accept (almost) nothing
        base_rate_incorrect = float((labels == 0).mean())

        alpha_results = {}
        for alpha in TARGET_ALPHAS:
            feasible_idx = np.where(curve.risk <= alpha)[0]
            if len(feasible_idx) == 0:
                alpha_results[alpha] = {"feasible": False, "threshold": None, "empirical_risk": None, "n_accepted": 0}
            else:
                # loosest (smallest) threshold meeting the risk bound -> best hit-rate among feasible thresholds
                idx = feasible_idx[0]
                n_accepted = n - idx
                alpha_results[alpha] = {
                    "feasible": True,
                    "threshold": float(curve.thresholds[idx]),
                    "empirical_risk": float(curve.risk[idx]),
                    "n_accepted": int(n_accepted),
                    "accept_fraction_of_gray_zone": n_accepted / n,
                }

        report[name] = {
            "n_gray_zone": n,
            "base_rate_incorrect_in_gray_zone": base_rate_incorrect,
            "risk_accept_everything": risk_loosest,
            "risk_accept_almost_nothing": risk_strictest,
            "alpha_sweep": alpha_results,
        }

        print(f"\n=== {name} ===")
        print(f"n_gray_zone={n}  base_rate_incorrect={base_rate_incorrect:.4f}")
        print(f"R_GZ(accept everything)={risk_loosest:.4f}  R_GZ(accept ~nothing)={risk_strictest:.4f}")
        for alpha, res in alpha_results.items():
            if res["feasible"]:
                print(
                    f"  alpha={alpha:.3f}: threshold={res['threshold']:.4f}  "
                    f"empirical_risk={res['empirical_risk']:.4f}  "
                    f"accepts {res['n_accepted']}/{n} ({res['accept_fraction_of_gray_zone']:.1%} of gray zone)"
                )
            else:
                print(f"  alpha={alpha:.3f}: INFEASIBLE (no threshold on this sample achieves this risk)")

    out_path = Path("results/gray_zone_risk_curve_check.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
