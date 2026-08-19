"""Protocol T (CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md): chronological
calibration/test split -- deployment-transfer / drift-robustness experiment.
Explicitly NOT a CRC theorem claim (exchangeability doesn't hold under a
chronological split when the stream is non-stationary -- confirmed by the
state-dependence audit, scripts/state_dependence_audit.py). This asks a
different, still-useful question: does a threshold calibrated on the first
half of the stream transfer to the second half, under whatever real drift is
actually present?

Same 50/50 split point `threshold_calibration_ablation.py` already uses
(first half = calibration, second half = test, in original stream order) --
reused deliberately so this is directly comparable to that script's
established convention, not a new split methodology.

No GPU: reuses cached match trace + verifier scores.

Usage:
    python scripts/crc_protocol_t_chronological.py --dataset lmarena
    python scripts/crc_protocol_t_chronological.py --dataset quora
    python scripts/crc_protocol_t_chronological.py --dataset search_queries_corrected
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import crc_select_threshold, gray_zone_risk_curve

CACHE_DIR = Path("results/.cache")

DATASET_FILES = {
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


def realized_risk_and_reuse(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    accepted = scores > threshold
    incorrect = labels == 0
    n = len(labels)
    risk = float(np.mean(accepted & incorrect)) if n else 0.0
    reuse = float(np.mean(accepted)) if n else 0.0
    return risk, reuse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_FILES.keys()))
    args = parser.parse_args()

    trace_file, scored_file = DATASET_FILES[args.dataset]
    trace = load_match_trace(CACHE_DIR / trace_file)
    scored = load_scored(CACHE_DIR / scored_file)

    # Chronological order = original stream order, exactly as
    # threshold_calibration_ablation.py's `gz_indices` construction does.
    gz_indices = [i for i in range(len(trace)) if i in scored]
    scores_all = np.array([scored[i].score for i in gz_indices])
    labels_all = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
    n = len(labels_all)
    half = n // 2
    cal_scores, cal_labels = scores_all[:half], labels_all[:half]
    test_scores, test_labels = scores_all[half:], labels_all[half:]

    cal_incorrect_rate = float(np.mean(cal_labels == 0))
    test_incorrect_rate = float(np.mean(test_labels == 0))

    print(f"=== {args.dataset}: Protocol T (chronological split) ===")
    print(f"n_gray_zone={n}  n_calibration={half}  n_test={n - half}")
    print(f"calibration-half incorrect rate={cal_incorrect_rate:.4f}  test-half incorrect rate={test_incorrect_rate:.4f}"
          f"  (ratio test/cal={test_incorrect_rate / cal_incorrect_rate if cal_incorrect_rate else float('nan'):.3f})")

    print(f"\n{'alpha':>7} | {'CRC_lambda':>10} {'test_risk':>10} {'exceeds?':>9} {'test_reuse':>11}")
    results = []
    for alpha in TARGET_ALPHAS:
        lam = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)
        risk, reuse = realized_risk_and_reuse(test_scores, test_labels, lam)
        exceeds = "YES" if risk > alpha else "no"
        print(f"{alpha:>7.3f} | {lam:>10.4f} {risk:>10.4f} {exceeds:>9} {reuse:>11.4f}")
        results.append({"alpha": alpha, "lambda": lam, "test_risk": risk, "exceeds_alpha": risk > alpha, "test_reuse": reuse})

    out = {
        "dataset": args.dataset, "n_gray_zone": n, "n_calibration": half, "n_test": n - half,
        "calibration_incorrect_rate": cal_incorrect_rate, "test_incorrect_rate": test_incorrect_rate,
        "alpha_sweep": results,
    }
    out_path = Path(f"results/crc_protocol_t_{args.dataset}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
