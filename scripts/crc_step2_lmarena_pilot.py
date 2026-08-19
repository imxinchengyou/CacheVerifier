"""Step 2 of the locked CRC protocol (CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md):
LmArena pilot comparing Youden's J vs full-data empirical oracle vs real CRC
(Theorem 1 / eq. 4), Protocol R (uniform random calibration/test split of the
fixed gray-zone pool).

Frozen policy (Step 0): tau_low=0.80 (widest grid value -- also happens to be
exactly what's already cached, so no new scoring needed), tau_high=0.97
(original anchor, never grid-searched -- avoids the leakage risk documented
for the grid-search-recommended value).

No GPU / new model inference: reuses the match trace + off-the-shelf
cross-encoder scores already cached by earlier diagnostics this session.

Usage:
    python scripts/crc_step2_lmarena_pilot.py
"""

import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import crc_select_threshold, gray_zone_risk_curve

CACHE_DIR = Path("results/.cache")
TRACE_FILE = "lmarena__precomputed__n60000.trace.json"
SCORED_FILE = "lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"

TARGET_ALPHAS = [0.05, 0.02, 0.01, 0.005]
SEED = 0


def youden_j_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Classification-optimal operating point (max TPR-FPR), ported verbatim
    from cacheverifier-service/verifier_core/finetune.py::select_threshold --
    same formula used throughout this project's honest-eval scripts."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Youden's J requires both classes present")
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr, fpr = tp / n_pos, fp / n_neg
    youden_j = tpr - fpr
    return float(sorted_scores[int(np.argmax(youden_j))])


def realized_risk_and_reuse(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    """Evaluate a FIXED threshold on a held-out set: strict accept rule
    (score > threshold), matching gray_zone_risk_curve's convention."""
    n = len(labels)
    accepted = scores > threshold
    incorrect = labels == 0
    risk = float(np.mean(accepted & incorrect)) if n else 0.0
    reuse = float(np.mean(accepted)) if n else 0.0
    return risk, reuse


def main() -> None:
    trace = load_match_trace(CACHE_DIR / TRACE_FILE)
    scored = load_scored(CACHE_DIR / SCORED_FILE)

    gz_indices = sorted(scored.keys())
    scores_all = np.array([scored[i].score for i in gz_indices])
    labels_all = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
    n = len(labels_all)
    print(f"LmArena gray zone (tau_low=0.80, tau_high=0.97): n={n}")

    # Protocol R: uniform random split of the FIXED pool (see protocol doc --
    # exchangeability of the split, not of the underlying stream, is what's
    # being invoked here).
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    half = n // 2
    calib_idx, test_idx = perm[:half], perm[half:]
    cal_scores, cal_labels = scores_all[calib_idx], labels_all[calib_idx]
    test_scores, test_labels = scores_all[test_idx], labels_all[test_idx]
    print(f"Random split: n_calibration={len(calib_idx)}  n_test={len(test_idx)}")

    # --- Youden's J (classification-optimal, single operating point) ---
    lam_youden = youden_j_threshold(cal_scores, cal_labels)
    risk_youden, reuse_youden = realized_risk_and_reuse(test_scores, test_labels, lam_youden)
    print(f"\nYouden J: lambda={lam_youden:.4f}  test_risk={risk_youden:.4f}  test_reuse={reuse_youden:.4f}")

    # --- Full-data empirical oracle (pooled calib+test as a stand-in for the
    # true population -- same 16102-candidate pool already used in
    # gray_zone_risk_curve_check.py, so these numbers should closely match
    # that script's earlier output as an internal consistency check) ---
    oracle_curve = gray_zone_risk_curve(scores_all, labels_all)

    results = []
    print(f"\n{'alpha':>7} | {'oracle_lambda':>13} {'oracle_risk':>11} {'oracle_reuse':>12} | "
          f"{'CRC_lambda':>10} {'CRC_test_risk':>13} {'CRC_test_reuse':>15} | {'eta(reuse ratio)':>17}")
    for alpha in TARGET_ALPHAS:
        # Oracle: loosest threshold on the FULL pool with empirical risk <= alpha
        feasible = np.where(oracle_curve.risk <= alpha)[0]
        if len(feasible) == 0:
            oracle_lambda, oracle_risk, oracle_reuse = float(oracle_curve.thresholds[-1]), 0.0, 0.0
        else:
            idx = feasible[0]
            oracle_lambda = float(oracle_curve.thresholds[idx])
            oracle_risk = float(oracle_curve.risk[idx])
            oracle_reuse = float(np.mean(scores_all > oracle_lambda))

        # CRC: real eq.(4) selector on CALIBRATION ONLY, evaluated on the
        # independent TEST half -- this is the number that actually tests
        # whether the finite-sample guarantee holds.
        lam_crc = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)
        crc_test_risk, crc_test_reuse = realized_risk_and_reuse(test_scores, test_labels, lam_crc)

        eta = (crc_test_reuse / oracle_reuse) if oracle_reuse > 0 else float("nan")

        print(f"{alpha:>7.3f} | {oracle_lambda:>13.4f} {oracle_risk:>11.4f} {oracle_reuse:>12.4f} | "
              f"{lam_crc:>10.4f} {crc_test_risk:>13.4f} {crc_test_reuse:>15.4f} | {eta:>17.3f}")

        results.append({
            "alpha": alpha,
            "oracle_lambda": oracle_lambda, "oracle_risk": oracle_risk, "oracle_reuse": oracle_reuse,
            "crc_lambda": lam_crc, "crc_test_risk": crc_test_risk, "crc_test_reuse": crc_test_reuse,
            "eta_reuse_ratio": eta,
        })

    out = {
        "dataset": "lmarena", "n_gray_zone": n, "n_calibration": len(calib_idx), "n_test": len(test_idx),
        "seed": SEED,
        "youden_j": {"lambda": lam_youden, "test_risk": risk_youden, "test_reuse": reuse_youden},
        "alpha_sweep": results,
    }
    out_path = Path("results/crc_step2_lmarena_pilot.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
