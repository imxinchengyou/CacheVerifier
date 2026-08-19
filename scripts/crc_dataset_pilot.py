"""Steps 2+3 of the locked CRC protocol (CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md),
generalized to any of the three datasets (LmArena done separately as the
original pilot; this is the reusable version for Quora / SearchQueries).

For the given dataset: single-split pilot (Youden J vs full-data oracle vs
CRC, Protocol R) + M repeated splits (mean realized risk, 95% CI,
P(risk>alpha), eta vs oracle).

No GPU: reuses cached match trace + verifier scores.

Usage:
    python scripts/crc_dataset_pilot.py --dataset quora
    python scripts/crc_dataset_pilot.py --dataset search_queries_corrected
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
M_TRIALS = 200
SEED = 0


def youden_j_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
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
    accepted = scores > threshold
    incorrect = labels == 0
    n = len(labels)
    risk = float(np.mean(accepted & incorrect)) if n else 0.0
    reuse = float(np.mean(accepted)) if n else 0.0
    return risk, reuse


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    m = len(x)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(m))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_FILES.keys()))
    args = parser.parse_args()

    trace_file, scored_file = DATASET_FILES[args.dataset]
    trace = load_match_trace(CACHE_DIR / trace_file)
    scored = load_scored(CACHE_DIR / scored_file)
    gz_indices = sorted(scored.keys())
    scores_all = np.array([scored[i].score for i in gz_indices])
    labels_all = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
    n = len(labels_all)
    print(f"=== {args.dataset} gray zone (tau_low=0.80, tau_high=0.97): n={n} ===")

    # --- Single-split pilot (Step 2 equivalent) ---
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    half = n // 2
    calib_idx, test_idx = perm[:half], perm[half:]
    cal_scores, cal_labels = scores_all[calib_idx], labels_all[calib_idx]
    test_scores, test_labels = scores_all[test_idx], labels_all[test_idx]
    print(f"Single-split pilot: n_calibration={len(calib_idx)}  n_test={len(test_idx)}")

    lam_youden = youden_j_threshold(cal_scores, cal_labels)
    risk_youden, reuse_youden = realized_risk_and_reuse(test_scores, test_labels, lam_youden)
    print(f"Youden J: lambda={lam_youden:.4f}  test_risk={risk_youden:.4f}  test_reuse={reuse_youden:.4f}")

    oracle_curve = gray_zone_risk_curve(scores_all, labels_all)
    pilot_rows = {}
    print(f"\n{'alpha':>7} | {'oracle_risk':>11} {'oracle_reuse':>12} | {'CRC_test_risk':>13} {'CRC_test_reuse':>15}")
    for alpha in TARGET_ALPHAS:
        feasible = np.where(oracle_curve.risk <= alpha)[0]
        if len(feasible) == 0:
            oracle_lambda, oracle_risk, oracle_reuse = float(oracle_curve.thresholds[-1]), 0.0, 0.0
        else:
            idx = feasible[0]
            oracle_lambda = float(oracle_curve.thresholds[idx])
            oracle_risk = float(oracle_curve.risk[idx])
            oracle_reuse = float(np.mean(scores_all > oracle_lambda))
        lam_crc = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)
        crc_risk, crc_reuse = realized_risk_and_reuse(test_scores, test_labels, lam_crc)
        print(f"{alpha:>7.3f} | {oracle_risk:>11.4f} {oracle_reuse:>12.4f} | {crc_risk:>13.4f} {crc_reuse:>15.4f}")
        pilot_rows[alpha] = {"oracle_lambda": oracle_lambda, "oracle_risk": oracle_risk, "oracle_reuse": oracle_reuse,
                              "crc_lambda": lam_crc, "crc_test_risk": crc_risk, "crc_test_reuse": crc_reuse}

    # --- Repeated splits (Step 3 equivalent) ---
    print(f"\nRepeated splits: M={M_TRIALS}")
    crc_risks = {alpha: [] for alpha in TARGET_ALPHAS}
    crc_reuses = {alpha: [] for alpha in TARGET_ALPHAS}
    youden_risks, youden_reuses = [], []

    rng2 = np.random.default_rng(SEED)
    for _m in range(M_TRIALS):
        perm = rng2.permutation(n)
        ci, ti = perm[:half], perm[half:]
        cs, cl = scores_all[ci], labels_all[ci]
        ts, tl = scores_all[ti], labels_all[ti]

        ly = youden_j_threshold(cs, cl)
        ry, uy = realized_risk_and_reuse(ts, tl, ly)
        youden_risks.append(ry)
        youden_reuses.append(uy)

        for alpha in TARGET_ALPHAS:
            lam = crc_select_threshold(cs, cl, alpha=alpha)
            r, u = realized_risk_and_reuse(ts, tl, lam)
            crc_risks[alpha].append(r)
            crc_reuses[alpha].append(u)

    ym, ylo, yhi = mean_ci(np.array(youden_risks))
    print(f"Youden J: mean_test_risk={ym:.4f}  95% CI=[{ylo:.4f},{yhi:.4f}]  mean_reuse={np.mean(youden_reuses):.4f}")

    print(f"\n{'alpha':>7} | {'mean_risk':>10} {'95%_CI_lo':>10} {'95%_CI_hi':>10} | "
          f"{'P(risk>alpha)':>14} | {'mean_reuse':>11} | {'eta':>6}")
    repeated_rows = {}
    for alpha in TARGET_ALPHAS:
        risks = np.array(crc_risks[alpha])
        reuses = np.array(crc_reuses[alpha])
        mean, lo, hi = mean_ci(risks)
        p_exceed = float(np.mean(risks > alpha))
        oracle_reuse = pilot_rows[alpha]["oracle_reuse"]
        eta = (float(np.mean(reuses)) / oracle_reuse) if oracle_reuse > 0 else float("nan")
        print(f"{alpha:>7.3f} | {mean:>10.4f} {lo:>10.4f} {hi:>10.4f} | {p_exceed:>14.3f} | {np.mean(reuses):>11.4f} | {eta:>6.3f}")
        repeated_rows[alpha] = {"mean_test_risk": mean, "ci_95_lo": lo, "ci_95_hi": hi,
                                 "p_exceed_alpha": p_exceed, "mean_reuse": float(np.mean(reuses)), "eta": eta}

    out = {
        "dataset": args.dataset, "n_gray_zone": n, "seed": SEED, "m_trials": M_TRIALS,
        "pilot": {"youden_j": {"lambda": lam_youden, "test_risk": risk_youden, "test_reuse": reuse_youden},
                  "alpha_sweep": {str(a): v for a, v in pilot_rows.items()}},
        "repeated": {"youden_j": {"mean_test_risk": ym, "ci_95_lo": ylo, "ci_95_hi": yhi, "mean_reuse": float(np.mean(youden_reuses))},
                     "alpha_sweep": {str(a): v for a, v in repeated_rows.items()}},
    }
    out_path = Path(f"results/crc_pilot_{args.dataset}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
