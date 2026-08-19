"""Step 3 of the locked CRC protocol (CRC_RISK_CONTROLLED_CACHING_PROTOCOL.md):
repeated calibration/test splits (Protocol R) to empirically validate CRC's
guarantee, properly -- not the 8-seed spot check from the Step 2 report.

Reports mean realized risk +/- CI and P(realized risk > alpha) across M
independent random splits, for each target alpha, plus Youden J's risk
distribution on the SAME splits for comparison. Deliberately called
"empirical validation of risk control," not "coverage" -- CRC's guarantee is
on E[L_{n+1}(lambda_hat)], not a per-run probabilistic coverage statement.

No GPU: reuses cached match trace + verifier scores, pure array math.

Usage:
    python scripts/crc_step3_lmarena_repeated_splits.py
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
M_TRIALS = 200
BASE_SEED = 0


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
    """Normal-approximation 95% CI on the mean across M independent trials."""
    m = len(x)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(m))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def main() -> None:
    trace = load_match_trace(CACHE_DIR / TRACE_FILE)
    scored = load_scored(CACHE_DIR / SCORED_FILE)
    gz_indices = sorted(scored.keys())
    scores_all = np.array([scored[i].score for i in gz_indices])
    labels_all = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
    n = len(labels_all)
    print(f"LmArena gray zone: n={n}, M={M_TRIALS} repeated 50/50 random splits")

    crc_risks = {alpha: [] for alpha in TARGET_ALPHAS}
    crc_reuses = {alpha: [] for alpha in TARGET_ALPHAS}
    youden_risks = []
    youden_reuses = []

    rng = np.random.default_rng(BASE_SEED)
    half = n // 2
    for m in range(M_TRIALS):
        perm = rng.permutation(n)
        calib_idx, test_idx = perm[:half], perm[half:]
        cal_scores, cal_labels = scores_all[calib_idx], labels_all[calib_idx]
        test_scores, test_labels = scores_all[test_idx], labels_all[test_idx]

        lam_youden = youden_j_threshold(cal_scores, cal_labels)
        ry, uy = realized_risk_and_reuse(test_scores, test_labels, lam_youden)
        youden_risks.append(ry)
        youden_reuses.append(uy)

        for alpha in TARGET_ALPHAS:
            lam = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)
            r, u = realized_risk_and_reuse(test_scores, test_labels, lam)
            crc_risks[alpha].append(r)
            crc_reuses[alpha].append(u)

    print(f"\nYouden J across {M_TRIALS} splits (not alpha-targeted, reference only):")
    ym, ylo, yhi = mean_ci(np.array(youden_risks))
    print(f"  mean_test_risk={ym:.4f}  95% CI=[{ylo:.4f}, {yhi:.4f}]  "
          f"std={np.std(youden_risks):.4f}  mean_reuse={np.mean(youden_reuses):.4f}")

    print(f"\n{'alpha':>7} | {'mean_risk':>10} {'95%_CI_lo':>10} {'95%_CI_hi':>10} | "
          f"{'P(risk>alpha)':>14} | {'mean_reuse':>11}")
    results = {}
    for alpha in TARGET_ALPHAS:
        risks = np.array(crc_risks[alpha])
        reuses = np.array(crc_reuses[alpha])
        mean, lo, hi = mean_ci(risks)
        p_exceed = float(np.mean(risks > alpha))
        print(f"{alpha:>7.3f} | {mean:>10.4f} {lo:>10.4f} {hi:>10.4f} | {p_exceed:>14.3f} | {np.mean(reuses):>11.4f}")
        results[alpha] = {
            "mean_test_risk": mean, "ci_95_lo": lo, "ci_95_hi": hi,
            "p_exceed_alpha": p_exceed, "mean_reuse": float(np.mean(reuses)),
            "std_test_risk": float(np.std(risks)),
        }

    out = {
        "dataset": "lmarena", "n_gray_zone": n, "m_trials": M_TRIALS, "base_seed": BASE_SEED,
        "youden_j": {"mean_test_risk": ym, "ci_95_lo": ylo, "ci_95_hi": yhi, "mean_reuse": float(np.mean(youden_reuses))},
        "crc_by_alpha": {str(a): v for a, v in results.items()},
    }
    out_path = Path("results/crc_step3_lmarena_repeated_splits.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
