"""Direction 19 continuation 7: the paper's Section 5.25 originally claimed
that comcast's real deployment window sitting near the theoretical crossover
W* = (score range)/(gamma * max(alpha, 1-alpha)) was "the precise mathematical
reason" ACI showed no clear advantage there. An adversarial review correctly
flagged this as a causal overclaim: the theorem only proves a worst-case
UPPER BOUND is near-vacuous at rho = W/W* ~ 1; it does not by itself prove
that ACI's *empirical* advantage over a static threshold actually vanishes
as rho crosses 1. Comcast alone cannot settle this either way -- its full
held-out stream (1,106 rows) only ever gives one (T, rho) pair near the
crossover, with no way to see what happens on the same data at rho >> 1 or
rho << 1.

This script runs the controlled experiment the review asked for: hold
everything else fixed (same gray-zone trace/scored cache, same CRC-calibrated
starting threshold lambda0, same gamma, same alpha=0.05) and sweep ONLY the
window length W used for one "redeployment" of ACI, on SearchQueries and
Quora (both long enough to sweep rho from well below 1 to well above 1) and
on comcast (whose own stream is too short to reach rho >> 1, but can still be
checked for a monotonic trend up to its own ceiling). For each window length,
the historical stream is chopped into non-overlapping windows of that length;
static (frozen lambda0) and ACI (fresh restart, theta0=lambda0, same gamma)
are each run independently on every window; the advantage (static's mean
|risk-alpha| minus ACI's) is averaged across windows with a std/CI, giving a
real distribution rather than one anecdote per window length.

If the theorem's crossover is doing genuine explanatory work (not just a
numerical coincidence at the one comcast operating point), ACI's advantage
over static should shrink towards ~0 as rho falls towards and below 1 --
mirroring what was previously only observed as a single data point at
comcast's actual (T=1106, rho=1.18).

Usage:
    python scripts/aci_window_sweep_causal_test.py
"""

import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import crc_select_threshold

CACHE_DIR = Path("results/.cache")
ALPHA = 0.05
GAMMA_SCALE = 0.05

DATASET_FILES = {
    "search_queries_corrected": (
        "search_queries_corrected__precomputed__n150000.trace.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
    "quora": (
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
}


def run_static(scores: np.ndarray, labels: np.ndarray, theta: float):
    accepted = scores > theta
    err = accepted & (labels == 0)
    return err


def run_aci(scores: np.ndarray, labels: np.ndarray, theta0: float, gamma: float, alpha: float):
    theta = theta0
    err_hist = np.zeros(len(scores), dtype=bool)
    for t in range(len(scores)):
        accept = scores[t] > theta
        err = accept and (labels[t] == 0)
        err_hist[t] = err
        theta = theta + gamma * ((1.0 if err else 0.0) - alpha)
    return err_hist


def sweep_grid(t_total: int, lo: int = 20, n_points: int = 16) -> list[int]:
    hi = t_total
    if hi <= lo:
        return [hi]
    grid = np.unique(np.round(np.geomspace(lo, hi, n_points)).astype(int))
    return [int(g) for g in grid if g >= 5]


def run_sweep(name: str, scores: np.ndarray, labels: np.ndarray, lam0: float, gamma: float,
              range_width: float, max_windows: int = 300) -> dict:
    t_total = len(scores)
    w_star = range_width / (gamma * max(ALPHA, 1 - ALPHA))
    rows = []
    for t_sub in sweep_grid(t_total):
        n_windows_avail = t_total // t_sub
        n_windows = min(n_windows_avail, max_windows)
        if n_windows < 1:
            continue
        static_devs, aci_devs = [], []
        for w in range(n_windows):
            start = w * t_sub
            end = start + t_sub
            s_win, l_win = scores[start:end], labels[start:end]
            static_err = run_static(s_win, l_win, lam0)
            aci_err = run_aci(s_win, l_win, theta0=lam0, gamma=gamma, alpha=ALPHA)
            static_devs.append(abs(float(static_err.mean()) - ALPHA))
            aci_devs.append(abs(float(aci_err.mean()) - ALPHA))
        static_devs = np.array(static_devs)
        aci_devs = np.array(aci_devs)
        advantage = static_devs - aci_devs
        rho = t_sub / w_star
        bound = min(range_width / (t_sub * gamma), max(ALPHA, 1 - ALPHA))
        n = len(advantage)
        adv_mean = float(advantage.mean())
        adv_sem = float(advantage.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({
            "T_sub": t_sub, "rho": rho, "theoretical_bound": bound, "n_windows": n,
            "static_dev_mean": float(static_devs.mean()), "static_dev_std": float(static_devs.std(ddof=1)) if n > 1 else 0.0,
            "aci_dev_mean": float(aci_devs.mean()), "aci_dev_std": float(aci_devs.std(ddof=1)) if n > 1 else 0.0,
            "advantage_mean": adv_mean, "advantage_sem": adv_sem,
            "advantage_significant": bool(n > 1 and abs(adv_mean) > 1.96 * adv_sem),
        })
        print(f"{name:>25} T_sub={t_sub:>6} rho={rho:6.2f} n_win={n:>4} bound={bound:.4f} "
              f"static_dev={static_devs.mean():.4f} aci_dev={aci_devs.mean():.4f} "
              f"advantage={adv_mean:+.4f} (sem={adv_sem:.4f}) "
              f"{'SIG' if rows[-1]['advantage_significant'] else 'n.s.'}")
    return {"w_star": w_star, "t_total": t_total, "range_width": range_width, "gamma": gamma,
            "lambda0": lam0, "alpha": ALPHA, "rows": rows}


def score_range_from_cache(trace_file: str, scored_file: str) -> float:
    trace = load_match_trace(CACHE_DIR / trace_file)
    scored = load_scored(CACHE_DIR / scored_file)
    gz_indices = [i for i in range(len(trace)) if i in scored]
    scores_all = np.array([scored[i].score for i in gz_indices])
    return float(scores_all.max() - scores_all.min())


def main() -> None:
    results = {}

    for dataset, (trace_file, scored_file) in DATASET_FILES.items():
        print(f"\n{'=' * 100}\n{dataset}\n{'=' * 100}")
        trace = load_match_trace(CACHE_DIR / trace_file)
        scored = load_scored(CACHE_DIR / scored_file)
        gz_indices = [i for i in range(len(trace)) if i in scored]
        scores_all = np.array([scored[i].score for i in gz_indices])
        labels_all = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
        n = len(labels_all)
        half = n // 2
        cal_scores, cal_labels = scores_all[:half], labels_all[:half]
        test_scores, test_labels = scores_all[half:], labels_all[half:]

        lam0 = crc_select_threshold(cal_scores, cal_labels, alpha=ALPHA)
        gamma = GAMMA_SCALE * cal_scores.std()
        range_width = score_range_from_cache(trace_file, scored_file)
        print(f"n_test={len(test_scores)}  lambda0={lam0:.4f}  gamma={gamma:.4f}  range_width={range_width:.2f}")

        results[dataset] = run_sweep(dataset, test_scores, test_labels, lam0, gamma, range_width)

    print(f"\n{'=' * 100}\ncomcast (real production drift)\n{'=' * 100}")
    comcast_scored = json.loads(Path("results/aci_real_production_drift_scored_comcast.json").read_text(encoding="utf-8"))
    scores = np.array(comcast_scored["scores"])
    labels = np.array(comcast_scored["labels"])
    n_train = comcast_scored["n_train_split"]
    cal_scores, cal_labels = scores[:n_train], labels[:n_train]
    test_scores, test_labels = scores[n_train:], labels[n_train:]
    lam0 = crc_select_threshold(cal_scores, cal_labels, alpha=ALPHA)
    gamma = GAMMA_SCALE * cal_scores.std()
    range_width = float(scores.max() - scores.min())
    print(f"n_test={len(test_scores)}  lambda0={lam0:.4f}  gamma={gamma:.4f}  range_width={range_width:.2f}")
    results["comcast"] = run_sweep("comcast", test_scores, test_labels, lam0, gamma, range_width, max_windows=50)

    Path("results/aci_window_sweep_causal_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote results/aci_window_sweep_causal_test.json")


if __name__ == "__main__":
    main()
