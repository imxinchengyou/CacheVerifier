"""Direction 19 follow-up: Section 5.24's conclusion claims ACI "upgrades
Section 5.21's coarse 'forced recalibration cadence' recommendation into a
per-request, online-corrected refinement" -- but ACI has only ever been
validated on the Top-K cascade's rank-1 decision, a side branch. This script
closes that loop by testing ACI directly on the ORIGINAL problem that
motivated citing Section 5.21 in the first place: SearchQueries' real,
independently-confirmed Conformal Risk Control guarantee violation under
Protocol T (scripts/crc_protocol_t_chronological.py), where a threshold
CRC-calibrated on the first half of the stream exceeds the target risk
budget alpha on the second half at ALL FOUR tested alpha levels
(0.05/0.02/0.01/0.005; see results/crc_protocol_t_search_queries_corrected.json).

Method: identical data and split to crc_protocol_t_chronological.py (same
gray-zone trace/scored cache, same 50/50 chronological split). For each
alpha:
  1. Static CRC baseline: calibrate lambda via crc_select_threshold on the
     first half (exactly Protocol T's own procedure), freeze it, apply to
     the second half -- reproduces the already-published violation.
  2. ACI: use that same CRC-calibrated lambda as the starting threshold
     theta_0, then run online over the second half in stream order,
     updating theta_t after every decision using the observed loss
     (accepted AND incorrect, matching CRC's own 0/1 risk definition
     exactly) relative to the target alpha -- no held-out split beyond the
     same warm-up point Protocol T already uses.
Reports realized risk (overall and windowed, to see whether it stays near
alpha rather than just averaging out) and reuse rate for both, at all four
alpha levels.

Usage:
    python scripts/aci_crc_violation_fix.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import crc_select_threshold

CACHE_DIR = Path("results/.cache")

DATASET_FILES = {
    "search_queries_corrected": (
        "search_queries_corrected__precomputed__n150000.trace.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
    # Quora included as a control: Section 5.21's audit found Quora's Protocol-T violation was a
    # QQP annotation batch-effect artifact, not real drift, once independently rechecked -- ACI
    # should show no particular benefit there (nothing real to adapt to).
    "quora": (
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    ),
}

TARGET_ALPHAS = [0.05, 0.02, 0.01, 0.005]


def run_static(scores: np.ndarray, labels: np.ndarray, theta: float):
    accepted = scores > theta
    err = accepted & (labels == 0)
    return accepted, err


def run_aci(scores: np.ndarray, labels: np.ndarray, theta0: float, gamma: float, alpha: float):
    theta = theta0
    accepted_hist = np.zeros(len(scores), dtype=bool)
    err_hist = np.zeros(len(scores), dtype=bool)
    theta_hist = np.empty(len(scores))
    for t in range(len(scores)):
        accept = scores[t] > theta
        err = accept and (labels[t] == 0)
        accepted_hist[t] = accept
        err_hist[t] = err
        theta_hist[t] = theta
        theta = theta + gamma * ((1.0 if err else 0.0) - alpha)
    return accepted_hist, err_hist, theta_hist


def windowed_risk(err: np.ndarray, n_windows: int = 10):
    n = len(err)
    window = max(1, n // n_windows)
    rates = []
    for start in range(0, n, window):
        end = min(start + window, n)
        rates.append(float(err[start:end].mean()))
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gamma-scale", type=float, default=0.05)
    parser.add_argument("--output", default="results/aci_crc_violation_fix.json")
    args = parser.parse_args()

    results = {}
    for dataset, (trace_file, scored_file) in DATASET_FILES.items():
        print(f"\n{'=' * 90}\n{dataset}\n{'=' * 90}")
        trace = load_match_trace(CACHE_DIR / trace_file)
        scored = load_scored(CACHE_DIR / scored_file)

        gz_indices = [i for i in range(len(trace)) if i in scored]
        scores_all = np.array([scored[i].score for i in gz_indices])
        labels_all = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])
        n = len(labels_all)
        half = n // 2
        cal_scores, cal_labels = scores_all[:half], labels_all[:half]
        test_scores, test_labels = scores_all[half:], labels_all[half:]
        print(f"n_gray_zone={n}  n_calibration={half}  n_test={n - half}")

        gamma = args.gamma_scale * cal_scores.std()
        dataset_results = {}
        for alpha in TARGET_ALPHAS:
            lam0 = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)

            static_accepted, static_err = run_static(test_scores, test_labels, lam0)
            static_risk = float(static_err.mean())
            static_reuse = float(static_accepted.mean())

            aci_accepted, aci_err, aci_theta_hist = run_aci(test_scores, test_labels, theta0=lam0, gamma=gamma, alpha=alpha)
            aci_risk = float(aci_err.mean())
            aci_reuse = float(aci_accepted.mean())

            static_windowed = windowed_risk(static_err)
            aci_windowed = windowed_risk(aci_err)
            static_dev = float(np.mean([abs(w - alpha) for w in static_windowed]))
            aci_dev = float(np.mean([abs(w - alpha) for w in aci_windowed]))
            static_exceeds_windows = sum(1 for w in static_windowed if w > alpha)
            aci_exceeds_windows = sum(1 for w in aci_windowed if w > alpha)

            print(f"\nalpha={alpha}: lambda0={lam0:.4f} (CRC-calibrated on first half)")
            print(f"  STATIC (frozen lambda0): overall_risk={static_risk:.4f} ({'EXCEEDS' if static_risk > alpha else 'ok'})  "
                  f"reuse={static_reuse:.4f}  windows_exceeding={static_exceeds_windows}/10  mean|dev|={static_dev:.4f}")
            print(f"  ACI (online-adapted):    overall_risk={aci_risk:.4f} ({'EXCEEDS' if aci_risk > alpha else 'ok'})  "
                  f"reuse={aci_reuse:.4f}  windows_exceeding={aci_exceeds_windows}/10  mean|dev|={aci_dev:.4f}")

            dataset_results[str(alpha)] = {
                "alpha": alpha, "lambda0": lam0, "gamma": gamma,
                "static_overall_risk": static_risk, "static_exceeds": static_risk > alpha,
                "static_reuse": static_reuse, "static_windowed_risk": static_windowed,
                "static_windows_exceeding": static_exceeds_windows, "static_mean_abs_deviation": static_dev,
                "aci_overall_risk": aci_risk, "aci_exceeds": aci_risk > alpha,
                "aci_reuse": aci_reuse, "aci_windowed_risk": aci_windowed,
                "aci_windows_exceeding": aci_exceeds_windows, "aci_mean_abs_deviation": aci_dev,
            }
        results[dataset] = dataset_results

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n\nWrote {args.output}")

    print(f"\n{'=' * 100}\nSUMMARY: windows exceeding alpha (out of 10), static vs ACI\n{'=' * 100}")
    for dataset, dres in results.items():
        for alpha_str, r in dres.items():
            print(f"{dataset:>25} alpha={r['alpha']:.3f}: static {r['static_windows_exceeding']}/10 exceeding "
                  f"(overall_risk={r['static_overall_risk']:.4f})  |  ACI {r['aci_windows_exceeding']}/10 exceeding "
                  f"(overall_risk={r['aci_overall_risk']:.4f})  |  reuse static={r['static_reuse']:.4f} ACI={r['aci_reuse']:.4f}")


if __name__ == "__main__":
    main()
