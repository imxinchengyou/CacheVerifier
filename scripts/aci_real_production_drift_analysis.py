"""Direction 19 follow-up (candidate B), part 2: static-vs-ACI comparison on
the real comcastcares Twitter drift (Section 5.8), using the off-the-shelf
verifier scores from aci_real_production_drift_score.py. Unlike every other
ACI test so far (Top-K cascade's cache-density effect, direction 18; the
benchmark CRC violations, direction-19-continuation, one of which turned out
to be a labeling artifact), this is the paper's only case of REAL,
independently-documented production drift: gray-zone positive rate moves
from 1.0% (fine-tuning window) to 5.2% (later held-out traffic) on this one
brand, already identified as the root cause of Group E's fine-tuning
turning harmful there (Section 5.8).

Uses the SAME split point Section 5.8's actual Group E experiment used
(n_train = 2579, the fine-tuning window boundary), not an arbitrary 50/50 --
calibrates a CRC threshold on the train window, then compares a frozen
static threshold against ACI's online adaptation over the held-out test
window, at several target alpha levels appropriate to this domain's low
base correctness rate.

Usage:
    python scripts/aci_real_production_drift_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.metrics.core import crc_select_threshold

TARGET_ALPHAS = [0.05, 0.02, 0.01]


def run_static(scores, labels, theta):
    accepted = scores > theta
    err = accepted & (labels == 0)
    return accepted, err


def run_aci(scores, labels, theta0, gamma, alpha):
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


def windowed_risk(err, n_windows=6):
    n = len(err)
    window = max(1, n // n_windows)
    rates = []
    for start in range(0, n, window):
        end = min(start + window, n)
        rates.append(float(err[start:end].mean()))
    return rates


def main() -> None:
    d = json.loads(Path("results/aci_real_production_drift_scored_comcast.json").read_text(encoding="utf-8"))
    scores = np.array(d["scores"])
    labels = np.array(d["labels"])
    n_train = d["n_train_split"]
    cal_scores, cal_labels = scores[:n_train], labels[:n_train]
    test_scores, test_labels = scores[n_train:], labels[n_train:]

    print(f"n_total={len(scores)}  n_train={n_train}  n_test={len(test_scores)}")
    print(f"train positive rate={cal_labels.mean():.4f}  test positive rate={test_labels.mean():.4f}")

    gamma_scale = 0.05
    gamma = gamma_scale * cal_scores.std()

    results = {}
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

        print(f"\nalpha={alpha}: lambda0={lam0:.4f} (CRC-calibrated on the real Group E train window)")
        print(f"  STATIC (frozen, matches actual Group E deployment): overall_risk={static_risk:.4f} "
              f"({'EXCEEDS' if static_risk > alpha else 'ok'})  reuse={static_reuse:.4f}  "
              f"windowed={[round(w,4) for w in static_windowed]}  mean|dev|={static_dev:.4f}")
        print(f"  ACI (online-adapted):                              overall_risk={aci_risk:.4f} "
              f"({'EXCEEDS' if aci_risk > alpha else 'ok'})  reuse={aci_reuse:.4f}  "
              f"windowed={[round(w,4) for w in aci_windowed]}  mean|dev|={aci_dev:.4f}")

        results[str(alpha)] = {
            "alpha": alpha, "lambda0": lam0, "gamma": gamma,
            "static_overall_risk": static_risk, "static_exceeds": static_risk > alpha,
            "static_reuse": static_reuse, "static_windowed_risk": static_windowed, "static_mean_abs_deviation": static_dev,
            "aci_overall_risk": aci_risk, "aci_exceeds": aci_risk > alpha,
            "aci_reuse": aci_reuse, "aci_windowed_risk": aci_windowed, "aci_mean_abs_deviation": aci_dev,
        }

    Path("results/aci_real_production_drift_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote results/aci_real_production_drift_analysis.json")


if __name__ == "__main__":
    main()
