"""Direction 19's flagged limitation ("no rigorous derivation was done of
under what assumptions this specific ACI adaptation retains a formal
long-run coverage guarantee") is closed here with an actual proof, not
another empirical test -- then checked against every ACI run this paper has
already done, to see whether the bound is a valid (if sometimes loose)
upper bound in every case, and whether its tightness explains the qualitative
differences already observed between the "clean win" cases (SearchQueries,
Quora CRC-violation fix) and the "no clear benefit" case (comcast real
production drift, direction 19 continuation 2).

THEOREM (finite-time long-run risk control for threshold-space ACI).
Let (score_t, label_t) for t=1..T be an ARBITRARY sequence (no i.i.d. or
stationarity assumption -- holds even under adversarial drift). Define:
    accept_t = 1{score_t > theta_t}
    err_t    = 1{accept_t = 1 and label_t = 0}
    theta_{t+1} = theta_t + gamma * (err_t - alpha)
If theta_t in [theta_min, theta_max] for all t (true whenever theta_min <=
min_t score_t and theta_max >= max_t score_t, since crossing outside the
observed score range never changes any accept/reject decision, so theta_t
can always be clipped to this range without altering the induced sequence),
then:
    | (1/T) sum_{t=1}^T err_t - alpha | <= (theta_max - theta_min) / (T * gamma)

PROOF. Telescoping the update rule:
    theta_{T+1} - theta_1 = gamma * sum_{t=1}^T (err_t - alpha)
                           = gamma * sum(err_t) - gamma * T * alpha
    => sum(err_t) = T*alpha + (theta_{T+1} - theta_1) / gamma
    => (1/T) sum(err_t) - alpha = (theta_{T+1} - theta_1) / (T * gamma)
Since theta_1, theta_{T+1} are both in [theta_min, theta_max]:
    |theta_{T+1} - theta_1| <= theta_max - theta_min.  QED.

The identical argument applied to any sub-window [t0+1, t0+W] gives the
same bound with W in place of T -- explaining why SHORTER windows have a
LOOSER (larger) guaranteed bound for the same gamma, and why the bound
becomes VACUOUS (exceeds 1, the maximum possible deviation, hence
uninformative) once W < (theta_max - theta_min) / gamma.

This script recomputes the bound for every (dataset, alpha) combination
already tested by aci_crc_violation_fix.py and aci_real_production_drift_analysis.py,
and checks it against the actually observed deviation -- both overall and
per-window.

Usage:
    python scripts/aci_theoretical_bound_verification.py
"""

import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

CACHE_DIR = Path("results/.cache")


def score_range(trace_file: str, scored_file: str) -> float:
    trace = load_match_trace(CACHE_DIR / trace_file)
    scored = load_scored(CACHE_DIR / scored_file)
    gz_indices = [i for i in range(len(trace)) if i in scored]
    scores_all = np.array([scored[i].score for i in gz_indices])
    return float(scores_all.max() - scores_all.min())


def check(name: str, alpha: float, gamma: float, T: int, range_width: float,
          observed_overall_dev: float, observed_windowed_devs: list[float], n_windows: int = 10):
    bound = range_width / (T * gamma)
    holds = observed_overall_dev <= bound + 1e-9
    W = T // n_windows
    window_bound = range_width / (W * gamma)
    max_window_dev = max(observed_windowed_devs) if observed_windowed_devs else float("nan")
    window_holds = max_window_dev <= window_bound + 1e-9
    vacuous = window_bound >= 1.0
    print(f"{name:>42} alpha={alpha:.3f}  T={T:>7}  overall_bound={bound:.5f} observed={observed_overall_dev:.5f} "
          f"[{'HOLDS' if holds else 'VIOLATED'}]  |  window_bound(W={W})={window_bound:.4f} "
          f"observed_max={max_window_dev:.5f} [{'HOLDS' if window_holds else 'VIOLATED'}]"
          f"{'  <-- VACUOUS (>1, uninformative)' if vacuous else ''}")
    return {
        "name": name, "alpha": alpha, "T": T, "gamma": gamma, "range_width": range_width,
        "overall_bound": bound, "observed_overall_dev": observed_overall_dev, "overall_holds": holds,
        "window": W, "window_bound": window_bound, "observed_max_window_dev": max_window_dev,
        "window_holds": window_holds, "window_bound_vacuous": vacuous,
    }


def main() -> None:
    rows = []

    sq_range = score_range(
        "search_queries_corrected__precomputed__n150000.trace.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    )
    quora_range = score_range(
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
    )
    print(f"Score ranges: SearchQueries={sq_range:.2f}  Quora={quora_range:.2f}")

    crc_fix = json.loads(Path("results/aci_crc_violation_fix.json").read_text(encoding="utf-8"))
    for dataset, dres in crc_fix.items():
        range_width = sq_range if dataset == "search_queries_corrected" else quora_range
        T = 43140 if dataset == "search_queries_corrected" else 8520
        for alpha_str, r in dres.items():
            observed_dev = abs(r["aci_overall_risk"] - r["alpha"])
            windowed_devs = [abs(w - r["alpha"]) for w in r["aci_windowed_risk"]]
            rows.append(check(f"CRC-fix/{dataset}", r["alpha"], r["gamma"], T, range_width, observed_dev, windowed_devs))

    comcast_scored = json.loads(Path("results/aci_real_production_drift_scored_comcast.json").read_text(encoding="utf-8"))
    comcast_range = float(np.array(comcast_scored["scores"]).max() - np.array(comcast_scored["scores"]).min())
    comcast_analysis = json.loads(Path("results/aci_real_production_drift_analysis.json").read_text(encoding="utf-8"))
    T_comcast = comcast_scored["n_total"] - comcast_scored["n_train_split"]
    for alpha_str, r in comcast_analysis.items():
        observed_dev = abs(r["aci_overall_risk"] - r["alpha"])
        windowed_devs = [abs(w - r["alpha"]) for w in r["aci_windowed_risk"]]
        rows.append(check("comcast_real_drift", r["alpha"], r["gamma"], T_comcast, comcast_range, observed_dev,
                           windowed_devs, n_windows=6))

    n_holds = sum(1 for r in rows if r["overall_holds"] and r["window_holds"])
    print(f"\n{n_holds}/{len(rows)} (dataset, alpha) combinations: theorem holds both overall and per-window.")
    n_vacuous = sum(1 for r in rows if r["window_bound_vacuous"])
    print(f"{n_vacuous}/{len(rows)} combinations have a vacuous (uninformative, >1) per-window bound "
          f"-- all should be the comcast (small-T) condition.")

    Path("results/aci_theoretical_bound_verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nWrote results/aci_theoretical_bound_verification.json")


if __name__ == "__main__":
    main()
