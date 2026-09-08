"""Direction 17/18's unified reflection concluded: LmArena's K=2 candidate-2
decision is a real case where the underlying ranking/label relationship
drifts too severely (positive rate 0% -> 83% across the stream) for ANY
static-threshold recalibration protocol tried so far (single split, forward-
chaining, explicit detrending) to give a trustworthy number -- all of them
still fit ONE threshold on a window and apply it to the next, which is
fragile whenever the calibration/deployment relationship itself shifts, not
just the optimal threshold's value. This mirrors Section 5.21's independent
finding that CRC's per-request validity gate fails under covariate shift for
the same underlying reason.

This script tests the natural fix flagged in that reflection: Adaptive
Conformal Inference (in the spirit of Gibbs & Candes 2021) -- instead of
calibrating a threshold once and freezing it, maintain it ONLINE, adjusting
after every single decision based on whether the realized outcome was an
error relative to a target error rate alpha. This makes no stationarity
assumption about the ranking or the threshold; it only assumes you can
observe ground truth shortly after each decision (true throughout this
project's experimental design, which always has ground-truth correctness
available).

Implementation (a direct, explained adaptation, not a literal reproduction
of the cited paper's regression-interval formalism): maintain a threshold
theta_t on the verifier score (or on a joint fused probability, in the
--joint variant). At each step:
  1. Decide: accept iff score_t >= theta_t.
  2. Observe true label_t. err_t = 1 if (accepted AND label_t == 0) else 0
     (an error only occurs on a false accept, matching this project's
     error_rate definition throughout; a reject is never an error by this
     definition -- monotone loss, same convention as the CRC sections).
  3. Update: theta_{t+1} = theta_t + gamma * (err_t - alpha)
     (an error pushes the threshold up / more conservative; no error pushes
     it down / more permissive, at a rate proportional to how far err_t is
     from the target alpha -- the same control-loop principle behind ACI's
     alpha_t update, applied directly to the decision threshold instead of
     to a quantile level, since no separate conformal calibration set of
     nonconformity scores exists in this project's classification setting).

Compared against the STATIC baseline (a threshold calibrated once on an
initial warm-up window and then frozen, matching what every protocol tested
so far in this investigation effectively does), on LmArena's K=2 candidate-2
decision (off-the-shelf verifier) -- the specific case direction 18 could
not get a trustworthy static number for -- and, as a secondary check, on
SearchQueries' candidate-2 decision, where direction 18 found the drift mild
and the static protocols DID work, to see whether ACI is at least
competitive there too (a fix for severe drift should not regress a case
where static methods already work fine).

Usage:
    python scripts/aci_online_threshold_cascade.py
    python scripts/aci_online_threshold_cascade.py --joint   # use (sim_1, score_1) fused probability instead of score_1 alone
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_LOW, TAU_HIGH = 0.80, 0.97
WARMUP_FRACTION = 0.10  # first 10% of the stream calibrates the static baseline AND ACI's starting point
TARGET_ALPHA = 0.05

DATASETS = {
    "lmarena_offtheshelf": {
        "trace": "results/.cache/lmarena__precomputed__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": -1.2026,
    },
    "searchqueries_offtheshelf": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000__k2.cascade_trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 6.7433,
    },
}


def select_threshold_for_alpha(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    """Smallest threshold whose empirical false-accept rate (among accepted rows) is <= alpha,
    matching this project's error_rate definition and the RCPS-style monotone-loss framing
    used throughout Sections 5.13/5.20/5.21."""
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    accepted_wrong = np.cumsum(l == 0)
    accepted_total = np.arange(1, len(s) + 1)
    err_rate = accepted_wrong / accepted_total
    ok = err_rate <= alpha
    if not ok.any():
        return float(s[0]) + 1.0  # nothing satisfies alpha; be maximally conservative
    last_ok = np.where(ok)[0].max()
    return float(s[last_ok])


def collect_rows(trace_path, scored_path, rank0_thr):
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    scored = json.loads(Path(scored_path).read_text(encoding="utf-8"))
    rows = []
    for i, row in enumerate(trace):
        sims, corrects, idxs = row
        if len(sims) < 2:
            continue
        s0 = sims[0]
        if not (TAU_LOW <= s0 < TAU_HIGH):
            continue
        key0 = f"{i}_0"
        if key0 not in scored:
            continue
        score0 = scored[key0][0]
        if score0 >= rank0_thr:
            continue
        s1 = sims[1]
        if not (TAU_LOW <= s1 < TAU_HIGH):
            continue
        key1 = f"{i}_1"
        if key1 not in scored:
            continue
        score1 = scored[key1][0]
        label1 = 1 if corrects[1] else 0
        rows.append((i, s0, score0, s1, score1, label1))
    rows.sort(key=lambda r: r[0])
    return rows


def run_static(scores: np.ndarray, labels: np.ndarray, theta: float):
    accepted = scores >= theta
    err = accepted & (labels == 0)
    return accepted, err


def run_aci(scores: np.ndarray, labels: np.ndarray, theta0: float, gamma: float, alpha: float):
    theta = theta0
    accepted_hist = np.zeros(len(scores), dtype=bool)
    err_hist = np.zeros(len(scores), dtype=bool)
    theta_hist = np.empty(len(scores))
    for t in range(len(scores)):
        accept = scores[t] >= theta
        err = accept and (labels[t] == 0)
        accepted_hist[t] = accept
        err_hist[t] = err
        theta_hist[t] = theta
        theta = theta + gamma * ((1.0 if err else 0.0) - alpha)
    return accepted_hist, err_hist, theta_hist


def windowed_error_rate(err: np.ndarray, accepted: np.ndarray, window: int):
    """Realized error rate = false_accepts / total_requests_in_window (matches this project's
    error_rate convention: denominator is all requests, not just accepted ones)."""
    n = len(err)
    rates = []
    for start in range(0, n, window):
        end = min(start + window, n)
        rates.append(err[start:end].sum() / (end - start))
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint", action="store_true", help="use a (sim_1, score_1) fused probability instead of score_1 alone")
    parser.add_argument("--gamma-scale", type=float, default=0.05, help="ACI step size, as a fraction of the score's std dev")
    parser.add_argument("--alpha", type=float, default=TARGET_ALPHA)
    parser.add_argument("--output", default="results/aci_online_threshold_cascade.json")
    args = parser.parse_args()

    results = {}
    for name, cfg in DATASETS.items():
        print(f"\n{'=' * 90}\n{name} ({'joint' if args.joint else 'score-only'})\n{'=' * 90}")
        rows = collect_rows(cfg["trace"], cfg["scored"], cfg["rank0_threshold"])
        n = len(rows)
        idxs = np.array([r[0] for r in rows])
        sims1 = np.array([r[3] for r in rows])
        scores1 = np.array([r[4] for r in rows])
        labels = np.array([r[5] for r in rows])
        print(f"n={n}, overall positive rate={labels.mean():.4f}")

        n_warmup = max(50, int(n * WARMUP_FRACTION))
        if args.joint:
            # LmArena's early stream is so sparse the first 10% can be single-class (no positives
            # at all yet, matching the 0.001 positive rate in decile 0 found in direction 18) --
            # grow the fit window (independent of n_warmup, which still gates when ACI/static start
            # adapting) until it contains at least MIN_POS positives, so the joint classifier can be
            # fit at all.
            MIN_POS = 20
            n_fit = n_warmup
            while n_fit < n and labels[:n_fit].sum() < MIN_POS:
                n_fit = min(n, n_fit + max(50, n_warmup))
            print(f"  joint classifier fit window grown to {n_fit} rows ({labels[:n_fit].sum()} positives) "
                  f"to escape single-class warm-up")
            scaler = StandardScaler().fit(np.column_stack([sims1[:n_fit], scores1[:n_fit]]))
            clf = LogisticRegression(max_iter=2000).fit(
                scaler.transform(np.column_stack([sims1[:n_fit], scores1[:n_fit]])), labels[:n_fit])
            fused = clf.predict_proba(scaler.transform(np.column_stack([sims1, scores1])))[:, 1]
            decision_scores = fused
            n_warmup = n_fit  # static/ACI can only start once the ranking function itself exists
        else:
            decision_scores = scores1

        # Static baseline: calibrate once on warm-up window, freeze for the rest.
        theta_static = select_threshold_for_alpha(decision_scores[:n_warmup], labels[:n_warmup], args.alpha)
        static_accepted, static_err = run_static(decision_scores[n_warmup:], labels[n_warmup:], theta_static)
        static_hit_rate = static_accepted.mean()
        static_err_rate = static_err.mean()
        print(f"Static baseline: theta={theta_static:.4f} (from first {n_warmup} rows)  "
              f"post-warmup hit_rate={static_hit_rate:.4f}  error_rate={static_err_rate:.4f}  "
              f"(target alpha={args.alpha})")

        gamma = args.gamma_scale * decision_scores[:n_warmup].std()
        aci_accepted, aci_err, aci_theta_hist = run_aci(
            decision_scores[n_warmup:], labels[n_warmup:], theta0=theta_static, gamma=gamma, alpha=args.alpha)
        aci_hit_rate = aci_accepted.mean()
        aci_err_rate = aci_err.mean()
        print(f"ACI (gamma={gamma:.4f}): post-warmup hit_rate={aci_hit_rate:.4f}  error_rate={aci_err_rate:.4f}")

        window = max(200, (n - n_warmup) // 10)
        static_windowed = windowed_error_rate(static_err, static_accepted, window)
        aci_windowed = windowed_error_rate(aci_err, aci_accepted, window)
        print(f"Windowed error_rate (window={window}), static vs ACI:")
        for i, (s, a) in enumerate(zip(static_windowed, aci_windowed)):
            print(f"  window {i}: static={s:.4f}  ACI={a:.4f}  (target={args.alpha})")

        # How far, on average, does each method's realized windowed error rate deviate from target?
        static_dev = float(np.mean([abs(w - args.alpha) for w in static_windowed]))
        aci_dev = float(np.mean([abs(w - args.alpha) for w in aci_windowed]))
        print(f"Mean |windowed_error_rate - alpha|: static={static_dev:.4f}  ACI={aci_dev:.4f}  "
              f"({'ACI closer to target' if aci_dev < static_dev else 'static closer to target'})")

        results[name] = {
            "joint": args.joint,
            "n": n,
            "n_warmup": n_warmup,
            "alpha": args.alpha,
            "gamma": gamma,
            "static_theta": theta_static,
            "static_hit_rate": static_hit_rate,
            "static_error_rate": static_err_rate,
            "aci_hit_rate": aci_hit_rate,
            "aci_error_rate": aci_err_rate,
            "static_windowed_error_rates": static_windowed,
            "aci_windowed_error_rates": aci_windowed,
            "static_mean_abs_deviation_from_target": static_dev,
            "aci_mean_abs_deviation_from_target": aci_dev,
            "aci_theta_trajectory_sample": aci_theta_hist[::max(1, len(aci_theta_hist) // 20)].tolist(),
        }

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
