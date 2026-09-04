"""Direction 15 v2, Step 0 (RQ1 load-bearing validation): is the CRC Protocol-T
violation reported in Section 5.13 real and stable, or an artifact of one
split point / one contiguous test block / the reconstructed-label noise?

Section 5.13's Protocol T did ONE 50/50 chronological split per dataset and
found: LmArena safe (drift favours conservatism), Quora violates all four
alpha (up to 80% relative overshoot), SearchQueries-corrected violates all
four but mildly (~20-25%). Before RQ1 builds a validity-gate apparatus on
top of "CRC validity fails under drift", this script stress-tests that
finding:

  1. split-point sweep       -- does the violation survive cal-fraction in
                                {0.2,...,0.8}, not just 0.5?
  2. reversed split          -- calibrate on the LAST fraction, test on the
                                FIRST. Symmetric violation => two halves just
                                differ (covariate structure), not a
                                directional "future is harder" drift.
  3. random-split anchor     -- M random 50/50 splits. If these ALSO violate,
                                CRC is simply miscalibrated on this dataset
                                and "chronological" is not the cause
                                (reproduces Section 5.13's Protocol R).
  4. decile trajectory       -- per stream-position decile: gray-zone error
                                rate, and CRC(first-half) realized risk on
                                that decile alone. Shows WHERE a violation
                                concentrates.
  5. block bootstrap         -- canonical 50/50 chronological split, test
                                half resampled in CONTIGUOUS blocks (not
                                iid). CI on test_risk and on relative
                                overshoot (risk - alpha) / alpha.

IMPORTANT CAVEAT baked into the output for Quora: Quora's stream order is
"first-appearance order of each unique question in the QQP training file"
(scripts/convert_quora_dataset.py) -- it is NOT chronological. So a Quora
Protocol-T split tests "do the first and second halves of the QQP file
differ enough to break CRC transfer", which is a real question about CRC
fragility but CANNOT be called a temporal / drift failure. LmArena and
SearchQueries ARE real request streams (vCache benchmarks).

No GPU: pure array math over the cached match trace + verifier scores.

Usage:
    python scripts/crc_step0_violation_stability.py --dataset quora
    python scripts/crc_step0_violation_stability.py --dataset search_queries_corrected
    python scripts/crc_step0_violation_stability.py --dataset lmarena
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

FAKE_STREAM_ORDER = {"quora"}  # QQP file order, not chronological -- see docstring

TARGET_ALPHAS = [0.05, 0.02, 0.01, 0.005]
CAL_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_RANDOM_SPLITS = 300
N_BOOTSTRAP = 3000
N_DECILES = 10
BASE_SEED = 0


def realized_risk_and_reuse(scores, labels, threshold):
    accepted = scores > threshold
    n = len(labels)
    if n == 0:
        return float("nan"), float("nan")
    risk = float(np.mean(accepted & (labels == 0)))
    reuse = float(np.mean(accepted))
    return risk, reuse


def crc_transfer(cal_scores, cal_labels, test_scores, test_labels, alpha):
    lam = crc_select_threshold(cal_scores, cal_labels, alpha=alpha)
    risk, reuse = realized_risk_and_reuse(test_scores, test_labels, lam)
    return {
        "alpha": alpha,
        "lambda": float(lam),
        "test_risk": risk,
        "test_reuse": reuse,
        "exceeds": bool(risk > alpha),
        "rel_overshoot": (risk - alpha) / alpha if alpha else float("nan"),
        "n_cal": int(len(cal_labels)),
        "n_test": int(len(test_labels)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_FILES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace_file, scored_file = DATASET_FILES[args.dataset]
    trace = load_match_trace(CACHE_DIR / trace_file)
    scored = load_scored(CACHE_DIR / scored_file)

    gz = [i for i in range(len(trace)) if i in scored]
    scores = np.array([scored[i].score for i in gz])
    labels = np.array([1 if trace[i].would_be_correct else 0 for i in gz])
    n = len(labels)
    overall_err = float(np.mean(labels == 0))
    fake = args.dataset in FAKE_STREAM_ORDER

    print(f"=== {args.dataset}: CRC Protocol-T violation stability (Step 0) ===")
    print(f"n_gray_zone={n}  overall gray-zone incorrect rate={overall_err:.4f}")
    if fake:
        print("!! stream order is QQP-file first-appearance order, NOT chronological.")
        print("!! A split here tests 'do the file halves differ', not temporal drift.")
    print()

    out = {
        "dataset": args.dataset, "n_gray_zone": n, "overall_incorrect_rate": overall_err,
        "stream_order_is_chronological": not fake,
        "alphas": TARGET_ALPHAS,
    }

    # ---- 1. forward split-point sweep (chronological) -----------------------
    print("--- 1. forward split-point sweep (cal = first f*n, test = rest) ---")
    print(f"{'cal_f':>6} {'cal_err':>8} {'test_err':>8} | " +
          " ".join(f"a={a:<6}" for a in TARGET_ALPHAS))
    sweep_fwd = []
    for f in CAL_FRACTIONS:
        k = int(round(f * n))
        cs, cl = scores[:k], labels[:k]
        ts, tl = scores[k:], labels[k:]
        row = {"cal_fraction": f, "cal_err": float(np.mean(cl == 0)), "test_err": float(np.mean(tl == 0)),
               "by_alpha": [crc_transfer(cs, cl, ts, tl, a) for a in TARGET_ALPHAS]}
        sweep_fwd.append(row)
        flags = " ".join(f"{r['rel_overshoot']:+6.2f}{'*' if r['exceeds'] else ' '}" for r in row["by_alpha"])
        print(f"{f:>6.2f} {row['cal_err']:>8.4f} {row['test_err']:>8.4f} | {flags}")
    out["forward_split_sweep"] = sweep_fwd
    print("  (values = relative overshoot (risk-alpha)/alpha; * = exceeds alpha)")

    # ---- 2. reversed split ------------------------------------------------
    print("\n--- 2. reversed split (cal = LAST f*n, test = FIRST (1-f)*n) ---")
    print(f"{'cal_f':>6} {'cal_err':>8} {'test_err':>8} | " +
          " ".join(f"a={a:<6}" for a in TARGET_ALPHAS))
    sweep_rev = []
    for f in CAL_FRACTIONS:
        k = int(round(f * n))
        cs, cl = scores[n - k:], labels[n - k:]
        ts, tl = scores[:n - k], labels[:n - k]
        row = {"cal_fraction": f, "cal_err": float(np.mean(cl == 0)), "test_err": float(np.mean(tl == 0)),
               "by_alpha": [crc_transfer(cs, cl, ts, tl, a) for a in TARGET_ALPHAS]}
        sweep_rev.append(row)
        flags = " ".join(f"{r['rel_overshoot']:+6.2f}{'*' if r['exceeds'] else ' '}" for r in row["by_alpha"])
        print(f"{f:>6.2f} {row['cal_err']:>8.4f} {row['test_err']:>8.4f} | {flags}")
    out["reversed_split_sweep"] = sweep_rev

    # ---- 3. random-split anchor (Protocol R reproduction) ----------------
    print(f"\n--- 3. random 50/50 split anchor ({N_RANDOM_SPLITS} splits) ---")
    rng = np.random.default_rng(BASE_SEED)
    half = n // 2
    rand = {a: [] for a in TARGET_ALPHAS}
    for _ in range(N_RANDOM_SPLITS):
        perm = rng.permutation(n)
        ci, ti = perm[:half], perm[half:]
        for a in TARGET_ALPHAS:
            r = crc_transfer(scores[ci], labels[ci], scores[ti], labels[ti], a)
            rand[a].append(r["test_risk"])
    out["random_split_anchor"] = {}
    print(f"{'alpha':>7} {'mean_risk':>10} {'P(risk>a)':>10} {'mean_rel_over':>14}")
    for a in TARGET_ALPHAS:
        arr = np.array(rand[a])
        rec = {"mean_risk": float(arr.mean()), "p_exceed": float(np.mean(arr > a)),
               "mean_rel_overshoot": float((arr.mean() - a) / a)}
        out["random_split_anchor"][str(a)] = rec
        print(f"{a:>7.3f} {rec['mean_risk']:>10.4f} {rec['p_exceed']:>10.3f} {rec['mean_rel_overshoot']:>+14.3f}")
    print("  (Protocol R should sit ~on target: mean_rel_overshoot near 0, P(risk>a) near 0.5 -- a")
    print("   tightly-but-correctly-bound guarantee, NOT a violation. Systematic > 0 here would mean")
    print("   CRC is miscalibrated on this dataset regardless of split.)")

    # ---- 4. decile trajectory ------------------------------------------
    print(f"\n--- 4. stream-position decile trajectory ---")
    edges = np.linspace(0, n, N_DECILES + 1).astype(int)
    cs_half, cl_half = scores[:half], labels[:half]
    lam_by_alpha = {a: crc_select_threshold(cs_half, cl_half, alpha=a) for a in TARGET_ALPHAS}
    deciles = []
    print(f"{'decile':>7} {'idx_range':>16} {'err_rate':>9} | " +
          " ".join(f"a={a:<6}" for a in TARGET_ALPHAS))
    for d in range(N_DECILES):
        lo, hi = edges[d], edges[d + 1]
        ds, dl = scores[lo:hi], labels[lo:hi]
        err = float(np.mean(dl == 0))
        risks = {}
        for a in TARGET_ALPHAS:
            r, _ = realized_risk_and_reuse(ds, dl, lam_by_alpha[a])
            risks[str(a)] = r
        deciles.append({"decile": d, "idx_lo": int(lo), "idx_hi": int(hi), "err_rate": err,
                        "risk_from_firsthalf_lambda": risks})
        flags = " ".join(f"{risks[str(a)]:>7.4f}" for a in TARGET_ALPHAS)
        print(f"{d:>7} {f'[{lo},{hi})':>16} {err:>9.4f} | {flags}")
    out["decile_trajectory"] = {
        "first_half_lambda_by_alpha": {str(a): float(v) for a, v in lam_by_alpha.items()},
        "deciles": deciles,
    }
    print("  (risk columns: CRC lambda calibrated on the FIRST HALF, realized risk on that decile alone.")
    print("   Compare against alpha; the last 5 deciles are the Protocol-T test half.)")

    # ---- 5. block bootstrap on canonical 50/50 -------------------------
    print(f"\n--- 5. block bootstrap, canonical 50/50 chronological split "
          f"({N_BOOTSTRAP} resamples) ---")
    cs, cl = scores[:half], labels[:half]          # fixed calibration
    ts, tl = scores[half:], labels[half:]          # test half, resampled in blocks
    n_test = len(tl)
    block_len = max(1, n_test // 40)
    n_blocks = int(np.ceil(n_test / block_len))
    starts = np.arange(0, n_test, block_len)
    rng_b = np.random.default_rng(BASE_SEED + 1)
    boot = {a: {"risk": [], "rel_over": []} for a in TARGET_ALPHAS}
    lam_canon = {a: crc_select_threshold(cs, cl, alpha=a) for a in TARGET_ALPHAS}
    for _ in range(N_BOOTSTRAP):
        pick = rng_b.integers(0, len(starts), size=n_blocks)
        idx = np.concatenate([np.arange(starts[p], min(starts[p] + block_len, n_test)) for p in pick])[:n_test]
        bs, bl = ts[idx], tl[idx]
        for a in TARGET_ALPHAS:
            r, _ = realized_risk_and_reuse(bs, bl, lam_canon[a])
            boot[a]["risk"].append(r)
            boot[a]["rel_over"].append((r - a) / a)
    out["block_bootstrap_5050"] = {"block_len": int(block_len), "n_blocks": int(n_blocks)}
    print(f"{'alpha':>7} {'risk_median':>12} {'risk_CI95':>22} {'relover_CI95':>22} {'P(exceed)':>10}")
    for a in TARGET_ALPHAS:
        rk = np.array(boot[a]["risk"])
        ro = np.array(boot[a]["rel_over"])
        rec = {
            "risk_median": float(np.median(rk)),
            "risk_ci95": [float(np.percentile(rk, 2.5)), float(np.percentile(rk, 97.5))],
            "rel_overshoot_ci95": [float(np.percentile(ro, 2.5)), float(np.percentile(ro, 97.5))],
            "p_exceed": float(np.mean(rk > a)),
        }
        out["block_bootstrap_5050"][str(a)] = rec
        ci = f"[{rec['risk_ci95'][0]:.4f}, {rec['risk_ci95'][1]:.4f}]"
        cro = f"[{rec['rel_overshoot_ci95'][0]:+.2f}, {rec['rel_overshoot_ci95'][1]:+.2f}]"
        print(f"{a:>7.3f} {rec['risk_median']:>12.4f} {ci:>22} {cro:>22} {rec['p_exceed']:>10.3f}")
    print("  (rel_overshoot CI excluding 0 on the low end => violation robust at this split point.)")

    out_path = Path(args.out) if args.out else Path(f"results/crc_step0_violation_stability_{args.dataset}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
