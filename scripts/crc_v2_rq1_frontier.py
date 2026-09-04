"""Direction 15 v2, RQ1 (shrunk per Step 0): on the one surviving natural
Protocol-T CRC violation (SearchQueries-corrected), does a per-request
validity gate beat the simple global policies it would replace, on the
(R_cert, C_cert) frontier?

Step 0: Quora's violation was a QQP-label artifact; SearchQueries' is real
but is a weak covariate score-distribution drift plus a P(err|accepted)
climb that Step 0's judge relabel showed is mostly benchmark id_set label
noise. So RQ1 is a head-to-head:

  R_cert = mean over deployed gray-zone requests of 1[wrong AND certified]
           (= CRC's own E[L]; target <= alpha; low-variance mean over ~60k)
  C_cert = mean of 1[certified]   (certified-for-reuse rate; higher better)

Policies (all evaluated on the same deployment stream, in order):
  naive               CRC lambda on the calibration window at target alpha,
                      fixed (this is what violates)
  global_conservative CRC lambda at alpha*(1-m), margin grid m -- Primary
                      Baseline B; a family of (R,C) points
  periodic_recal      re-run CRC on a trailing window W every W deployed
                      requests, at alpha*(1-m), realized labels lagged by L
                      grid (W, L, m)
  weighted_conformal  Tibshirani et al. 2019: density ratio
                      w(x) = p(x|recent deploy)/p(x|cal) from a balanced
                      sklearn logistic on (score, similarity), reweight the
                      cal risk curve. Refit on a rolling window. No deploy
                      labels needed.
  per_request_gate    the thing RQ1 tests. For each request, w(x) from the
                      same running classifier; if w(x) > tau, don't certify
                      (drifted region). Grid over tau and base lambda
                      (naive / a conservative margin).

Verdict test: does per_request_gate's (R_cert, C_cert) frontier DOMINATE
global_conservative / weighted_conformal / periodic_recal? The script also
reports mean w among {certified & error} vs {certified & correct} -- if w
does not separate the errors, the gate cannot work no matter its tuning.

No GPU. sklearn logistic for the density ratio.

Usage:
    python scripts/crc_v2_rq1_frontier.py --dataset search_queries_corrected --cal-frac 0.3
    python scripts/crc_v2_rq1_frontier.py --dataset search_queries_corrected --cal-frac 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import crc_select_threshold

CACHE_DIR = Path("results/.cache")
DATASET_FILES = {
    "lmarena": ("lmarena__precomputed__n60000.trace.json",
                "lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
              "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"),
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000.trace.json",
                                 "search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"),
}
ALPHAS = [0.05, 0.02, 0.01]
MARGINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RECAL_WINDOWS = [2000, 5000, 10000]
RECAL_LAGS = [0, 5000]
RECAL_MARGINS = [0.0, 0.2, 0.4]
GATE_TAUS = [1.1, 1.2, 1.3, 1.5, 2.0]
GATE_BASES = ["naive", "m0.3"]
REFIT_EVERY = 4000
DRC_WINDOW = 8000   # rolling window for the density-ratio classifier


def risk_cov(scores, labels, acc_mask):
    return float(np.mean(acc_mask & (labels == 0))), float(np.mean(acc_mask))


def weighted_crc_lambda(cs, cl, w, alpha):
    order = np.argsort(cs)
    cs_s, cl_s, w_s = cs[order], cl[order], w[order]
    inc = (cl_s == 0).astype(float) * w_s
    suffix = np.concatenate([np.cumsum(inc[::-1])[::-1], [0.0]])
    idxr = np.searchsorted(cs_s, cs_s, side="right")
    wrisk = suffix[idxr] / w_s.sum()
    nn = len(cl_s)
    crc_val = (nn / (nn + 1)) * wrisk + 1.0 / (nn + 1)
    feas = np.where(crc_val <= alpha)[0]
    return float(cs_s[feas[0]]) if len(feas) else float(cs_s[-1])


def density_ratio_fn(feat_cal, feat_recent):
    """w(x) = P(recent|x)/P(cal|x), balanced logistic on standardized feats."""
    m = min(len(feat_cal), len(feat_recent))
    rng = np.random.default_rng(0)
    a = feat_cal[rng.choice(len(feat_cal), m, replace=False)]
    b = feat_recent[rng.choice(len(feat_recent), m, replace=False)]
    X = np.vstack([a, b]); yb = np.r_[np.zeros(m), np.ones(m)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(max_iter=500).fit((X - mu) / sd, yb)

    def w(feat):
        p = lr.predict_proba((np.atleast_2d(feat) - mu) / sd)[:, 1]
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return p / (1 - p)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="search_queries_corrected", choices=list(DATASET_FILES))
    ap.add_argument("--cal-frac", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tf, sf = DATASET_FILES[args.dataset]
    trace = load_match_trace(CACHE_DIR / tf)
    scored = load_scored(CACHE_DIR / sf)
    gz = [i for i in range(len(trace)) if i in scored]
    s = np.array([scored[i].score for i in gz])
    sim = np.array([trace[i].similarity if trace[i].similarity is not None else 0.0 for i in gz])
    y = np.array([1 if trace[i].would_be_correct else 0 for i in gz])
    feat = np.column_stack([s, sim])
    n = len(s)
    n_cal = int(round(args.cal_frac * n))
    cs, cl, cf = s[:n_cal], y[:n_cal], feat[:n_cal]
    ds, dl, df = s[n_cal:], y[n_cal:], feat[n_cal:]
    n_dep = len(ds)
    print(f"=== {args.dataset}: RQ1 (R_cert,C_cert) frontier  cal_frac={args.cal_frac} ===")
    print(f"n_gz={n}  n_cal={n_cal}  n_deploy={n_dep}  "
          f"cal mean_s={cs.mean():.3f} deploy mean_s={ds.mean():.3f}  "
          f"cal gz-err={np.mean(cl==0):.3f} deploy gz-err={np.mean(dl==0):.3f}")

    # precompute the running density-ratio evaluated on every deploy point,
    # refit on a trailing window every REFIT_EVERY (shared by wconf + gate)
    w_dep = np.ones(n_dep)
    lam_wconf_series = []
    for start in range(0, n_dep, REFIT_EVERY):
        lo = max(0, start - DRC_WINDOW)
        recent = df[lo:start] if start > 0 else cf[-DRC_WINDOW:]
        wfn = density_ratio_fn(cf, recent)
        end = min(n_dep, start + REFIT_EVERY)
        w_dep[start:end] = wfn(df[start:end])
        lam_wconf_series.append((start, end, wfn))
    print(f"density ratio w on deploy: median {np.median(w_dep):.2f}  p90 {np.percentile(w_dep,90):.2f}  "
          f"p99 {np.percentile(w_dep,99):.2f}  max {w_dep.max():.2f}")

    out = {"dataset": args.dataset, "cal_frac": args.cal_frac, "n_cal": n_cal, "n_deploy": n_dep,
           "w_deploy_quantiles": {q: float(np.percentile(w_dep, q)) for q in (50, 90, 99, 100)},
           "alphas": ALPHAS, "by_alpha": {}}

    for alpha in ALPHAS:
        rows = []
        lam0 = crc_select_threshold(cs, cl, alpha=alpha)
        lam_m03 = crc_select_threshold(cs, cl, alpha=alpha * 0.7)

        acc = ds > lam0
        r, c = risk_cov(ds, dl, acc)
        rows.append(dict(policy="naive", cfg={}, R_cert=r, C_cert=c, meets=r <= alpha))

        for m in MARGINS:
            lam = crc_select_threshold(cs, cl, alpha=alpha * (1 - m))
            r, c = risk_cov(ds, dl, ds > lam)
            rows.append(dict(policy="global_conservative", cfg={"margin": m}, R_cert=r, C_cert=c, meets=r <= alpha))

        for W in RECAL_WINDOWS:
            for L in RECAL_LAGS:
                for m in RECAL_MARGINS:
                    accw = np.zeros(n_dep, dtype=bool)
                    lam = lam0
                    for t in range(n_dep):
                        if t % W == 0:
                            hi = t - L
                            lo = hi - W
                            if lo >= 0:
                                lam = crc_select_threshold(ds[lo:hi], dl[lo:hi], alpha=alpha * (1 - m))
                            elif hi >= W // 2:
                                lam = crc_select_threshold(
                                    np.concatenate([cs, ds[:hi]]), np.concatenate([cl, dl[:hi]]),
                                    alpha=alpha * (1 - m))
                        accw[t] = ds[t] > lam
                    r, c = risk_cov(ds, dl, accw)
                    rows.append(dict(policy="periodic_recal", cfg={"W": W, "lag": L, "margin": m},
                                     R_cert=r, C_cert=c, meets=r <= alpha))

        # weighted_conformal: recompute lambda per refit block from reweighted cal
        accw = np.zeros(n_dep, dtype=bool)
        lam = lam0
        for (start, end, wfn) in lam_wconf_series:
            w_on_cal = wfn(cf)
            w_on_cal = np.clip(w_on_cal, 1e-3, 1e3)
            w_on_cal = w_on_cal / w_on_cal.mean()
            lam = weighted_crc_lambda(cs, cl, w_on_cal, alpha)
            accw[start:end] = ds[start:end] > lam
        r, c = risk_cov(ds, dl, accw)
        rows.append(dict(policy="weighted_conformal", cfg={"refit": REFIT_EVERY}, R_cert=r, C_cert=c, meets=r <= alpha))

        # per_request_gate
        for base_name, base_lam in (("naive", lam0), ("m0.3", lam_m03)):
            for tau in GATE_TAUS:
                cert = (ds > base_lam) & (w_dep <= tau)
                r, c = risk_cov(ds, dl, cert)
                rows.append(dict(policy="per_request_gate", cfg={"tau": tau, "base": base_name},
                                 R_cert=r, C_cert=c, meets=r <= alpha))

        # diagnostic: does w separate the certified errors?
        cert0 = ds > lam0
        ce = cert0 & (dl == 0)
        ck = cert0 & (dl == 1)
        w_err = float(w_dep[ce].mean()) if ce.any() else float("nan")
        w_ok = float(w_dep[ck].mean()) if ck.any() else float("nan")

        out["by_alpha"][str(alpha)] = {
            "rows": rows,
            "w_certified_error_mean": w_err, "w_certified_correct_mean": w_ok,
        }

        print(f"\n----- alpha = {alpha}  (naive R={rows[0]['R_cert']:.4f} vs {alpha}) -----")
        print(f"  w | certified&error = {w_err:.3f}   w | certified&correct = {w_ok:.3f}  "
              f"({'separates errors' if w_err - w_ok > 0.05 else 'does NOT separate errors'})")
        print(f"  {'policy':20} {'best cfg (meets R<=a, max C)':32} {'R_cert':>8} {'C_cert':>8}")
        for pol in ["naive", "global_conservative", "periodic_recal", "weighted_conformal", "per_request_gate"]:
            pr = [x for x in rows if x["policy"] == pol]
            meet = [x for x in pr if x["meets"]]
            best = max(meet, key=lambda x: x["C_cert"]) if meet else min(pr, key=lambda x: x["R_cert"])
            note = "" if best["meets"] else "  <- none meet"
            print(f"  {pol:20} {str(best['cfg']):32} {best['R_cert']:>8.4f} {best['C_cert']:>8.4f}{note}")

    out_path = Path(args.out) if args.out else Path(
        f"results/crc_v2_rq1_frontier_{args.dataset}_cf{int(args.cal_frac*100)}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
