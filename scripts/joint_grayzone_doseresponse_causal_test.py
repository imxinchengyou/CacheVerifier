"""Direction 17 causal follow-up: is "verifier discriminative power weak ->
joint (similarity, verifier_score) decision beats the verifier-only cascade"
actually CAUSAL, or is it confounded by dataset domain?

The audited result (RESEARCH_PROPOSAL.md direction 17 + post-hoc audit) found
this pattern across only 5 conditions spanning 3 datasets: LmArena (long
conversational text, strong verifier) loses significantly from joint
decision, Quora/SearchQueries (short questions, weak verifier) gain
significantly. With n=3 datasets, "verifier discriminative power" and
"domain / text length" are confounded -- exactly the kind of small-n
correlational claim this project's own conventions (e.g. RESEARCH_PROPOSAL.md
direction 16 "再续之三", flagged explicitly as "排列顺序吻合,但 n=3 无法排除
混淆变量") warn against over-interpreting.

This script removes the confound with a single-dataset dose-response design:
take LmArena's off-the-shelf verifier scores (the strongest, most negative
condition -- most room to show a reversal) and inject increasing amounts of
Gaussian noise directly into the SCORE itself (not the label), simulating a
"weaker verifier" while holding the dataset, the similarity signal, the
labels, and every other variable completely fixed. If the joint-decision
advantage over the cascade grows monotonically as the injected noise
increases (i.e. as the verifier's own discriminative power degrades), that
is clean causal evidence for the mechanism proposed in direction 17 -- not
just a correlational pattern across 3 different datasets.

Method: exact replication of joint_grayzone_decision_analysis_v2.py's
established-protocol split/denominator (verified there to reproduce this
paper's own published Group E honest-calibration numbers), swept across
noise levels sigma in {0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0} (in units of the
calibration-half verifier score's own standard deviation). At each sigma:
  1. verifier_score_noisy = verifier_score + N(0, sigma * std(verifier_score))
     (same noise draw applied consistently to calibration and test halves,
     fixed seed per sigma for reproducibility).
  2. Report the noisy verifier's own standalone AUC (to confirm the
     manipulation actually degrades discriminative power monotonically).
  3. Re-run the baseline (Youden's J on noisy verifier score alone) vs joint
     (logistic regression on (similarity, noisy verifier score)) comparison,
     with the same nested bootstrap CI as v2.

Usage:
    python scripts/joint_grayzone_doseresponse_causal_test.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_LOW = 0.80
TAU_HIGH = 0.97
NOISE_SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

TRACE_PATH = "results/.cache/lmarena__precomputed__n60000.trace.json"
SCORED_PATH = "results/.cache/lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json"


def select_threshold_youden(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr, fpr = tp / n_pos, fp / n_neg
    youden_j = tpr - fpr
    return float(sorted_scores[int(np.argmax(youden_j))])


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def evaluate_at_noise(sigma: float, trace, gz_indices, calib_idx, test_idx, cal_sim, cal_v_raw, cal_y,
                       test_sim, test_v_raw, test_y, n_total, n_direct_accept, fp_direct_accept, seed: int):
    rng = np.random.default_rng(seed)
    noise_std = sigma * cal_v_raw.std()
    cal_v = cal_v_raw + rng.normal(0, noise_std, size=len(cal_v_raw)) if sigma > 0 else cal_v_raw.copy()
    test_v = test_v_raw + rng.normal(0, noise_std, size=len(test_v_raw)) if sigma > 0 else test_v_raw.copy()

    noisy_verifier_auc = roc_auc(test_v, test_y)

    baseline_thr = select_threshold_youden(cal_v, cal_y)
    base_pred = test_v >= baseline_thr

    def outcomes(gray_selected, gray_y, n_calib):
        gray_tp = int((gray_selected & (gray_y == 1)).sum())
        gray_fp = int((gray_selected & (gray_y == 0)).sum())
        n_test_total = n_total - n_calib
        hit_rate = (n_direct_accept + gray_selected.sum()) / n_test_total
        error_rate = (fp_direct_accept + gray_fp) / n_test_total
        return hit_rate, error_rate

    base_hit, base_err = outcomes(base_pred, test_y, len(calib_idx))

    scaler = StandardScaler().fit(np.column_stack([cal_sim, cal_v]))
    X_cal, X_test = scaler.transform(np.column_stack([cal_sim, cal_v])), scaler.transform(np.column_stack([test_sim, test_v]))
    clf = LogisticRegression(max_iter=2000).fit(X_cal, cal_y)
    joint_prob_cal, joint_prob_test = clf.predict_proba(X_cal)[:, 1], clf.predict_proba(X_test)[:, 1]
    joint_thr = select_threshold_youden(joint_prob_cal, cal_y)
    joint_pred = joint_prob_test >= joint_thr
    joint_hit, joint_err = outcomes(joint_pred, test_y, len(calib_idx))

    def frontier(scores):
        pts = []
        for t in np.unique(np.percentile(scores, np.arange(1, 100, 1))):
            pred = scores >= t
            h, e = outcomes(pred, test_y, len(calib_idx))
            pts.append((e, h))
        return sorted(pts)

    base_frontier, joint_frontier = frontier(test_v), frontier(joint_prob_test)

    def interp(pts, target_err):
        errs, hits = [p[0] for p in pts], [p[1] for p in pts]
        return float(np.interp(target_err, errs, hits))

    point_delta = interp(joint_frontier, base_err) - interp(base_frontier, base_err)

    # nested bootstrap (lighter: 300 iters, matches v2's approach)
    n_cal, n_test = len(calib_idx), len(test_idx)
    n_boot = 300
    deltas = np.empty(n_boot)
    boot_rng = np.random.default_rng(1000 + seed)
    for b in range(n_boot):
        cal_boot = boot_rng.integers(0, n_cal, size=n_cal)
        bc_sim, bc_v, bc_y = cal_sim[cal_boot], cal_v[cal_boot], cal_y[cal_boot]
        if len(np.unique(bc_y)) < 2:
            deltas[b] = np.nan
            continue
        b_base_thr = select_threshold_youden(bc_v, bc_y)
        b_scaler = StandardScaler().fit(np.column_stack([bc_sim, bc_v]))
        b_X_cal = b_scaler.transform(np.column_stack([bc_sim, bc_v]))
        b_clf = LogisticRegression(max_iter=2000).fit(b_X_cal, bc_y)
        b_joint_thr = select_threshold_youden(b_clf.predict_proba(b_X_cal)[:, 1], bc_y)

        test_boot = boot_rng.integers(0, n_test, size=n_test)
        bt_sim, bt_v, bt_y = test_sim[test_boot], test_v[test_boot], test_y[test_boot]
        bt_joint_prob = b_clf.predict_proba(b_scaler.transform(np.column_stack([bt_sim, bt_v])))[:, 1]

        def boot_outcomes(selected, y):
            gray_fp = int((selected & (y == 0)).sum())
            n_test_total_b = n_total - n_cal
            h = (n_direct_accept + selected.sum()) / n_test_total_b
            e = (fp_direct_accept + gray_fp) / n_test_total_b
            return h, e

        b_base_pred = bt_v >= b_base_thr
        b_base_h, b_base_e = boot_outcomes(b_base_pred, bt_y)

        def boot_frontier(scores):
            pts = []
            for t in np.unique(np.percentile(scores, np.arange(2, 100, 4))):
                pred = scores >= t
                h, e = boot_outcomes(pred, bt_y)
                pts.append((e, h))
            return sorted(pts)

        deltas[b] = interp(boot_frontier(bt_joint_prob), b_base_e) - interp(boot_frontier(bt_v), b_base_e)

    deltas = deltas[~np.isnan(deltas)]
    ci = (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))
    significant = ci[0] > 0 or ci[1] < 0

    return {
        "sigma": sigma,
        "noisy_verifier_auc": noisy_verifier_auc,
        "baseline_hit_rate": base_hit,
        "baseline_error_rate": base_err,
        "joint_hit_rate": joint_hit,
        "joint_error_rate": joint_err,
        "hit_rate_gain_at_matched_error": point_delta,
        "hit_rate_gain_ci95": ci,
        "significant": significant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="results/joint_grayzone_doseresponse_causal_test.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    trace = json.loads(Path(TRACE_PATH).read_text(encoding="utf-8"))
    scored = json.loads(Path(SCORED_PATH).read_text(encoding="utf-8"))
    n_total = len(trace)

    gz_indices = sorted(int(k) for k in scored.keys())
    split = len(gz_indices) // 2
    calib_idx, test_idx = gz_indices[:split], gz_indices[split:]
    log(f"LmArena off-the-shelf. Gray-zone: {len(gz_indices)} rows -> "
        f"{len(calib_idx)} calibration / {len(test_idx)} test")

    def features(idx_list):
        sims = np.array([trace[i][0] for i in idx_list])
        labels = np.array([1 if trace[i][1] else 0 for i in idx_list])
        vscores = np.array([scored[str(i)][0] for i in idx_list])
        return sims, vscores, labels

    cal_sim, cal_v_raw, cal_y = features(calib_idx)
    test_sim, test_v_raw, test_y = features(test_idx)

    n_direct_accept, fp_direct_accept = 0, 0
    for row in trace:
        sim = row[0]
        if sim is not None and sim >= TAU_HIGH:
            n_direct_accept += 1
            if not row[1]:
                fp_direct_accept += 1

    log(f"Clean (sigma=0) verifier AUC on test half: {roc_auc(test_v_raw, test_y):.4f}")
    log(f"\n{'sigma':>6} | {'noisy_AUC':>9} | {'gain@matched_err':>17} | {'95% CI':>22} | verdict")

    results = []
    for sigma in NOISE_SIGMAS:
        r = evaluate_at_noise(sigma, trace, gz_indices, calib_idx, test_idx, cal_sim, cal_v_raw, cal_y,
                               test_sim, test_v_raw, test_y, n_total, n_direct_accept, fp_direct_accept,
                               seed=int(sigma * 1000) + 1)
        results.append(r)
        ci = r["hit_rate_gain_ci95"]
        verdict = "SIGNIFICANT" if r["significant"] else "not significant"
        sign = "+" if r["hit_rate_gain_at_matched_error"] >= 0 else ""
        log(f"{sigma:6.2f} | {r['noisy_verifier_auc']:9.4f} | {sign}{r['hit_rate_gain_at_matched_error']:16.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {verdict}")

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
