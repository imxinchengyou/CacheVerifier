"""Rigor-audit rewrite of joint_grayzone_decision_analysis.py.

That first version made two methodological deviations from this project's
own established honest-calibration protocol
(scripts/threshold_calibration_ablation_groupE.py), discovered by comparing
its recomputed baseline threshold (0.6921, LmArena finetuned) against the
already-published, peer-reviewed-within-this-project value (0.6241, in
results/lmarena_groupE_honest_calibration_merged.json) and finding they did
not match:

  1. WRONG SPLIT UNIT. The established protocol splits the GRAY-ZONE ROWS
     THEMSELVES in half by count (`gz_indices[:split]` / `gz_indices[split:]`,
     where gz_indices is the sorted list of gray-zone row indices only). The
     first version instead split the FULL TRACE's index range in half and
     filtered gray-zone rows into whichever half their raw index fell in --
     these are NOT the same partition whenever gray-zone density varies
     across the stream (plausible, e.g. as the cache fills up), so the
     calibration SET itself differed, which is sufficient on its own to
     produce a different Youden's J threshold.
  2. WRONG DENOMINATOR. The established protocol's test-set outcome list is
     "every non-gray-zone row in the ENTIRE trace (direct-accept/reject,
     unaffected by calibration) PLUS only the held-out half of gray-zone
     rows" -- the calibration half of gray-zone rows is excluded from the
     metric entirely, not folded into a temporal "second half of the trace."
     The first version used "second half of the full trace" as N_test,
     which is a different, non-standard denominator.

This version fixes both, replicating scripts/threshold_calibration_
ablation_groupE.py's split/denominator exactly (verified to reproduce its
published baseline threshold and hit_rate/error_rate before trusting any
new joint-model number), and adds a NESTED bootstrap (resampling the
calibration half too, refitting both the Youden's J threshold and the
logistic regression each time, not just resampling the test half against a
fixed fitted model) -- the first version's bootstrap only captured test-
sample variance, not calibration/model-fitting variance, so its CIs likely
understated the true uncertainty.

Usage:
    python scripts/joint_grayzone_decision_analysis_v2.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_LOW = 0.80
TAU_HIGH = 0.97

DATASETS = {
    "lmarena_offtheshelf": {
        "trace": "results/.cache/lmarena__precomputed__n60000.trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "published_reference": None,
    },
    "lmarena_finetuned": {
        "trace": "results/.cache/lmarena__precomputed__n60000.trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json",
        "published_reference": {"threshold": 0.6240989565849304, "hit_rate": 0.9823480721476833, "error_rate": 0.0631580973647231},
    },
    "quora_offtheshelf": {
        "trace": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "scored": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "published_reference": None,
    },
    "searchqueries_offtheshelf": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000.trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "published_reference": None,
    },
    "searchqueries_finetuned": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000.trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json",
        "published_reference": None,
    },
}


def select_threshold_youden(scores: np.ndarray, labels: np.ndarray) -> float:
    """Verbatim port of threshold_calibration_ablation_groupE.py::select_threshold."""
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr, fpr = tp / n_pos, fp / n_neg
    youden_j = tpr - fpr
    return float(sorted_scores[int(np.argmax(youden_j))])


def evaluate_condition(name: str, trace_path: str, scored_path: str, published_reference: dict | None) -> dict:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    scored = json.loads(Path(scored_path).read_text(encoding="utf-8"))
    n_total = len(trace)

    # --- EXACT replication of threshold_calibration_ablation_groupE.py's split ---
    gz_indices = sorted(int(k) for k in scored.keys())
    split = len(gz_indices) // 2
    calib_idx, test_idx = gz_indices[:split], gz_indices[split:]
    print(f"Trace: {n_total} requests. Gray-zone: {len(gz_indices)} rows -> "
          f"{len(calib_idx)} calibration / {len(test_idx)} test (established-protocol split)")

    def features(idx_list):
        sims = np.array([trace[i][0] for i in idx_list])
        labels = np.array([1 if trace[i][1] else 0 for i in idx_list])
        vscores = np.array([scored[str(i)][0] for i in idx_list])
        return sims, vscores, labels

    cal_sim, cal_v, cal_y = features(calib_idx)
    test_sim, test_v, test_y = features(test_idx)

    # --- Baseline: verbatim-matched Youden's J on verifier score alone ---
    baseline_thr = select_threshold_youden(cal_v, cal_y)
    print(f"Baseline (verifier-only) threshold: {baseline_thr:.4f}")
    if published_reference is not None:
        print(f"  Published reference threshold (tau_low={TAU_LOW}): {published_reference['threshold']:.4f}  "
              f"{'MATCH' if abs(baseline_thr - published_reference['threshold']) < 1e-6 else 'MISMATCH -- investigate'}")

    # --- Established-protocol outcome list: every non-gray-zone row in the FULL trace,
    # plus only the held-out (test) half of gray-zone rows. Calibration-half gray-zone
    # rows are excluded from the metric entirely (they were "spent" on calibration). ---
    def full_outcomes_hit_error(gray_zone_test_selected: np.ndarray, gray_zone_test_y: np.ndarray):
        n_direct_accept, fp_direct_accept = 0, 0
        for i, row in enumerate(trace):
            sim = row[0]
            if sim is None:
                continue
            if sim >= TAU_HIGH:
                n_direct_accept += 1
                if not row[1]:
                    fp_direct_accept += 1
        gray_tp = int((gray_zone_test_selected & (gray_zone_test_y == 1)).sum())
        gray_fp = int((gray_zone_test_selected & (gray_zone_test_y == 0)).sum())
        n_test_total = n_total - len(calib_idx)  # full trace minus the "spent" calibration gray-zone rows
        hit_rate = (n_direct_accept + gray_zone_test_selected.sum()) / n_test_total
        error_rate = (fp_direct_accept + gray_fp) / n_test_total
        return hit_rate, error_rate, n_test_total, n_direct_accept, fp_direct_accept

    base_pred = test_v >= baseline_thr
    base_hit, base_err, n_test_total, n_direct_accept, fp_direct_accept = full_outcomes_hit_error(base_pred, test_y)
    print(f"  hit_rate={base_hit:.4f}  error_rate={base_err:.4f}  (n_test_total={n_test_total}, "
          f"direct_accept={n_direct_accept}, fp_direct_accept={fp_direct_accept})")
    if published_reference is not None:
        print(f"  Published reference: hit_rate={published_reference['hit_rate']:.4f}  "
              f"error_rate={published_reference['error_rate']:.4f}  "
              f"{'MATCH' if abs(base_hit - published_reference['hit_rate']) < 1e-3 and abs(base_err - published_reference['error_rate']) < 1e-3 else 'MISMATCH -- investigate'}")

    # --- Joint: 2-feature logistic regression, fit on the SAME calibration set ---
    scaler = StandardScaler().fit(np.column_stack([cal_sim, cal_v]))
    X_cal, X_test = scaler.transform(np.column_stack([cal_sim, cal_v])), scaler.transform(np.column_stack([test_sim, test_v]))
    clf = LogisticRegression(max_iter=2000).fit(X_cal, cal_y)
    joint_prob_cal, joint_prob_test = clf.predict_proba(X_cal)[:, 1], clf.predict_proba(X_test)[:, 1]
    joint_thr = select_threshold_youden(joint_prob_cal, cal_y)
    joint_pred = joint_prob_test >= joint_thr
    joint_hit, joint_err, _, _, _ = full_outcomes_hit_error(joint_pred, test_y)
    print(f"Joint (similarity+verifier) threshold: {joint_thr:.4f}  hit_rate={joint_hit:.4f}  error_rate={joint_err:.4f}")
    print(f"  Point comparison at each method's OWN Youden's J: joint hit_rate {joint_hit - base_hit:+.4f}, "
          f"error_rate {joint_err - base_err:+.4f} vs baseline")

    # --- Frontier-based matched-error-rate comparison (same as v1) ---
    def frontier(scores):
        pts = []
        for t in np.unique(np.percentile(scores, np.arange(1, 100, 1))):
            pred = scores >= t
            h, e, *_ = full_outcomes_hit_error(pred, test_y)
            pts.append((e, h))
        return sorted(pts)

    base_frontier, joint_frontier = frontier(test_v), frontier(joint_prob_test)

    def interp(pts, target_err):
        errs, hits = [p[0] for p in pts], [p[1] for p in pts]
        return float(np.interp(target_err, errs, hits))

    base_hit_at_base_err = interp(base_frontier, base_err)
    joint_hit_at_base_err = interp(joint_frontier, base_err)
    point_delta = joint_hit_at_base_err - base_hit_at_base_err
    print(f"  At baseline's error_rate ({base_err:.4f}): baseline frontier hit_rate={base_hit_at_base_err:.4f}, "
          f"joint frontier hit_rate={joint_hit_at_base_err:.4f}  (delta={point_delta:+.4f})")

    # --- NESTED bootstrap: resample CALIBRATION rows (refit threshold + LR each time) AND
    # resample TEST rows independently, to capture both calibration- and test-sample variance. ---
    rng = np.random.default_rng(0)
    n_cal, n_test = len(calib_idx), len(test_idx)
    n_boot = 500
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        cal_boot = rng.integers(0, n_cal, size=n_cal)
        bc_sim, bc_v, bc_y = cal_sim[cal_boot], cal_v[cal_boot], cal_y[cal_boot]
        if len(np.unique(bc_y)) < 2:
            deltas[b] = np.nan
            continue
        b_base_thr = select_threshold_youden(bc_v, bc_y)
        b_scaler = StandardScaler().fit(np.column_stack([bc_sim, bc_v]))
        b_X_cal = b_scaler.transform(np.column_stack([bc_sim, bc_v]))
        b_clf = LogisticRegression(max_iter=2000).fit(b_X_cal, bc_y)
        b_joint_prob_cal = b_clf.predict_proba(b_X_cal)[:, 1]
        b_joint_thr = select_threshold_youden(b_joint_prob_cal, bc_y)

        test_boot = rng.integers(0, n_test, size=n_test)
        bt_sim, bt_v, bt_y = test_sim[test_boot], test_v[test_boot], test_y[test_boot]
        bt_X = b_scaler.transform(np.column_stack([bt_sim, bt_v]))
        bt_joint_prob = b_clf.predict_proba(bt_X)[:, 1]

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

        b_base_frontier = boot_frontier(bt_v)
        b_joint_frontier = boot_frontier(bt_joint_prob)
        b_base_hit_at_err = interp(b_base_frontier, b_base_e)
        b_joint_hit_at_err = interp(b_joint_frontier, b_base_e)
        deltas[b] = b_joint_hit_at_err - b_base_hit_at_err

    deltas = deltas[~np.isnan(deltas)]
    nested_ci = (float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)))
    nested_significant = nested_ci[0] > 0 or nested_ci[1] < 0
    print(f"  NESTED bootstrap 95% CI (resampling calibration AND test, {len(deltas)} valid iters): "
          f"[{nested_ci[0]:+.4f}, {nested_ci[1]:+.4f}]  "
          f"({'SIGNIFICANT' if nested_significant else 'not significant, CI includes 0'})")

    return {
        "dataset": name,
        "n_total": n_total,
        "n_gray_zone_calibration": len(calib_idx),
        "n_gray_zone_test": len(test_idx),
        "baseline_threshold": baseline_thr,
        "baseline_hit_rate": base_hit,
        "baseline_error_rate": base_err,
        "published_reference": published_reference,
        "published_reference_match": (
            None if published_reference is None else
            bool(abs(base_hit - published_reference["hit_rate"]) < 1e-3 and abs(base_err - published_reference["error_rate"]) < 1e-3)
        ),
        "joint_threshold": joint_thr,
        "joint_hit_rate": joint_hit,
        "joint_error_rate": joint_err,
        "hit_rate_gain_at_matched_error": point_delta,
        "hit_rate_gain_ci95_nested_bootstrap": nested_ci,
        "hit_rate_gain_significant_nested": nested_significant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="results/joint_grayzone_decision_analysis_v2.json")
    args = parser.parse_args()

    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["trace"]).exists() or not Path(cfg["scored"]).exists():
            print(f"Skipping {name}: missing data")
            continue
        results[name] = evaluate_condition(name, cfg["trace"], cfg["scored"], cfg["published_reference"])

    print(f"\n\n{'=' * 100}\nSUMMARY (v2, established-protocol split, nested bootstrap CI)\n{'=' * 100}")
    print(f"{'dataset':>28} | {'gain':>8} | {'nested 95% CI':>24} | {'verdict':>16}")
    for name, r in results.items():
        ci = r["hit_rate_gain_ci95_nested_bootstrap"]
        verdict = "SIGNIFICANT" if r["hit_rate_gain_significant_nested"] else "not significant"
        print(f"{name:>28} | {r['hit_rate_gain_at_matched_error']:+8.4f} | "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {verdict:>16}")

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
