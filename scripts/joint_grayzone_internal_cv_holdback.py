"""Direction 17 mathematical follow-up #2: the Wald-test shortcut (previous
script) was cleanly refuted -- calibration-time coefficient significance
does not predict whether joint decision helps or hurts on held-out data.
The natural next candidate, flagged as this project's own suggested
alternative in that script's limitations, is not a free closed-form
shortcut but a proper (still free of any NEW data) internal cross-
validation: split the CALIBRATION set itself further into a sub-calibration
and a sub-validation half, fit both the baseline and joint decision rules
on sub-calibration, and measure the matched-error-rate gain on sub-
validation using the exact same established-protocol machinery. If this
internal estimate's SIGN correctly predicts the sign of the true (genuinely
held-out, already-audited) test-half result across all 5 conditions, that
validates a concrete, deployable procedure: before turning on joint
decision for a new tenant, run this internal split-and-check on existing
calibration data alone -- no live A/B test or new held-out data required.

Method: within each condition's calibration set (the same one used to fit
the audited baseline/joint models in joint_grayzone_decision_analysis_v2.py),
apply the identical chronological-split logic one level deeper (split
calibration's gray-zone rows in half by count, sub_calib = first half,
sub_valid = second half), fit baseline Youden's J and the joint logistic
regression on sub_calib only, then measure the matched-error-rate gain on
sub_valid using the same full-trace-relative hit_rate/error_rate definition
(non-gray-zone rows contribute their fixed, method-independent baseline;
sub_calib's own rows are excluded from the denominator, matching the
established protocol's convention of never scoring on rows spent on
fitting).

Usage:
    python scripts/joint_grayzone_internal_cv_holdback.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_HIGH = 0.97

DATASETS = {
    "lmarena_offtheshelf": {
        "trace": "results/.cache/lmarena__precomputed__n60000.trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "audited_test_gain": -0.0105,
    },
    "lmarena_finetuned": {
        "trace": "results/.cache/lmarena__precomputed__n60000.trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json",
        "audited_test_gain": -0.0009,
    },
    "quora_offtheshelf": {
        "trace": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "scored": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "audited_test_gain": +0.0056,
    },
    "searchqueries_offtheshelf": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000.trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "audited_test_gain": +0.0169,
    },
    "searchqueries_finetuned": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000.trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json",
        "audited_test_gain": +0.0017,
    },
}


def select_threshold_youden(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr, fpr = tp / n_pos, fp / n_neg
    youden_j = tpr - fpr
    return float(sorted_scores[int(np.argmax(youden_j))])


def main() -> None:
    print(f"{'dataset':>28} | {'internal_cv_gain':>16} | {'audited_test_gain':>17} | verdict")
    rows = []
    for name, cfg in DATASETS.items():
        trace = json.loads(Path(cfg["trace"]).read_text(encoding="utf-8"))
        scored = json.loads(Path(cfg["scored"]).read_text(encoding="utf-8"))
        n_total = len(trace)
        gz_indices = sorted(int(k) for k in scored.keys())
        split = len(gz_indices) // 2
        calib_idx = gz_indices[:split]  # this condition's established-protocol calibration set

        # One level deeper: split calibration itself in half by count (same convention).
        sub_split = len(calib_idx) // 2
        sub_calib_idx, sub_valid_idx = calib_idx[:sub_split], calib_idx[sub_split:]

        def features(idx_list):
            sims = np.array([trace[i][0] for i in idx_list])
            labels = np.array([1 if trace[i][1] else 0 for i in idx_list])
            vscores = np.array([scored[str(i)][0] for i in idx_list])
            return sims, vscores, labels

        sc_sim, sc_v, sc_y = features(sub_calib_idx)
        sv_sim, sv_v, sv_y = features(sub_valid_idx)

        n_direct_accept, fp_direct_accept = 0, 0
        for row in trace:
            sim = row[0]
            if sim is not None and sim >= TAU_HIGH:
                n_direct_accept += 1
                if not row[1]:
                    fp_direct_accept += 1

        baseline_thr = select_threshold_youden(sc_v, sc_y)

        scaler = StandardScaler().fit(np.column_stack([sc_sim, sc_v]))
        X_sc = scaler.transform(np.column_stack([sc_sim, sc_v]))
        clf = LogisticRegression(max_iter=2000).fit(X_sc, sc_y)
        joint_thr = select_threshold_youden(clf.predict_proba(X_sc)[:, 1], sc_y)

        X_sv = scaler.transform(np.column_stack([sv_sim, sv_v]))
        joint_prob_sv = clf.predict_proba(X_sv)[:, 1]

        n_spent = len(sub_calib_idx)  # rows "used up" fitting, excluded from the denominator

        def outcomes(gray_selected, gray_y):
            gray_fp = int((gray_selected & (gray_y == 0)).sum())
            n_denom = n_total - n_spent
            hit_rate = (n_direct_accept + gray_selected.sum()) / n_denom
            error_rate = (fp_direct_accept + gray_fp) / n_denom
            return hit_rate, error_rate

        base_pred = sv_v >= baseline_thr
        base_hit, base_err = outcomes(base_pred, sv_y)

        def frontier(scores):
            pts = []
            for t in np.unique(np.percentile(scores, np.arange(1, 100, 1))):
                pred = scores >= t
                h, e = outcomes(pred, sv_y)
                pts.append((e, h))
            return sorted(pts)

        base_frontier, joint_frontier = frontier(sv_v), frontier(joint_prob_sv)

        def interp(pts, target_err):
            errs, hits = [p[0] for p in pts], [p[1] for p in pts]
            return float(np.interp(target_err, errs, hits))

        internal_cv_gain = interp(joint_frontier, base_err) - interp(base_frontier, base_err)

        predicted_sign = "helps" if internal_cv_gain > 0 else "hurts"
        actual_sign = "helps" if cfg["audited_test_gain"] > 0 else "hurts"
        match = "MATCH" if predicted_sign == actual_sign else "MISMATCH"

        row = {
            "dataset": name,
            "n_sub_calibration": len(sub_calib_idx),
            "n_sub_validation": len(sub_valid_idx),
            "internal_cv_gain": internal_cv_gain,
            "audited_test_gain": cfg["audited_test_gain"],
            "sign_match": match,
        }
        rows.append(row)
        print(f"{name:>28} | {internal_cv_gain:+16.4f} | {cfg['audited_test_gain']:+17.4f} | {match}")

    n_match = sum(1 for r in rows if r["sign_match"] == "MATCH")
    print(f"\n{n_match}/{len(rows)} conditions: internal-CV sign matches the true held-out sign.")

    Path("results/joint_grayzone_internal_cv_holdback.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote results/joint_grayzone_internal_cv_holdback.json")


if __name__ == "__main__":
    main()
