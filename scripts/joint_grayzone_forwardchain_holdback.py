"""Direction 17 mathematical follow-up #3: the single-split internal
cross-validation (previous script) had 2/5 false positives, both on the
LmArena conditions where the true effect is small and the sub-calibration
sample was noisy (half of an already-limited calibration set). The natural
fix flagged there is averaging over multiple internal splits to reduce
variance -- but a NAIVE random k-fold split on this chronologically-ordered
gray-zone stream would reintroduce exactly the optimism bias this paper has
repeatedly caught and corrected elsewhere (Section 5.13's Protocol R vs
Protocol T: random splits look artificially stable/safe, chronological
splits reveal the real risk). Random shuffling here would let "future"
calibration rows help fit a model validated against "past" rows (or vice
versa), which does not reflect the real deployment question ("having seen
calibration data up to now, would enabling joint decision have helped on
what came next").

This script instead implements FORWARD-CHAINING (blocked, chronological)
cross-validation, the standard honest approach for time-ordered data: split
each condition's calibration set into K contiguous blocks in stream order;
for each split point k=1..K-1, fit on blocks[0:k] concatenated (only past
data) and validate the matched-error-rate gain on block k alone (the next
chronological block, never seen during fitting); average the per-fold gains
(weighted by validation-block size) into a single forward-chaining estimate,
and check whether ITS sign matches the true, already-audited held-out test
result across all 5 conditions -- and whether it is at least as reliable as
the single-split internal CV that produced 2 dangerous false positives.

Usage:
    python scripts/joint_grayzone_forwardchain_holdback.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_HIGH = 0.97
K_BLOCKS = 5

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
    print(f"{'dataset':>28} | {'fwd_chain_gain':>14} | {'single_split_gain':>17} | {'audited_test_gain':>17} | verdict")
    rows = []
    for name, cfg in DATASETS.items():
        trace = json.loads(Path(cfg["trace"]).read_text(encoding="utf-8"))
        scored = json.loads(Path(cfg["scored"]).read_text(encoding="utf-8"))
        n_total = len(trace)
        gz_indices = sorted(int(k) for k in scored.keys())
        split = len(gz_indices) // 2
        calib_idx = gz_indices[:split]

        n_direct_accept, fp_direct_accept = 0, 0
        for row in trace:
            sim = row[0]
            if sim is not None and sim >= TAU_HIGH:
                n_direct_accept += 1
                if not row[1]:
                    fp_direct_accept += 1

        def features(idx_list):
            sims = np.array([trace[i][0] for i in idx_list])
            labels = np.array([1 if trace[i][1] else 0 for i in idx_list])
            vscores = np.array([scored[str(i)][0] for i in idx_list])
            return sims, vscores, labels

        # K contiguous blocks over the calibration set, in stream order.
        block_bounds = np.linspace(0, len(calib_idx), K_BLOCKS + 1, dtype=int)
        blocks = [calib_idx[block_bounds[i]:block_bounds[i + 1]] for i in range(K_BLOCKS)]

        fold_gains, fold_weights = [], []
        for k in range(1, K_BLOCKS):
            fit_idx = [i for b in blocks[:k] for i in b]
            val_idx = blocks[k]
            if len(val_idx) < 30:
                continue
            f_sim, f_v, f_y = features(fit_idx)
            v_sim, v_v, v_y = features(val_idx)
            if len(np.unique(f_y)) < 2 or len(np.unique(v_y)) < 2:
                continue

            baseline_thr = select_threshold_youden(f_v, f_y)
            scaler = StandardScaler().fit(np.column_stack([f_sim, f_v]))
            X_fit = scaler.transform(np.column_stack([f_sim, f_v]))
            clf = LogisticRegression(max_iter=2000).fit(X_fit, f_y)
            joint_thr = select_threshold_youden(clf.predict_proba(X_fit)[:, 1], f_y)
            joint_prob_val = clf.predict_proba(scaler.transform(np.column_stack([v_sim, v_v])))[:, 1]

            n_spent = len(fit_idx)  # data used to fit this fold, excluded from the denominator

            def outcomes(gray_selected, gray_y):
                gray_fp = int((gray_selected & (gray_y == 0)).sum())
                n_denom = n_total - n_spent
                hit_rate = (n_direct_accept + gray_selected.sum()) / n_denom
                error_rate = (fp_direct_accept + gray_fp) / n_denom
                return hit_rate, error_rate

            base_pred = v_v >= baseline_thr
            base_hit, base_err = outcomes(base_pred, v_y)

            def frontier(scores):
                pts = []
                for t in np.unique(np.percentile(scores, np.arange(2, 100, 2))):
                    pred = scores >= t
                    h, e = outcomes(pred, v_y)
                    pts.append((e, h))
                return sorted(pts)

            base_frontier, joint_frontier = frontier(v_v), frontier(joint_prob_val)

            def interp(pts, target_err):
                errs, hits = [p[0] for p in pts], [p[1] for p in pts]
                return float(np.interp(target_err, errs, hits))

            fold_gain = interp(joint_frontier, base_err) - interp(base_frontier, base_err)
            fold_gains.append(fold_gain)
            fold_weights.append(len(val_idx))

        fwd_chain_gain = float(np.average(fold_gains, weights=fold_weights))

        predicted_sign = "helps" if fwd_chain_gain > 0 else "hurts"
        actual_sign = "helps" if cfg["audited_test_gain"] > 0 else "hurts"
        match = "MATCH" if predicted_sign == actual_sign else "MISMATCH"

        row = {
            "dataset": name,
            "n_folds_used": len(fold_gains),
            "fold_gains": fold_gains,
            "forward_chaining_gain": fwd_chain_gain,
            "audited_test_gain": cfg["audited_test_gain"],
            "sign_match": match,
        }
        rows.append(row)
        print(f"{name:>28} | {fwd_chain_gain:+14.4f} | {'(see prior script)':>17} | "
              f"{cfg['audited_test_gain']:+17.4f} | {match}  (folds: {[f'{g:+.4f}' for g in fold_gains]})")

    n_match = sum(1 for r in rows if r["sign_match"] == "MATCH")
    print(f"\n{n_match}/{len(rows)} conditions: forward-chaining sign matches the true held-out sign.")

    Path("results/joint_grayzone_forwardchain_holdback.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote results/joint_grayzone_forwardchain_holdback.json")


if __name__ == "__main__":
    main()
