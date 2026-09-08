"""Direction 17 mathematical follow-up #4: forward-chaining CV (previous
script) found that LmArena's calibration window has an internal temporal
trend -- early blocks say "joint decision helps," the block closest to the
real test period says "hurts," matching the true held-out sign -- but
equal-weighted averaging across blocks washes this out, and using only the
last block alone is directionally right yet not statistically significant
(CI crosses zero, block too small).

This script tries the natural middle ground flagged there: instead of the
two extremes (average everything equally, or throw away everything but the
most recent block), fit the SAME baseline threshold and joint logistic
regression using ALL calibration data up to each fold's cutoff, but with
EXPONENTIALLY RECENCY-WEIGHTED sample weights (w_i = decay_rate^(rank of
row i from most recent, within the fit set)). This uses more data than
"last block only" (reducing variance) while still concentrating influence
on the most recent, most deployment-relevant rows (addressing the
non-stationarity forward-chaining CV exposed) -- a single estimator that
should, in principle, beat both prior extremes on both variance and bias.

Youden's J itself is extended to its weighted form (weighted TPR/FPR); the
logistic regression uses sklearn's native `sample_weight` support. Decay is
parameterized by a half-life in rows (weight halves every `half_life` rows
of recency); two half-lives are tried (one aggressive, one gentle) to check
sensitivity rather than picking a single arbitrary value post-hoc.

Usage:
    python scripts/joint_grayzone_recency_weighted_holdback.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_HIGH = 0.97
K_BLOCKS = 5
HALF_LIVES = ["block_size", "half_block_size"]  # relative to len(fit_idx) at each fold, recomputed per fold

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


def weighted_youden_threshold(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(-scores)
    s, l, w = scores[order], labels[order], weights[order]
    w_pos_total, w_neg_total = w[l == 1].sum(), w[l == 0].sum()
    tp_w = np.cumsum(np.where(l == 1, w, 0.0))
    fp_w = np.cumsum(np.where(l == 0, w, 0.0))
    tpr, fpr = tp_w / w_pos_total, fp_w / w_neg_total
    j = tpr - fpr
    return float(s[int(np.argmax(j))])


def recency_weights(n: int, half_life: float) -> np.ndarray:
    """Row 0 = oldest, row n-1 = most recent. Weight decays geometrically going backward from the end."""
    rank_from_recent = np.arange(n - 1, -1, -1)  # most recent row -> 0
    return 0.5 ** (rank_from_recent / half_life)


def main() -> None:
    print(f"{'dataset':>28} | {'equal_avg (ref)':>15} | {'half_life=block':>16} | {'half_life=block/2':>18} | {'audited_test_gain':>17}")
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

        block_bounds = np.linspace(0, len(calib_idx), K_BLOCKS + 1, dtype=int)
        blocks = [calib_idx[block_bounds[i]:block_bounds[i + 1]] for i in range(K_BLOCKS)]

        result_by_hl = {"block_size": [], "half_block_size": []}
        for k in range(1, K_BLOCKS):
            fit_idx = [i for b in blocks[:k] for i in b]
            val_idx = blocks[k]
            if len(val_idx) < 30:
                continue
            f_sim, f_v, f_y = features(fit_idx)
            v_sim, v_v, v_y = features(val_idx)
            if len(np.unique(f_y)) < 2 or len(np.unique(v_y)) < 2:
                continue

            block_size = len(blocks[k - 1]) if k >= 1 else len(fit_idx)
            n_spent = len(fit_idx)

            def outcomes(gray_selected, gray_y):
                gray_fp = int((gray_selected & (gray_y == 0)).sum())
                n_denom = n_total - n_spent
                hit_rate = (n_direct_accept + gray_selected.sum()) / n_denom
                error_rate = (fp_direct_accept + gray_fp) / n_denom
                return hit_rate, error_rate

            def frontier(scores):
                pts = []
                for t in np.unique(np.percentile(scores, np.arange(2, 100, 2))):
                    pred = scores >= t
                    h, e = outcomes(pred, v_y)
                    pts.append((e, h))
                return sorted(pts)

            def interp(pts, target_err):
                errs, hits = [p[0] for p in pts], [p[1] for p in pts]
                return float(np.interp(target_err, errs, hits))

            for hl_name, hl_value in [("block_size", block_size), ("half_block_size", block_size / 2)]:
                w = recency_weights(len(fit_idx), hl_value)
                baseline_thr = weighted_youden_threshold(f_v, f_y, w)
                scaler = StandardScaler().fit(np.column_stack([f_sim, f_v]))
                X_fit = scaler.transform(np.column_stack([f_sim, f_v]))
                clf = LogisticRegression(max_iter=2000).fit(X_fit, f_y, sample_weight=w)
                joint_prob_fit = clf.predict_proba(X_fit)[:, 1]
                joint_thr = weighted_youden_threshold(joint_prob_fit, f_y, w)
                joint_prob_val = clf.predict_proba(scaler.transform(np.column_stack([v_sim, v_v])))[:, 1]

                base_pred = v_v >= baseline_thr
                _, base_err = outcomes(base_pred, v_y)
                base_frontier, joint_frontier = frontier(v_v), frontier(joint_prob_val)
                fold_gain = interp(joint_frontier, base_err) - interp(base_frontier, base_err)
                result_by_hl[hl_name].append((len(val_idx), fold_gain))

        avg_by_hl = {
            hl: float(np.average([g for _, g in vals], weights=[n for n, _ in vals])) if vals else float("nan")
            for hl, vals in result_by_hl.items()
        }

        row = {
            "dataset": name,
            "recency_weighted_gain_halflife_block": avg_by_hl["block_size"],
            "recency_weighted_gain_halflife_halfblock": avg_by_hl["half_block_size"],
            "audited_test_gain": cfg["audited_test_gain"],
        }
        rows.append(row)
        true_sign = "helps" if cfg["audited_test_gain"] > 0 else "hurts"
        hl1_sign = "helps" if avg_by_hl["block_size"] > 0 else "hurts"
        hl2_sign = "helps" if avg_by_hl["half_block_size"] > 0 else "hurts"
        m1 = "MATCH" if hl1_sign == true_sign else "MISMATCH"
        m2 = "MATCH" if hl2_sign == true_sign else "MISMATCH"
        print(f"{name:>28} | {'--':>15} | {avg_by_hl['block_size']:+14.4f} {m1:>1} | "
              f"{avg_by_hl['half_block_size']:+16.4f} {m2:>1} | {cfg['audited_test_gain']:+17.4f}")

    n_match_hl1 = sum(1 for r in rows if (r["recency_weighted_gain_halflife_block"] > 0) == (r["audited_test_gain"] > 0))
    n_match_hl2 = sum(1 for r in rows if (r["recency_weighted_gain_halflife_halfblock"] > 0) == (r["audited_test_gain"] > 0))
    print(f"\nhalf_life=block_size: {n_match_hl1}/{len(rows)} sign matches")
    print(f"half_life=block_size/2: {n_match_hl2}/{len(rows)} sign matches")

    Path("results/joint_grayzone_recency_weighted_holdback.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nWrote results/joint_grayzone_recency_weighted_holdback.json")


if __name__ == "__main__":
    main()
