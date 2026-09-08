"""Direction 18 follow-up: forward-chaining validation (previous script)
showed LmArena's rank-1-reachable subset has a positive rate that climbs
from 0% to 84% across the stream, and the "joint decision" gain tracks that
climb almost in lockstep fold-by-fold (score_1-only baseline: +1.6% -> +8.9%
-> +20.5% -> +22.6%) -- strong circumstantial evidence that a large share of
the apparent gain is really "the validation block happens to be easier,"
not "similarity/joint signals genuinely help more."

This script tests that directly with EXPLICIT detrending: add stream
position (i / n_total, a direct proxy for how mature/dense the cache was
when the request arrived) as an additional feature to BOTH the baseline and
the joint model, so both get to "know" the trend and the comparison
isolates whatever incremental value similarity/joint signals add ON TOP OF
that trend, rather than conflating trend-awareness with joint-decision
quality:
  (a') trend-aware baseline: (stream_position, score_1) -> label
  (b') trend-aware simple joint: (stream_position, sim_1, score_1) -> label
  (c') trend-aware full joint: (stream_position, sim_0, score_0, sim_1, score_1) -> label
Compared against the ORIGINAL (non-trend-aware) baseline and joint numbers
from topk_cascade_forwardchain_analysis.py, on the same forward-chaining
folds, to see how much of the previously observed gain survives once trend
awareness is given to everyone (including the baseline).

Usage:
    python scripts/topk_cascade_detrended_analysis.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_LOW, TAU_HIGH = 0.80, 0.97
K_BLOCKS = 5

DATASETS = {
    "lmarena_offtheshelf": {
        "trace": "results/.cache/lmarena__precomputed__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": -1.2026,
        "n_total": 60000,
    },
    "lmarena_finetuned": {
        "trace": "results/.cache/lmarena__precomputed__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__k2__cross_encoder_finetuned_verifier_model_lmarena__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 0.6241,
        "n_total": 60000,
    },
    "quora_offtheshelf": {
        "trace": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 3.7386,
        "n_total": 60000,
    },
    "searchqueries_offtheshelf": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000__k2.cascade_trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 6.7433,
        "n_total": 150000,
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


def local_outcomes(selected, y):
    n = len(y)
    return selected.sum() / n, int((selected & (y == 0)).sum()) / n


def frontier(scores, y):
    pts = []
    for th in np.unique(np.percentile(scores, np.arange(2, 100, 2))):
        pred = scores >= th
        h, e = local_outcomes(pred, y)
        pts.append((e, h))
    return sorted(pts)


def interp(pts, target_err):
    errs, hits = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(target_err, errs, hits))


def fit_and_score(train_X, train_y, val_X):
    """Fit logistic regression on train; return (val_probabilities, train_calibrated_threshold)
    where the threshold is Youden's J on the model's OWN fitted probabilities on the training set."""
    scaler = StandardScaler().fit(train_X)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(train_X), train_y)
    prob_train = clf.predict_proba(scaler.transform(train_X))[:, 1]
    thr = select_threshold_youden(prob_train, train_y)
    prob_val = clf.predict_proba(scaler.transform(val_X))[:, 1]
    return prob_val, thr


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


def main() -> None:
    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["trace"]).exists() or not Path(cfg["scored"]).exists():
            print(f"Skipping {name}: missing data")
            continue

        rows = collect_rows(cfg["trace"], cfg["scored"], cfg["rank0_threshold"])
        n = len(rows)
        if n < 500:
            continue
        n_total = cfg["n_total"]

        bounds = np.linspace(0, n, K_BLOCKS + 1, dtype=int)
        blocks = [rows[bounds[i]:bounds[i + 1]] for i in range(K_BLOCKS)]
        print(f"\n{'=' * 90}\n{name}: n={n}\n{'=' * 90}")

        def arrays(rs):
            arr = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rs])
            pos = arr[:, 0] / n_total  # stream position feature
            return pos, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5].astype(int)

        fold_data = []
        for k in range(1, K_BLOCKS):
            fit_rows = [r for b in blocks[:k] for r in b]
            val_rows = blocks[k]
            if len(val_rows) < 30:
                continue
            f_pos, f_s0, f_v0, f_s1, f_v1, f_y = arrays(fit_rows)
            v_pos, v_s0, v_v0, v_s1, v_v1, v_y = arrays(val_rows)
            if len(np.unique(f_y)) < 2 or len(np.unique(v_y)) < 2:
                continue

            # (a) original baseline: score_1 alone
            base_thr = select_threshold_youden(f_v1, f_y)
            base_pred = v_v1 >= base_thr
            base_hit, base_err = local_outcomes(base_pred, v_y)

            # (a') trend-aware baseline: (stream_pos, score_1)
            prob_a2, thr_a2 = fit_and_score(np.column_stack([f_pos, f_v1]), f_y, np.column_stack([v_pos, v_v1]))
            trend_base_pred = prob_a2 >= thr_a2
            trend_base_hit, trend_base_err = local_outcomes(trend_base_pred, v_y)
            gain_a2 = interp(frontier(prob_a2, v_y), base_err) - base_hit

            # (b) original simple joint: (sim_1, score_1)
            prob_b, _ = fit_and_score(np.column_stack([f_s1, f_v1]), f_y, np.column_stack([v_s1, v_v1]))
            gain_b = interp(frontier(prob_b, v_y), base_err) - base_hit

            # (b') trend-aware simple joint: (stream_pos, sim_1, score_1)
            prob_b2, _ = fit_and_score(np.column_stack([f_pos, f_s1, f_v1]), f_y, np.column_stack([v_pos, v_s1, v_v1]))
            gain_b2 = interp(frontier(prob_b2, v_y), base_err) - base_hit

            # (c') trend-aware full joint: (stream_pos, sim_0, score_0, sim_1, score_1)
            prob_c2, _ = fit_and_score(
                np.column_stack([f_pos, f_s0, f_v0, f_s1, f_v1]), f_y,
                np.column_stack([v_pos, v_s0, v_v0, v_s1, v_v1]),
            )
            gain_c2 = interp(frontier(prob_c2, v_y), base_err) - base_hit

            # The key comparison: incremental value of joint signals OVER a trend-aware baseline,
            # re-anchored at the trend-aware baseline's own (hit_rate, error_rate) operating point.
            incremental_b2 = interp(frontier(prob_b2, v_y), trend_base_err) - trend_base_hit
            incremental_c2 = interp(frontier(prob_c2, v_y), trend_base_err) - trend_base_hit

            fold_data.append({
                "k": k, "val_pos_rate": float(v_y.mean()), "n_val": len(val_rows),
                "gain_original_simple_joint": gain_b,
                "gain_trend_aware_baseline": gain_a2,
                "gain_trend_aware_simple_joint": gain_b2,
                "gain_trend_aware_full_joint": gain_c2,
                "incremental_simple_joint_over_trend_baseline": incremental_b2,
                "incremental_full_joint_over_trend_baseline": incremental_c2,
            })
            print(f"  fold k={k}: val_pos_rate={v_y.mean():.3f}  "
                  f"orig_simple_joint={gain_b:+.4f}  trend_baseline={gain_a2:+.4f}  "
                  f"trend_simple_joint={gain_b2:+.4f}  trend_full_joint={gain_c2:+.4f}  "
                  f"| incremental(simple)={incremental_b2:+.4f}  incremental(full)={incremental_c2:+.4f}")

        if not fold_data:
            continue

        weights = [f["n_val"] for f in fold_data]
        avg_orig_gain = float(np.average([f["gain_original_simple_joint"] for f in fold_data], weights=weights))
        avg_trend_baseline_gain = float(np.average([f["gain_trend_aware_baseline"] for f in fold_data], weights=weights))
        avg_incremental_simple = float(np.average([f["incremental_simple_joint_over_trend_baseline"] for f in fold_data], weights=weights))
        avg_incremental_full = float(np.average([f["incremental_full_joint_over_trend_baseline"] for f in fold_data], weights=weights))

        results[name] = {
            "n_rows": n,
            "folds": fold_data,
            "avg_original_gain_vs_naive_baseline": avg_orig_gain,
            "avg_trend_aware_baseline_gain_vs_naive_baseline": avg_trend_baseline_gain,
            "avg_incremental_gain_simple_joint_over_trend_baseline": avg_incremental_simple,
            "avg_incremental_gain_full_joint_over_trend_baseline": avg_incremental_full,
        }
        print(f"  -> AVG: original gain (vs naive baseline)={avg_orig_gain:+.4f}  "
              f"trend-aware baseline alone (vs naive baseline)={avg_trend_baseline_gain:+.4f}  "
              f"|  INCREMENTAL joint value once trend is controlled: simple={avg_incremental_simple:+.4f}  full={avg_incremental_full:+.4f}")

    print(f"\n\n{'=' * 110}\nSUMMARY: how much of the original gain survives once BOTH sides get trend-awareness\n{'=' * 110}")
    print(f"{'dataset':>28} | {'orig_gain':>10} | {'trend_baseline_alone':>20} | {'incremental_simple':>18} | {'incremental_full':>16}")
    for name, r in results.items():
        print(f"{name:>28} | {r['avg_original_gain_vs_naive_baseline']:+10.4f} | "
              f"{r['avg_trend_aware_baseline_gain_vs_naive_baseline']:+20.4f} | "
              f"{r['avg_incremental_gain_simple_joint_over_trend_baseline']:+18.4f} | "
              f"{r['avg_incremental_gain_full_joint_over_trend_baseline']:+16.4f}")

    Path("results/topk_cascade_detrended_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote results/topk_cascade_detrended_analysis.json")


if __name__ == "__main__":
    main()
