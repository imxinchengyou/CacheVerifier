"""Direction 18 follow-up: the single 50/50 chronological split used in
topk_cascade_joint_decision_analysis.py turned out to badly mishandle
LmArena's rank-1-reachable subset, whose positive rate climbs from 0% to
83% across the stream (a real cache-density effect: rank-1 candidates are
almost never correct while the cache is sparse early on, and often correct
once it's dense late on) -- a single split puts almost all of one regime in
calibration and almost all of the other in test, inflating the apparent
"joint decision" gain with a large chunk of pure distributional easiness,
not genuine decision-rule quality.

This script reruns the same three-way comparison (baseline: score_2 alone;
simple joint: sim_2+score_2; full joint: sim_1+score_1+sim_2+score_2) using
FORWARD-CHAINING (blocked, chronological) validation instead, the same
honest protocol direction 17 settled on for exactly this kind of
non-stationary subset: split the rank-1-reachable rows into K contiguous
blocks, fit on blocks[0:k] (only past data), validate on block k (never
seen while fitting), and report both the per-fold breakdown (to see whether
the gain itself drifts with cache density, as direction 17's forward-
chaining diagnostic did for the K=1 case) and the weighted-average gain
across folds -- which should be far less inflated by the calibration/test
base-rate gap than the single-split result, since each fold's validation
block is much closer in time (and therefore in cache density / base rate)
to what it was fit on.

Usage:
    python scripts/topk_cascade_forwardchain_analysis.py
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
    },
    "lmarena_finetuned": {
        "trace": "results/.cache/lmarena__precomputed__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__k2__cross_encoder_finetuned_verifier_model_lmarena__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 0.6241,
    },
    "quora_offtheshelf": {
        "trace": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 3.7386,
    },
    "searchqueries_offtheshelf": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000__k2.cascade_trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 6.7433,
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


def local_outcomes(selected: np.ndarray, y: np.ndarray):
    n = len(y)
    hit_rate = selected.sum() / n
    error_rate = int((selected & (y == 0)).sum()) / n
    return hit_rate, error_rate


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
    print(f"{'dataset':>28} | {'fwd_chain_simple':>16} | {'fwd_chain_full':>14} | verdict")
    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["trace"]).exists() or not Path(cfg["scored"]).exists():
            print(f"Skipping {name}: missing data")
            continue

        rows = collect_rows(cfg["trace"], cfg["scored"], cfg["rank0_threshold"])
        n = len(rows)
        if n < 500:
            print(f"Skipping {name}: only {n} rows")
            continue

        bounds = np.linspace(0, n, K_BLOCKS + 1, dtype=int)
        blocks = [rows[bounds[i]:bounds[i + 1]] for i in range(K_BLOCKS)]
        print(f"\n{name}: n={n}, block positive rates: "
              f"{[round(np.mean([r[5] for r in b]), 3) for b in blocks]}")

        def arrays(rs):
            arr = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in rs])
            return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4].astype(int)

        fold_simple, fold_full, fold_weights = [], [], []
        for k in range(1, K_BLOCKS):
            fit_rows = [r for b in blocks[:k] for r in b]
            val_rows = blocks[k]
            if len(val_rows) < 30:
                continue
            f_s0, f_v0, f_s1, f_v1, f_y = arrays(fit_rows)
            v_s0, v_v0, v_s1, v_v1, v_y = arrays(val_rows)
            if len(np.unique(f_y)) < 2 or len(np.unique(v_y)) < 2:
                continue

            base_thr = select_threshold_youden(f_v1, f_y)
            base_pred = v_v1 >= base_thr
            base_hit, base_err = local_outcomes(base_pred, v_y)
            base_frontier = frontier(v_v1, v_y)

            scaler_b = StandardScaler().fit(np.column_stack([f_s1, f_v1]))
            clf_b = LogisticRegression(max_iter=2000).fit(scaler_b.transform(np.column_stack([f_s1, f_v1])), f_y)
            prob_b_val = clf_b.predict_proba(scaler_b.transform(np.column_stack([v_s1, v_v1])))[:, 1]
            gain_simple = interp(frontier(prob_b_val, v_y), base_err) - base_hit

            scaler_c = StandardScaler().fit(np.column_stack([f_s0, f_v0, f_s1, f_v1]))
            clf_c = LogisticRegression(max_iter=2000).fit(scaler_c.transform(np.column_stack([f_s0, f_v0, f_s1, f_v1])), f_y)
            prob_c_val = clf_c.predict_proba(scaler_c.transform(np.column_stack([v_s0, v_v0, v_s1, v_v1])))[:, 1]
            gain_full = interp(frontier(prob_c_val, v_y), base_err) - base_hit

            fold_simple.append(gain_simple)
            fold_full.append(gain_full)
            fold_weights.append(len(val_rows))
            print(f"  fold k={k}: fit_n={len(fit_rows)} val_n={len(val_rows)} val_pos_rate={v_y.mean():.3f}  "
                  f"gain_simple={gain_simple:+.4f}  gain_full={gain_full:+.4f}")

        if not fold_simple:
            print(f"  No usable folds for {name}")
            continue

        avg_simple = float(np.average(fold_simple, weights=fold_weights))
        avg_full = float(np.average(fold_full, weights=fold_weights))
        last_simple, last_full = fold_simple[-1], fold_full[-1]

        results[name] = {
            "n_rows": n,
            "block_positive_rates": [float(np.mean([r[5] for r in b])) for b in blocks],
            "fold_gain_simple": fold_simple,
            "fold_gain_full": fold_full,
            "forward_chain_avg_simple": avg_simple,
            "forward_chain_avg_full": avg_full,
            "last_fold_simple": last_simple,
            "last_fold_full": last_full,
            "single_split_reference_simple": None,  # filled from prior script's already-published numbers manually
        }
        print(f"  -> forward-chain weighted avg: simple={avg_simple:+.4f}  full={avg_full:+.4f}  "
              f"(last fold: simple={last_simple:+.4f}  full={last_full:+.4f})")

    print(f"\n\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    for name, r in results.items():
        print(f"{name:>28} | fwd_chain_avg simple={r['forward_chain_avg_simple']:+.4f} full={r['forward_chain_avg_full']:+.4f} "
              f"| last_fold simple={r['last_fold_simple']:+.4f} full={r['last_fold_full']:+.4f}")

    Path("results/topk_cascade_forwardchain_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote results/topk_cascade_forwardchain_analysis.json")


if __name__ == "__main__":
    main()
