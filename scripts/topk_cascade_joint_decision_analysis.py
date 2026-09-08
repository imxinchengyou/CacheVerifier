"""Direction 17 generalized to Top-K cascade (Section 5.15): does the same
"cascade discards information a joint decision could use" principle, just
causally validated for the K=1 gray-zone gate, also apply to the K=2
candidate cascade?

`cacheverifier/experiments/topk_sweep.py::replay_cascade` decides rank 1
(the second candidate, only reached if rank 0 was rejected) using ONLY
`score_1 >= threshold` -- discarding rank 1's own similarity AND everything
already learned about rank 0 (its similarity and verifier score, both of
which were computed and are sitting right there, since rank 0 had to be
scored and rejected before rank 1 is ever reached). This mirrors exactly the
K=1 gray-zone architecture direction 17 found leaves value on the table
when the verifier's own discriminative power is limited -- except here
there is EVEN MORE discarded information (four signals: sim_0, score_0,
sim_1, score_1), not just two.

This script isolates the rank-1 decision as the sole experimental variable:
rank 0's policy is held FIXED at the already-established, already-audited
K=1 Youden's J threshold from joint_grayzone_decision_analysis_v2.py (since
rank 0's situation IS the K=1 gray-zone situation on the same verifier/
dataset/tau bounds) -- this avoids the confound of different rank-0 policies
changing which rows even reach rank 1. Within the population that reaches a
genuine rank-1 gray-zone decision, three candidate decision rules are
compared:
  (a) baseline: accept iff score_1 >= threshold (today's cascade logic)
  (b) simple joint: accept iff f(sim_1, score_1) >= threshold (direction 17's
      principle applied fresh at rank 1, ignoring rank 0's history)
  (c) full joint: accept iff f(sim_0, score_0, sim_1, score_1) >= threshold
      (uses everything already known, including the rejected rank-0
      candidate's own evidence)

Metrics are LOCAL to this rank-1-reachable subpopulation (not renormalized
against the full trace) since every row outside this subpopulation is
policy-invariant across (a)/(b)/(c) by construction -- a fixed additive
term identical for all three, so comparing local hit_rate/error_rate
frontiers is sufficient to determine direction and relative magnitude.

Usage:
    python scripts/topk_cascade_joint_decision_analysis.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TAU_LOW, TAU_HIGH = 0.80, 0.97

DATASETS = {
    "lmarena_offtheshelf": {
        "trace": "results/.cache/lmarena__precomputed__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__k2__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": -1.2026,  # from joint_grayzone_decision_analysis_v2.py's audited K=1 baseline
    },
    "lmarena_finetuned": {
        "trace": "results/.cache/lmarena__precomputed__n60000__k2.cascade_trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__k2__cross_encoder_finetuned_verifier_model_lmarena__lo0.8__hi0.97.cascade_scored.json",
        "rank0_threshold": 0.6241,  # from joint_grayzone_decision_analysis_v2.py's audited K=1 baseline
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


def evaluate_condition(name: str, cfg: dict) -> dict:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    trace = json.loads(Path(cfg["trace"]).read_text(encoding="utf-8"))
    scored = json.loads(Path(cfg["scored"]).read_text(encoding="utf-8"))
    rank0_thr = cfg["rank0_threshold"]

    rows = []  # (i, sim0, score0, sim1, score1, label1)
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
            continue  # accepted at rank 0, never reaches rank 1
        s1 = sims[1]
        if not (TAU_LOW <= s1 < TAU_HIGH):
            continue
        key1 = f"{i}_1"
        if key1 not in scored:
            continue
        score1 = scored[key1][0]
        label1 = 1 if corrects[1] else 0
        rows.append((i, s0, score0, s1, score1, label1))

    print(f"Rows reaching a genuine rank-1 gray-zone decision: {len(rows)}")
    if len(rows) < 200:
        print("Too few rows, skipping.")
        return None

    rows.sort(key=lambda r: r[0])  # chronological order (original stream index)
    split = len(rows) // 2
    calib, test = rows[:split], rows[split:]

    def arrays(rs):
        arr = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in rs])
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4].astype(int)

    c_s0, c_v0, c_s1, c_v1, c_y = arrays(calib)
    t_s0, t_v0, t_s1, t_v1, t_y = arrays(test)
    print(f"Calibration: {len(c_y)} rows, {c_y.mean():.4f} positive. Test: {len(t_y)} rows, {t_y.mean():.4f} positive.")

    def local_outcomes(selected: np.ndarray, y: np.ndarray):
        n = len(y)
        hit_rate = selected.sum() / n
        error_rate = int((selected & (y == 0)).sum()) / n
        return hit_rate, error_rate

    def frontier(scores, y):
        pts = []
        for th in np.unique(np.percentile(scores, np.arange(1, 100, 1))):
            pred = scores >= th
            h, e = local_outcomes(pred, y)
            pts.append((e, h))
        return sorted(pts)

    def interp(pts, target_err):
        errs, hits = [p[0] for p in pts], [p[1] for p in pts]
        return float(np.interp(target_err, errs, hits))

    # (a) baseline: score_1 alone
    base_thr = select_threshold_youden(c_v1, c_y)
    base_pred = t_v1 >= base_thr
    base_hit, base_err = local_outcomes(base_pred, t_y)
    base_frontier = frontier(t_v1, t_y)

    # (b) simple joint: (sim_1, score_1)
    scaler_b = StandardScaler().fit(np.column_stack([c_s1, c_v1]))
    Xb_c, Xb_t = scaler_b.transform(np.column_stack([c_s1, c_v1])), scaler_b.transform(np.column_stack([t_s1, t_v1]))
    clf_b = LogisticRegression(max_iter=2000).fit(Xb_c, c_y)
    thr_b = select_threshold_youden(clf_b.predict_proba(Xb_c)[:, 1], c_y)
    prob_b_test = clf_b.predict_proba(Xb_t)[:, 1]
    b_frontier = frontier(prob_b_test, t_y)
    b_hit_at_base_err = interp(b_frontier, base_err)

    # (c) full joint: (sim_0, score_0, sim_1, score_1)
    scaler_c = StandardScaler().fit(np.column_stack([c_s0, c_v0, c_s1, c_v1]))
    Xc_c = scaler_c.transform(np.column_stack([c_s0, c_v0, c_s1, c_v1]))
    Xc_t = scaler_c.transform(np.column_stack([t_s0, t_v0, t_s1, t_v1]))
    clf_c = LogisticRegression(max_iter=2000).fit(Xc_c, c_y)
    thr_c = select_threshold_youden(clf_c.predict_proba(Xc_c)[:, 1], c_y)
    prob_c_test = clf_c.predict_proba(Xc_t)[:, 1]
    c_frontier = frontier(prob_c_test, t_y)
    c_hit_at_base_err = interp(c_frontier, base_err)

    print(f"Baseline (score_1 only): hit_rate={base_hit:.4f} error_rate={base_err:.4f}")
    print(f"At baseline's error_rate ({base_err:.4f}):")
    print(f"  (b) simple joint (sim1,score1): hit_rate={b_hit_at_base_err:.4f}  delta={b_hit_at_base_err - base_hit:+.4f}")
    print(f"  (c) full joint (sim0,v0,sim1,v1): hit_rate={c_hit_at_base_err:.4f}  delta={c_hit_at_base_err - base_hit:+.4f}")

    # nested bootstrap for (c) vs baseline, the headline comparison
    rng = np.random.default_rng(0)
    n_cal, n_test = len(calib), len(test)
    n_boot = 500
    deltas_b, deltas_c = np.empty(n_boot), np.empty(n_boot)
    for bi in range(n_boot):
        cal_boot = rng.integers(0, n_cal, size=n_cal)
        bc_s0, bc_v0, bc_s1, bc_v1, bc_y = c_s0[cal_boot], c_v0[cal_boot], c_s1[cal_boot], c_v1[cal_boot], c_y[cal_boot]
        if len(np.unique(bc_y)) < 2:
            deltas_b[bi] = deltas_c[bi] = np.nan
            continue
        b_base_thr = select_threshold_youden(bc_v1, bc_y)
        bs_b = StandardScaler().fit(np.column_stack([bc_s1, bc_v1]))
        bclf_b = LogisticRegression(max_iter=2000).fit(bs_b.transform(np.column_stack([bc_s1, bc_v1])), bc_y)
        bs_c = StandardScaler().fit(np.column_stack([bc_s0, bc_v0, bc_s1, bc_v1]))
        bclf_c = LogisticRegression(max_iter=2000).fit(bs_c.transform(np.column_stack([bc_s0, bc_v0, bc_s1, bc_v1])), bc_y)

        test_boot = rng.integers(0, n_test, size=n_test)
        bt_s0, bt_v0, bt_s1, bt_v1, bt_y = t_s0[test_boot], t_v0[test_boot], t_s1[test_boot], t_v1[test_boot], t_y[test_boot]

        bt_base_pred = bt_v1 >= b_base_thr
        _, bt_base_err = local_outcomes(bt_base_pred, bt_y)
        bt_prob_b = bclf_b.predict_proba(bs_b.transform(np.column_stack([bt_s1, bt_v1])))[:, 1]
        bt_prob_c = bclf_c.predict_proba(bs_c.transform(np.column_stack([bt_s0, bt_v0, bt_s1, bt_v1])))[:, 1]

        bt_base_frontier = frontier(bt_v1, bt_y)
        bt_b_frontier = frontier(bt_prob_b, bt_y)
        bt_c_frontier = frontier(bt_prob_c, bt_y)
        deltas_b[bi] = interp(bt_b_frontier, bt_base_err) - interp(bt_base_frontier, bt_base_err)
        deltas_c[bi] = interp(bt_c_frontier, bt_base_err) - interp(bt_base_frontier, bt_base_err)

    deltas_b, deltas_c = deltas_b[~np.isnan(deltas_b)], deltas_c[~np.isnan(deltas_c)]
    ci_b = (float(np.quantile(deltas_b, 0.025)), float(np.quantile(deltas_b, 0.975)))
    ci_c = (float(np.quantile(deltas_c, 0.025)), float(np.quantile(deltas_c, 0.975)))
    sig_b = ci_b[0] > 0 or ci_b[1] < 0
    sig_c = ci_c[0] > 0 or ci_c[1] < 0
    print(f"  (b) bootstrap 95% CI: [{ci_b[0]:+.4f}, {ci_b[1]:+.4f}]  {'SIGNIFICANT' if sig_b else 'not significant'}")
    print(f"  (c) bootstrap 95% CI: [{ci_c[0]:+.4f}, {ci_c[1]:+.4f}]  {'SIGNIFICANT' if sig_c else 'not significant'}")

    return {
        "dataset": name,
        "n_reaching_rank1": len(rows),
        "n_calibration": len(calib),
        "n_test": len(test),
        "baseline_hit_rate": base_hit,
        "baseline_error_rate": base_err,
        "simple_joint_gain_at_matched_error": b_hit_at_base_err - base_hit,
        "simple_joint_ci95": ci_b,
        "simple_joint_significant": sig_b,
        "full_joint_gain_at_matched_error": c_hit_at_base_err - base_hit,
        "full_joint_ci95": ci_c,
        "full_joint_significant": sig_c,
    }


def main() -> None:
    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["trace"]).exists() or not Path(cfg["scored"]).exists():
            print(f"Skipping {name}: missing data")
            continue
        r = evaluate_condition(name, cfg)
        if r:
            results[name] = r

    print(f"\n\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    print(f"{'dataset':>28} | {'simple_joint_gain':>18} | {'full_joint_gain':>16} | verdict")
    for name, r in results.items():
        v = "full joint SIGNIFICANT" if r["full_joint_significant"] else "not significant"
        print(f"{name:>28} | {r['simple_joint_gain_at_matched_error']:+18.4f} | "
              f"{r['full_joint_gain_at_matched_error']:+16.4f} | {v}")

    Path("results/topk_cascade_joint_decision_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nWrote results/topk_cascade_joint_decision_analysis.json")


if __name__ == "__main__":
    main()
