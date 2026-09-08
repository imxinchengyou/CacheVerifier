"""Direction 17 mathematical follow-up: WHY does the joint (similarity,
verifier_score) decision rule hurt on LmArena but help on Quora/SearchQueries
-- and can that be predicted from calibration data ALONE, without ever
touching a held-out test set?

Working hypothesis: when similarity carries near-zero incremental
information beyond the verifier score (conditionally independent of the
label given verifier_score), the population-optimal 2-D decision boundary
should reduce to using verifier_score alone -- the joint model should not be
WORSE in the population limit. The observed harm on LmArena is therefore
suspected to be a FINITE-SAMPLE artifact of a specific mechanism: selecting
a fresh Youden's J threshold on the FUSED probability (a noisy, nearly
monotonic reparameterization of verifier_score when similarity adds little)
stacks a second layer of threshold-selection noise on top of the first,
making the joint pipeline strictly noisier than thresholding verifier_score
directly -- even though the two rankings are nearly identical.

If this is right, a simple, closed-form, calibration-time-only diagnostic
should predict which regime a given condition is in: a Wald test on whether
the similarity coefficient in the fitted logistic regression is
significantly different from zero. This requires no held-out validation
split at all (unlike the "try it and see if it wins on held-out data" hold-
back approach) -- it is computed directly from the SAME calibration data
already used to fit the model, via the standard asymptotic theory for
maximum-likelihood logistic regression (Wald test using the inverse
observed Fisher information as the coefficient covariance estimate).

Method: refit an (approximately) UNREGULARIZED logistic regression on each
of joint_grayzone_decision_analysis_v2.py's 5 calibration sets (large C to
approximate MLE), compute the Wald z-statistic and two-sided p-value for the
similarity coefficient by hand (no statsmodels dependency), and check
whether p-value magnitude tracks the already-established "joint model helps
vs hurts" pattern from the audited v2 results.

Usage:
    python scripts/joint_grayzone_calibration_time_pretest.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATASETS = {
    "lmarena_offtheshelf": {
        "trace": "results/.cache/lmarena__precomputed__n60000.trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "audited_gain": -0.0105,
    },
    "lmarena_finetuned": {
        "trace": "results/.cache/lmarena__precomputed__n60000.trace.json",
        "scored": "results/.cache/lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json",
        "audited_gain": -0.0009,
    },
    "quora_offtheshelf": {
        "trace": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        "scored": "results/.cache/quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "audited_gain": +0.0056,
    },
    "searchqueries_offtheshelf": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000.trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json",
        "audited_gain": +0.0169,
    },
    "searchqueries_finetuned": {
        "trace": "results/.cache/search_queries_corrected__precomputed__n150000.trace.json",
        "scored": "results/.cache/search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json",
        "audited_gain": +0.0017,
    },
}


def wald_test_logistic(X: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float):
    """X: (n, 2) standardized [similarity, verifier_score]. Returns per-coefficient
    (including intercept) standard errors and two-sided p-values via the inverse
    observed Fisher information (X_aug^T W X_aug)^-1, W = diag(p(1-p))."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    beta = np.concatenate([[intercept], coef])
    logits = X_aug @ beta
    p = 1.0 / (1.0 + np.exp(-logits))
    W = p * (1 - p)
    fisher_info = (X_aug * W[:, None]).T @ X_aug
    cov = np.linalg.inv(fisher_info)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pvals = 2 * (1 - norm.cdf(np.abs(z)))
    return se, z, pvals  # order: [intercept, similarity, verifier_score]


def main() -> None:
    print(f"{'dataset':>28} | {'sim_coef':>9} {'sim_se':>8} {'sim_z':>7} {'sim_p':>10} | "
          f"{'v_coef':>7} {'v_p':>10} | {'audited_gain':>13}")
    rows = []
    for name, cfg in DATASETS.items():
        trace = json.loads(Path(cfg["trace"]).read_text(encoding="utf-8"))
        scored = json.loads(Path(cfg["scored"]).read_text(encoding="utf-8"))
        gz_indices = sorted(int(k) for k in scored.keys())
        split = len(gz_indices) // 2
        calib_idx = gz_indices[:split]

        sims = np.array([trace[i][0] for i in calib_idx])
        labels = np.array([1 if trace[i][1] else 0 for i in calib_idx])
        vscores = np.array([scored[str(i)][0] for i in calib_idx])

        scaler = StandardScaler().fit(np.column_stack([sims, vscores]))
        X = scaler.transform(np.column_stack([sims, vscores]))

        # Large C -> approximates unregularized MLE (needed for valid asymptotic Wald inference).
        clf = LogisticRegression(max_iter=5000, C=1e6).fit(X, labels)
        coef, intercept = clf.coef_[0], clf.intercept_[0]
        se, z, pvals = wald_test_logistic(X, labels, coef, intercept)

        row = {
            "dataset": name,
            "n_calibration": len(calib_idx),
            "similarity_coef": float(coef[0]),
            "similarity_se": float(se[1]),
            "similarity_z": float(z[1]),
            "similarity_pvalue": float(pvals[1]),
            "verifier_coef": float(coef[1]),
            "verifier_pvalue": float(pvals[2]),
            "audited_hit_rate_gain": cfg["audited_gain"],
        }
        rows.append(row)
        print(f"{name:>28} | {coef[0]:9.4f} {se[1]:8.4f} {z[1]:7.2f} {pvals[1]:10.2e} | "
              f"{coef[1]:7.4f} {pvals[2]:10.2e} | {cfg['audited_gain']:+13.4f}")

    print("\nDoes calibration-time significance of the similarity coefficient predict "
          "whether joint decision helps (audited_gain > 0) or hurts (< 0)?")
    for r in rows:
        predicted = "HELPS (p<0.05)" if r["similarity_pvalue"] < 0.05 else "NO SIGNAL (p>=0.05)"
        actual = "helped" if r["audited_hit_rate_gain"] > 0 else "hurt"
        match = "MATCH" if (r["similarity_pvalue"] < 0.05) == (r["audited_hit_rate_gain"] > 0) else "MISMATCH"
        print(f"  {r['dataset']:>28}: calibration-time predicts {predicted:>20}, actually {actual:>6}  [{match}]")

    Path("results/joint_grayzone_calibration_time_pretest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nWrote results/joint_grayzone_calibration_time_pretest.json")


if __name__ == "__main__":
    main()
