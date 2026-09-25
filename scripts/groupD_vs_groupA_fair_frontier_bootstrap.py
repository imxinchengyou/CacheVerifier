"""Bootstrap CIs for the same-population Group D/E vs Group A deltas in
scripts/groupD_vs_groupA_fair_frontier.py. Each replicate resamples requests
from the evaluation population (all non-gray-zone requests + the test half of
the gray zone at that tau_low), recomputes the verifier policy's hit/error
and Group A's dense step frontier on the SAME resample, and records
hit(policy) - best Group A hit with error <= error(policy)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored, replay

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "groupD_vs_groupA_fair_frontier.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
REPS = 300
FT = {
    "lmarena": "lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json",
    "search_queries_corrected": "search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json",
    # Re-scored 2026-09-24 on the T4 with the same checkpoint; the rerun reproduced
    # results/quora_groupE_honest_calibration.json exactly (thresholds, hit, error).
    "quora": "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__cross_encoder_results_finetuned_verifier_model_quora__lo0.8__hi0.97.scored.json",
}
PUBLISHED_D = {
    "lmarena": "threshold_calibration_lmarena_60k_merged.json",
    "search_queries_corrected": "threshold_calibration_sq_corrected_merged.json",
    "quora": "threshold_calibration_quora_merged.json",
}
CASES = [(d, grp, tl) for d in ("search_queries_corrected", "lmarena", "quora") for grp in ("D", "E") for tl in g.TAU_LOW_GRID]


def main():
    rng = np.random.default_rng(0)
    res = []
    for d, grp, tl in CASES:
        trace_f, ce_f, _ = g.SETS[d]
        trace = load_match_trace(g.CACHE / trace_f)
        base_scored = load_scored(g.CACHE / ce_f)
        gz = [i for i, t in enumerate(trace) if t.similarity is not None and tl <= t.similarity < g.TAU_HIGH and i in base_scored]
        calib, test = gz[: len(gz) // 2], set(gz[len(gz) // 2:])
        if grp == "D":
            scored = base_scored
            thr = g.select_threshold(np.array([scored[i].score for i in calib]), np.array([1 if trace[i].would_be_correct else 0 for i in calib]))
        else:
            scored = load_scored(g.CACHE / FT[d])
            thr = next(e["calibrated_threshold"] for e in json.loads((g.ROOT / "results" / f"{d}_groupE_honest_calibration.json").read_text()) if e["tau_low"] == tl)
        outcomes = replay(trace, scored, tau_low=tl, tau_high=g.TAU_HIGH, threshold=thr)
        keep = np.array([i for i in range(len(trace)) if trace[i].similarity is None or not (tl <= trace[i].similarity < g.TAU_HIGH) or i in test])
        pub_file = PUBLISHED_D[d] if grp == "D" else f"{d}_groupE_honest_calibration.json"
        pub = next(e for e in json.loads((g.ROOT / "results" / pub_file).read_text()) if e["tau_low"] == tl)
        served = np.array([outcomes[i].action == "hit" for i in keep])
        wrong = np.array([outcomes[i].action == "hit" and not outcomes[i].correct for i in keep])
        has = np.array([trace[i].similarity is not None for i in keep])
        sim = np.array([trace[i].similarity if trace[i].similarity is not None else -1.0 for i in keep])
        cor = np.array([bool(trace[i].would_be_correct) for i in keep])

        def delta(ix):
            hr, er = served[ix].mean(), wrong[ix].mean()
            e, h = g.dense_frontier(sim[ix], cor[ix], has[ix])
            return hr - g.hit_at_error(e, h, er)

        n = len(keep)
        # Reproduction guard: the recomputed policy must be the published one.
        assert abs(served.mean() - pub["hit_rate"]) < 1e-9 and abs(wrong.mean() - pub["error_rate"]) < 1e-9, (d, grp, tl, served.mean(), pub["hit_rate"])
        point = delta(np.arange(n))
        boots = [delta(rng.integers(0, n, n)) for _ in range(REPS)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "win" if lo > 0 else ("loss" if hi < 0 else "tie")
        res.append({"dataset": d, "group": grp, "tau_low": tl, "hit_rate": float(served.mean()), "error_rate": float(wrong.mean()),
                    "delta_same_population": float(point), "ci95": [float(lo), float(hi)], "verdict": verdict})
        print(f"{d:26s} {grp} tau_low={tl:.2f}  delta {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}", flush=True)
    (ROOT / "results" / "groupD_vs_groupA_fair_frontier_bootstrap.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
