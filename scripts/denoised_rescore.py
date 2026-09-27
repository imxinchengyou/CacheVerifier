"""RESEARCH_PROPOSAL.md 方向 26.17: re-score the paper's key numbers after
correcting for the direct-hit label noise confirmed by human review (26.16).

Noise model (expected-value correction): a request served as a direct hit
(similarity >= tau_high = 0.97) and labelled wrong by equivalence_id counts
as an error with probability 1 - pi_ds, where pi_ds is the estimated share
of labelled-wrong direct hits that are actually reusable. Gray-zone labels
are left as they are (their noise is not yet measured). pi_ds combines the
LLM category shares (26.15) with the human "reusable" rates (26.16):
    pi = P(F) * p_F + (1 - P(F)) * p_other
Low / high scenarios use the Wilson 95% bounds of p_F (per dataset) and
p_other (pooled, n=12).

Numbers re-scored, each first reproduced exactly from the noisy labels:
  1. Group A static-threshold error rates on the published grid;
  2. same-population Group D/E vs Group A deltas (2026-09-24 erratum protocol);
  3. Group C (oracle verifier) vs Group A hit rate at matched error.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored, replay

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "groupD_vs_groupA_fair_frontier.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
spec_b = importlib.util.spec_from_file_location("b", ROOT / "scripts" / "groupD_vs_groupA_fair_frontier_bootstrap.py")
b = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(b)

TAU_HIGH = 0.97
P_F = {"lmarena": (21 / 25, 0.65, 0.94), "search_queries_corrected": (23 / 25, 0.75, 0.98), "quora": (22 / 25, 0.70, 0.96)}
P_OTHER = (6 / 12, 0.25, 0.75)
SHARE_F = {"lmarena": 0.935, "search_queries_corrected": 0.910, "quora": 0.818}
A_GRID = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]


def pis(d):
    f = SHARE_F[d]
    return {s: f * P_F[d][k] + (1 - f) * P_OTHER[k] for k, s in enumerate(("point", "low", "high"))}


def err_weights(sim, correct, pi):
    """Expected error contribution of serving each request."""
    w = (~correct).astype(float)
    w[(sim >= TAU_HIGH) & ~correct] = 1 - pi
    return w


def dense_frontier_w(sim, w, has):
    s = np.where(has, sim, -np.inf)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    ws = np.where(np.isfinite(s_sorted), w[order], 0.0)
    n = len(s)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True] & np.isfinite(s_sorted)
    return (np.cumsum(ws) / n)[last], (np.arange(1, n + 1) / n)[last]


def main():
    out = {}
    for d, (trace_f, ce_f, a_f) in g.SETS.items():
        trace = load_match_trace(g.CACHE / trace_f)
        n = len(trace)
        has = np.array([t.similarity is not None for t in trace])
        sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
        cor = np.array([bool(t.would_be_correct) for t in trace])
        P = pis(d)
        res = {"pi": P}
        print(f"\n===== {d}  pi point {P['point']:.3f} (low {P['low']:.3f}, high {P['high']:.3f})")

        # 1. Group A on the published grid (whole population)
        pubA = {round(e["threshold"], 2): e for e in json.loads((ROOT / "results" / a_f).read_text())}
        rowsA = []
        for t in A_GRID:
            acc = has & (sim >= t)
            hit = acc.mean()
            e0 = (acc & ~cor).mean()
            if t in pubA:
                # SearchQueries' published Group A table comes from an earlier trace run that differs from the
                # current trace by a handful of records (HNSW build); tolerance 5e-4, max gap recorded.
                gap = max(abs(pubA[t]["hit_rate"] - hit), abs(pubA[t]["error_rate"] - e0))
                assert gap < 5e-4, (d, t, hit, e0, pubA[t])
                res.setdefault("groupA_max_gap_vs_published", 0.0)
                res["groupA_max_gap_vs_published"] = max(res["groupA_max_gap_vs_published"], float(gap))
            ec = {s: float((err_weights(sim, cor, p) * acc).mean()) for s, p in P.items()}
            rowsA.append({"tau": t, "hit": float(hit), "err_noisy": float(e0), "err_denoised": ec,
                          "direct_hit_share_of_errors": float(((acc & ~cor & (sim >= TAU_HIGH)).sum()) / max((acc & ~cor).sum(), 1))})
            print(f"  A tau {t:.2f}: hit {hit:.4f}  err noisy {e0:.4f} -> denoised {ec['point']:.4f} [{ec['high']:.4f}, {ec['low']:.4f}]  "
                  f"({100 * rowsA[-1]['direct_hit_share_of_errors']:.0f}% of its errors are direct hits)")
        res["groupA"] = rowsA

        # 3. Group C (oracle) vs A at matched error, whole population
        rowsC = []
        for tl in g.TAU_LOW_GRID:
            gray = has & (sim >= tl) & (sim < TAU_HIGH)
            served = (has & (sim >= TAU_HIGH)) | (gray & cor)
            hit = served.mean()
            r = {"tau_low": tl, "hit": float(hit)}
            for s, p in [("noisy", None)] + list(P.items()):
                w = (~cor).astype(float) if p is None else err_weights(sim, cor, p)
                eC = float((w * served).mean())
                eA, hA = dense_frontier_w(sim, w, has)
                r[s] = {"err": eC, "delta_vs_A": float(hit - g.hit_at_error(eA, hA, eC))}
            rowsC.append(r)
            print(f"  C tau_low {tl:.2f}: hit {hit:.4f} | noisy err {r['noisy']['err']:.4f} delta {100 * r['noisy']['delta_vs_A']:+.2f}pp"
                  f" -> denoised err {r['point']['err']:.4f} delta {100 * r['point']['delta_vs_A']:+.2f}pp"
                  f" [{100 * r['high']['delta_vs_A']:+.2f}, {100 * r['low']['delta_vs_A']:+.2f}]")
        res["groupC_vs_A"] = rowsC

        # 2. Same-population D/E vs A (erratum protocol), point estimates
        rowsDE = []
        base_scored = load_scored(g.CACHE / ce_f)
        for grp in ("D", "E"):
            for tl in g.TAU_LOW_GRID:
                gz = [i for i, t in enumerate(trace) if t.similarity is not None and tl <= t.similarity < TAU_HIGH and i in base_scored]
                calib, test = gz[: len(gz) // 2], set(gz[len(gz) // 2:])
                if grp == "D":
                    scored = base_scored
                    thr = g.select_threshold(np.array([scored[i].score for i in calib]), np.array([1 if trace[i].would_be_correct else 0 for i in calib]))
                else:
                    scored = load_scored(g.CACHE / b.FT[d])
                    thr = next(e["calibrated_threshold"] for e in json.loads((ROOT / "results" / f"{d}_groupE_honest_calibration.json").read_text()) if e["tau_low"] == tl)
                outcomes = replay(trace, scored, tau_low=tl, tau_high=TAU_HIGH, threshold=thr)
                keep = np.array([i for i in range(n) if trace[i].similarity is None or not (tl <= trace[i].similarity < TAU_HIGH) or i in test])
                served = np.array([outcomes[i].action == "hit" for i in keep])
                wrong = np.array([outcomes[i].action == "hit" and not outcomes[i].correct for i in keep])
                pub_file = b.PUBLISHED_D[d] if grp == "D" else f"{d}_groupE_honest_calibration.json"
                pub = next(e for e in json.loads((ROOT / "results" / pub_file).read_text()) if e["tau_low"] == tl)
                assert abs(served.mean() - pub["hit_rate"]) < 1e-9 and abs(wrong.mean() - pub["error_rate"]) < 1e-9
                s_k, c_k, h_k = sim[keep], cor[keep], has[keep]
                r = {"group": grp, "tau_low": tl, "hit": float(served.mean())}
                for s, p in [("noisy", None)] + list(P.items()):
                    w = (~c_k).astype(float) if p is None else err_weights(s_k, c_k, p)
                    ePol = float((w * wrong).sum() / len(keep)) if p is None else float((w * served).mean())
                    eA, hA = dense_frontier_w(s_k, w, h_k)
                    r[s] = {"err": ePol, "delta_vs_A": float(served.mean() - g.hit_at_error(eA, hA, ePol))}
                rowsDE.append(r)
                print(f"  {grp} tau_low {tl:.2f}: noisy delta {100 * r['noisy']['delta_vs_A']:+.2f}pp -> denoised {100 * r['point']['delta_vs_A']:+.2f}pp"
                      f" [{100 * r['high']['delta_vs_A']:+.2f}, {100 * r['low']['delta_vs_A']:+.2f}]   err {r['noisy']['err']:.4f} -> {r['point']['err']:.4f}")
        res["groupDE_vs_A"] = rowsDE
        out[d] = res
    (ROOT / "results" / "denoised_rescore.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
