"""Side check behind the 2026-09-24 erratum in PAPER.md: on the pooled default gray
zone [0.80, 0.97), raw similarity has a HIGHER AUC than the off-the-shelf
cross-encoder on all three datasets. AUC is a whole-ranking statistic,
while the paper's Group A vs Group D comparison lives at the high-precision
end, so this compares the two rankings where it matters.

Within the gray zone, "accept iff similarity >= t" is exactly what Group A's
static threshold does there (both groups treat >= tau_high the same way), and
"accept iff ce >= t" is Group D. So at a fixed gray-zone error budget, the
score that accepts more gray-zone requests is the one that wins that segment
of the frontier.

Three views, per dataset and per score (similarity, off-the-shelf CE, and the
fine-tuned CE where cached -- in-sample caveat):
  1. precision@top-k% of the gray zone (k = 5, 10, 20, 30, 50)
  2. oracle hit@alpha: the largest accepted share of gray-zone requests whose
     marginal risk mean(accept * wrong) <= alpha, alpha in {1, 2, 5, 10}%
  3. honest: threshold picked on the chronological first half for each alpha,
     realized hit rate / risk on the second half (the paper's §5.4 protocol)
Bootstrap 95% CIs (1000 reps) for sim-minus-CE differences in views 1 and 3.
"""

import json
from pathlib import Path

import numpy as np

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
CE = "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2"
TOPK = (0.05, 0.10, 0.20, 0.30, 0.50)
ALPHAS = (0.01, 0.02, 0.05, 0.10)
REPS = 1000

SETS = {
    "lmarena": (
        "lmarena__precomputed__n60000.trace.json",
        f"lmarena__precomputed__n60000__{CE}__lo0.8__hi0.97.scored.json",
        "lmarena__precomputed__n60000__cross_encoder_results_finetuned_verifier_model__lo0.8__hi0.97.scored.json",
    ),
    "search_queries_corrected": (
        "search_queries_corrected__precomputed__n150000.trace.json",
        f"search_queries_corrected__precomputed__n150000__{CE}__lo0.8__hi0.97.scored.json",
        "search_queries_corrected__precomputed__n150000__cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected__lo0.8__hi0.97.scored.json",
    ),
    "quora": (
        "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
        f"quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000__{CE}__lo0.8__hi0.97.scored.json",
        None,
    ),
}


def load(dataset):
    trace_f, ce_f, ft_f = SETS[dataset]
    trace = load_match_trace(CACHE / trace_f)
    ce = load_scored(CACHE / ce_f)
    ft = load_scored(CACHE / ft_f) if ft_f and (CACHE / ft_f).exists() else None
    idx = [i for i in sorted(ce) if i < len(trace) and (ft is None or i in ft)]
    y = np.array([1 if trace[i].would_be_correct else 0 for i in idx])  # 1 = reuse is correct
    scores = {"similarity": np.array([trace[i].similarity for i in idx]), "ce": np.array([ce[i].score for i in idx])}
    if ft is not None:
        scores["ce_ft*"] = np.array([ft[i].score for i in idx])
    return y, scores


def precision_at(s, y, k):
    n = max(1, int(round(k * len(y))))
    top = np.argsort(-s, kind="stable")[:n]
    return float(y[top].mean())


def oracle_hit(s, y, alpha):
    order = np.argsort(-s, kind="stable")
    risk = np.cumsum(1 - y[order]) / len(y)
    k = int(np.searchsorted(risk, alpha, side="right"))
    return k / len(y)


def pick_threshold(s, y, alpha):
    """Lowest threshold whose marginal risk on (s, y) is <= alpha."""
    order = np.argsort(-s, kind="stable")
    risk = np.cumsum(1 - y[order]) / len(y)
    k = int(np.searchsorted(risk, alpha, side="right"))
    return np.inf if k == 0 else float(s[order[k - 1]])


def honest(s, y, alpha, h):
    t = pick_threshold(s[:h], y[:h], alpha)
    acc = s[h:] >= t
    yt = y[h:]
    return float(acc.mean()), float((acc & (yt == 0)).mean())


def boot(fn, n, rng):
    vals = [fn(rng.integers(0, n, n)) for _ in range(REPS)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    rng = np.random.default_rng(0)
    out = {}
    for d in SETS:
        y, S = load(d)
        n, h = len(y), len(y) // 2
        r = {"n": n, "positive_rate": float(y.mean()), "precision_at": {}, "oracle_hit_at_alpha": {}, "honest": {}}
        for k in TOPK:
            row = {name: precision_at(s, y, k) for name, s in S.items()}
            row["sim_minus_ce_ci95"] = boot(
                lambda i, k=k: precision_at(S["similarity"][i], y[i], k) - precision_at(S["ce"][i], y[i], k), n, rng
            )
            r["precision_at"][str(k)] = row
        for a in ALPHAS:
            r["oracle_hit_at_alpha"][str(a)] = {name: oracle_hit(s, y, a) for name, s in S.items()}
            row = {}
            for name, s in S.items():
                hit, risk = honest(s, y, a, h)
                row[name] = {"hit": hit, "risk": risk}
            yt = y[h:]
            thr = {name: pick_threshold(s[:h], y[:h], a) for name, s in S.items()}

            def diff(i, thr=thr):
                acc_s = S["similarity"][h:][i] >= thr["similarity"]
                acc_c = S["ce"][h:][i] >= thr["ce"]
                return float(acc_s.mean() - acc_c.mean())

            row["sim_minus_ce_hit_ci95"] = boot(diff, n - h, rng)
            r["honest"][str(a)] = row
        out[d] = r

        print(f"\n== {d}  n={n}  reuse-correct rate={y.mean():.3f}")
        names = list(S)
        print("  precision@top-k:  " + "  ".join(
            f"{k:.0%}: " + "/".join(f"{r['precision_at'][str(k)][m]:.3f}" for m in names)
            + " [sim-ce %+.3f,%+.3f]" % tuple(r["precision_at"][str(k)]["sim_minus_ce_ci95"]) for k in TOPK))
        print("  oracle hit@alpha: " + "  ".join(
            f"{a:.0%}: " + "/".join(f"{r['oracle_hit_at_alpha'][str(a)][m]:.3f}" for m in names) for a in ALPHAS))
        for a in ALPHAS:
            row = r["honest"][str(a)]
            print(f"  honest alpha={a:.0%}: " + "  ".join(f"{m} hit {row[m]['hit']:.3f} risk {row[m]['risk']:.3f}" for m in names)
                  + "  [sim-ce hit %+.3f,%+.3f]" % tuple(row["sim_minus_ce_hit_ci95"]))
        print("  (order: " + "/".join(names) + ")")

    path = ROOT / "results" / "gray_zone_high_precision_compare.json"
    path.write_text(json.dumps({"config": {"gray_zone": [0.80, 0.97], "topk": TOPK, "alphas": ALPHAS,
                                           "note": "ce_ft* scores include the fine-tune training split (in-sample)"},
                                "results": out}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
