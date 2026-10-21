"""Reddit follow-up on post 18 (2026-10-21): does query length explain why raw
cosine and the off-the-shelf cross-encoder misbehave in the default gray zone
[0.80, 0.97), and does a per-length cosine threshold beat one global cutoff?

Same rows as scripts/gray_zone_high_precision_compare.py. Length = word count
of the SHORTER side of the pair (incoming query vs matched cached query): the
hypothesis is "two short queries sit at high cosine even when they differ".

  1. descriptive: per length tercile (cut points from the first half), the
     reuse-correct rate overall and inside cosine bins -- is a short pair
     really less often correct at the same cosine?
  2. honest per-length threshold: on the chronological first half, calibrate
     P(correct | cosine) separately per tercile (isotonic), rank by that, pick
     the alpha cutoff on the first half, realize hit/risk on the second half.
     A cutoff on the calibrated probability == one cosine threshold per
     tercile, chosen to equalize marginal precision. Compared to the global
     cosine threshold and the off-the-shelf CE under the same protocol.
Bootstrap 95% CIs (1000 reps) on the second-half hit differences.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gray_zone_high_precision_compare import CACHE, SETS, pick_threshold

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {"lmarena": "lmarena", "search_queries_corrected": "search_queries_corrected", "quora": "quora"}
ALPHAS = (0.01, 0.02, 0.05, 0.10)
SIM_BINS = (0.80, 0.85, 0.90, 0.93, 0.97)
REPS = 1000


def load(dataset):
    cfg = load_dataset_config(f"configs/{CONFIGS[dataset]}.yaml")
    records = load_jsonl(Path(cfg.processed_path))[: cfg.max_samples]
    trace_f, ce_f, ft_f = SETS[dataset]
    trace = load_match_trace(CACHE / trace_f)
    ce = load_scored(CACHE / ce_f)
    ft = load_scored(CACHE / ft_f) if ft_f and (CACHE / ft_f).exists() else None
    idx = [i for i in sorted(ce) if i < len(trace) and (ft is None or i in ft)]
    y = np.array([1 if trace[i].would_be_correct else 0 for i in idx])
    sim = np.array([trace[i].similarity for i in idx])
    cs = np.array([ce[i].score for i in idx])
    ln = np.array([min(len(records[i].query.split()), len(records[trace[i].candidate_index].query.split())) for i in idx])
    return y, sim, cs, ln


def tercile(ln, cuts):
    return np.digitize(ln, cuts, right=True)  # 0 = short, 1 = mid, 2 = long


def calibrated(sim, ln, y, h, cuts):
    """Per-tercile isotonic P(correct | cosine) fit on [:h], applied to all rows."""
    b = tercile(ln, cuts)
    p = np.zeros(len(sim))
    for g in range(3):
        tr, m = (b[:h] == g), (b == g)
        if tr.sum() < 20:
            p[m] = y[:h].mean()
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sim[:h][tr], y[:h][tr])
        # tiny cosine tiebreak so the isotonic plateaus still rank like cosine within a tercile
        p[m] = iso.predict(sim[m]) + 1e-6 * sim[m]
    return p


def main():
    rng = np.random.default_rng(0)
    out = {}
    for d in SETS:
        y, sim, cs, ln = load(d)
        n, h = len(y), len(y) // 2
        cuts = np.unique(np.quantile(ln[:h], [1 / 3, 2 / 3]))
        if len(cuts) < 2:
            cuts = np.array([cuts[0], cuts[0] + 1])
        b = tercile(ln, cuts)
        r = {"n": n, "tercile_cuts_words_le": cuts.tolist(), "terciles": {}, "honest": {}}
        print(f"\n== {d}  n={n}  shorter-side word-count tercile cuts <= {cuts.tolist()}")
        for g, name in enumerate(("short", "mid", "long")):
            m = b == g
            row = {"n": int(m.sum()), "reuse_correct": float(y[m].mean()), "mean_cosine": float(sim[m].mean()), "by_cosine_bin": {}}
            for lo, hi in zip(SIM_BINS[:-1], SIM_BINS[1:]):
                mm = m & (sim >= lo) & (sim < hi)
                row["by_cosine_bin"][f"{lo:.2f}-{hi:.2f}"] = {"n": int(mm.sum()), "reuse_correct": float(y[mm].mean()) if mm.any() else None}
            r["terciles"][name] = row
            print(f"  {name:5s} n={row['n']:6d} correct={row['reuse_correct']:.3f} mean_cos={row['mean_cosine']:.3f}  by cos bin: "
                  + "  ".join(f"{k}:{v['reuse_correct']:.3f}(n={v['n']})" if v["reuse_correct"] is not None else f"{k}:-" for k, v in row["by_cosine_bin"].items()))

        p = calibrated(sim, ln, y, h, cuts)
        S = {"global_cos": sim, "per_length_cos": p, "ce": cs}
        yt = y[h:]
        for a in ALPHAS:
            acc = {k: s[h:] >= pick_threshold(s[:h], y[:h], a) for k, s in S.items()}
            row = {k: {"hit": float(v.mean()), "risk": float((v & (yt == 0)).mean())} for k, v in acc.items()}
            for other in ("global_cos", "ce"):
                vals = []
                for _ in range(REPS):
                    i = rng.integers(0, n - h, n - h)
                    vals.append(acc["per_length_cos"][i].mean() - acc[other][i].mean())
                row[f"per_length_minus_{other}_hit_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            r["honest"][str(a)] = row
            print(f"  honest alpha={a:.0%}: " + "  ".join(f"{k} hit {v['hit']:.3f} risk {v['risk']:.3f}" for k, v in row.items() if isinstance(v, dict))
                  + "  [perlen-global %+.3f,%+.3f]" % tuple(row["per_length_minus_global_cos_hit_ci95"])
                  + "  [perlen-ce %+.3f,%+.3f]" % tuple(row["per_length_minus_ce_hit_ci95"]))
        out[d] = r

    path = ROOT / "results" / "gray_zone_length_sliced_threshold.json"
    path.write_text(json.dumps({"config": {"gray_zone": [0.80, 0.97], "length": "word count of shorter side of pair",
                                           "alphas": ALPHAS, "protocol": "calibrate+pick on chronological first half, report second half"},
                                "results": out}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
