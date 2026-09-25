"""2026-09-24 erratum side check (PAPER.md erratum; scripts/groupD_vs_groupA_fair_frontier.py):
the post-hoc grid-search comparisons (§5.4 first half, §5.6, e.g. "SearchQueries
Group E 53/54 beat") replay the whole stream for both groups, so they do NOT have
the evaluation-set mismatch the honest-calibration comparison had. The only open
question is Group A's 9-point linear interpolation. This recounts every grid-search
point against (a) the published 9-point interpolated frontier and (b) a dense step
frontier over every distinct similarity value, on all requests.

Win/tie/loss rule used here: win if the point's hit-rate CI lower bound is above
Group A's hit rate at the same error rate, loss if its upper bound is below, else
tie. The paper's own counts used a slightly different rule, so the absolute counts
here differ a little from the published ones; what matters is how many verdicts
change between (a) and (b).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

from cacheverifier.experiments.verified_sweep import load_match_trace

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "groupD_vs_groupA_fair_frontier.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

FILES = {
    "lmarena": ("lmarena__precomputed__n60000.trace.json",
                ["lmarena_groupD_cross_encoder.json", "lmarena_groupE_finetuned.json"], "lmarena_groupA.json"),
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000.trace.json",
                                 ["search_queries_corrected_groupD.json", "search_queries_corrected_groupE_finetuned.json"],
                                 "search_queries_groupA.json"),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
              ["quora_groupD_cross_encoder.json", "quora_groupE_finetuned.json"], "quora_groupA.json"),
}


def classify(ci, a_hit):
    return "win" if ci[0] > a_hit else ("loss" if ci[1] < a_hit else "tie")


def main():
    out = {}
    for d, (trace_f, files, a_f) in FILES.items():
        trace = load_match_trace(g.CACHE / trace_f)
        has = np.array([t.similarity is not None for t in trace])
        sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
        cor = np.array([bool(t.would_be_correct) for t in trace])
        dense_err, dense_hit = g.dense_frontier(sim, cor, has)
        pub = [(r["error_rate"], r["hit_rate"]) for r in json.loads((ROOT / "results" / a_f).read_text())]
        for f in files:
            rows = json.loads((ROOT / "results" / f).read_text())
            rows = rows if isinstance(rows, list) else rows.get("rows", rows)
            counts = {"published": {"win": 0, "tie": 0, "loss": 0}, "dense": {"win": 0, "tie": 0, "loss": 0}}
            best = {"published": -9.0, "dense": -9.0}
            n = flips = 0
            for r in rows:
                if r.get("n_records") not in (None, len(trace)):
                    continue
                err, hit = r["error_rate"], r["hit_rate"]
                ci = r.get("hit_rate_ci", [hit, hit])
                a_pub = g.interpolate(pub, err)
                if a_pub is None:
                    continue
                a_dense = g.hit_at_error(dense_err, dense_hit, err)
                n += 1
                cp, cd = classify(ci, a_pub), classify(ci, a_dense)
                counts["published"][cp] += 1
                counts["dense"][cd] += 1
                flips += cp != cd
                best["published"] = max(best["published"], hit - a_pub)
                best["dense"] = max(best["dense"], hit - a_dense)
            out[f] = {"n": n, "counts": counts, "best": best, "flips": flips}
            print(f, n, "published", counts["published"], "best %+.4f" % best["published"],
                  "| dense", counts["dense"], "best %+.4f" % best["dense"], "| verdict changes", flips)
    (ROOT / "results" / "grid_search_vs_dense_groupA_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
