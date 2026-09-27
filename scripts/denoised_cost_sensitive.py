"""RESEARCH_PROPOSAL.md 方向 26.17: the same-population cost-sensitive
reanalysis (scripts/cost_sensitive_reanalysis_fair.py, PAPER §5.17 / Paper B
§4.3) under the direct-hit noise correction of scripts/denoised_rescore.py.

A served request with similarity >= 0.97 that is labelled wrong counts as
an error with probability 1 - pi. No verifier scores are needed:
- tau_low >= 0.97 points (tau_high = 0.999): every served request has
  similarity >= 0.97, so the corrected error is exactly (1 - pi) * published;
- tau_low < 0.97 points (tau_high = 0.97): the served requests with
  similarity >= 0.97 are exactly the direct hits of the point's population,
  countable from the match trace.
The primary noisy-label result is reproduced first."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_sensitive_reanalysis_fair as cf  # noqa: E402
from denoised_rescore import TAU_HIGH, pis  # noqa: E402

from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def a_grid9_w(sim, w, has):
    n = len(sim)
    err, hit = [], []
    for t in cf.GRID9:
        acc = has & (sim >= t)
        hit.append(acc.sum() / n)
        err.append((w * acc).sum() / n)
    return np.array(err), np.array(hit)


def main():
    out = {}
    published = json.loads((ROOT / "results" / "cost_sensitive_reanalysis_fair.json").read_text())
    for key, cfg in cf.DATASETS.items():
        trace = load_match_trace(cf.CACHE / cfg["trace"])
        scored_idx = set(load_scored(cf.CACHE / cfg["scored"]))
        has = np.array([t.similarity is not None for t in trace])
        sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
        correct = np.array([bool(t.would_be_correct) for t in trace])
        wrong_direct = has & (sim >= TAU_HIGH) & ~correct
        P = pis("search_queries_corrected" if key == "search_queries" else key)
        out[key] = {"pi": P}
        print(f"\n== {key}")
        for grp in ("D", "E"):
            points = json.loads((ROOT / "results" / cfg[grp]).read_text())
            out[key][grp] = {}
            for scen, pi in [("noisy", 0.0)] + list(P.items()):
                w = (~correct).astype(float)
                w[wrong_direct] = 1 - pi
                margins = []
                for p in points:
                    keep, n_test = cf.population(trace, scored_idx, p["tau_low"], p["tau_high"])
                    assert n_test == p["n_test"]
                    if p["tau_low"] >= TAU_HIGH:
                        err_p = p["error_rate"] * (1 - pi)
                    else:
                        err_p = p["error_rate"] - pi * wrong_direct[keep].sum() / len(keep)
                    err_a, hit_a = a_grid9_w(sim[keep], w[keep], has[keep])
                    margins.append(np.array([np.min(r * err_a + (1 - hit_a)) - (r * err_p + (1 - p["hit_rate"])) for r in cf.FINE_GRID]))
                win = cf.windows(np.max(np.vstack(margins), axis=0) > 0)
                out[key][grp][scen] = win
                if scen == "noisy":
                    pub = [tuple(x) for x in published[key][grp]["grid9"]["winning_r_windows"]]
                    assert [tuple(x) for x in win] == pub, (key, grp, win, pub)
                print(f"  {grp} [{scen:5s} pi={pi:.3f}]: " + (", ".join(f"[{lo}, {hi}]" for lo, hi in win) if win else "never beats A"))
    (ROOT / "results" / "denoised_cost_sensitive.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
