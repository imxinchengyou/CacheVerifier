"""RESEARCH_PROPOSAL.md 方向 26.19: PAPER §5.11 (tau_high sweep) under the
direct-hit noise correction of scripts/denoised_rescore.py.

Verifier scores: the on-disk [0.80, 0.97) band plus the [0.97, 0.99) band
re-scored on the GPU box (scripts/score_band_097_099.py). Every grid point of
results/<ds>_groupD_tauhigh_grid.json is replayed and must reproduce its
published hit and error rate exactly before anything is corrected. Then a
served request with similarity >= 0.97 labelled wrong counts as an error with
probability 1 - pi (both for Group D and for the Group A frontier), and the
§5.11 statistics are recomputed: best net lead over the interpolated Group A
frontier and wins (hit-rate CI lower bound above Group A), per tau_high."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoised_rescore import pis  # noqa: E402

from cacheverifier.experiments.verified_sweep import ScoredCandidate, load_match_trace, load_scored, replay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
spec = importlib.util.spec_from_file_location("c", ROOT / "scripts" / "compare_groupE_honest_to_groupA.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
CE = "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2"
SETS = {
    "lmarena": ("lmarena__precomputed__n60000", "lmarena_groupA.json"),
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000", "search_queries_groupA.json"),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000", "quora_groupA.json"),
}
A_GRID = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99]


def main():
    out = {}
    for d, (prefix, a_file) in SETS.items():
        trace = load_match_trace(CACHE / f"{prefix}.trace.json")
        scored = load_scored(CACHE / f"{prefix}__{CE}__lo0.8__hi0.97.scored.json")
        band = json.loads((CACHE / f"band097_099_{d}.scored.json").read_text())
        scored.update({int(i): ScoredCandidate(score=s, latency_ms=l) for i, (s, l) in band.items()})
        n = len(trace)
        has = np.array([t.similarity is not None for t in trace])
        sim = np.array([t.similarity if t.similarity is not None else -1.0 for t in trace])
        cor = np.array([bool(t.would_be_correct) for t in trace])
        grid = json.loads((ROOT / "results" / f"{d}_groupD_tauhigh_grid.json").read_text())
        pubA = json.loads((ROOT / "results" / a_file).read_text())
        res = {}
        for scen, pi in [("noisy", 0.0)] + list(pis(d).items()):
            w = (~cor).astype(float)
            w[has & (sim >= 0.97) & ~cor] = 1 - pi
            if scen == "noisy":
                frontier = [(p["error_rate"], p["hit_rate"]) for p in pubA]
            else:
                frontier = [(float((w * (has & (sim >= t))).mean()), float((has & (sim >= t)).mean())) for t in A_GRID]
            by_th = {}
            for g in grid:
                outs = replay(trace, scored, tau_low=g["tau_low"], tau_high=g["tau_high"], threshold=g["threshold"])
                served = np.array([o.action == "hit" for o in outs])
                if scen == "noisy":
                    wrong = served & ~cor
                    assert abs(served.mean() - g["hit_rate"]) < 1e-9 and abs(wrong.mean() - g["error_rate"]) < 1e-9, (d, g["tau_low"], g["tau_high"], g["threshold"], served.mean(), g["hit_rate"], wrong.mean(), g["error_rate"])
                err = float((w * served).mean())
                a_hr = c.interpolate(frontier, err)
                row = by_th.setdefault(g["tau_high"], {"n": 0, "in_range": 0, "wins": 0, "best_lead": None})
                row["n"] += 1
                if a_hr is None:
                    continue
                row["in_range"] += 1
                lead = g["hit_rate"] - a_hr
                row["best_lead"] = lead if row["best_lead"] is None else max(row["best_lead"], lead)
                row["wins"] += g["hit_rate_ci"][0] > a_hr
            res[scen] = {str(k): v for k, v in sorted(by_th.items())}
            print(f"{d:25s} [{scen:5s} pi={pi:.3f}] " + "  ".join(
                f"th{k}: {v['wins']}/{v['n']} win, best {100 * v['best_lead']:+.2f}pp" + (f" ({v['n'] - v['in_range']} out of range)" if v['in_range'] < v['n'] else "")
                for k, v in sorted(by_th.items())), flush=True)
        out[d] = res
    (ROOT / "results" / "denoised_tau_high.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
