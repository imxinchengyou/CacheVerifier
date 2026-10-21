"""RESEARCH_PROPOSAL.md 26.20 (candidate A): do gray-zone reuse errors cluster
by the reused cache entry? Measures the design effect of the gray-zone loss
mean under entry clustering (full stream and within 2,000-row CRC-style
windows), plus a ceiling for "audit finds an error -> quarantine the entry".

Preregistered rule: median window deff of L_0.02 < 1.2 on all 3 datasets and
both verifiers closes candidate A; >= 1.5 on at least 2 datasets passes."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
LO, HI = 0.80, 0.97
WIN = 2000
ALPHAS = (0.01, 0.02, 0.05)
AUDIT_P, SEEDS = 0.05, 20
SETS = {
    "lmarena": ("lmarena__precomputed__n60000", "lmarena",
                {"ots": "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2",
                 "ft": "cross_encoder_results_finetuned_verifier_model"}),
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000", "search_queries_corrected",
                                 {"ots": "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2",
                                  "ft": "cross_encoder__root_workspace_finetuned_verifier_model_searchqueries_corrected"}),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000", None,
              {"ots": "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2",
               "ft": "cross_encoder_results_finetuned_verifier_model_quora"}),
}


def deff(loss, groups):
    """Cluster-robust / iid variance ratio of the mean; ANOVA ICC."""
    r = loss - loss.mean()
    iid = float((r ** 2).sum())
    if iid == 0:
        return None, None
    sums = defaultdict(float)
    sizes = defaultdict(int)
    for v, g in zip(r, groups):
        sums[g] += v
        sizes[g] += 1
    clu = float(sum(v * v for v in sums.values()))
    m = np.array(list(sizes.values()), dtype=float)
    mbar = (m ** 2).sum() / m.sum()  # size-weighted mean cluster size
    icc = (clu / iid - 1) / (mbar - 1) if mbar > 1 else None
    return clu / iid, icc


def window_deffs(loss, groups):
    out = []
    for a in range(0, len(loss) - WIN + 1, WIN):
        d, _ = deff(loss[a:a + WIN], groups[a:a + WIN])
        if d is not None:
            out.append(d)
    return np.array(out)


def threshold_for(score, wrong, alpha):
    """Most permissive verifier threshold whose pooled gray-zone risk <= alpha."""
    order = np.argsort(-score)
    cum = np.cumsum(wrong[order]) / len(score)  # risk if we approve the top-k
    ok = np.nonzero(cum <= alpha)[0]
    if len(ok) == 0:
        return np.inf
    k = ok.max()
    return score[order][k]


def quarantine(approve, wrong, entries, rng):
    blocked = set()
    loss = avoided = blocked_ok = 0
    for a, w, e in zip(approve, wrong, entries):
        if not a:
            continue
        if e in blocked:
            avoided += w
            blocked_ok += (not w)
            continue
        if w:
            loss += 1
            if rng.random() < AUDIT_P:
                blocked.add(e)
    return loss, avoided, blocked_ok


def main():
    out = {}
    for name, (stem, geom, vers) in SETS.items():
        T = json.loads((CACHE / f"{stem}.trace.json").read_text())
        rows = [i for i, t in enumerate(T) if t[0] is not None and LO <= t[0] < HI]
        wrong = np.array([not T[i][1] for i in rows], dtype=float)
        entry = np.array([T[i][2] for i in rows])
        if geom:
            meta = json.loads((CACHE / f"geom_{geom}_meta.json").read_text())
            cls = np.array([meta[e][1] for e in entry])
        else:
            eq = [json.loads(l)["equivalence_id"] for l in open(ROOT / "data/processed/quora.jsonl")][:60000]
            cls = np.array([eq[e] for e in entry])
        sizes = np.bincount(np.unique(entry, return_inverse=True)[1])
        res = {"gray_rows": len(rows), "wrong_rate": float(wrong.mean()), "entries": int(len(sizes)),
               "entry_size_mean": float(sizes.mean()), "entry_size_p90": float(np.quantile(sizes, 0.9)),
               "rows_in_entries_ge2": float(sizes[sizes >= 2].sum() / sizes.sum())}
        d, icc = deff(wrong, entry)
        dc, _ = deff(wrong, cls)
        wd = window_deffs(wrong, entry)
        res["L0"] = {"deff_full_entry": d, "icc_entry": icc, "deff_full_class": dc,
                     "win_deff_median": float(np.median(wd)), "win_deff_iqr": [float(np.quantile(wd, .25)), float(np.quantile(wd, .75))]}
        print(f"\n{name}: gray {len(rows)}, wrong {wrong.mean():.3f}, entries {len(sizes)}, "
              f"mean size {sizes.mean():.2f}, rows in entries>=2 {res['rows_in_entries_ge2']:.2f}")
        print(f"  L0: full deff entry {d:.3f} (icc {icc:.3f}) class {dc:.3f}; window median {np.median(wd):.3f}")
        for v, tag in vers.items():
            S = json.loads((CACHE / f"{stem}__{tag}__lo0.8__hi0.97.scored.json").read_text())
            score = np.array([S[str(i)][0] for i in rows])
            for alpha in ALPHAS:
                t = threshold_for(score, wrong, alpha)
                approve = score >= t
                L = (approve & (wrong == 1)).astype(float)
                d, icc = deff(L, entry)
                dc, _ = deff(L, cls)
                wd = window_deffs(L, entry)
                losses_by_entry = defaultdict(int)
                for e, l in zip(entry, L):
                    losses_by_entry[e] += int(l)
                tot = sum(losses_by_entry.values())
                repeat = sum(c for c in losses_by_entry.values() if c >= 2) / tot if tot else None
                after_first = sum(c - 1 for c in losses_by_entry.values() if c >= 1) / tot if tot else None
                q = [quarantine(approve, wrong == 1, entry, np.random.default_rng(s)) for s in range(SEEDS)]
                q = np.array(q, dtype=float)
                r = {"threshold": float(t), "approve_rate": float(approve.mean()), "risk": float(L.mean()),
                     "deff_full_entry": d, "icc_entry": icc, "deff_full_class": dc,
                     "win_deff_median": float(np.median(wd)) if len(wd) else None,
                     "win_deff_iqr": [float(np.quantile(wd, .25)), float(np.quantile(wd, .75))] if len(wd) else None,
                     "repeat_loss_share": repeat, "oracle_quarantine_ceiling": after_first,
                     "quarantine_avoided_share": float((q[:, 1] / (q[:, 0] + q[:, 1])).mean()),
                     "quarantine_blocked_correct_per_avoided": float(q[:, 2].sum() / max(q[:, 1].sum(), 1)),
                     "quarantine_blocked_correct_share_of_approved": float((q[:, 2] / approve.sum()).mean())}
                res[f"{v}_a{alpha}"] = r
                fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
                print(f"  {v} a={alpha}: approve {approve.mean():.3f} risk {L.mean():.4f} | deff full {fmt(d)} "
                      f"class {fmt(dc)} win-median {fmt(r['win_deff_median'])} | repeat {fmt(repeat)} "
                      f"ceiling {fmt(after_first)} | 5%-audit quarantine avoids {r['quarantine_avoided_share']:.3f}, "
                      f"blocks {r['quarantine_blocked_correct_per_avoided']:.2f} correct per avoided")
        out[name] = res
    (ROOT / "results" / "entry_error_clustering.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
