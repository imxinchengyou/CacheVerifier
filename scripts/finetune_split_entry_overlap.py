"""Does cache-entry overlap between the fine-tune split's training and held-out
rows inflate held-out AUC?

The fine-tune splits chronologically (first 80% train, the rest calibrate/test).
One cached entry can be matched by several queries, so a held-out row's cached
answer may already have appeared in training. This trains once per dataset on
the first N gray-zone rows (similarity 0.80-0.97) in stream order and compares,
on the held-out rows, the fine-tune's AUC gain over the stock model for rows
whose cache entry was seen in training vs. rows whose was not, with a bootstrap
95% CI on the difference.

Run 2026-11-28 with N=3000; results in results/finetune_split_entry_overlap.json
(listed in results/INDEX.md). Finding: no consistent inflation across the three
datasets, so the split was left unchanged.

Reproducibility: `train()` comes from `verifier_core`, the training package of
the commercial CacheVerifier service, which is not public. The script stops
with a message if it is not installed. The method above is fully described so
it can be re-implemented on top of any cross-encoder fine-tune.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    from verifier_core.cross_encoder import DEFAULT_BASE_MODEL, CrossEncoderVerifier
    from verifier_core.finetune import GrayZoneExample, train
except ImportError:
    sys.exit("needs verifier_core, the CacheVerifier service's training package, which is not public")

R = Path(__file__).resolve().parent.parent
CACHE = R / "results/.cache"
DATA = {
    "lmarena": ("lmarena__precomputed__n60000", "data/processed/lmarena.jsonl"),
    "search_queries_corrected": ("search_queries_corrected__precomputed__n150000", "data/processed/search_queries_corrected.jsonl"),
    "quora": ("quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000", "data/processed/quora.jsonl"),
}
LO, HI = 0.80, 0.97
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
ONLY = sys.argv[2:] or list(DATA)
OUT_DIR = R / "results" / ".cache" / "finetune_split_entry_overlap"  # model checkpoints
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = R / "results" / "finetune_split_entry_overlap.json"


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_gain_gap(tuned, stock, y, seen, n=1000, seed=0):
    """CI of (gain on seen rows) - (gain on unseen rows), gain = AUC tuned - AUC stock."""
    rng = np.random.default_rng(seed)
    idx_s, idx_u = np.where(seen)[0], np.where(~seen)[0]
    out = []
    for _ in range(n):
        a = rng.choice(idx_s, len(idx_s))
        b = rng.choice(idx_u, len(idx_u))
        gs = auc(tuned[a], y[a]) - auc(stock[a], y[a])
        gu = auc(tuned[b], y[b]) - auc(stock[b], y[b])
        if np.isfinite(gs) and np.isfinite(gu):
            out.append(gs - gu)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None


report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
stock = CrossEncoderVerifier(DEFAULT_BASE_MODEL)
for ds in ONLY:
    prefix, path = DATA[ds]
    T = json.loads((CACHE / f"{prefix}.trace.json").read_text())
    rows = [i for i, t in enumerate(T) if t[0] is not None and LO <= t[0] < HI][:N]
    need = set(rows) | {int(T[i][2]) for i in rows}
    text = {}
    with open(R / path, encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j > max(need):
                break
            if j in need:
                o = json.loads(line)
                text[j] = (o["query"], o["answer"])
    ex, entry = [], []
    for i in rows:
        c = int(T[i][2])
        ex.append(GrayZoneExample(text[i][0], text[c][1], bool(T[i][1]), similarity_score=float(T[i][0]), cached_query=text[c][0]))
        entry.append(c)
    y_all = np.array([int(e.was_correct) for e in ex])
    print(f"{ds}: {len(ex)} rows, pos rate {y_all.mean():.3f}, distinct entries {len(set(entry))}", flush=True)

    res = train(ex, output_dir=str(OUT_DIR / ds))
    split = max(1, int(len(ex) * 0.8))  # same as train()'s default test_fraction=0.2
    train_entries = set(entry[:split])
    hold = list(range(split, len(ex)))
    seen = np.array([entry[i] in train_entries for i in hold])
    y = y_all[hold]
    tuned = CrossEncoderVerifier(res.model_path, uses_cached_query=res.uses_cached_query)
    pairs = [(ex[i].query, ex[i].candidate_answer) for i in hold]
    cqs = [ex[i].cached_query for i in hold]
    s_t = np.array([r.score for r in tuned.verify_batch(pairs, cached_queries=cqs)])
    s_s = np.array([r.score for r in stock.verify_batch(pairs)])

    def block(mask):
        return {
            "n": int(mask.sum()),
            "pos_rate": float(y[mask].mean()) if mask.any() else None,
            "auc_tuned": auc(s_t[mask], y[mask]),
            "auc_stock": auc(s_s[mask], y[mask]),
        }

    allm = np.ones(len(hold), bool)
    report[ds] = {
        "n": len(ex),
        "holdout_n": len(hold),
        "holdout_overlap_share": float(seen.mean()),
        "train_reported_auc_tuned": res.auc_tuned,
        "train_reported_auc_baseline": res.auc_baseline,
        "holdout_all": block(allm),
        "holdout_seen_entry": block(seen),
        "holdout_unseen_entry": block(~seen),
        "gain_gap_seen_minus_unseen_ci95": boot_gain_gap(s_t, s_s, y, seen) if seen.any() and (~seen).any() else None,
    }
    for k in ("holdout_all", "holdout_seen_entry", "holdout_unseen_entry"):
        b = report[ds][k]
        b["gain"] = b["auc_tuned"] - b["auc_stock"]
    print(json.dumps(report[ds], indent=1), flush=True)
    REPORT.write_text(json.dumps(report, indent=1))
print("DONE", flush=True)
