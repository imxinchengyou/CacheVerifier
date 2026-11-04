"""Enterprise comparison with the thinking-mode (low) judge labels: three checks.

1. Matched-error frontier on the test half under the new labels (from
   results/enterprise_comparison_{ds}_judge_thinklow.json).
2. The same frontier re-scored with P(human yes | thinklow verdict, dataset),
   estimated from the 99 usable human labels, for static threshold and the
   CacheVerifier fine-tuned on old (judge) vs new (judge_thinklow) labels.
3. AUC against the human labels on the test-half gray-zone audit items, which no
   model was trained on: stock vs ft on old labels vs ft on new labels.

Usage:
    .venv/bin/python -u scripts/enterprise_compare_thinklow_eval.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": 60000, "twitter_comcast": 30180}
LO, HI = 0.80, 0.97


def roc_auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(len(s))
    # average ties
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum()))


def verdicts(ds, tag):
    out = {}
    for line in open(CACHE / f"enterprise_{ds}_{tag}.jsonl"):
        r = json.loads(line)
        if r["kind"] == "above" and r["verdict"] in ("YES", "PARTIAL", "NO"):
            out[r["i"]] = r["verdict"]
    return out


def main():
    human = [json.loads(line) for line in open(ROOT / "results/enterprise_audit_human_labels_notext.jsonl", encoding="utf-8")]
    human = [h for h in human if h["human_label"] != "unsure"]
    out = {"frontier_judge_thinklow": {}, "human_reweighted": {}, "rates": {}, "auc_vs_human": {}}
    for ds, n in DATASETS.items():
        half = n // 2
        res = json.loads((ROOT / f"results/enterprise_comparison_{ds}_judge_thinklow.json").read_text())["label_sets"]["judge"]["methods"]
        fr = {}
        for m in ("redis_static", "gptcache_onnx", "cv_stock", "cv_ft", "vcache"):
            pts = [(r["error_rate"], r["hit_rate"]) for r in res[m]["frontier_test"]]
            fr[m] = {str(e): max([h for er, h in pts if er <= e] or [0.0]) for e in (0.01, 0.02, 0.03, 0.05)}
        out["frontier_judge_thinklow"][ds] = fr

        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim = np.nan_to_num(z["sim"], nan=-1.0)
        test = np.arange(n) >= half
        new_v = verdicts(ds, "judge_thinklow")
        rates = {}
        for c in ("YES", "PARTIAL", "NO"):
            hs = [h["human_label"] == "yes" for h in human if h["dataset"] == ds and new_v.get(h["request_index"]) == c]
            rates[c] = (float(np.mean(hs)) if hs else 0.0, len(hs))
        out["rates"][ds] = rates
        p = np.zeros(n)
        for i, c in new_v.items():
            p[i] = rates[c][0]
        gray_idx = {}
        rw = {}
        for tag, key in (("judge", "ft_judge"), ("judge_thinklow", "ft_judge")):
            v = np.load(CACHE / f"enterprise_{ds}_verifier{'' if tag == 'judge' else '_' + tag}.npz")
            gray = v["gray"]
            s = np.full(n, -np.inf)
            s[gray] = v[key]
            ing = np.zeros(n, dtype=bool)
            ing[gray] = True
            grid = np.percentile(v[key][gray < half], np.arange(0, 100.5, 1))
            pts = []
            for t in list(grid) + [np.inf]:
                h = ((sim >= HI) | (ing & (s >= t)))[test]
                pts.append((float(((1 - p[test]) * h).mean()), float(h.mean())))
            rw[f"cv_ft_on_{tag}"] = {str(e): max([hr for er, hr in pts if er <= e] or [0.0]) for e in (0.01, 0.02, 0.03, 0.05)}
            gray_idx[tag] = {int(g): k for k, g in enumerate(gray)}
            out.setdefault("_scores", {})[(ds, tag)] = v
        pts = []
        for t in np.arange(0.5, 0.9951, 0.005):
            h = (sim >= t)[test]
            pts.append((float(((1 - p[test]) * h).mean()), float(h.mean())))
        rw["redis_static"] = {str(e): max([hr for er, hr in pts if er <= e] or [0.0]) for e in (0.01, 0.02, 0.03, 0.05)}
        out["human_reweighted"][ds] = rw

    items = [h for h in human if h["request_index"] >= DATASETS[h["dataset"]] // 2 and LO <= h["similarity"] < HI]
    y = [h["human_label"] == "yes" for h in items]
    sc = {"stock": [], "ft_on_judge": [], "ft_on_judge_thinklow": []}
    for h in items:
        ds, i = h["dataset"], h["request_index"]
        v_old, v_new = out["_scores"][(ds, "judge")], out["_scores"][(ds, "judge_thinklow")]
        k_old = {int(g): k for k, g in enumerate(v_old["gray"])}[i]
        k_new = {int(g): k for k, g in enumerate(v_new["gray"])}[i]
        sc["stock"].append(v_old["stock"][k_old])
        sc["ft_on_judge"].append(v_old["ft_judge"][k_old])
        sc["ft_on_judge_thinklow"].append(v_new["ft_judge"][k_new])
    rng = np.random.default_rng(0)
    y = np.array(y, int)
    for m, s in sc.items():
        s = np.array(s)
        boots = []
        for _ in range(2000):
            b = rng.integers(0, len(y), len(y))
            if 0 < y[b].sum() < len(b):
                boots.append(roc_auc(s[b], y[b]))
        out["auc_vs_human"][m] = {"auc": roc_auc(s, y), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    # paired bootstrap: new - old
    so, sn = np.array(sc["ft_on_judge"]), np.array(sc["ft_on_judge_thinklow"])
    diffs = []
    for _ in range(2000):
        b = rng.integers(0, len(y), len(y))
        if 0 < y[b].sum() < len(b):
            diffs.append(roc_auc(sn[b], y[b]) - roc_auc(so[b], y[b]))
    out["auc_vs_human"]["new_minus_old"] = {"diff": out["auc_vs_human"]["ft_on_judge_thinklow"]["auc"] - out["auc_vs_human"]["ft_on_judge"]["auc"],
                                            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    out["auc_vs_human"]["n"] = len(y)
    del out["_scores"]
    print(json.dumps(out, indent=1))
    (ROOT / "results/enterprise_thinklow_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
