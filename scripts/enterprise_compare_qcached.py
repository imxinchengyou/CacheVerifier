"""Enterprise comparison: give the verifier the earlier customer message too.

The judge and the human reviewer both saw the earlier customer message (the
cache entry's own query); the verifier only saw (new message, cached reply).
This trains the same cross-encoder recipe (enterprise_compare_verifier.py:
ms-marco-MiniLM-L6-v2, warmup 10%, 3 epochs, lr 2e-5, first 70% of the
calibration half's gray zone) with text_a built two ways from PAPER §5.22:

  concat  "{query}\\n[cached_query] {cached_query}"
  diff    "{query}\\n[diff vs cached_query] {word-level diff}"

once per judge label set (judge = non-thinking, judge_thinklow = thinking low),
and compares against the existing text_only models (ft_judge in
enterprise_{ds}_verifier[_judge_thinklow].npz) on:

  1. AUC against the human labels on the test-half gray-zone audit items
     (none used in training), with a paired bootstrap difference vs text_only;
  2. the test-half matched-error frontier (best hit rate at error <= e) under
     that label set, next to the static threshold.

Models are saved as scores only: results/.cache/enterprise_{ds}_verifier_qc_{arm}_{tag}.npz
(skipped when present, so the run resumes).

With GPU_SCORES set to a directory of scores_{arm}_{tag}_{ds}.npy from
enterprise_compare_qcached_gpu.py, training is skipped and all three arms,
text_only included, are taken from that run, so every arm shares one training
environment. Output: results/enterprise_qcached[_gpu].json.

Usage:
    [GPU_SCORES=results/.cache/qcached_gpu_scores] .venv/bin/python -u scripts/enterprise_compare_qcached.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from enterprise_compare_thinklow_eval import roc_auc  # noqa: E402
from enterprise_compare_verifier import BASE, HI, LO, TRAIN_FRACTION, finetune, score  # noqa: E402
from finetune_verifier_qcached_train_eval import DIFF_TEMPLATE, QCACHED_TEMPLATE, word_diff_summary  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
TAGS = ("judge", "judge_thinklow")
ARMS = {"concat": lambda q, cq: QCACHED_TEMPLATE.format(query=q, cached_query=cq),
        "diff": lambda q, cq: DIFF_TEMPLATE.format(query=q, diff=word_diff_summary(cq, q))}
ERRS = (0.01, 0.02, 0.03, 0.05)
GPU_SCORES = os.environ.get("GPU_SCORES")


def labels(ds: str, tag: str, n: int) -> np.ndarray:
    lab = np.zeros(n, dtype=bool)
    for line in open(CACHE / f"enterprise_{ds}_{tag}.jsonl"):
        r = json.loads(line)
        if r["kind"] == "above" and r["verdict"] in ("YES", "PARTIAL", "NO"):
            lab[r["i"]] = r["verdict"] == "YES"
    return lab


def best_hit(pts, e):
    return max([h for er, h in pts if er <= e] or [0.0])


def main() -> None:
    data = {}
    for ds, path in DATASETS.items():
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        sim, nn = np.nan_to_num(z["sim"], nan=-1.0), z["nn"]
        gray = np.where((sim >= LO) & (sim < HI))[0]
        n = len(rows)
        calib_idx = np.where(gray < n // 2)[0]
        train_pos = calib_idx[: int(TRAIN_FRACTION * len(calib_idx))]
        data[ds] = (rows, sim, nn, gray, train_pos, n)

    # 1. train and score
    for tag in TAGS if not GPU_SCORES else ():
        for arm, fn in ARMS.items():
            for ds, (rows, sim, nn, gray, train_pos, n) in data.items():
                out = CACHE / f"enterprise_{ds}_verifier_qc_{arm}_{tag}.npz"
                if out.exists():
                    continue
                pairs = [(fn(rows[i]["query"], rows[nn[i]]["query"]), rows[nn[i]]["answer"]) for i in gray]
                y = labels(ds, tag, n)[gray[train_pos]]
                model = finetune([pairs[k] for k in train_pos], y)
                np.savez_compressed(out, gray=gray, scores=score(model, pairs))
                print(f"{tag}/{arm}/{ds}: trained on {len(train_pos)} pairs ({y.mean():.3f} positive)", flush=True)

    # 2. evaluate
    human = [json.loads(line) for line in open(ROOT / "results/enterprise_audit_human_labels_notext.jsonl", encoding="utf-8")]
    items = [h for h in human if h["human_label"] != "unsure"
             and h["request_index"] >= data[h["dataset"]][5] // 2 and LO <= h["similarity"] < HI]
    yh = np.array([h["human_label"] == "yes" for h in items], int)
    rng = np.random.default_rng(0)
    boot = [b for b in (rng.integers(0, len(yh), len(yh)) for _ in range(2000)) if 0 < yh[b].sum() < len(b)]
    report = {"n_human_items": len(items), "auc_vs_human": {}, "frontier": {}}

    def model_scores(ds, tag, arm):
        if GPU_SCORES:
            return data[ds][3], np.load(Path(GPU_SCORES) / f"scores_{arm}_{tag}_{ds}.npy")
        if arm == "text_only":
            v = np.load(CACHE / f"enterprise_{ds}_verifier{'' if tag == 'judge' else '_' + tag}.npz")
            return v["gray"], v["ft_judge"]
        v = np.load(CACHE / f"enterprise_{ds}_verifier_qc_{arm}_{tag}.npz")
        return v["gray"], v["scores"]

    for tag in TAGS:
        per_arm = {}
        for arm in ("text_only", *ARMS):
            s = []
            for h in items:
                g, sc = model_scores(h["dataset"], tag, arm)
                s.append(sc[int(np.where(g == h["request_index"])[0][0])])
            per_arm[arm] = np.array(s, float)
        res = {}
        for arm, s in per_arm.items():
            r = {"auc": roc_auc(s, yh), "ci95": list(np.percentile([roc_auc(s[b], yh[b]) for b in boot], [2.5, 97.5]))}
            if arm != "text_only":
                d = [roc_auc(s[b], yh[b]) - roc_auc(per_arm["text_only"][b], yh[b]) for b in boot]
                r["minus_text_only"] = {"diff": r["auc"] - roc_auc(per_arm["text_only"], yh), "ci95": list(np.percentile(d, [2.5, 97.5]))}
            res[arm] = r
        report["auc_vs_human"][tag] = res
        print(f"\nAUC vs human ({len(items)} test-half gray items), trained on {tag} labels:")
        for arm, r in res.items():
            extra = f"  vs text_only {r['minus_text_only']['diff']:+.3f} [{r['minus_text_only']['ci95'][0]:+.3f}, {r['minus_text_only']['ci95'][1]:+.3f}]" if "minus_text_only" in r else ""
            print(f"  {arm:10s} {r['auc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]{extra}")

        for ds, (rows, sim, nn, gray, train_pos, n) in data.items():
            test = np.arange(n) >= n // 2
            correct = labels(ds, tag, n)
            fr = {}
            pts = []
            for t in np.arange(0.5, 0.9951, 0.005):
                h = (sim >= t)[test]
                pts.append(((h & ~correct[test]).mean(), h.mean()))
            fr["redis_static"] = {str(e): best_hit(pts, e) for e in ERRS}
            for arm in ("text_only", *ARMS):
                g, sc = model_scores(ds, tag, arm)
                s = np.full(n, -np.inf)
                s[g] = sc
                ing = np.zeros(n, dtype=bool)
                ing[g] = True
                pts = []
                for t in list(np.percentile(sc[g < n // 2], np.arange(0, 100.5, 1))) + [np.inf]:
                    h = ((sim >= HI) | (ing & (s >= t)))[test]
                    pts.append(((h & ~correct[test]).mean(), h.mean()))
                fr[arm] = {str(e): best_hit(pts, e) for e in ERRS}
            report["frontier"][f"{ds}/{tag}"] = fr
            print(f"  frontier {ds}: " + " | ".join(f"{m} " + "/".join(f"{v * 100:.2f}" for v in f.values()) for m, f in fr.items()))
    (ROOT / f"results/enterprise_qcached{'_gpu' if GPU_SCORES else ''}.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
