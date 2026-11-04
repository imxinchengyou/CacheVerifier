"""Enterprise comparison: does fine-tuning on the 100 human audit labels help?

The human labels are also the only ground truth, so train and test never share
an item: items from the calibration half (first half of each stream) may be
used for training, items from the test half are the evaluation set.

Models, all scored on the test-half human items that fall in the gray zone
(0.80 <= sim < 0.97), against the binary human label ("can send as-is" vs not;
"unsure" dropped):
  stock         off-the-shelf ms-marco cross-encoder (saved scores)
  ft_judge      fine-tuned on the judge labels (saved scores, enterprise_compare_verifier.py)
  ft_human      fine-tuned only on the calibration-half human items (both datasets pooled)
  ft_judge_human  ft_judge's training set with human labels substituted where they exist,
                plus the remaining calibration-half human items; one model per dataset

Same recipe as enterprise_compare_verifier.py (warmup 10%, 3 epochs, lr 2e-5).

Usage:
    .venv/bin/python -u scripts/enterprise_compare_human_finetune.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from enterprise_compare_verifier import DATASETS, HI, LO, TRAIN_FRACTION, finetune, judge_labels, score  # noqa: E402
from finetune_verifier_train_eval import roc_auc  # noqa: E402

CACHE = ROOT / "results" / ".cache"


def main() -> None:
    audit = [json.loads(line) for line in open(ROOT / "results/enterprise_audit_human_labels_notext.jsonl")]
    data = {}
    for ds, path in DATASETS.items():
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        data[ds] = (z["sim"], z["nn"], rows, len(rows))

    def pair(ds, i):
        sim, nn, rows, _ = data[ds]
        return rows[i]["query"], rows[nn[i]]["answer"]

    items = [(r["id"], r["dataset"], r["request_index"], r["similarity"], r["human_label"] == "yes")
             for r in audit if r["human_label"] != "unsure"]
    train_h = [it for it in items if it[2] < data[it[1]][3] // 2 and LO <= it[3] < HI]
    test_h = [it for it in items if it[2] >= data[it[1]][3] // 2 and LO <= it[3] < HI]
    print(f"human items: train (calibration half, gray zone) {len(train_h)}, test (test half, gray zone) {len(test_h)}", flush=True)

    scores = {m: np.zeros(len(test_h)) for m in ("stock", "ft_judge", "ft_human", "ft_judge_human")}
    for ds in DATASETS:
        v = np.load(CACHE / f"enterprise_{ds}_verifier.npz")
        pos = {int(g): k for k, g in enumerate(v["gray"])}
        for t, it in enumerate(test_h):
            if it[1] == ds:
                scores["stock"][t] = v["stock"][pos[it[2]]]
                scores["ft_judge"][t] = v["ft_judge"][pos[it[2]]]

    model = finetune([pair(ds, i) for _, ds, i, _, _ in train_h], np.array([y for *_, y in train_h]))
    scores["ft_human"] = score(model, [pair(ds, i) for _, ds, i, _, _ in test_h])

    for ds in DATASETS:
        sim, nn, rows, n = data[ds]
        lab = judge_labels(ds, n, sim)
        gray = np.where((sim >= LO) & (sim < HI))[0]
        calib_gray = gray[gray < n // 2]
        train_idx = list(calib_gray[: int(TRAIN_FRACTION * len(calib_gray))])
        y = {int(i): bool(lab[i]) for i in train_idx}
        replaced = 0
        for _, d, i, _, yes in train_h:
            if d == ds:
                replaced += int(i in y)
                y[int(i)] = yes
        model = finetune([pair(ds, i) for i in y], np.array(list(y.values())))
        sel = [t for t, it in enumerate(test_h) if it[1] == ds]
        scores["ft_judge_human"][sel] = score(model, [pair(ds, test_h[t][2]) for t in sel])
        print(f"  {ds}: ft_judge_human trained on {len(y)} pairs ({replaced} judge labels replaced by human labels)", flush=True)

    labels = np.array([yes for *_, yes in test_h], dtype=int)
    out = {"n_test": len(test_h), "test_positive_rate": float(labels.mean()), "auc": {}, "auc_by_dataset": {}}
    rng = np.random.default_rng(0)
    for m, s in scores.items():
        boots = []
        for _ in range(2000):
            b = rng.integers(0, len(labels), len(labels))
            if 0 < labels[b].sum() < len(b):
                boots.append(roc_auc(s[b], labels[b]))
        out["auc"][m] = {"auc": roc_auc(s, labels), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        out["auc_by_dataset"][m] = {ds: roc_auc(s[[t for t, it in enumerate(test_h) if it[1] == ds]],
                                                 labels[[t for t, it in enumerate(test_h) if it[1] == ds]]) for ds in DATASETS}
    verdict = {r["id"]: r["judge_verdict"] for r in audit}
    judge_yes = np.array([verdict[it[0]] == "YES" for it in test_h], dtype=float)
    out["auc"]["judge_verdict_itself"] = {"auc": roc_auc(judge_yes + 0.5 * np.array([verdict[it[0]] == "PARTIAL" for it in test_h]), labels)}
    print(json.dumps(out, indent=1), flush=True)
    (ROOT / "results/enterprise_human_finetune.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
