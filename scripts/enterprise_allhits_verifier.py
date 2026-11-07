"""Score every candidate hit (sim >= 0.80, gray zone AND direct-hit zone) with a fine-tuned verifier.

enterprise_compare_verifier.py scores only the gray zone [0.80, 0.97), because the direct-hit
zone is served without verification. To test whether the risk guarantee can be extended to all
hits (scripts/enterprise_allhits_guarantee.py), the direct-hit pairs need scores too. Same
recipe as enterprise_compare_verifier.py (ms-marco-MiniLM-L6-v2, 3 epochs, lr 2e-5, warmup 10%),
but trained on the earliest 70% of the calibration half's pairs with sim >= 0.80, so the
model has seen near-duplicates as well.

Usage:
    JUDGE_TAG=judge          .venv/bin/python -u scripts/enterprise_allhits_verifier.py
    JUDGE_TAG=judge_thinklow .venv/bin/python -u scripts/enterprise_allhits_verifier.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from enterprise_compare_verifier import (CACHE, DATASETS, LO, SUFFIX, TRAIN_FRACTION,  # noqa: E402
                                         finetune, judge_labels, score)


def main() -> None:
    for ds, path in DATASETS.items():
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn = z["sim"], z["nn"]
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        n = len(rows)
        half = n // 2
        cand = np.where(np.nan_to_num(sim, nan=-1.0) >= LO)[0]
        pairs = [(rows[i]["query"], rows[nn[i]]["answer"]) for i in cand]
        calib_idx = np.where(cand < half)[0]
        n_train = int(TRAIN_FRACTION * len(calib_idx))
        train_pos = calib_idx[:n_train]
        lab = judge_labels(ds, n, sim)
        y = lab[cand[train_pos]]
        t0 = time.time()
        model = finetune([pairs[k] for k in train_pos], y)
        sc = score(model, pairs)
        np.savez_compressed(CACHE / f"enterprise_{ds}_verifier_allhits{SUFFIX}.npz", cand=cand, ft=sc,
                            train_until=np.array(cand[train_pos[-1]]))
        print(f"{ds}: {len(cand)} candidates (sim >= {LO}), trained on {n_train} ({y.mean():.3f} positive), "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
