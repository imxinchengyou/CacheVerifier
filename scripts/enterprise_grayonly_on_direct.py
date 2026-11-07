"""Does a verifier fine-tuned on gray-zone pairs only still work on the direct-hit zone?

When a deployment switches from verifying the gray zone only to verifying every hit, its
verifier was trained on gray-zone labels only. This trains the enterprise_compare_verifier.py
recipe on the earliest 70% of the calibration half's GRAY-ZONE pairs (judge labels),
scores every candidate (sim >= 0.80), and saves it in the same layout as
enterprise_allhits_verifier.py, so enterprise_allhits_guarantee.py can be rerun on it.

Usage:
    .venv/bin/python -u scripts/enterprise_grayonly_on_direct.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from enterprise_compare_verifier import (CACHE, DATASETS, HI, LO, TRAIN_FRACTION,  # noqa: E402
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
        gray_calib = np.where((cand < half) & (sim[cand] < HI))[0]
        train_pos = gray_calib[: int(TRAIN_FRACTION * len(gray_calib))]
        lab = judge_labels(ds, n, sim)
        y = lab[cand[train_pos]]
        t0 = time.time()
        model = finetune([pairs[k] for k in train_pos], y)
        # evaluation must start after the same window as the all-hits model, so both are compared on the same requests
        until = int(np.load(CACHE / f"enterprise_{ds}_verifier_allhits.npz")["train_until"])
        np.savez_compressed(CACHE / f"enterprise_{ds}_verifier_allhits_grayonly.npz", cand=cand, ft=score(model, pairs),
                            train_until=np.array(max(until, int(cand[train_pos[-1]]))))
        print(f"{ds}: trained on {len(train_pos)} gray-zone pairs ({y.mean():.3f} positive), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
