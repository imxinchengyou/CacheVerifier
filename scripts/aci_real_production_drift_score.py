"""Direction 19 follow-up (candidate B): everything ACI has been tested
against so far is either a structural/synthetic drift (Top-K cascade's
cache-density effect, direction 18) or a benchmark's own construction
artifact (Quora's CRC violation, confirmed by Section 5.21 to be a labeling
batch effect, not real production drift). This is the first test against
REAL production drift: the comcastcares Twitter customer-support brand
(Section 5.8), whose gray-zone positive rate genuinely shifts between the
fine-tuning window and later traffic (1.0% train vs 5.2% test, already
documented as the root cause of Group E's fine-tuning turning harmful on
this brand).

This script only does the missing step -- scoring every (query, answer) row
in results/finetune_verifier_experiment_twitter_comcast.examples.json with
the off-the-shelf ms-marco verifier (no cached scores exist for this stash,
unlike the benchmark datasets' Group D/E caches) -- so the resulting
(position, score, label) stream can feed a static-vs-ACI comparison exactly
like the other direction-19 experiments. Requires GPU (or a working local
torch); run on the remote server.

Usage:
    python scripts/aci_real_production_drift_score.py \
        --stash results/finetune_verifier_experiment_twitter_comcast.examples.json \
        --output results/aci_real_production_drift_scored_comcast.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stash", required=True)
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    stash = json.loads(Path(args.stash).read_text(encoding="utf-8"))
    all_rows = stash["train"] + stash["test"]
    all_rows.sort(key=lambda r: r[0])  # true chronological order by stream position
    n_train = len(stash["train"])
    log(f"{len(all_rows)} total rows ({n_train} train + {len(stash['test'])} test), sorted by stream position")

    model = CrossEncoder(args.model, device=device)
    pairs = [(r[1], r[2]) for r in all_rows]
    t0 = time.time()
    scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
    log(f"Scored {len(pairs)} pairs in {time.time() - t0:.1f}s")

    output = {
        "stash": args.stash,
        "model": args.model,
        "n_total": len(all_rows),
        "n_train_split": n_train,
        "positions": [r[0] for r in all_rows],
        "scores": [float(s) for s in scores],
        "labels": [1 if r[3] else 0 for r in all_rows],
    }
    Path(args.output).write_text(json.dumps(output), encoding="utf-8")
    log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
