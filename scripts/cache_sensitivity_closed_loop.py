"""RESEARCH_PROPOSAL.md 方向 26.9: how sensitive is the gray-zone distribution
of a write-on-miss semantic cache to the deployed threshold lambda, and is it
stateless (depends only on the current lambda) or stateful (depends on
history)? This is the core assumption of Performative Risk Control
(arXiv 2505.24097), which needs the distribution map to be gamma-sensitive.

Each run is a full write-on-miss closed loop (PAPER_B §4.2 regime 2) with a
FIXED threshold schedule on the off-the-shelf verifier score V: constant
lambda for the grid runs, lambda_a for the first half and lambda_b for the
second half for the hysteresis runs. All runs of one dataset share one
verifier-score memo (scores are deterministic). Per run we keep every
gray-zone observation (stream position, V, correct, accepted).

Usage (GPU box): python scripts/cache_sensitivity_closed_loop.py --dataset quora
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cacheverifier.cache.store import CacheEntry, VectorCacheStore
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier
from crc_closed_loop_calibration_sources import ScoreCache
from crc_closed_loop_self_selection import TAU_HIGH, TAU_LOW

GRID_Q = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99)
A_Q, B_Q = 0.97, 0.85
PREFIX = {
    "quora": "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000",
    "lmarena": "lmarena__precomputed__n60000",
    "search_queries_corrected": "search_queries_corrected__precomputed__n150000",
}
CE = "cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2"


def run(records, embeddings, scorer, lam_first, lam_second, label):
    store = VectorCacheStore(dim=embeddings.shape[1])
    half = len(records) // 2
    obs = []
    t0 = time.time()
    for i, (record, emb) in enumerate(zip(records, embeddings)):
        lam = lam_first if i < half else lam_second
        match = store.query(emb)
        if match is None or match.similarity < TAU_LOW:
            outcome = "miss"
        elif match.similarity >= TAU_HIGH:
            outcome = "hit"
        else:
            v = scorer.score(record, match.entry)
            correct = match.entry.equivalence_id == record.equivalence_id
            accepted = v > lam
            obs.append((i, v, correct, accepted))
            outcome = "hit" if accepted else "miss"
        if outcome == "miss":
            store.insert(CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer,
                                    equivalence_id=record.equivalence_id), emb)
        if (i + 1) % 20000 == 0:
            print(f"  [{label}] {i + 1}/{len(records)} gray {len(obs)} memo {len(scorer.memo)} {time.time() - t0:.0f}s", flush=True)
    return {"label": label, "lam_first": lam_first, "lam_second": lam_second, "final_store": len(store),
            "obs": [[int(p), float(v), bool(c), bool(a)] for p, v, c, a in obs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(PREFIX))
    args = p.parse_args()
    cfg = load_dataset_config(f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    embeddings = build_embedder(cfg.embedder, cfg.embedder_model).embed(records)
    scored = json.loads(Path(f"results/.cache/{PREFIX[args.dataset]}__{CE}__lo0.8__hi0.97.scored.json").read_text())
    v_open = np.array([s[0] for s in scored.values()])  # scored rows are [score, latency_ms]
    lam = {q: float(np.quantile(v_open, q)) for q in GRID_Q}
    print("lambda grid:", {q: round(x, 3) for q, x in lam.items()}, flush=True)
    scorer = ScoreCache(CrossEncoderVerifier())
    runs = []
    for q in GRID_Q:
        runs.append(run(records, embeddings, scorer, lam[q], lam[q], f"fixed_q{q}"))
    runs.append(run(records, embeddings, scorer, lam[A_Q], lam[B_Q], "A_then_B"))
    runs.append(run(records, embeddings, scorer, lam[B_Q], lam[A_Q], "B_then_A"))
    out = Path(f"results/cache_sensitivity_{args.dataset}.json")
    out.write_text(json.dumps({"dataset": args.dataset, "n_records": len(records), "grid_q": GRID_Q,
                               "lambda": {str(q): x for q, x in lam.items()}, "A_q": A_Q, "B_q": B_Q, "runs": runs}))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
