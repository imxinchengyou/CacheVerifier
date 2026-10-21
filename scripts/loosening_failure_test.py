"""RESEARCH_PROPOSAL.md 方向 26.13: does the current sliding-window CRC
(PAPER_B §4.4 / direction 24) exceed its target after a large loosening,
because of the cache's statefulness?

Write-on-miss closed loop, no eviction. alpha is 0.01 for the first half of
the stream and 0.03 from the midpoint ("switch"), against constant 0.03
("fixed_hi") and constant 0.01 ("fixed_lo"). Labels: S2 (every rejected
gray-zone request + a 5% audit of approved ones, weighted 1/p) or S0 (every
gray-zone request). Recalibration exactly as run_source in
crc_closed_loop_calibration_sources.py, except that alpha follows the
schedule.

Usage (GPU box): python scripts/loosening_failure_test.py --dataset lmarena
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
from cacheverifier.metrics.core import crc_select_threshold, crc_select_threshold_weighted
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier
from crc_closed_loop_calibration_sources import MIN_POOL, RECAL_EVERY, RECAL_WINDOW, TAU_HIGH, TAU_LOW, WARMUP_N, ScoreCache

RATE = 0.05
SCHEDULES = {"switch": (0.01, 0.03), "fixed_hi": (0.03, 0.03), "fixed_lo": (0.01, 0.01)}


def run(records, embeddings, scorer, frozen, source, alpha_first, alpha_second, label, seed=0):
    store = VectorCacheStore(dim=embeddings.shape[1])
    side_rng = np.random.default_rng(seed + 1_000_003)
    half = len(records) // 2
    threshold, since = frozen, 0
    ps, pl, pw = [], [], []
    obs, n_audits, thresholds = [], 0, []
    t0 = time.time()
    for i, (record, emb) in enumerate(zip(records, embeddings)):
        alpha = alpha_first if i < half else alpha_second
        match = store.query(emb)
        if match is None or match.similarity < TAU_LOW:
            outcome = "miss"
        elif match.similarity >= TAU_HIGH:
            outcome = "hit"
        else:
            s = scorer.score(record, match.entry)
            correct = match.entry.equivalence_id == record.equivalence_id
            approved = s > threshold
            audited = source == "S2" and approved and side_rng.random() < RATE
            n_audits += audited
            obs.append((i, bool(approved), bool(correct)))
            outcome = "hit" if approved and not audited else "miss"
            label_ = 1 if correct else 0
            if source == "S0":
                ps.append(s), pl.append(label_), pw.append(1.0)
            elif not approved:
                ps.append(s), pl.append(label_), pw.append(1.0)
            elif audited:
                ps.append(s), pl.append(label_), pw.append(1.0 / RATE)
            since += 1
            ready = len(obs) >= WARMUP_N if source == "S0" else (len(obs) >= WARMUP_N and len(ps) >= MIN_POOL)
            if ready and since >= RECAL_EVERY:
                sc, lb, wt = (np.array(x[-RECAL_WINDOW:]) for x in (ps, pl, pw))
                picked = crc_select_threshold(sc, lb, alpha) if source == "S0" else crc_select_threshold_weighted(sc, lb, wt, alpha)
                if picked is not None:
                    threshold = picked
                    thresholds.append((i, float(threshold)))
                since = 0
        if outcome == "miss":
            store.insert(CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer,
                                    equivalence_id=record.equivalence_id), emb)
    print(f"  [{label}] gray {len(obs)} audits {n_audits} store {len(store)} {time.time() - t0:.0f}s", flush=True)
    return {"label": label, "source": source, "alpha_first": alpha_first, "alpha_second": alpha_second,
            "n_audits": int(n_audits), "store": len(store), "thresholds": thresholds,
            "obs": [[p, a, c] for p, a, c in obs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["quora", "lmarena", "search_queries_corrected"])
    args = p.parse_args()
    cfg = load_dataset_config(f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    embeddings = build_embedder(cfg.embedder, cfg.embedder_model).embed(records)
    frozen = json.loads(Path(f"results/crc_closed_loop_calibration_sources_{args.dataset}.json").read_text())["frozen_threshold"]
    scorer = ScoreCache(CrossEncoderVerifier())
    runs = []
    for source in ("S2", "S0"):
        for name, (a1, a2) in SCHEDULES.items():
            runs.append(run(records, embeddings, scorer, frozen, source, a1, a2, f"{source}_{name}"))
    out = Path(f"results/loosening_failure_{args.dataset}.json")
    out.write_text(json.dumps({"dataset": args.dataset, "n_records": len(records), "audit_rate": RATE, "runs": runs}))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
