"""RESEARCH_PROPOSAL.md 方向 26.11: does a TTL on cache entries cut off the
influence of past thresholds, or does it outlive the TTL through cascades?

Same write-on-miss closed loop and A/B thresholds as
cache_sensitivity_closed_loop.py (26.9), with every cache entry expiring TTL
requests after it was written. Nearest-neighbour search is EXACT here (see
ExactTTLStore), unlike the HNSW search of every other run in this project, so
an infinite-TTL run is close to but not bit-identical with 26.9. Per TTL: constant A, constant B, A->B and B->A
(switch at the midpoint). LmArena only.

Usage (GPU box): python scripts/history_decay_ttl.py --dataset lmarena
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cacheverifier.cache.store import CacheEntry
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier
from cache_sensitivity_closed_loop import A_Q, B_Q, CE, PREFIX
from crc_closed_loop_calibration_sources import ScoreCache
from crc_closed_loop_self_selection import TAU_HIGH, TAU_LOW

TTLS = (2000, 5000, 10000, 20000)


class ExactTTLStore:
    """Exact (brute-force, on GPU) nearest-neighbour cache whose entries
    expire `ttl` requests after insertion. hnswlib's mark_deleted keeps
    deleted nodes in the graph, so with a short TTL almost every node is
    deleted and search slows to a crawl; exact search over the live entries
    has no such problem. Entries are identified by record position, so the
    whole embedding matrix lives on the GPU once and a live mask selects
    the current cache contents."""

    def __init__(self, all_embeddings, ttl):
        import torch

        self.torch = torch
        self.E = torch.as_tensor(all_embeddings, dtype=torch.float32, device="cuda")
        self.live = torch.zeros(len(all_embeddings), dtype=torch.bool, device="cuda")
        self.ttl = ttl
        self._born = deque()
        self.n_live = 0

    def insert(self, pos):
        self.live[pos] = True
        self._born.append(pos)
        self.n_live += 1

    def expire(self, now):
        while self._born and now - self._born[0] >= self.ttl:
            self.live[self._born.popleft()] = False
            self.n_live -= 1

    def query(self, pos):
        """(best cached position, similarity) or None."""
        if self.n_live == 0:
            return None
        sims = self.E[:pos] @ self.E[pos]
        sims = sims.masked_fill(~self.live[:pos], -2.0)
        best = int(self.torch.argmax(sims))
        return best, float(sims[best])


def run(records, embeddings, scorer, ttl, lam_first, lam_second, label, store_cls=ExactTTLStore):
    store = store_cls(embeddings, ttl)
    half = len(records) // 2
    obs = []
    t0 = time.time()
    for i, record in enumerate(records):
        store.expire(i)
        lam = lam_first if i < half else lam_second
        match = store.query(i)
        if match is None or match[1] < TAU_LOW:
            outcome = "miss"
        elif match[1] >= TAU_HIGH:
            outcome = "hit"
        else:
            cached = records[match[0]]
            entry = CacheEntry(query_id=cached.query_id, query=cached.query, answer=cached.answer,
                               equivalence_id=cached.equivalence_id)
            v = scorer.score(record, entry)
            correct = cached.equivalence_id == record.equivalence_id
            accepted = v > lam
            obs.append((i, v, correct, accepted))
            outcome = "hit" if accepted else "miss"
        if outcome == "miss":
            store.insert(i)
    print(f"  [ttl={ttl} {label}] gray {len(obs)} live {store.n_live} memo {len(scorer.memo)} {time.time() - t0:.0f}s", flush=True)
    return {"ttl": ttl, "label": label, "lam_first": lam_first, "lam_second": lam_second, "live_final": store.n_live,
            "obs": [[int(p), float(v), bool(c), bool(a)] for p, v, c, a in obs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lmarena", choices=list(PREFIX))
    args = p.parse_args()
    cfg = load_dataset_config(f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    embeddings = build_embedder(cfg.embedder, cfg.embedder_model).embed(records)
    scored = json.loads(Path(f"results/.cache/{PREFIX[args.dataset]}__{CE}__lo0.8__hi0.97.scored.json").read_text())
    v_open = np.array([s[0] for s in scored.values()])  # scored rows are [score, latency_ms]
    lamA, lamB = float(np.quantile(v_open, A_Q)), float(np.quantile(v_open, B_Q))
    lam95 = float(np.quantile(v_open, 0.95))
    print(f"A {lamA:.3f} B {lamB:.3f} q95 {lam95:.3f}", flush=True)
    scorer = ScoreCache(CrossEncoderVerifier())
    runs = [run(records, embeddings, scorer, 10**9, lamA, lamB, "A_then_B_ttl_inf_exact_check")]
    for ttl in TTLS:
        for first, second, label in ((lamA, lamA, "fixed_A"), (lamB, lamB, "fixed_B"),
                                     (lamA, lamB, "A_then_B"), (lamB, lamA, "B_then_A")):
            runs.append(run(records, embeddings, scorer, ttl, first, second, label))
    out = Path(f"results/history_decay_ttl_{args.dataset}.json")
    out.write_text(json.dumps({"dataset": args.dataset, "n_records": len(records), "ttls": TTLS,
                               "lamA": lamA, "lamB": lamB, "lam95": lam95, "runs": runs}))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
