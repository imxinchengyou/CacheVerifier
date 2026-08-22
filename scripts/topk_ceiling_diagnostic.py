"""Phase 0 diagnostic for the Top-K retrieval direction (memory:
future_research_directions.md, "Top-1 retrieval ceiling"; discussed
2026-08-22).

`cacheverifier/cache/store.py::VectorCacheStore.query` only ever returns the
single nearest neighbor. `SynchronousVerifiedPolicy` (Groups C/D) therefore
never gets a second chance: if the top-1 candidate falls in the gray zone
and is rejected (or would be rejected by an oracle, i.e. is genuinely
wrong), the request is a MISS even if a *correct* candidate existed at
rank 2..K but was never surfaced.

This script answers the cheap, model-free question that should gate whether
a real K-candidate cascade (Group C-K / D-K) is worth building at all: among
requests whose top-1 candidate is in the gray zone [tau_low, tau_high) and
WRONG, how often does a correct candidate exist at rank 2..K?

Two numbers are reported, and the gap between them matters:

  - `raw_ceiling`: correct candidate exists anywhere in ranks 2..K, ignoring
    whether a sequential cascade could ever reach it. An upper bound only.
  - `reachable_ceiling`: correct candidate exists at some rank m in 2..K
    AND every candidate in ranks 1..m has similarity >= tau_low. This is
    what actually matters, because similarity is non-increasing in rank
    (HNSW returns nearest first) -- a cascade that walks candidates in
    order and stops the first time it sees similarity < tau_low (see the
    design discussion: rank 1 < tau_low means EVERY later rank is also
    < tau_low, so continuing is provably useless there) can only ever reach
    a correct candidate that sits inside the tau_low-contiguous prefix.

No verifier is called here -- `would_be_correct` is ground truth
(equivalence_id match), same convention as MatchTrace. This just measures
the retrieval-recall ceiling a perfect verifier could ever benefit from; it
says nothing about whether a real verifier could actually pick out the
correct candidate once surfaced (that's Phase 1/2, gated on this result
being non-trivial).

Usage:
    python scripts/topk_ceiling_diagnostic.py --config configs/lmarena.yaml \
        --k 5 --output results/topk_ceiling_lmarena.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.cache.store import CacheEntry, VectorCacheStore
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.data.schema import QueryRecord
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.run_verified import CACHE_DIR, _slug

TAU_LOW = 0.80
TAU_HIGH = 0.97


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class TopKOutcome:
    rank1_similarity: float
    rank1_correct: bool
    similarities: list[float]  # rank 1..len, descending
    corrects: list[bool]


def build_topk_trace(records: list[QueryRecord], embedder, k: int) -> list[TopKOutcome | None]:
    """Same streaming/insertion discipline as `verified_sweep.build_match_trace`
    (every record is inserted regardless of what would have happened to it),
    but keeps the full top-k neighbor list per record instead of just rank 1.
    """
    embeddings = embedder.embed(records)
    store = VectorCacheStore(dim=embeddings.shape[1])
    trace: list[TopKOutcome | None] = []

    for record, embedding in zip(records, embeddings):
        matches = store.query_topk(embedding, k)
        if not matches:
            trace.append(None)
        else:
            sims = [m.similarity for m in matches]
            corrects = [m.entry.equivalence_id == record.equivalence_id for m in matches]
            trace.append(TopKOutcome(rank1_similarity=sims[0], rank1_correct=corrects[0], similarities=sims, corrects=corrects))
        store.insert(
            CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer, equivalence_id=record.equivalence_id),
            embedding,
        )

    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tau-low", type=float, default=TAU_LOW)
    parser.add_argument("--tau-high", type=float, default=TAU_HIGH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    dataset_path = Path(cfg.processed_path)
    log(f"Loading records from {dataset_path}...")
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    log(f"Loaded {len(records)} records")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__k{args.k}.topk_trace.json"

    if cache_path.exists():
        log(f"Loading cached top-{args.k} trace from {cache_path}...")
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        trace = [
            TopKOutcome(rank1_similarity=r[0], rank1_correct=r[1], similarities=r[2], corrects=r[3]) if r is not None else None
            for r in raw
        ]
    else:
        log(f"Building top-{args.k} trace (one HNSW pass, no verifier calls)...")
        trace = build_topk_trace(records, embedder, args.k)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [[t.rank1_similarity, t.rank1_correct, t.similarities, t.corrects] if t is not None else None for t in trace]
        cache_path.write_text(json.dumps(rows), encoding="utf-8")
        log(f"Cached to {cache_path}")

    tau_low, tau_high = args.tau_low, args.tau_high

    # Population of interest: rank-1 candidate in the gray zone AND wrong.
    # (rank-1 < tau_low: no K helps, monotonicity argument. rank-1 >= tau_high:
    # policy hits directly today regardless of K, so K changes nothing there.)
    gray_zone_wrong = [t for t in trace if t is not None and tau_low <= t.rank1_similarity < tau_high and not t.rank1_correct]
    n_gray_zone_wrong = len(gray_zone_wrong)
    log(f"Rank-1-in-gray-zone-and-wrong: {n_gray_zone_wrong} requests (out of {len(records)} total)")

    results_by_k: dict[int, dict] = {}
    for k_probe in range(2, args.k + 1):
        n_raw = 0
        n_reachable = 0
        found_at_rank: list[int] = []
        for t in gray_zone_wrong:
            sims = t.similarities[:k_probe]
            corrects = t.corrects[:k_probe]

            # raw ceiling: correct anywhere in ranks 2..k_probe, no reachability constraint
            if any(corrects[1:]):
                n_raw += 1

            # reachable ceiling: walk ranks in order, stop at first sim < tau_low
            reachable_rank = None
            for rank_idx in range(1, k_probe):  # 0-indexed rank 2..k_probe
                if sims[rank_idx] < tau_low:
                    break
                if corrects[rank_idx]:
                    reachable_rank = rank_idx + 1  # 1-indexed rank
                    break
            if reachable_rank is not None:
                n_reachable += 1
                found_at_rank.append(reachable_rank)

        results_by_k[k_probe] = {
            "n_gray_zone_wrong": n_gray_zone_wrong,
            "raw_ceiling_count": n_raw,
            "raw_ceiling_fraction": n_raw / n_gray_zone_wrong if n_gray_zone_wrong else 0.0,
            "reachable_ceiling_count": n_reachable,
            "reachable_ceiling_fraction": n_reachable / n_gray_zone_wrong if n_gray_zone_wrong else 0.0,
            "found_at_rank_histogram": {r: found_at_rank.count(r) for r in sorted(set(found_at_rank))},
        }
        log(
            f"K={k_probe}: raw_ceiling={n_raw}/{n_gray_zone_wrong} "
            f"({results_by_k[k_probe]['raw_ceiling_fraction']:.2%})  "
            f"reachable_ceiling={n_reachable}/{n_gray_zone_wrong} "
            f"({results_by_k[k_probe]['reachable_ceiling_fraction']:.2%})"
        )

    result = {
        "dataset": cfg.name,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "k": args.k,
        "n_total_requests": len(records),
        "n_gray_zone_wrong": n_gray_zone_wrong,
        "by_k": results_by_k,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
