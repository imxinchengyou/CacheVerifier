"""Top-K cascade sweep (Top-1 retrieval ceiling direction, memory:
future_research_directions.md; design discussed 2026-08-22; Phase 0 ceiling
diagnostic: `scripts/topk_ceiling_diagnostic.py`).

`cacheverifier.experiments.verified_sweep` and `SynchronousVerifiedPolicy`
only ever look at the single nearest cache neighbor: if it's in the gray
zone and the verifier rejects it, the request is a MISS even if a correct
candidate existed at rank 2..K but was never surfaced. This module extends
the same "one expensive pass, cheap replay" design to a K-candidate
cascade:

    for candidate in top_K (similarity descending):
        if candidate.similarity < tau_low: break   # monotonicity: later
                                                     # ranks are only ever
                                                     # LESS similar, so none
                                                     # of them can be >= tau_low
                                                     # either -- see store.py
                                                     # docstring / design
                                                     # discussion.
        if candidate.similarity >= tau_high: HIT
        if verifier(query, candidate).approved: HIT
        # else: rejected, try the next-ranked candidate

Deliberately NOT wired into `CachePolicy`/`ExperimentRunner`/
`SynchronousVerifiedPolicy` (the K=1 interface every already-published A/B/
C/D/E result depends on) -- this is a self-contained parallel path, same
posture as `scripts/bucketing_ablation.py`'s standalone ablation. `run_cascade_slow`
is the naive one-candidate-at-a-time reference implementation, used only to
cross-check the fast trace-once/score-once/replay-many path (see
tests/test_topk_sweep.py, same cross-check discipline as
tests/test_verified_sweep.py::test_fast_sweep_matches_slow_policy_run_exactly).
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from cacheverifier.cache.store import CacheEntry, VectorCacheStore
from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.base import Embedder
from cacheverifier.experiments.verified_sweep import ScoredCandidate
from cacheverifier.metrics.core import RequestOutcome
from cacheverifier.verifiers.base import Verifier


@dataclass(frozen=True)
class CascadeTrace:
    """One record's top-K neighbor list against the cache as it existed just
    before that record, most-similar first -- the K-candidate generalization
    of `verified_sweep.MatchTrace`. Same policy-independence argument
    applies (every record is inserted regardless of hit/miss, so the growing
    cache's contents never depend on any cascade policy's decisions), so
    this can be built once and reused across an entire (tau_low, tau_high,
    threshold, K-probe) sweep.

    All three lists are the same length (0..k, shorter than k only when the
    cache held fewer than k entries yet, e.g. near the start of the stream).
    `candidate_indices[r]` is `records[...]`-indexable, same convention as
    `MatchTrace.candidate_index`.
    """

    similarities: list[float]
    would_be_corrects: list[bool]
    candidate_indices: list[int]


def resolve_candidate_at_rank(records: list[QueryRecord], trace_entry: CascadeTrace, rank: int) -> CacheEntry:
    record = records[trace_entry.candidate_indices[rank]]
    return CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer, equivalence_id=record.equivalence_id)


def build_cascade_trace(records: list[QueryRecord], embedder: Embedder, k: int) -> list[CascadeTrace]:
    """The one expensive pass for the cascade direction: build the growing
    HNSW cache once, keeping the top-k neighbors (not just rank 1) for every
    record. Reused across an entire cascade grid sweep so ANN search only
    happens once per (dataset, k)."""
    embeddings = embedder.embed(records)
    store = VectorCacheStore(dim=embeddings.shape[1])
    trace: list[CascadeTrace] = []

    for record, embedding in zip(records, embeddings):
        matches = store.query_topk(embedding, k)
        sims = [m.similarity for m in matches]
        corrects = [m.entry.equivalence_id == record.equivalence_id for m in matches]
        indices = [m.index for m in matches]
        trace.append(CascadeTrace(similarities=sims, would_be_corrects=corrects, candidate_indices=indices))
        store.insert(
            CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer, equivalence_id=record.equivalence_id),
            embedding,
        )

    return trace


def save_cascade_trace(trace: list[CascadeTrace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[t.similarities, t.would_be_corrects, t.candidate_indices] for t in trace]
    path.write_text(json.dumps(rows), encoding="utf-8")


def load_cascade_trace(path: Path) -> list[CascadeTrace]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [CascadeTrace(similarities=s, would_be_corrects=c, candidate_indices=idx) for s, c, idx in rows]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score_cascade_candidates(
    records: list[QueryRecord],
    trace: list[CascadeTrace],
    verifier: Verifier,
    gray_zone_lo: float,
    gray_zone_hi: float,
    log_every: int = 500,
) -> dict[tuple[int, int], ScoredCandidate]:
    """Run `verifier.score` once for every (record_index, rank) the cascade
    could actually reach for SOME (tau_low, threshold) combo in a sweep that
    fixes tau_high at `gray_zone_hi` and lets tau_low range down to
    `gray_zone_lo` -- i.e. rank r is only scored if ranks 0..r-1 ALL have
    similarity in [gray_zone_lo, gray_zone_hi) too (a prior rank below
    gray_zone_lo means the walk breaks there for every tau_low in the sweep;
    a prior rank >= gray_zone_hi means it's a direct hit there for every run,
    since tau_high is fixed). This is the same reachability walk
    `replay_cascade` does at replay time -- computing it here as well means
    `scored` never holds an entry `replay_cascade` couldn't reach, without
    changing any result (unreached entries are never looked up either way).
    Keeps the score-once/replay-many-times property `verified_sweep.
    score_gray_zone` has for K=1: correctness and thresholding are re-derived
    cheaply per grid point in `replay_cascade`, not by re-running the
    verifier.
    """
    total = 0
    for t in trace:
        for sim in t.similarities:
            if not (gray_zone_lo <= sim < gray_zone_hi):
                break
            total += 1

    log(f"[score_cascade_candidates] {total} (request, rank) candidates to score")

    scored: dict[tuple[int, int], ScoredCandidate] = {}
    t_start = time.time()
    for i, (record, t) in enumerate(zip(records, trace)):
        for rank, sim in enumerate(t.similarities):
            if not (gray_zone_lo <= sim < gray_zone_hi):
                break  # unreachable at this rank for every combo in the sweep -- and so is every later rank
            candidate = resolve_candidate_at_rank(records, t, rank)
            score, latency_ms = verifier.score(record, candidate)
            scored[(i, rank)] = ScoredCandidate(score=score, latency_ms=latency_ms)

            done = len(scored)
            if done % log_every == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                log(
                    f"[score_cascade_candidates] {done}/{total} ({100 * done / max(1, total):5.1f}%)  "
                    f"{rate:.1f} ex/s  elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m"
                )
    return scored


def save_scored_cascade(scored: dict[tuple[int, int], ScoredCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {f"{i}_{r}": [s.score, s.latency_ms] for (i, r), s in scored.items()}
    path.write_text(json.dumps(rows), encoding="utf-8")


def load_scored_cascade(path: Path) -> dict[tuple[int, int], ScoredCandidate]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for key, (score, latency_ms) in rows.items():
        i_str, r_str = key.rsplit("_", 1)
        result[(int(i_str), int(r_str))] = ScoredCandidate(score=score, latency_ms=latency_ms)
    return result


def replay_cascade(
    trace: list[CascadeTrace],
    scored: dict[tuple[int, int], ScoredCandidate],
    tau_low: float,
    tau_high: float,
    threshold: float,
    k: int,
) -> list[RequestOutcome]:
    """Cheaply re-derive the online cascade decision sequence for one
    (tau_low, tau_high, threshold, k) combination, without touching the ANN
    index or the verifier model again. Requires every (i, rank) the walk
    below needs to already be a key in `scored` -- i.e. `tau_low` must be >=
    whatever `gray_zone_lo` was used to build `scored`, `tau_high` <=
    `gray_zone_hi`, and `k` <= the k used to build `trace`.
    """
    outcomes: list[RequestOutcome] = []
    for i, t in enumerate(trace):
        if not t.similarities:
            outcomes.append(RequestOutcome(action="miss", correct=True, would_be_correct=None))
            continue

        action = "miss"
        correct = True
        would_be_correct_rank1 = t.would_be_corrects[0]
        verifier_invoked = False
        total_latency_ms = 0.0
        calls = 0

        for rank in range(min(k, len(t.similarities))):
            sim = t.similarities[rank]
            if sim < tau_low:
                break
            if sim >= tau_high:
                action, correct = "hit", t.would_be_corrects[rank]
                break

            candidate_score = scored[(i, rank)]
            verifier_invoked = True
            calls += 1
            total_latency_ms += candidate_score.latency_ms
            if candidate_score.score >= threshold:
                action, correct = "hit", t.would_be_corrects[rank]
                break
            # rejected -- fall through to the next-ranked candidate

        outcomes.append(
            RequestOutcome(
                action=action,
                correct=correct,
                would_be_correct=would_be_correct_rank1,
                verifier_invoked=verifier_invoked,
                verifier_latency_ms=total_latency_ms,
                verifier_calls=calls,
            )
        )
    return outcomes


def run_cascade_slow(
    records: list[QueryRecord],
    embedder: Embedder,
    verifier: Verifier,
    tau_low: float,
    tau_high: float,
    threshold: float,
    k: int,
) -> list[RequestOutcome]:
    """Naive one-candidate-at-a-time reference implementation: streams the
    cache exactly like `ExperimentRunner.run` but queries `query_topk` and
    walks the cascade directly, calling the verifier live instead of
    precomputing/replaying scores. Used only to cross-check
    `build_cascade_trace` + `score_cascade_candidates` + `replay_cascade`
    (see tests/test_topk_sweep.py) -- not meant for real sweeps, since it
    re-runs the verifier from scratch for every grid point.
    """
    if not records:
        return []

    embeddings = embedder.embed(records)
    store = VectorCacheStore(dim=embeddings.shape[1])
    outcomes: list[RequestOutcome] = []

    for record, embedding in zip(records, embeddings):
        matches = store.query_topk(embedding, k)

        action = "miss"
        correct = True
        would_be_correct_rank1 = None
        verifier_invoked = False
        total_latency_ms = 0.0
        calls = 0

        if matches:
            would_be_correct_rank1 = matches[0].entry.equivalence_id == record.equivalence_id
            for match in matches:
                if match.similarity < tau_low:
                    break
                if match.similarity >= tau_high:
                    action, correct = "hit", match.entry.equivalence_id == record.equivalence_id
                    break
                score, latency_ms = verifier.score(record, match.entry)
                verifier_invoked = True
                calls += 1
                total_latency_ms += latency_ms
                if score >= threshold:
                    action, correct = "hit", match.entry.equivalence_id == record.equivalence_id
                    break

        outcomes.append(
            RequestOutcome(
                action=action,
                correct=correct,
                would_be_correct=would_be_correct_rank1,
                verifier_invoked=verifier_invoked,
                verifier_latency_ms=total_latency_ms,
                verifier_calls=calls,
            )
        )

        store.insert(
            CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer, equivalence_id=record.equivalence_id),
            embedding,
        )

    return outcomes
