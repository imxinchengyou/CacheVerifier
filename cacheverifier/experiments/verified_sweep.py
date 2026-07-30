import json
import time
from dataclasses import dataclass
from pathlib import Path

from cacheverifier.cache.store import CacheEntry, VectorCacheStore
from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.base import Embedder
from cacheverifier.metrics.core import RequestOutcome
from cacheverifier.verifiers.base import Verifier


@dataclass(frozen=True)
class MatchTrace:
    """One record's nearest-neighbor match against the cache as it existed
    just before that record — and whether that match would have been
    correct — computed independently of any policy.

    This is possible because every record gets inserted into the cache
    regardless of whether it was a hit or a miss (see
    `cacheverifier.experiments.runner.ExperimentRunner.run`): the growing cache's
    contents, and therefore each record's nearest neighbor, never depend on
    what any policy decided to do with previous matches. So Group A/C/D
    (none of which carry state that feeds back into future decisions) can
    all share ONE pass over the stream instead of one pass per grid point.

    NOT valid for Group B: `AdaptiveThresholdPolicy` learns from its own
    past decisions (`observe`), so its match sequence — and therefore its
    results — are NOT reproducible from a policy-independent trace.

    `candidate_index` is the matched entry's position in the ORIGINAL
    records list (entries are inserted in that same order — see
    `build_match_trace`), not the entry itself: resolving it back to a
    `CacheEntry` only needs `records[candidate_index]`, so `save_match_trace`
    doesn't have to duplicate every cached answer's text on disk.
    """

    similarity: float | None
    would_be_correct: bool | None
    candidate_index: int | None


def resolve_candidate(records: list[QueryRecord], trace_entry: MatchTrace) -> CacheEntry | None:
    if trace_entry.candidate_index is None:
        return None
    record = records[trace_entry.candidate_index]
    return CacheEntry(
        query_id=record.query_id, query=record.query, answer=record.answer, equivalence_id=record.equivalence_id
    )


def build_match_trace(records: list[QueryRecord], embedder: Embedder) -> list[MatchTrace]:
    """The one expensive pass: build the growing HNSW cache once and record
    each request's nearest-neighbor match. Reused across an entire Group
    A/C/D grid sweep so ANN search only happens once per dataset."""
    embeddings = embedder.embed(records)
    store = VectorCacheStore(dim=embeddings.shape[1])
    trace: list[MatchTrace] = []

    for record, embedding in zip(records, embeddings):
        match = store.query(embedding)
        if match is None:
            trace.append(MatchTrace(similarity=None, would_be_correct=None, candidate_index=None))
        else:
            would_be_correct = match.entry.equivalence_id == record.equivalence_id
            trace.append(MatchTrace(similarity=match.similarity, would_be_correct=would_be_correct, candidate_index=match.index))
        store.insert(
            CacheEntry(
                query_id=record.query_id,
                query=record.query,
                answer=record.answer,
                equivalence_id=record.equivalence_id,
            ),
            embedding,
        )

    return trace


def save_match_trace(trace: list[MatchTrace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[t.similarity, t.would_be_correct, t.candidate_index] for t in trace]
    path.write_text(json.dumps(rows), encoding="utf-8")


def load_match_trace(path: Path) -> list[MatchTrace]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [MatchTrace(similarity=s, would_be_correct=c, candidate_index=idx) for s, c, idx in rows]


@dataclass(frozen=True)
class ScoredCandidate:
    score: float
    latency_ms: float


def score_gray_zone(
    records: list[QueryRecord],
    trace: list[MatchTrace],
    verifier: Verifier,
    gray_zone_lo: float,
    gray_zone_hi: float,
    log_every: int = 200,
) -> dict[int, ScoredCandidate]:
    """Run `verifier.score` once for every trace index whose similarity
    falls in [gray_zone_lo, gray_zone_hi). Callers should set this range to
    the union of every (tau_low, tau_high) pair they're about to sweep with
    `replay`, so the model runs on each candidate exactly once no matter how
    many grid points (or verifier thresholds) end up reusing the result.

    Prints periodic progress (count/rate/ETA) since this is the one loop in
    the sweep that calls a real (slow, CPU-bound) model once per candidate
    instead of being a cheap re-thresholding pass.
    """
    total = sum(1 for t in trace if t.similarity is not None and gray_zone_lo <= t.similarity < gray_zone_hi)
    print(f"[score_gray_zone] {total} candidates to score", flush=True)

    scored: dict[int, ScoredCandidate] = {}
    t_start = time.time()
    for i, (record, t) in enumerate(zip(records, trace)):
        if t.similarity is not None and gray_zone_lo <= t.similarity < gray_zone_hi:
            candidate = resolve_candidate(records, t)
            score, latency_ms = verifier.score(record, candidate)
            scored[i] = ScoredCandidate(score=score, latency_ms=latency_ms)

            done = len(scored)
            if done % log_every == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else float("inf")
                print(
                    f"[score_gray_zone] {done}/{total} ({100 * done / max(1, total):5.1f}%)  "
                    f"{rate:.1f} ex/s  elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m  "
                    f"last_score={score:.3f}  last_latency={latency_ms:.1f}ms",
                    flush=True,
                )
    return scored


def save_scored(scored: dict[int, ScoredCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {str(i): [s.score, s.latency_ms] for i, s in scored.items()}
    path.write_text(json.dumps(rows), encoding="utf-8")


def load_scored(path: Path) -> dict[int, ScoredCandidate]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(i): ScoredCandidate(score=s, latency_ms=lat) for i, (s, lat) in rows.items()}


def replay(
    trace: list[MatchTrace],
    scored: dict[int, ScoredCandidate],
    tau_low: float,
    tau_high: float,
    threshold: float,
) -> list[RequestOutcome]:
    """Cheaply re-derive the online decision sequence for one
    (tau_low, tau_high, threshold) combination from a precomputed `trace` +
    `scored` map, without touching the ANN index or the verifier model
    again. Every index the gray-zone branch below needs must already be a
    key in `scored` — i.e. `tau_low` must be >= whatever `gray_zone_lo` was
    used to build `scored`, and `tau_high` <= `gray_zone_hi`.
    """
    outcomes: list[RequestOutcome] = []
    for i, t in enumerate(trace):
        if t.similarity is None:
            outcomes.append(RequestOutcome(action="miss", correct=True, would_be_correct=None))
            continue

        if t.similarity >= tau_high:
            action, verifier_invoked, latency_ms = "hit", False, 0.0
        elif t.similarity < tau_low:
            action, verifier_invoked, latency_ms = "miss", False, 0.0
        else:
            candidate = scored[i]
            action = "hit" if candidate.score >= threshold else "miss"
            verifier_invoked, latency_ms = True, candidate.latency_ms

        correct = t.would_be_correct if action == "hit" else True
        outcomes.append(
            RequestOutcome(
                action=action,
                correct=correct,
                would_be_correct=t.would_be_correct,
                verifier_invoked=verifier_invoked,
                verifier_latency_ms=latency_ms,
            )
        )
    return outcomes
