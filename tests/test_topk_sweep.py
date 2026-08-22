from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.hash_embedder import HashEmbedder
from cacheverifier.experiments.topk_sweep import (
    build_cascade_trace,
    load_cascade_trace,
    load_scored_cascade,
    replay_cascade,
    run_cascade_slow,
    save_cascade_trace,
    save_scored_cascade,
    score_cascade_candidates,
)
from cacheverifier.metrics.core import error_rate, hit_rate, mean_verifier_calls
from cacheverifier.verifiers.oracle_verifier import OracleVerifier


def make_mixed_records(n_classes: int, repeats: int) -> list[QueryRecord]:
    records = []
    for c in range(n_classes):
        for r in range(repeats):
            records.append(
                QueryRecord(
                    query_id=f"c{c}-{r}",
                    query=f"duplicate query number {c}" if r > 0 else f"duplicate query number {c} variant",
                    answer=f"answer-{c}",
                    equivalence_id=f"class-{c}",
                )
            )
    return records


def test_fast_cascade_replay_matches_slow_cascade_exactly():
    records = make_mixed_records(n_classes=15, repeats=6)
    embedder = HashEmbedder()
    tau_low, tau_high, k = 0.4, 0.9, 3

    slow_outcomes = run_cascade_slow(
        records, embedder, OracleVerifier(latency_ms=11.0), tau_low=tau_low, tau_high=tau_high, threshold=0.5, k=k
    )

    trace = build_cascade_trace(records, embedder, k)
    scored = score_cascade_candidates(records, trace, OracleVerifier(latency_ms=11.0), gray_zone_lo=tau_low, gray_zone_hi=tau_high)
    fast_outcomes = replay_cascade(trace, scored, tau_low=tau_low, tau_high=tau_high, threshold=0.5, k=k)

    assert len(fast_outcomes) == len(slow_outcomes)
    for slow, fast in zip(slow_outcomes, fast_outcomes):
        assert slow.action == fast.action
        assert slow.correct == fast.correct
        assert slow.would_be_correct == fast.would_be_correct
        assert slow.verifier_invoked == fast.verifier_invoked
        assert slow.verifier_calls == fast.verifier_calls
        assert slow.verifier_latency_ms == fast.verifier_latency_ms

    assert hit_rate(fast_outcomes) == hit_rate(slow_outcomes)
    assert error_rate(fast_outcomes) == error_rate(slow_outcomes)


def test_cascade_k1_matches_k1_single_candidate_behavior():
    """A k=1 cascade should behave exactly like there being no cascade at
    all -- same sanity check as "K=1 is the existing Group C/D mechanism"."""
    records = make_mixed_records(n_classes=15, repeats=6)
    embedder = HashEmbedder()
    tau_low, tau_high = 0.4, 0.9

    trace = build_cascade_trace(records, embedder, k=1)
    scored = score_cascade_candidates(records, trace, OracleVerifier(latency_ms=5.0), gray_zone_lo=tau_low, gray_zone_hi=tau_high)
    outcomes = replay_cascade(trace, scored, tau_low=tau_low, tau_high=tau_high, threshold=0.5, k=1)

    for o in outcomes:
        assert o.verifier_calls <= 1


def test_cascade_never_reaches_a_rank_below_tau_low():
    """Monotonicity argument from the design discussion: once a rank's
    similarity drops below tau_low, no later (necessarily-less-similar) rank
    should ever get a verifier call."""
    records = make_mixed_records(n_classes=10, repeats=8)
    embedder = HashEmbedder()
    tau_low, tau_high, k = 0.6, 0.9, 5

    trace = build_cascade_trace(records, embedder, k)
    scored = score_cascade_candidates(records, trace, OracleVerifier(latency_ms=1.0), gray_zone_lo=tau_low, gray_zone_hi=tau_high)
    outcomes = replay_cascade(trace, scored, tau_low=tau_low, tau_high=tau_high, threshold=0.5, k=k)

    for t, o in zip(trace, outcomes):
        # number of ranks with similarity >= tau_low, capped at k
        n_reachable = 0
        for sim in t.similarities[:k]:
            if sim < tau_low:
                break
            n_reachable += 1
        assert o.verifier_calls <= n_reachable


def test_cascade_trace_survives_save_and_load(tmp_path):
    records = make_mixed_records(n_classes=15, repeats=6)
    trace = build_cascade_trace(records, HashEmbedder(), k=3)

    path = tmp_path / "cascade_trace.json"
    save_cascade_trace(trace, path)
    loaded = load_cascade_trace(path)

    assert loaded == trace


def test_scored_cascade_survives_save_and_load(tmp_path):
    records = make_mixed_records(n_classes=15, repeats=6)
    trace = build_cascade_trace(records, HashEmbedder(), k=3)
    scored = score_cascade_candidates(records, trace, OracleVerifier(latency_ms=6.0), gray_zone_lo=0.4, gray_zone_hi=0.9)

    path = tmp_path / "scored_cascade.json"
    save_scored_cascade(scored, path)
    loaded = load_scored_cascade(path)

    assert loaded == scored


def test_mean_verifier_calls_is_one_for_a_k1_policy_and_can_exceed_one_for_cascade():
    records = make_mixed_records(n_classes=15, repeats=6)
    embedder = HashEmbedder()
    tau_low, tau_high = 0.4, 0.9

    trace_k1 = build_cascade_trace(records, embedder, k=1)
    scored_k1 = score_cascade_candidates(records, trace_k1, OracleVerifier(latency_ms=5.0), gray_zone_lo=tau_low, gray_zone_hi=tau_high)
    outcomes_k1 = replay_cascade(trace_k1, scored_k1, tau_low=tau_low, tau_high=tau_high, threshold=0.5, k=1)
    if any(o.verifier_invoked for o in outcomes_k1):
        assert mean_verifier_calls(outcomes_k1) == 1.0

    trace_k5 = build_cascade_trace(records, embedder, k=5)
    scored_k5 = score_cascade_candidates(records, trace_k5, OracleVerifier(latency_ms=5.0), gray_zone_lo=tau_low, gray_zone_hi=tau_high)
    outcomes_k5 = replay_cascade(trace_k5, scored_k5, tau_low=tau_low, tau_high=tau_high, threshold=0.5, k=5)
    assert mean_verifier_calls(outcomes_k5) >= mean_verifier_calls(outcomes_k1)
