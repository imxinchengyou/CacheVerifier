from cacheverifier.cache.synchronous_verified import SynchronousVerifiedPolicy
from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.hash_embedder import HashEmbedder
from cacheverifier.experiments.runner import ExperimentRunner
from cacheverifier.experiments.verified_sweep import (
    build_match_trace,
    load_match_trace,
    load_scored,
    replay,
    save_match_trace,
    save_scored,
    score_gray_zone,
)
from cacheverifier.metrics.core import error_rate, hit_rate, verifier_fidelity
from cacheverifier.verifiers.oracle_verifier import OracleVerifier


def make_mixed_records(n_classes: int, repeats: int) -> list[QueryRecord]:
    """Mostly-exact-duplicate classes plus a few off-pattern queries, so the
    match sequence has a realistic mix of empty-cache, exact, and near-miss
    similarities to exercise all three branches of the gray-zone gate."""
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


def test_fast_sweep_matches_slow_policy_run_exactly():
    records = make_mixed_records(n_classes=15, repeats=6)
    embedder = HashEmbedder()
    tau_low, tau_high = 0.4, 0.9
    verifier_for_slow = OracleVerifier(latency_ms=33.0)

    slow_outcomes = ExperimentRunner(embedder).run(
        records, SynchronousVerifiedPolicy(tau_low=tau_low, tau_high=tau_high, verifier=verifier_for_slow)
    )

    trace = build_match_trace(records, embedder)
    scored = score_gray_zone(records, trace, OracleVerifier(latency_ms=33.0), gray_zone_lo=tau_low, gray_zone_hi=tau_high)
    fast_outcomes = replay(trace, scored, tau_low=tau_low, tau_high=tau_high, threshold=0.5)

    assert len(fast_outcomes) == len(slow_outcomes)
    for slow, fast in zip(slow_outcomes, fast_outcomes):
        assert slow.action == fast.action
        assert slow.correct == fast.correct
        assert slow.would_be_correct == fast.would_be_correct
        assert slow.verifier_invoked == fast.verifier_invoked
        assert slow.verifier_latency_ms == fast.verifier_latency_ms

    assert hit_rate(fast_outcomes) == hit_rate(slow_outcomes)
    assert error_rate(fast_outcomes) == error_rate(slow_outcomes)


def test_replay_can_reuse_one_score_pass_across_a_tau_low_grid():
    records = make_mixed_records(n_classes=15, repeats=6)
    embedder = HashEmbedder()
    tau_high = 0.9
    tau_low_grid = [0.3, 0.5, 0.7]

    trace = build_match_trace(records, embedder)
    verifier = OracleVerifier(latency_ms=10.0)
    scored_once = score_gray_zone(records, trace, verifier, gray_zone_lo=min(tau_low_grid), gray_zone_hi=tau_high)

    for tau_low in tau_low_grid:
        fast_outcomes = replay(trace, scored_once, tau_low=tau_low, tau_high=tau_high, threshold=0.5)
        slow_outcomes = ExperimentRunner(embedder).run(
            records, SynchronousVerifiedPolicy(tau_low=tau_low, tau_high=tau_high, verifier=OracleVerifier(latency_ms=10.0))
        )
        assert hit_rate(fast_outcomes) == hit_rate(slow_outcomes)
        assert error_rate(fast_outcomes) == error_rate(slow_outcomes)


def test_match_trace_survives_save_and_load(tmp_path):
    records = make_mixed_records(n_classes=15, repeats=6)
    trace = build_match_trace(records, HashEmbedder())

    path = tmp_path / "trace.json"
    save_match_trace(trace, path)
    loaded = load_match_trace(path)

    assert loaded == trace


def test_scored_survives_save_and_load(tmp_path):
    records = make_mixed_records(n_classes=15, repeats=6)
    trace = build_match_trace(records, HashEmbedder())
    scored = score_gray_zone(records, trace, OracleVerifier(latency_ms=7.0), gray_zone_lo=0.4, gray_zone_hi=0.9)

    path = tmp_path / "scored.json"
    save_scored(scored, path)
    loaded = load_scored(path)

    assert loaded == scored


def test_verifier_fidelity_is_scoped_to_verifier_invoked_only():
    records = make_mixed_records(n_classes=15, repeats=6)
    embedder = HashEmbedder()
    trace = build_match_trace(records, embedder)
    verifier = OracleVerifier(latency_ms=5.0)
    scored = score_gray_zone(records, trace, verifier, gray_zone_lo=0.4, gray_zone_hi=0.9)
    outcomes = replay(trace, scored, tau_low=0.4, tau_high=0.9, threshold=0.5)

    fidelity = verifier_fidelity(outcomes)
    verified_count = sum(o.verifier_invoked for o in outcomes)
    assert fidelity.n_verified == verified_count
    # Oracle is perfect: it never approves a wrong answer or rejects a right one.
    assert fidelity.false_approve_rate == 0.0
    assert fidelity.false_reject_rate == 0.0
