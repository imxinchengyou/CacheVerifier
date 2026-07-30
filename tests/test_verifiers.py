import numpy as np
import pytest

from cacheverifier.cache.store import CacheEntry, NeighborMatch
from cacheverifier.cache.synchronous_verified import SynchronousVerifiedPolicy
from cacheverifier.data.schema import QueryRecord
from cacheverifier.verifiers.base import Verifier
from cacheverifier.verifiers.oracle_verifier import OracleVerifier

QUERY_MATCHING = QueryRecord(query_id="q", query="hello", answer="a", equivalence_id="c1")
QUERY_MISMATCHED = QueryRecord(query_id="q2", query="hello", answer="a", equivalence_id="c2")
ENTRY = CacheEntry(query_id="h", query="hi", answer="a", equivalence_id="c1")


def test_oracle_approves_matching_equivalence_id():
    verifier = OracleVerifier(latency_ms=42.0)
    result = verifier.verify(QUERY_MATCHING, ENTRY)
    assert result.approved is True
    assert result.latency_ms == 42.0


def test_oracle_rejects_mismatched_equivalence_id():
    verifier = OracleVerifier()
    result = verifier.verify(QUERY_MISMATCHED, ENTRY)
    assert result.approved is False


class StubVerifier(Verifier):
    name = "stub"

    def __init__(self, approved: bool, latency_ms: float = 5.0):
        self.threshold = 0.5
        self._score = 1.0 if approved else 0.0
        self._latency_ms = latency_ms
        self.calls = 0

    def score(self, query, candidate):
        self.calls += 1
        return self._score, self._latency_ms


def test_rejects_invalid_tau_ordering():
    with pytest.raises(ValueError):
        SynchronousVerifiedPolicy(tau_low=0.9, tau_high=0.8, verifier=StubVerifier(True))


def test_decide_on_empty_cache_is_a_miss():
    policy = SynchronousVerifiedPolicy(tau_low=0.8, tau_high=0.95, verifier=StubVerifier(True))
    decision = policy.decide(QUERY_MATCHING, np.zeros(1), match=None)
    assert decision.action == "miss"
    assert decision.branch == "empty_cache"
    assert decision.verifier_invoked is False


def test_hit_above_tau_high_skips_verifier():
    verifier = StubVerifier(approved=False)  # would reject if called
    policy = SynchronousVerifiedPolicy(tau_low=0.8, tau_high=0.9, verifier=verifier)
    match = NeighborMatch(entry=ENTRY, similarity=0.95)

    decision = policy.decide(QUERY_MATCHING, np.zeros(1), match)
    assert decision.action == "hit"
    assert decision.branch == "above_tau_high"
    assert decision.verifier_invoked is False
    assert verifier.calls == 0


def test_miss_below_tau_low_skips_verifier():
    verifier = StubVerifier(approved=True)  # would approve if called
    policy = SynchronousVerifiedPolicy(tau_low=0.8, tau_high=0.9, verifier=verifier)
    match = NeighborMatch(entry=ENTRY, similarity=0.5)

    decision = policy.decide(QUERY_MATCHING, np.zeros(1), match)
    assert decision.action == "miss"
    assert decision.branch == "below_tau_low"
    assert decision.verifier_invoked is False
    assert verifier.calls == 0


def test_gray_zone_calls_verifier_and_follows_its_decision():
    approving_policy = SynchronousVerifiedPolicy(tau_low=0.8, tau_high=0.9, verifier=StubVerifier(True, 12.0))
    match = NeighborMatch(entry=ENTRY, similarity=0.85)

    decision = approving_policy.decide(QUERY_MATCHING, np.zeros(1), match)
    assert decision.action == "hit"
    assert decision.branch == "gray_zone_verifier_approved"
    assert decision.verifier_invoked is True
    assert decision.verifier_latency_ms == 12.0

    rejecting_policy = SynchronousVerifiedPolicy(tau_low=0.8, tau_high=0.9, verifier=StubVerifier(False))
    decision = rejecting_policy.decide(QUERY_MATCHING, np.zeros(1), match)
    assert decision.action == "miss"
    assert decision.branch == "gray_zone_verifier_rejected"


def test_policy_name_includes_verifier_name():
    policy = SynchronousVerifiedPolicy(tau_low=0.8, tau_high=0.9, verifier=OracleVerifier())
    assert policy.name == "synchronous_verified[oracle]"
