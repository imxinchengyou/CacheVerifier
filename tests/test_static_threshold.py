import numpy as np

from cacheverifier.cache.static_threshold import StaticThresholdPolicy
from cacheverifier.cache.store import CacheEntry, NeighborMatch
from cacheverifier.data.schema import QueryRecord

QUERY = QueryRecord(query_id="q", query="hello", answer="a", equivalence_id="c1")
ENTRY = CacheEntry(query_id="h", query="hi", answer="a", equivalence_id="c1")


def test_miss_on_empty_cache():
    policy = StaticThresholdPolicy(threshold=0.9)
    decision = policy.decide(QUERY, np.zeros(2), match=None)
    assert decision.action == "miss"
    assert decision.branch == "empty_cache"


def test_hit_above_threshold():
    policy = StaticThresholdPolicy(threshold=0.9)
    match = NeighborMatch(entry=ENTRY, similarity=0.95)
    decision = policy.decide(QUERY, np.zeros(2), match=match)
    assert decision.action == "hit"


def test_miss_below_threshold():
    policy = StaticThresholdPolicy(threshold=0.9)
    match = NeighborMatch(entry=ENTRY, similarity=0.85)
    decision = policy.decide(QUERY, np.zeros(2), match=match)
    assert decision.action == "miss"
