import numpy as np
import pytest

from cacheverifier.cache.store import CacheEntry, VectorCacheStore


def unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_query_on_empty_store_returns_none():
    store = VectorCacheStore(dim=2)
    assert store.query(unit([1.0, 0.0])) is None


def test_insert_and_query_returns_nearest_neighbor():
    store = VectorCacheStore(dim=2, initial_capacity=1)
    store.insert(CacheEntry("a", "qa", "answer-a", "class-a"), unit([1.0, 0.0]))
    store.insert(CacheEntry("b", "qb", "answer-b", "class-b"), unit([0.0, 1.0]))

    query_vec = unit([0.9, 0.1])
    match = store.query(query_vec)
    assert match is not None
    assert match.entry.query_id == "a"
    assert match.similarity == pytest.approx(float(query_vec @ unit([1.0, 0.0])))


def test_grows_beyond_initial_capacity():
    store = VectorCacheStore(dim=1, initial_capacity=1)
    for i in range(5):
        store.insert(CacheEntry(str(i), f"q{i}", f"a{i}", str(i)), unit([1.0]))
    assert len(store) == 5
