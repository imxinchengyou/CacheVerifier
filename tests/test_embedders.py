import numpy as np
import pytest

from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.hash_embedder import HashEmbedder
from cacheverifier.embeddings.precomputed_embedder import PrecomputedEmbedder

RECORD_WITH_EMBEDDING = QueryRecord(
    query_id="1", query="hi", answer="a", equivalence_id="c1", embedding=(3.0, 4.0)
)
RECORD_WITHOUT_EMBEDDING = QueryRecord(query_id="2", query="hey", answer="a", equivalence_id="c1")


def test_hash_embedder_returns_unit_norm_vectors():
    vectors = HashEmbedder().embed([RECORD_WITH_EMBEDDING, RECORD_WITHOUT_EMBEDDING])
    assert vectors.shape[0] == 2
    norms = np.linalg.norm(vectors, axis=1)
    assert norms == pytest.approx([1.0, 1.0])


def test_precomputed_embedder_normalizes_stored_vector():
    vectors = PrecomputedEmbedder().embed([RECORD_WITH_EMBEDDING])
    assert vectors.shape == (1, 2)
    assert vectors[0] == pytest.approx([0.6, 0.8])  # (3,4) normalized


def test_precomputed_embedder_raises_on_missing_embedding():
    with pytest.raises(ValueError, match="no precomputed embedding"):
        PrecomputedEmbedder().embed([RECORD_WITHOUT_EMBEDDING])
