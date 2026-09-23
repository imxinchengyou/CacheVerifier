from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRecord:
    """One request in a semantic-cache trace.

    `equivalence_id` is the ground-truth label used to decide whether a cached
    answer retrieved for this query is actually correct: two records are
    interchangeable iff they share the same `equivalence_id`. This mirrors the
    labeling scheme used by the vCache / Krites benchmarks (SemCacheLMArena,
    SemCacheSearchQueries) so a served cache hit's correctness can be checked
    without calling a real LLM judge.
    """

    query_id: str
    query: str
    answer: str
    equivalence_id: str
    embedding: tuple[float, ...] | None = None
    """Pre-computed embedding vector, if the trace carries one (see
    `scripts/convert_vcache_hf_dataset.py`). Required by
    `cacheverifier.embeddings.precomputed_embedder.PrecomputedEmbedder`; ignored by
    embedders that encode `query` themselves."""
