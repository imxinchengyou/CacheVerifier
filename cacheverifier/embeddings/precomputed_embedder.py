import numpy as np

from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.base import Embedder


class PrecomputedEmbedder(Embedder):
    """Uses each record's own `embedding` field instead of encoding text.

    This is what reproducing vCache's reported numbers actually requires:
    their benchmark harness reads embeddings straight out of dataset columns
    (`emb_e5_large_v2`, `emb_gte`, ...) produced by whatever exact model
    checkpoint and pooling they used — re-encoding the same model name
    ourselves via `SentenceTransformerEmbedder` is not guaranteed to
    reproduce those exact vectors. See `scripts/convert_vcache_hf_dataset.py`
    for how the `embedding` field gets populated.
    """

    def embed(self, records: list[QueryRecord]) -> np.ndarray:
        missing = [r.query_id for r in records if r.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} record(s) have no precomputed embedding "
                f"(e.g. query_id={missing[0]!r}); PrecomputedEmbedder requires "
                f"every record's `embedding` field to be populated — see "
                f"scripts/convert_vcache_hf_dataset.py."
            )
        vectors = np.array([r.embedding for r in records], dtype=np.float32)
        return self._normalize(vectors)
