import numpy as np

from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Re-encodes `query` text live. Useful for datasets with no pre-computed
    embeddings (e.g. `scripts/generate_synthetic_dataset.py` output), but NOT
    guaranteed to reproduce vCache's reported numbers on the real benchmarks
    even with a matching model name — use `PrecomputedEmbedder` for those
    (see `scripts/convert_vcache_hf_dataset.py`). Lazily imports
    sentence-transformers so the core pipeline has no heavy/optional
    dependency at import time.

    Install with: pip install -r requirements-embeddings.txt
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'sentence-transformers' "
                "package. Install it with: pip install -r requirements-embeddings.txt"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, records: list[QueryRecord]) -> np.ndarray:
        vectors = self._model.encode(
            [r.query for r in records], convert_to_numpy=True, show_progress_bar=False
        )
        return self._normalize(vectors.astype(np.float32))
