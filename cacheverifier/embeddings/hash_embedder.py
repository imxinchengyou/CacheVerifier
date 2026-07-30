import hashlib

import numpy as np

from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.base import Embedder


class HashEmbedder(Embedder):
    """Deterministic, dependency-free embedder for pipeline tests and CI.

    NOT semantically meaningful — it hashes word n-grams into a fixed-size
    bag-of-hashes vector, so paraphrase pairs will generally NOT score highly
    similar. Use `SentenceTransformerEmbedder` for any experiment whose
    numbers are meant to be compared against vCache/Krites baselines; this
    class only exists so the rest of the pipeline (cache store, policies,
    metrics) can be exercised without downloading a model.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.dim = dim
        self.ngram_range = ngram_range

    def _ngrams(self, text: str) -> list[str]:
        tokens = text.lower().split()
        grams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                grams.append(" ".join(tokens[i : i + n]))
        return grams or tokens

    def embed(self, records: list[QueryRecord]) -> np.ndarray:
        vectors = np.zeros((len(records), self.dim), dtype=np.float32)
        for row, record in enumerate(records):
            for gram in self._ngrams(record.query):
                digest = hashlib.sha1(gram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, bucket] += sign
        return self._normalize(vectors)
