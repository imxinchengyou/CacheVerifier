import time

from cacheverifier.cache.store import CacheEntry
from cacheverifier.data.schema import QueryRecord
from cacheverifier.verifiers.base import Verifier


class CrossEncoderVerifier(Verifier):
    """Group D's first real (non-oracle) verifier: a discriminative
    cross-encoder scoring (query, candidate_answer) relevance — the
    "~10ms" row in the proposal's Table 4.3 (measured latency on CPU is
    actually more like 55-100ms; see project notes).

    Default model is `cross-encoder/ms-marco-MiniLM-L6-v2`, trained for
    query-passage relevance ranking, which is a natural fit for "is this
    cached answer relevant to this new query" even though it wasn't
    purpose-built for cache verification. Its output is an unbounded
    relevance score, not a probability — `threshold` is a real hyperparameter
    that must be calibrated, not assumed universal; the default of 0.0 is a
    starting point. `cacheverifier.experiments.verified_sweep` sweeps it
    cheaply (score once, re-threshold many times) rather than trusting this
    default.

    Latency is measured as actual wall-clock time per call (CPU inference),
    not modeled — this is what the false-approve/false-reject-rate and
    latency numbers in Section 4.3/5 are supposed to be measured from.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        threshold: float = 0.0,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderVerifier requires the 'sentence-transformers' "
                "package. Install it with: pip install -r requirements-embeddings.txt"
            ) from exc
        self.model_name = model_name
        self.threshold = threshold
        self._model = CrossEncoder(model_name)

    def score(self, query: QueryRecord, candidate: CacheEntry) -> tuple[float, float]:
        start = time.perf_counter()
        score = float(self._model.predict([(query.query, candidate.answer)])[0])
        latency_ms = (time.perf_counter() - start) * 1000.0
        return score, latency_ms
