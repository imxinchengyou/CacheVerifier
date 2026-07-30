from cacheverifier.cache.adaptive_threshold import AdaptiveThresholdPolicy
from cacheverifier.cache.static_threshold import StaticThresholdPolicy
from cacheverifier.data.schema import QueryRecord
from cacheverifier.embeddings.hash_embedder import HashEmbedder
from cacheverifier.experiments.runner import ExperimentRunner
from cacheverifier.metrics.core import confusion_counts, error_rate, hit_rate


def make_duplicated_classes(n_classes: int, repeats: int) -> list[QueryRecord]:
    """Records where each class is the exact same string repeated `repeats`
    times, so even a non-semantic embedder gives similarity == 1.0 within a
    class and (with overwhelming likelihood) < 1.0 across classes."""
    records = []
    for c in range(n_classes):
        for r in range(repeats):
            records.append(
                QueryRecord(
                    query_id=f"c{c}-{r}",
                    query=f"exact duplicate query number {c}",
                    answer=f"answer-{c}",
                    equivalence_id=f"class-{c}",
                )
            )
    return records


def test_end_to_end_static_threshold_smoke():
    records = make_duplicated_classes(n_classes=10, repeats=8)
    runner = ExperimentRunner(HashEmbedder())

    strict_outcomes = runner.run(records, StaticThresholdPolicy(threshold=0.999999))
    loose_outcomes = runner.run(records, StaticThresholdPolicy(threshold=0.5))

    assert len(strict_outcomes) == len(records) == len(loose_outcomes)
    # Exact duplicates exist across the stream, so a near-1.0 threshold must
    # still register some hits, and a looser threshold can't hit less often.
    assert hit_rate(strict_outcomes) > 0.0
    assert hit_rate(loose_outcomes) >= hit_rate(strict_outcomes)
    # Only exact string duplicates clear the 0.999999 bar, and duplicates by
    # construction share an equivalence_id, so those hits must be correct.
    assert error_rate(strict_outcomes) == 0.0


def test_end_to_end_adaptive_threshold_smoke():
    records = make_duplicated_classes(n_classes=10, repeats=8)
    runner = ExperimentRunner(HashEmbedder())

    policy = AdaptiveThresholdPolicy(target_error_rate=0.1, min_observations=2)
    outcomes = runner.run(records, policy)

    assert len(outcomes) == len(records)
    assert 0.0 <= hit_rate(outcomes) <= 1.0
    assert 0.0 <= error_rate(outcomes) <= 1.0


def test_confusion_counts_match_hits_and_misses():
    records = make_duplicated_classes(n_classes=5, repeats=6)
    runner = ExperimentRunner(HashEmbedder())
    outcomes = runner.run(records, StaticThresholdPolicy(threshold=0.999999))

    counts = confusion_counts(outcomes)
    assert counts.tp + counts.fp + counts.tn + counts.fn + counts.no_neighbor == len(outcomes)
    # The very first request always hits an empty cache.
    assert counts.no_neighbor >= 1
