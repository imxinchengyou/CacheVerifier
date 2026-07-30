"""Prototype for Future Work item #1 (Section 5.8 / PAPER.md 8): an online
monitor for gray-zone positive-rate stability, tested against comcastcares'
own real drift to see whether it would have flagged the instability before
it degraded Group E's fine-tuning benefit (AUC delta went from +0.86 clean
to -0.186 after fine-tuning, traced to a 1.0% train vs. 5.2% test positive-
rate mismatch).

This is pure statistics over labels the online system already produces for
free (no embedding inference, no model training) — the same (position,
query, answer, label) stash finetune_verifier_experiment.py already writes.

Two methods, both operating on the binary "would_be_correct" label stream
in true chronological (stream-position) order:

  1. Chunked two-proportion z-test: split the stream into the same 8 equal
     chunks drift_experiment.py uses, and test each chunk's positive rate
     against the training-window baseline rate (the first n_train examples,
     matching Group E's actual fine-tuning split) via a two-proportion
     z-test. Directly comparable to the drift experiment's own chunking.
  2. Page-Hinkley test: a classical sequential change-point detector (used
     in streaming/concept-drift monitoring) that processes the held-out
     stream one example at a time and flags the first position where
     cumulative deviation from the training-window baseline rate exceeds a
     threshold — closer to what an actual online monitor would run.

Usage:
    python scripts/positive_rate_monitor.py \
        --stash results/finetune_verifier_experiment_twitter_comcast.examples.json \
        --label comcast
"""

import argparse
import json
import math
from pathlib import Path


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Returns (z, two-sided p-value) for H0: p1 == p2."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    # two-sided p-value via erfc (no scipy dependency)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def page_hinkley(labels: list[int], baseline_rate: float, delta: float = 0.005, lam: float = 8.0):
    """Classical Page-Hinkley sequential change-point test for a rate
    increase relative to `baseline_rate`. Returns the first index (into
    `labels`) where the test statistic crosses `lam`, or None if it never
    does. `delta` is the allowed slack (magnitude of change to ignore)."""
    cumulative = 0.0
    min_cumulative = 0.0
    for i, x in enumerate(labels):
        cumulative += (x - baseline_rate - delta)
        min_cumulative = min(min_cumulative, cumulative)
        ph_stat = cumulative - min_cumulative
        if ph_stat > lam:
            return i, ph_stat
    return None, cumulative - min_cumulative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stash", required=True)
    parser.add_argument("--label", required=True, help="Dataset label for output, e.g. comcast / amazon")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance threshold for the z-test")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stash = json.loads(Path(args.stash).read_text(encoding="utf-8"))
    train_examples = stash["train"]
    test_examples = stash["test"]
    n_train = len(train_examples)

    train_labels = [1 if ex[3] else 0 for ex in train_examples]
    p_train = sum(train_labels) / n_train
    print(f"[{args.label}] Training-window baseline: n={n_train}, positive_rate={p_train:.4f}")

    # Held-out stream, in true chronological order (already stream-position-sorted by construction).
    test_labels = [1 if ex[3] else 0 for ex in test_examples]
    n_test = len(test_labels)
    p_test_overall = sum(test_labels) / n_test
    print(f"[{args.label}] Full held-out window: n={n_test}, positive_rate={p_test_overall:.4f}")

    # Method 1: chunked two-proportion z-test against the training baseline.
    chunk_size = n_test // args.n_chunks
    chunks = [test_labels[i * chunk_size:(i + 1) * chunk_size] for i in range(args.n_chunks)]
    if n_test % args.n_chunks:
        chunks[-1] = test_labels[(args.n_chunks - 1) * chunk_size:]

    print(f"\n[{args.label}] Method 1 -- chunked two-proportion z-test vs. training baseline (alpha={args.alpha}):")
    first_flagged = None
    chunk_results = []
    for i, chunk in enumerate(chunks):
        p_chunk = sum(chunk) / len(chunk)
        z, p_value = two_proportion_z_test(p_train, n_train, p_chunk, len(chunk))
        flagged = p_value < args.alpha
        if flagged and first_flagged is None:
            first_flagged = i
        chunk_results.append({"chunk": i, "n": len(chunk), "positive_rate": p_chunk, "z": z, "p_value": p_value, "flagged": flagged})
        print(f"  chunk={i}  n={len(chunk)}  rate={p_chunk:.4f}  z={z:+.2f}  p={p_value:.4g}  {'*** FLAGGED ***' if flagged else ''}")
    print(f"  -> first chunk flagged as significantly different from training baseline: {first_flagged}")

    # Method 2: Page-Hinkley sequential test over the held-out stream, one example at a time.
    ph_index, ph_stat = page_hinkley(test_labels, p_train)
    ph_chunk = ph_index // chunk_size if ph_index is not None else None
    print(f"\n[{args.label}] Method 2 -- Page-Hinkley sequential test (delta=0.005, lambda=8.0):")
    if ph_index is not None:
        print(f"  -> flagged at held-out example #{ph_index} (falls in chunk {ph_chunk}), PH statistic={ph_stat:.3f}")
    else:
        print(f"  -> never flagged over the full held-out window (final PH statistic={ph_stat:.3f})")

    result = {
        "dataset": args.label,
        "n_train": n_train,
        "train_positive_rate": p_train,
        "n_test": n_test,
        "test_positive_rate_overall": p_test_overall,
        "chunk_size": chunk_size,
        "method1_chunked_z_test": chunk_results,
        "method1_first_flagged_chunk": first_flagged,
        "method2_page_hinkley_flag_index": ph_index,
        "method2_page_hinkley_flag_chunk": ph_chunk,
        "method2_page_hinkley_final_stat": ph_stat,
    }
    out_path = Path(args.output) if args.output else Path(f"results/positive_rate_monitor_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
