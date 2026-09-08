"""Stage 1 of the q_cached ablation: same gray-zone extraction as
finetune_verifier_experiment.py, but also stashes each example's CACHED
query text (the historical query whose answer is the reuse candidate) --
not just (query, answer, label).

Motivation: cacheverifier.verifiers.cross_encoder_verifier.CrossEncoderVerifier
only ever sees (current_query, candidate_answer). But CacheEntry (see
cacheverifier/cache/store.py) already carries the cached query's own text --
it's just never passed to the verifier. This script tests whether giving the
verifier that second query (via simple text concatenation into a single
text-pair segment -- see finetune_verifier_qcached_train_eval.py for why not
a true 3-segment cross-encoder input) helps it catch "current query and
cached answer look similar, but the cached answer doesn't actually transfer"
cases, as distinct from "current query and candidate answer are just
unrelated" -- the case an off-the-shelf/naturally-fine-tuned relevance
cross-encoder already loses to at 84-88% false-accept rate on the five
adversarial axes in results/llm_redteam_results.json (Section 5.18/5.19).

Same dataset, same split, same tau_low/tau_high as finetune_verifier_
experiment.py, so results are directly comparable to Section 5.6's published
LmArena numbers -- this script only ADDS a field, it does not change the
selection or split logic.

Usage:
    python scripts/finetune_verifier_qcached_experiment.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.verified_sweep import build_match_trace, resolve_candidate

TAU_LOW = 0.80
TAU_HIGH = 0.97


def collect_gray_zone_examples_with_cached_query(records, trace):
    """(position, query_text, cached_query_text, answer_text, label) for
    every record whose similarity falls in [TAU_LOW, TAU_HIGH), in stream
    order. Identical selection to finetune_verifier_experiment.py's
    collect_gray_zone_examples, with one extra field: candidate.query."""
    examples = []
    for i, (record, t) in enumerate(zip(records, trace)):
        if t.similarity is not None and TAU_LOW <= t.similarity < TAU_HIGH:
            candidate = resolve_candidate(records, t)
            examples.append((i, record.query, candidate.query, candidate.answer, t.would_be_correct))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/lmarena.yaml")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="Fraction of gray-zone examples (by stream position) used for fine-tuning")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--output", default="results/finetune_verifier_qcached_experiment.json")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    records = load_jsonl(Path(cfg.processed_path))
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    print(f"Loaded {len(records)} records from {cfg.processed_path}")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    print("Building match trace...")
    t0 = time.time()
    trace = build_match_trace(records, embedder)
    print(f"  done in {time.time() - t0:.1f}s")

    examples = collect_gray_zone_examples_with_cached_query(records, trace)
    print(f"Gray zone [{TAU_LOW}, {TAU_HIGH}): {len(examples)} labeled examples")

    split_idx = int(len(examples) * args.train_fraction)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    if args.train_size is not None:
        train_examples = train_examples[: args.train_size]
    if args.test_size is not None:
        test_examples = test_examples[: args.test_size]

    n_pos_train = sum(1 for _, _, _, _, label in train_examples if label)
    n_pos_test = sum(1 for _, _, _, _, label in test_examples if label)
    print(f"Train: {len(train_examples)} examples ({n_pos_train} correct, "
          f"{len(train_examples) - n_pos_train} incorrect)")
    print(f"Test:  {len(test_examples)} examples ({n_pos_test} correct, "
          f"{len(test_examples) - n_pos_test} incorrect)")

    result = {
        "config": vars(args),
        "n_gray_zone_total": len(examples),
        "n_train": len(train_examples),
        "n_test": len(test_examples),
        "train_positive_rate": n_pos_train / len(train_examples) if train_examples else None,
        "test_positive_rate": n_pos_test / len(test_examples) if test_examples else None,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    stash_path = out_path.with_suffix(".examples.json")
    stash_path.write_text(
        json.dumps({"train": train_examples, "test": test_examples}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote example stash to {stash_path}")


if __name__ == "__main__":
    main()
