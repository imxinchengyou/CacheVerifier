"""One-off experiment: does fine-tuning a small cross-encoder on domain-
specific gray-zone labels close more of the oracle-to-real verifier gap than
an off-the-shelf pretrained cross-encoder (Group D in PAPER.md)?

Uses LmArena's gray zone (tau_low=0.80, tau_high=0.97) — 16,102 labeled
(query, candidate_answer, would_be_correct) examples, already characterized
in results/lmarena_groupC.json / lmarena_groupD_cross_encoder.json. Splits
by STREAM POSITION (not randomly) into an early "calibration" segment and a
later held-out segment, mirroring a real "calibrate on past traffic, deploy
on future traffic" workflow rather than i.i.d. cross-validation.

Usage:
    python scripts/finetune_verifier_experiment.py --train-size 2000 --epochs 1
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


def collect_gray_zone_examples(records, trace):
    """(position, query_text, answer_text, label) for every record whose
    similarity falls in [TAU_LOW, TAU_HIGH), in stream order."""
    examples = []
    for i, (record, t) in enumerate(zip(records, trace)):
        if t.similarity is not None and TAU_LOW <= t.similarity < TAU_HIGH:
            candidate = resolve_candidate(records, t)
            examples.append((i, record.query, candidate.answer, t.would_be_correct))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/lmarena.yaml")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="Fraction of gray-zone examples (by stream position) used for fine-tuning")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Cap the training set to this many examples (from the front of the train split)")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Cap the held-out test set to this many examples (from the front of the test split)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output", default="results/finetune_verifier_experiment.json")
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

    examples = collect_gray_zone_examples(records, trace)
    print(f"Gray zone [{TAU_LOW}, {TAU_HIGH}): {len(examples)} labeled examples")

    split_idx = int(len(examples) * args.train_fraction)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    if args.train_size is not None:
        train_examples = train_examples[: args.train_size]
    if args.test_size is not None:
        test_examples = test_examples[: args.test_size]

    n_pos_train = sum(1 for _, _, _, label in train_examples if label)
    n_pos_test = sum(1 for _, _, _, label in test_examples if label)
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

    # Stash the raw examples for the training/eval step to consume without
    # having to rebuild the match trace again.
    stash_path = out_path.with_suffix(".examples.json")
    stash_path.write_text(
        json.dumps({"train": train_examples, "test": test_examples}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote example stash to {stash_path}")


if __name__ == "__main__":
    main()
