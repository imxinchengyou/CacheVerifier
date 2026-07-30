"""Generate a synthetic trace in the CacheVerifier schema for pipeline smoke-testing.

This does NOT stand in for SemCacheLMArena / SemCacheSearchQueries — it exists
so the pipeline (splits, cache store, policies, metrics) can be exercised and
unit-tested without network access. Week 1's "confirm we align with the
published baseline numbers" step still requires the real benchmark files
dropped into `data/raw/` and converted to this schema.

Each equivalence class contains a handful of paraphrases sharing one answer,
plus near-duplicate-wording-but-different-answer "confusable" pairs (e.g. the
"pause" vs "cancel" subscription example from the proposal) so that static
thresholds are forced to trade off hit rate against error rate.

Usage:
    python scripts/generate_synthetic_dataset.py --n-classes 500 --out data/processed/synthetic.jsonl
"""

import argparse
import json
import random
from pathlib import Path

TEMPLATES = [
    "How do I {action} my {noun}?",
    "What is the process to {action} a {noun}?",
    "Can you tell me how to {action} the {noun}?",
    "I want to {action} my {noun}, what should I do?",
    "Steps to {action} {noun}",
]

ACTIONS = ["cancel", "pause", "renew", "upgrade", "downgrade", "delete", "restore", "activate"]
NOUNS = ["subscription", "account", "order", "membership", "trial", "plan", "reservation"]


def build_equivalence_class(class_id: int, rng: random.Random) -> list[dict]:
    action = rng.choice(ACTIONS)
    noun = rng.choice(NOUNS)
    answer = f"To {action} your {noun}, go to Settings > {noun.title()} and select '{action.title()}'."

    n_paraphrases = rng.randint(2, len(TEMPLATES))
    templates = rng.sample(TEMPLATES, n_paraphrases)

    records = []
    for i, template in enumerate(templates):
        records.append(
            {
                "query_id": f"c{class_id}-{i}",
                "query": template.format(action=action, noun=noun),
                "answer": answer,
                "equivalence_id": f"class-{class_id}",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-classes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/processed/synthetic.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for class_id in range(args.n_classes):
            for record in build_equivalence_class(class_id, rng):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote synthetic trace to {out_path}")


if __name__ == "__main__":
    main()
