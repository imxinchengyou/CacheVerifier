"""Convert the Quora Question Pairs dataset (GLUE's QQP mirror) into
CacheVerifier's JSONL schema, as a third, independently-sourced real-world
dataset to check whether the Group D/E findings (Section 5.4/5.6 of
PAPER.md) generalize beyond vCache's own two benchmarks.

Unlike SemCacheLMArena/SemCacheSearchQueries, QQP has no LLM-generated
`answer` column and no dataset-native equivalence-class column — it's pairs
of (question1, question2, is_duplicate). This script reconstructs the
schema CacheVerifier expects:

  - Equivalence classes are built via union-find over every `label == 1`
    (duplicate) pair, exactly the same construction vCache's own harness
    used for SearchQueries' `id_set` (see scripts/convert_vcache_hf_dataset.py
    docstring). Two questions are "the same query" iff they end up in the
    same connected component.
  - There is no ground-truth `answer` text, so this script uses `answer =
    query` for every record: when a later request matches an earlier one,
    `resolve_candidate` returns that earlier record's own question text.
    This makes the verifier's job literally "is this candidate question a
    real duplicate of the new one" — the actual QQP task — rather than
    "would a cached LLM answer still be correct," which is a defensible
    substitution given no generated answers exist, and if anything a purer
    test of a semantic-similarity verifier than the original benchmarks.
  - The stream order is first-appearance order of each unique question
    text, scanning the pairs file top to bottom (question1 before
    question2 within a row) — the closest available analog to "requests
    arriving over time" for a dataset that wasn't collected as a stream.

No precomputed embeddings exist for this dataset, so configs/quora.yaml
must use `embedder: sentence-transformer` (re-encodes `query` text locally)
rather than `embedder: precomputed`.

Usage:
    python scripts/convert_quora_dataset.py \\
        --out data/processed/quora.jsonl --max-samples 60000
"""

import argparse
import json
from pathlib import Path


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-samples", type=int, default=60000,
                        help="Cap on the number of UNIQUE questions emitted (stream stops once reached)")
    parser.add_argument("--out", default="data/processed/quora.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading nyu-mll/glue (qqp, train split) from Hugging Face...")
    ds = load_dataset("nyu-mll/glue", "qqp", split="train")
    print(f"  {len(ds)} pairs loaded")

    uf = UnionFind()
    first_seen_order: list[str] = []
    seen: set[str] = set()

    print("Building union-find over duplicate pairs and first-appearance order...")
    for row in ds:
        q1, q2, label = row["question1"], row["question2"], row["label"]
        if not q1 or not q2:
            continue
        for q in (q1, q2):
            if q not in seen:
                seen.add(q)
                first_seen_order.append(q)
        if label == 1:
            uf.union(q1, q2)
        if len(first_seen_order) >= args.max_samples:
            break

    print(f"  {len(first_seen_order)} unique questions in stream order (capped at {args.max_samples})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes_multi = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, question in enumerate(first_seen_order):
            equivalence_id = uf.find(question) if question in uf.parent else question
            record = {
                "query_id": str(i),
                "query": question,
                "answer": question,
                "equivalence_id": equivalence_id,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    class_sizes: dict[str, int] = {}
    for question in first_seen_order:
        eq = uf.find(question) if question in uf.parent else question
        class_sizes[eq] = class_sizes.get(eq, 0) + 1
    n_classes_multi = sum(1 for size in class_sizes.values() if size > 1)

    print(f"Wrote {len(first_seen_order)} records to {out_path}")
    print(f"  {len(class_sizes)} equivalence classes total, {n_classes_multi} with 2+ members "
          f"({sum(v for v in class_sizes.values() if v > 1)} records in a non-singleton class)")


if __name__ == "__main__":
    main()
