"""Patch data/processed/search_queries.jsonl's `answer` field with the real
DeepSeek-generated answers (scripts/generate_searchqueries_answers_deepseek.py)
for the 103,188 records that actually get read by the verifier as either the
query- or candidate-side of a gray-zone match. Everything else (embedding,
equivalence_id, the remaining ~31% of records never touched by the
verifier) is carried over byte-for-byte unchanged.

Writes to a NEW file rather than overwriting the original, so the original
placeholder-only file stays available for provenance/audit -- anyone
checking this correction can diff the two.

Usage:
    python scripts/merge_searchqueries_deepseek_answers.py
"""

import json
from pathlib import Path

SRC = Path("data/processed/search_queries.jsonl")
ANSWERS = Path("data/processed/search_queries_answers_deepseek.jsonl")
OUT = Path("data/processed/search_queries_corrected.jsonl")

PLACEHOLDER = "Not required for the benchmark because of the id_set"


def main() -> None:
    answers = {}
    with ANSWERS.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[r["query_id"]] = r["answer"]
    print(f"Loaded {len(answers)} generated answers")

    n_total = 0
    n_patched = 0
    n_still_placeholder = 0
    with SRC.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            n_total += 1
            if record["query_id"] in answers:
                assert record["answer"] == PLACEHOLDER, (
                    f"query_id={record['query_id']} answer was not the expected placeholder "
                    f"(got {record['answer']!r}) -- refusing to overwrite unexpectedly"
                )
                record["answer"] = answers[record["query_id"]]
                n_patched += 1
            else:
                n_still_placeholder += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Total records: {n_total}")
    print(f"Patched with real DeepSeek answer: {n_patched}")
    print(f"Left as placeholder (never read by the verifier): {n_still_placeholder}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
