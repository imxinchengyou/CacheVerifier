"""Convert an already-downloaded vCache benchmark parquet file (see
convert_vcache_hf_dataset.py's docstring for the column mapping/rationale)
into CacheVerifier's JSONL schema, reading LOCALLY via pyarrow instead of
HuggingFace `datasets`' streaming reader.

Why this exists: `datasets`' streaming path for a single-file (unsharded)
parquet dataset buffers big HTTP byte ranges (~1GB chunks) in memory before
yielding rows, and on a flaky mirror connection that drops mid-transfer, the
retry logic didn't seem to free the previous attempt's buffer -- memory grew
unbounded across retries until it neared OOM on a 14GB box. Downloading the
raw file once with `curl -C -` (resumable, writes straight to disk, near-zero
memory) then reading it locally with pyarrow's memory-mapped reader sidesteps
both problems: disk, not RAM, holds the data, and a dropped connection only
costs a resumed download, not a restarted one.

Usage:
    curl -C - --retry 20 --retry-delay 3 -o /tmp/lmarena_train.parquet \\
        "https://hf-mirror.com/datasets/vCache/SemBenchmarkLmArena/resolve/<sha>/train.parquet"
    python scripts/convert_local_parquet_to_jsonl.py --dataset lmarena \\
        --parquet /tmp/lmarena_train.parquet --out data/processed/lmarena.jsonl --max-samples 15000
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Any

# Kept in sync with convert_vcache_hf_dataset.py's PRESETS/parse_embedding --
# duplicated rather than imported since scripts/ isn't a package (no
# __init__.py), and this is a one-off recovery script, not a permanent
# addition to the module graph.
PRESETS: dict[str, dict[str, str]] = {
    "lmarena": {
        "hf_id": "vCache/SemBenchmarkLmArena",
        "equivalence_column": "ID_Set",
        "embedding_column": "emb_e5_large_v2",
        "response_column": "response_gpt-4o-mini",
    },
    "search_queries": {
        "hf_id": "vCache/SemBenchmarkSearchQueries",
        "equivalence_column": "id_set",
        "embedding_column": "emb_gte",
        "response_column": "response_llama_3_8b",
    },
}


def parse_embedding(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(raw)
    return [float(x) for x in parsed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--parquet", required=True, help="Path to the already-downloaded train.parquet")
    parser.add_argument("--embedding-column", default=None)
    parser.add_argument("--response-column", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    preset = PRESETS[args.dataset]
    embedding_column = args.embedding_column or preset["embedding_column"]
    response_column = args.response_column or preset["response_column"]
    equivalence_column = preset["equivalence_column"]
    needed_columns = ["id", "prompt", equivalence_column, response_column, embedding_column]

    print(f"Opening {args.parquet} (memory-mapped, not loaded fully into RAM)...", flush=True)
    pf = pq.ParquetFile(args.parquet, memory_map=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch in pf.iter_batches(columns=needed_columns, batch_size=1000):
            table = batch.to_pylist()
            for row in table:
                record = {
                    "query_id": str(row["id"]),
                    "query": row["prompt"],
                    "answer": row[response_column],
                    "equivalence_id": str(row[equivalence_column]),
                    "embedding": parse_embedding(row[embedding_column]),
                }
                f.write(json.dumps(record) + "\n")
                n_written += 1
                if args.max_samples and n_written >= args.max_samples:
                    break
            if args.max_samples and n_written >= args.max_samples:
                break
            if n_written % 5000 == 0:
                print(f"  wrote {n_written} rows...", flush=True)

    print(f"Wrote {n_written} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
