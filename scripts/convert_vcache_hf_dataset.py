"""Convert a vCache HuggingFace benchmark dataset (SemBenchmarkLmArena /
SemBenchmarkSearchQueries — what the research proposal calls SemCacheLMArena
/ SemCacheSearchQueries) into CacheVerifier's JSONL schema, carrying along the
dataset's own pre-computed embedding vectors.

Why pre-computed embeddings, not re-encoded text: vCache's benchmark harness
(`vcache-project/vCache`, `benchmarks/benchmark.py`) reads embeddings
straight out of dataset columns (`emb_e5_large_v2`, `emb_gte`, ...) produced
by whatever exact model checkpoint/pooling the authors used when building
the benchmark. Re-encoding the same model name ourselves via
sentence-transformers is not guaranteed to reproduce those exact vectors, so
reproducing their reported numbers requires using their vectors directly —
see `cacheverifier.embeddings.precomputed_embedder.PrecomputedEmbedder`.

Ground-truth column mapping is taken directly from vCache's own harness
(`benchmarks/benchmark.py::run_benchmark_loop`, which reads
`data_entry.get("id_set", -1)` and falls back to `data_entry.get("ID_Set",
-1)`):
  - LmArena:        equivalence class = `ID_Set`  (capitalized)
  - SearchQueries:  equivalence class = `id_set`  (lowercase; `cluster_id`
                    also exists in the raw dataset but is NOT what the
                    official harness uses as ground truth — don't use it)
  - answer / response column: whichever `response_<model>` matches the
    RUN_COMBINATIONS entry used for the paper's figures — GPT-4o-mini for
    LmArena, Llama-3-8B for SearchQueries.

Requires the `datasets` package: pip install -r requirements-embeddings.txt

Usage:
    python scripts/convert_vcache_hf_dataset.py --dataset lmarena \\
        --out data/processed/lmarena.jsonl --max-samples 60000

    python scripts/convert_vcache_hf_dataset.py --dataset search_queries \\
        --out data/processed/search_queries.jsonl --max-samples 150000
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Any

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
    """Dataset embedding columns are stored as strings, not native lists.
    Mirrors the parsing vCache's own `benchmark.py::get_vcache_answer` does
    (try JSON first, fall back to `ast.literal_eval` for non-strict-JSON
    reprs like ones containing `nan`/`inf`)."""
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
    parser.add_argument("--embedding-column", default=None, help="Override the preset's embedding column")
    parser.add_argument("--response-column", default=None, help="Override the preset's response/answer column")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream rows instead of downloading the full split first — the "
        "datasets are several GB, so use this for a quick --max-samples smoke "
        "test. Full real runs (matching vCache's own `train[:N]` slicing) "
        "should omit this.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the 'datasets' package: pip install -r requirements-embeddings.txt"
        ) from exc

    preset = PRESETS[args.dataset]
    embedding_column = args.embedding_column or preset["embedding_column"]
    response_column = args.response_column or preset["response_column"]
    equivalence_column = preset["equivalence_column"]

    needed_columns = ["id", "prompt", equivalence_column, response_column, embedding_column]

    if args.streaming:
        from itertools import islice

        print(f"Streaming {preset['hf_id']} from Hugging Face (max_samples={args.max_samples})...")
        hf_dataset = load_dataset(preset["hf_id"], split="train", streaming=True)
        hf_dataset = hf_dataset.select_columns(needed_columns)
        if args.max_samples:
            hf_dataset = islice(hf_dataset, args.max_samples)
    else:
        split = f"train[:{args.max_samples}]" if args.max_samples else "train"
        print(f"Loading {preset['hf_id']} ({split}) from Hugging Face...")
        hf_dataset = load_dataset(preset["hf_id"], split=split)
        hf_dataset = hf_dataset.select_columns(needed_columns)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in hf_dataset:
            record = {
                "query_id": str(row["id"]),
                "query": row["prompt"],
                "answer": row[response_column],
                "equivalence_id": str(row[equivalence_column]),
                "embedding": parse_embedding(row[embedding_column]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(
        f"Wrote {n_written} records to {out_path} "
        f"(embedding_column={embedding_column!r}, response_column={response_column!r}, "
        f"equivalence_column={equivalence_column!r})"
    )


if __name__ == "__main__":
    main()
