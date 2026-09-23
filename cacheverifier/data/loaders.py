import json
from pathlib import Path
from typing import Any, Iterator

from cacheverifier.data.schema import QueryRecord

REQUIRED_FIELDS = ("query_id", "query", "answer", "equivalence_id")


def _build_record(obj: dict[str, Any], source: str, line_no: int) -> QueryRecord:
    missing = [k for k in REQUIRED_FIELDS if k not in obj]
    if missing:
        raise ValueError(f"{source}:{line_no}: record missing required fields {missing}")
    embedding = obj.get("embedding")
    return QueryRecord(
        query_id=str(obj["query_id"]),
        query=obj["query"],
        answer=obj["answer"],
        equivalence_id=str(obj["equivalence_id"]),
        embedding=tuple(float(x) for x in embedding) if embedding is not None else None,
    )


def load_jsonl(path: str | Path) -> list[QueryRecord]:
    """Load a trace in the CacheVerifier schema (one JSON object per line).

    Both bundled benchmarks (SemCacheLMArena, SemCacheSearchQueries) are
    expected to be pre-converted into this schema under `data/processed/`;
    see `scripts/convert_vcache_hf_dataset.py` (real benchmarks) or
    `scripts/generate_synthetic_dataset.py` (offline smoke-testing) for the
    exact field contract, including the optional `embedding` field.
    """
    path = Path(path)
    records: list[QueryRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(_build_record(json.loads(line), str(path), line_no))
    return records


def iter_jsonl(path: str | Path) -> Iterator[QueryRecord]:
    """Streaming variant of `load_jsonl` for traces too large to hold in memory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield _build_record(json.loads(line), str(path), line_no)
