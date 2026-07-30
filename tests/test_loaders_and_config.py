import json

import pytest

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl


def test_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {"query_id": "1", "query": "hi", "answer": "hello", "equivalence_id": "c1"},
        {"query_id": "2", "query": "hey", "answer": "hello", "equivalence_id": "c1"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    records = load_jsonl(path)
    assert len(records) == 2
    assert records[0].query_id == "1"
    assert records[1].equivalence_id == "c1"


def test_load_jsonl_parses_optional_embedding_field(tmp_path):
    path = tmp_path / "trace.jsonl"
    row = {
        "query_id": "1",
        "query": "hi",
        "answer": "hello",
        "equivalence_id": "c1",
        "embedding": [0.1, 0.2, 0.3],
    }
    path.write_text(json.dumps(row), encoding="utf-8")

    records = load_jsonl(path)
    assert records[0].embedding == (0.1, 0.2, 0.3)


def test_load_jsonl_missing_field_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"query_id": "1", "query": "hi"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(path)


def test_load_dataset_config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "name: demo\nprocessed_path: data/processed/demo.jsonl\nmax_samples: 3\n",
        encoding="utf-8",
    )
    cfg = load_dataset_config(path)
    assert cfg.name == "demo"
    assert cfg.max_samples == 3
    assert cfg.embedder == "hash"  # default preserved
