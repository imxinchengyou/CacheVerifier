import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.verified_sweep import build_match_trace

TAU_LOW, TAU_HIGH = 0.80, 0.97

cfg = load_dataset_config("configs/search_queries.yaml")
records = load_jsonl(Path(cfg.processed_path))
if cfg.max_samples is not None:
    records = records[: cfg.max_samples]
print(f"Loaded {len(records)} records")

embedder = build_embedder(cfg.embedder, cfg.embedder_model)
t0 = time.time()
trace = build_match_trace(records, embedder)
print(f"trace built in {time.time()-t0:.1f}s")

gray_zone_candidate_indices = set()
gray_zone_query_indices = set()
for i, t in enumerate(trace):
    if t.similarity is not None and TAU_LOW <= t.similarity < TAU_HIGH:
        gray_zone_query_indices.add(i)
        if t.candidate_index is not None:
            gray_zone_candidate_indices.add(t.candidate_index)

union = gray_zone_candidate_indices | gray_zone_query_indices
print(f"gray-zone pairs: {sum(1 for t in trace if t.similarity is not None and TAU_LOW <= t.similarity < TAU_HIGH)}")
print(f"unique QUERY-side records in gray zone: {len(gray_zone_query_indices)}")
print(f"unique CANDIDATE-side records in gray zone: {len(gray_zone_candidate_indices)}")
print(f"unique records needing a real answer (query-side UNION candidate-side): {len(union)}")
print(f"as fraction of full corpus: {len(union)/len(records):.1%}")

import json
Path("results/searchqueries_gray_zone_needed_indices.json").write_text(
    json.dumps(sorted(union)), encoding="utf-8"
)
print("Wrote results/searchqueries_gray_zone_needed_indices.json")
