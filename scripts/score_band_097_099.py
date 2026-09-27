"""RESEARCH_PROPOSAL.md 方向 26.19: re-create the off-the-shelf verifier
scores for the similarity band [0.97, 0.99) that PAPER §5.11's tau_high
sweep used (the original lo0.8__hi0.99 scored files no longer exist). The
[0.80, 0.97) band is already on disk and is merged locally.

Usage (GPU box): python scripts/score_band_097_099.py --dataset lmarena
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.verified_sweep import load_match_trace, score_gray_zone
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

TRACE = {
    "lmarena": "lmarena__precomputed__n60000.trace.json",
    "quora": "quora__sentence-transformer_sentence-transformers_all-MiniLM-L6-v2__n60000.trace.json",
    "search_queries_corrected": "search_queries_corrected__precomputed__n150000.trace.json",
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(TRACE))
    a = p.parse_args()
    cfg = load_dataset_config(f"configs/{a.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)[: cfg.max_samples]
    trace = load_match_trace(Path("results/.cache") / TRACE[a.dataset])
    scored = score_gray_zone(records, trace, CrossEncoderVerifier(), 0.97, 0.99, log_every=5000)
    out = Path(f"results/band097_099_{a.dataset}.scored.json")
    out.write_text(json.dumps({str(i): [s.score, s.latency_ms] for i, s in scored.items()}))
    print("wrote", out, len(scored), flush=True)
