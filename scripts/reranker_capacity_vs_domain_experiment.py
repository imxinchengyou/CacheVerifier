"""Direction 6: does a larger/differently-trained off-the-shelf reranker
close part of Group D's gap on SearchQueries, and does it separate "model
capacity" from "training-distribution mismatch" as competing explanations?

Two new verifiers, each isolating one variable relative to the paper's
existing Group D model (`cross-encoder/ms-marco-MiniLM-L6-v2`):
- `cross-encoder/ms-marco-MiniLM-L12-v2`: same MS MARCO training data/
  objective, just bigger (12 vs 6 layers) -- a pure capacity test. Already
  externally evaluated by Baral et al. (2026), CRR=0.427 on their own
  datasets/metric -- this reruns it on THIS paper's own three datasets and
  metric (AUC on gray-zone would_be_correct labels) for a direct comparison.
- `BAAI/bge-reranker-base`: different training lineage entirely (broad
  multi-domain retrieval data, not solely MS MARCO long-web-query/passage
  ranking), and larger (278M vs ~22M/33M params) -- if this beats L12 by
  more than its extra capacity alone would predict, that's evidence for
  "training distribution mismatch," not "model too small."

Reuses each dataset's already-cached match trace (`results/.cache/*.trace.json`,
built once by run_verified.py/run_baselines.py) and the paper's standard
gray zone (tau_low=0.80, tau_high=0.97, matching Section 4.3/5.4) -- no new
ANN search, only new verifier inference over the existing candidate set.

Usage:
    python scripts/reranker_capacity_vs_domain_experiment.py \
        --config configs/lmarena.yaml \
        --model cross-encoder/ms-marco-MiniLM-L12-v2 \
        --output results/lmarena_reranker_l12_auc.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.verified_sweep import build_match_trace, load_match_trace, save_match_trace, score_gray_zone
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

CACHE_DIR = Path("results/.cache")
GRAY_ZONE_LO = 0.80
GRAY_ZONE_HI = 0.97


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Plain rank-based AUC -- copied verbatim from
    scripts/finetune_verifier_train_eval.py::roc_auc for consistency with
    every other AUC number reported in this project."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="HF cross-encoder model id")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cfg = load_dataset_config(args.config)
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples

    dataset_path = Path(cfg.processed_path)
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    log(f"Loaded {len(records)} records from {dataset_path}")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}.trace.json"

    if trace_cache_path.exists():
        log(f"Loading cached match trace from {trace_cache_path}")
        trace = load_match_trace(trace_cache_path)
    else:
        log("No cached trace found -- building one (this is the one expensive ANN pass)")
        trace = build_match_trace(records, embedder)
        save_match_trace(trace, trace_cache_path)

    log(f"Scoring gray zone [{GRAY_ZONE_LO}, {GRAY_ZONE_HI}) with {args.model!r}...")
    verifier = CrossEncoderVerifier(model_name=args.model)
    scored = score_gray_zone(records, trace, verifier, gray_zone_lo=GRAY_ZONE_LO, gray_zone_hi=GRAY_ZONE_HI)

    scores = np.array([scored[i].score for i in scored])
    labels = np.array([1 if trace[i].would_be_correct else 0 for i in scored])
    auc = roc_auc(scores, labels)
    mean_latency_ms = float(np.mean([scored[i].latency_ms for i in scored]))

    result = {
        "dataset": dataset_path.stem,
        "model": args.model,
        "n_records": len(records),
        "n_gray_zone": len(scored),
        "n_positive": int(labels.sum()),
        "auc": auc,
        "mean_latency_ms": mean_latency_ms,
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
    }
    log(f"AUC = {auc:.4f}  (n={len(scored)}, {int(labels.sum())} positive, mean_latency={mean_latency_ms:.1f}ms)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
