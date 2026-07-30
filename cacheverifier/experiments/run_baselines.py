"""Week 1 CLI: run Group A (static threshold) or Group B (adaptive threshold)
over a processed trace and dump hit-rate/error-rate points to results/.

Follows vCache's own evaluation protocol: no offline history split. The
first `max_samples` records are streamed once, in the trace's original
order, through an initially empty cache (see
`cacheverifier.experiments.runner.ExperimentRunner`). Default threshold /
target-error-rate grids match vCache's `benchmarks/benchmark.py`
STATIC_THRESHOLDS / DELTAS so points line up with the paper's figures.

Examples:
    python -m cacheverifier.experiments.run_baselines \\
        --dataset data/processed/synthetic.jsonl --group A

    python -m cacheverifier.experiments.run_baselines \\
        --dataset data/processed/synthetic.jsonl --group B \\
        --embedder sentence-transformer
"""

import argparse
import json
import time
from pathlib import Path

from cacheverifier.cache.adaptive_threshold import AdaptiveThresholdPolicy
from cacheverifier.cache.static_threshold import StaticThresholdPolicy
from cacheverifier.config import DatasetConfig, load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.embeddings.base import Embedder
from cacheverifier.experiments.runner import ExperimentRunner
from cacheverifier.metrics.bootstrap import bootstrap_ci
from cacheverifier.metrics.core import RequestOutcome, confusion_counts, error_rate, false_accept_rate, hit_rate


def build_embedder(name: str, model_name: str) -> Embedder:
    if name == "hash":
        from cacheverifier.embeddings.hash_embedder import HashEmbedder

        return HashEmbedder()
    if name == "sentence-transformer":
        from cacheverifier.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=model_name)
    if name == "precomputed":
        from cacheverifier.embeddings.precomputed_embedder import PrecomputedEmbedder

        return PrecomputedEmbedder()
    raise ValueError(f"unknown embedder {name!r}")


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def summarize(
    outcomes: list[RequestOutcome], extra: dict, n_resamples: int, confidence: float
) -> dict:
    """One results row: point estimates + confusion matrix, plus bootstrap
    CIs on hit_rate/error_rate (skipped when `n_resamples <= 0`) — this is
    what the Go/No-Go check (proposal Section 6) compares across groups
    instead of eyeballing point estimates.
    """
    counts = confusion_counts(outcomes)
    row = {
        **extra,
        "hit_rate": hit_rate(outcomes),
        "error_rate": error_rate(outcomes),
        "false_accept_rate": false_accept_rate(outcomes),
        "precision": counts.precision,
        "recall": counts.recall,
        "tp": counts.tp,
        "fp": counts.fp,
        "tn": counts.tn,
        "fn": counts.fn,
        "n_records": len(outcomes),
    }
    if n_resamples > 0:
        hit_ci = bootstrap_ci(outcomes, hit_rate, n_resamples=n_resamples, confidence=confidence)
        err_ci = bootstrap_ci(outcomes, error_rate, n_resamples=n_resamples, confidence=confidence)
        row["hit_rate_ci"] = [hit_ci.ci_low, hit_ci.ci_high]
        row["error_rate_ci"] = [err_ci.ci_low, err_ci.ci_high]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="Path to a configs/*.yaml DatasetConfig; overrides --dataset and friends")
    parser.add_argument("--dataset", default=None, help="Path to a processed JSONL trace")
    parser.add_argument("--group", required=True, choices=["A", "B"])
    parser.add_argument("--max-samples", type=int, default=None, help="Records to stream (default: all)")
    parser.add_argument("--embedder", choices=["hash", "sentence-transformer", "precomputed"], default="hash")
    parser.add_argument("--embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold-grid", type=parse_float_list, default=None,
                        help="Comma-separated thresholds for Group A")
    parser.add_argument("--target-error-rates", type=parse_float_list, default=None,
                        help="Comma-separated target error rates (delta) for Group B")
    parser.add_argument("--output", default=None, help="Output JSON path (default: results/<dataset>_<group>.json)")
    parser.add_argument("--bootstrap-resamples", type=int, default=200,
                        help="Bootstrap resamples for hit_rate/error_rate CIs; 0 disables (default: 200)")
    parser.add_argument("--confidence", type=float, default=0.95, help="CI confidence level (default: 0.95)")
    args = parser.parse_args()

    if args.config:
        cfg = load_dataset_config(args.config)
        if args.max_samples is not None:
            cfg.max_samples = args.max_samples
    else:
        if not args.dataset:
            parser.error("either --config or --dataset is required")
        cfg = DatasetConfig(
            name=Path(args.dataset).stem,
            processed_path=args.dataset,
            max_samples=args.max_samples,
            embedder=args.embedder,
            embedder_model=args.embedder_model,
        )
        if args.threshold_grid:
            cfg.threshold_grid = args.threshold_grid
        if args.target_error_rates:
            cfg.target_error_rates = args.target_error_rates

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    dataset_path = Path(cfg.processed_path)
    log(f"Loading records from {dataset_path}...")
    t0 = time.time()
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    log(f"Loaded {len(records)} records to stream in {time.time() - t0:.1f}s (cache starts empty)")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    runner = ExperimentRunner(embedder)

    log(f"Embedding all {len(records)} records once with {cfg.embedder!r} "
        f"(shared across the whole grid, not re-embedded per point)...")
    t0 = time.time()
    embeddings = embedder.embed(records)
    embed_time = time.time() - t0
    log(f"  done in {embed_time:.1f}s ({len(records) / max(embed_time, 1e-9):.1f} records/s), "
        f"dim={embeddings.shape[1]}")

    grid = cfg.threshold_grid if args.group == "A" else cfg.target_error_rates
    grid_label = "threshold" if args.group == "A" else "target_error_rate"
    log(f"Sweeping {len(grid)} {grid_label} values: {grid}")

    results = []
    for i, value in enumerate(grid, start=1):
        t0 = time.time()
        if args.group == "A":
            policy = StaticThresholdPolicy(value)
            extra = {"group": "A", "policy": "static_threshold", "threshold": value}
        else:
            policy = AdaptiveThresholdPolicy(target_error_rate=value)
            extra = {"group": "B", "policy": "adaptive_threshold", "target_error_rate": value}

        outcomes = runner.run(records, policy, embeddings=embeddings)
        row = summarize(outcomes, extra=extra, n_resamples=args.bootstrap_resamples, confidence=args.confidence)
        results.append(row)
        log(f"[{i}/{len(grid)}] {grid_label}={value}  hit_rate={row['hit_rate']:.4f}  "
            f"error_rate={row['error_rate']:.4f}  ({time.time() - t0:.1f}s)")

    for row in results:
        print(row)

    output_path = Path(args.output) if args.output else Path("results") / f"{dataset_path.stem}_group{args.group}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} result rows to {output_path}")


if __name__ == "__main__":
    main()
