"""Phase 2 driver for the Top-K cascade direction (memory:
future_research_directions.md "Top-1 retrieval ceiling"; design discussed
2026-08-22; Phase 0 ceiling diagnostic: topk_ceiling_diagnostic.py; fast/slow
cross-check: tests/test_topk_sweep.py).

Mirrors `cacheverifier.experiments.run_verified`'s CLI and one-pass/cheap-
replay structure, but drives `cacheverifier.experiments.topk_sweep`'s
K-candidate cascade instead of `SynchronousVerifiedPolicy`'s single
candidate: the ANN build happens once per (dataset, k), the verifier scores
every gray-zone-range (record, rank) pair once, and each (tau_low, tau_high,
threshold) grid point is then a cheap replay.

Examples:
    python scripts/run_topk_cascade.py --config configs/lmarena.yaml \\
        --k 2 --verifier cross_encoder --tau-high 0.97 \\
        --output results/lmarena_cascade_k2_cross_encoder.json
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
from cacheverifier.experiments.run_baselines import build_embedder, parse_float_list
from cacheverifier.experiments.run_verified import (
    DEFAULT_CROSS_ENCODER_THRESHOLD_GRID,
    DEFAULT_TAU_HIGH,
    DEFAULT_TAU_LOW_GRID,
    CACHE_DIR,
    _slug,
    build_verifier,
)
from cacheverifier.experiments.topk_sweep import (
    build_cascade_trace,
    load_cascade_trace,
    load_scored_cascade,
    replay_cascade,
    save_cascade_trace,
    save_scored_cascade,
    score_cascade_candidates,
)
from cacheverifier.metrics.bootstrap import bootstrap_ci
from cacheverifier.metrics.core import (
    confusion_counts,
    error_rate,
    false_accept_rate,
    hit_rate,
    mean_verifier_calls,
    mean_verifier_latency_ms,
    verifier_fidelity,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def summarize_cascade(outcomes, extra: dict, n_resamples: int, confidence: float) -> dict:
    counts = confusion_counts(outcomes)
    fidelity = verifier_fidelity(outcomes)
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
        "verifier_call_rate": fidelity.n_verified / len(outcomes) if outcomes else 0.0,
        "mean_verifier_calls": mean_verifier_calls(outcomes),
        "mean_verifier_latency_ms": mean_verifier_latency_ms(outcomes),
        "expected_added_latency_ms": mean_verifier_latency_ms(outcomes) * (fidelity.n_verified / len(outcomes) if outcomes else 0.0),
        "n_verified": fidelity.n_verified,
        "false_approve_rate": fidelity.false_approve_rate,
        "false_reject_rate": fidelity.false_reject_rate,
    }
    if n_resamples > 0:
        hit_ci = bootstrap_ci(outcomes, hit_rate, n_resamples=n_resamples, confidence=confidence)
        err_ci = bootstrap_ci(outcomes, error_rate, n_resamples=n_resamples, confidence=confidence)
        row["hit_rate_ci"] = [hit_ci.ci_low, hit_ci.ci_high]
        row["error_rate_ci"] = [err_ci.ci_low, err_ci.ci_high]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--verifier", choices=["oracle", "cross_encoder"], default="cross_encoder")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tau-high", type=float, default=DEFAULT_TAU_HIGH)
    parser.add_argument("--tau-low-grid", type=parse_float_list, default=None)
    parser.add_argument("--cross-encoder-threshold-grid", type=parse_float_list, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    dataset_path = Path(cfg.processed_path)
    log(f"Loading records from {dataset_path}...")
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    log(f"Loaded {len(records)} records")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    tau_low_grid = args.tau_low_grid or DEFAULT_TAU_LOW_GRID
    tau_high = args.tau_high
    threshold_grid = [0.5] if args.verifier == "oracle" else (args.cross_encoder_threshold_grid or DEFAULT_CROSS_ENCODER_THRESHOLD_GRID)

    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__k{args.k}.cascade_trace.json"

    if not args.no_cache and trace_cache_path.exists():
        log(f"Loading cached k={args.k} cascade trace from {trace_cache_path}...")
        trace = load_cascade_trace(trace_cache_path)
    else:
        log(f"Building k={args.k} cascade trace (single HNSW pass)...")
        t0 = time.time()
        trace = build_cascade_trace(records, embedder, args.k)
        log(f"  done in {time.time() - t0:.1f}s")
        save_cascade_trace(trace, trace_cache_path)
        log(f"Cached to {trace_cache_path}")

    verifier_key = args.verifier if args.verifier != "cross_encoder" else f"cross_encoder_{_slug(args.cross_encoder_model)}"
    gray_zone_lo, gray_zone_hi = min(tau_low_grid), tau_high
    scored_cache_path = (
        CACHE_DIR
        / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__k{args.k}__{verifier_key}"
          f"__lo{gray_zone_lo}__hi{gray_zone_hi}.cascade_scored.json"
    )

    if not args.no_cache and scored_cache_path.exists():
        log(f"Loading cached cascade scores from {scored_cache_path}...")
        scored = load_scored_cascade(scored_cache_path)
    else:
        log(f"Scoring cascade candidates once with verifier={args.verifier!r} (similarity in [{gray_zone_lo}, {gray_zone_hi}))...")
        verifier = build_verifier(args.verifier, args.cross_encoder_model)
        scored = score_cascade_candidates(records, trace, verifier, gray_zone_lo=gray_zone_lo, gray_zone_hi=gray_zone_hi)
        save_scored_cascade(scored, scored_cache_path)
        log(f"Cached to {scored_cache_path}")

    if scored:
        raw_scores = np.array([s.score for s in scored.values()])
        log(f"Cascade score stats: n={len(raw_scores)}, min={raw_scores.min():.3f}, mean={raw_scores.mean():.3f}, max={raw_scores.max():.3f}")

    log(f"Sweeping {len(tau_low_grid)} tau_low x {len(threshold_grid)} threshold @ tau_high={tau_high}, k={args.k}...")
    t_sweep_start = time.time()
    results = []
    total_points = len(tau_low_grid) * len(threshold_grid)
    point_i = 0
    for tau_low in tau_low_grid:
        for threshold in threshold_grid:
            point_i += 1
            outcomes = replay_cascade(trace, scored, tau_low=tau_low, tau_high=tau_high, threshold=threshold, k=args.k)
            row = summarize_cascade(
                outcomes,
                extra={
                    "verifier": args.verifier,
                    "k": args.k,
                    "tau_low": tau_low,
                    "tau_high": tau_high,
                    "threshold": threshold,
                },
                n_resamples=args.bootstrap_resamples,
                confidence=args.confidence,
            )
            results.append(row)
            elapsed = time.time() - t_sweep_start
            rate = point_i / elapsed if elapsed > 0 else 0
            eta = (total_points - point_i) / rate if rate > 0 else float("inf")
            log(
                f"[{point_i}/{total_points}] tau_low={tau_low} threshold={threshold}  "
                f"hit_rate={row['hit_rate']:.4f}  error_rate={row['error_rate']:.4f}  "
                f"mean_calls={row['mean_verifier_calls']:.2f}  (eta={eta / 60:.1f}m)"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Wrote {len(results)} result rows to {output_path}")


if __name__ == "__main__":
    main()
