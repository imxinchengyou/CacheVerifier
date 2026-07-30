"""Week 2 CLI: run Group C (synchronous Krites / oracle verifier) or Group D
(synchronous + real verifier) — both use
`cacheverifier.cache.synchronous_verified.SynchronousVerifiedPolicy`'s gray-zone
gating logic, differing only in which `Verifier` is plugged in. See that
module's docstring for why sharing the mechanism is the point: it isolates
"sync vs async" (Group C, at Krites' own oracle verifier-fidelity
assumption) from "oracle vs real verifier" (Group D).

Performance note: this does NOT call `ExperimentRunner.run` once per grid
point. Because every record gets inserted into the cache regardless of
hit/miss (see `cacheverifier.experiments.verified_sweep`), the nearest-neighbor
match sequence for Groups A/C/D is independent of tau_low/tau_high/
threshold — so the ANN search runs exactly once (`build_match_trace`), the
verifier scores every gray-zone-range candidate exactly once
(`score_gray_zone`, covering the union of the whole sweep), and each
(tau_low, threshold) grid point is then just a cheap re-thresholding pass
(`replay`). `tau_high` is fixed per CLI invocation and therefore isn't part
of that union — only `tau_low` and the verifier's `threshold` are swept.

Examples:
    python -m cacheverifier.experiments.run_verified \\
        --config configs/lmarena.yaml --group C --tau-high 0.97 \\
        --output results/lmarena_groupC.json

    python -m cacheverifier.experiments.run_verified \\
        --config configs/lmarena.yaml --group D --verifier cross_encoder \\
        --tau-high 0.97 --output results/lmarena_groupD_cross_encoder.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder, parse_float_list
from cacheverifier.experiments.verified_sweep import (
    build_match_trace,
    load_match_trace,
    load_scored,
    replay,
    save_match_trace,
    save_scored,
    score_gray_zone,
)
from cacheverifier.metrics.bootstrap import bootstrap_ci
from cacheverifier.metrics.core import (
    confusion_counts,
    error_rate,
    false_accept_rate,
    hit_rate,
    mean_verifier_latency_ms,
    verifier_fidelity,
)
from cacheverifier.verifiers.base import Verifier

DEFAULT_TAU_HIGH = 0.97
DEFAULT_TAU_LOW_GRID = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95]
DEFAULT_CROSS_ENCODER_THRESHOLD_GRID = [-2.0, -1.0, 0.0, 1.0, 2.0]
CACHE_DIR = Path("results/.cache")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def build_verifier(verifier_name: str, cross_encoder_model: str) -> Verifier:
    if verifier_name == "oracle":
        from cacheverifier.verifiers.oracle_verifier import OracleVerifier

        return OracleVerifier()
    if verifier_name == "cross_encoder":
        from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

        return CrossEncoderVerifier(model_name=cross_encoder_model)
    raise ValueError(f"unknown verifier {verifier_name!r}")


def summarize_verified(outcomes, extra: dict, n_resamples: int, confidence: float) -> dict:
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
    parser.add_argument("--config", default=None, help="Path to a configs/*.yaml DatasetConfig")
    parser.add_argument("--dataset", default=None, help="Path to a processed JSONL trace (if not using --config)")
    parser.add_argument("--group", required=True, choices=["C", "D", "E"])
    parser.add_argument("--verifier", choices=["oracle", "cross_encoder"], default=None,
                        help="Default: oracle for Group C, cross_encoder for Group D")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--embedder", choices=["hash", "sentence-transformer", "precomputed"], default="hash")
    parser.add_argument("--embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--tau-high", type=float, default=DEFAULT_TAU_HIGH)
    parser.add_argument("--tau-low-grid", type=parse_float_list, default=None,
                        help=f"Comma-separated tau_low sweep (default: {DEFAULT_TAU_LOW_GRID})")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-threshold-grid", type=parse_float_list, default=None,
                        help=f"Comma-separated raw-score thresholds to sweep for Group D "
                             f"(default: {DEFAULT_CROSS_ENCODER_THRESHOLD_GRID}; NOT validated against any "
                             f"real score distribution — check the printed score stats and adjust)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute the match trace and verifier scores even if a matching "
                             f"cache file exists under {CACHE_DIR}/ (e.g. after changing the dataset "
                             "or embedder without changing the file path)")
    args = parser.parse_args()

    if args.verifier is None:
        args.verifier = "oracle" if args.group == "C" else "cross_encoder"

    if args.config:
        cfg = load_dataset_config(args.config)
        if args.max_samples is not None:
            cfg.max_samples = args.max_samples
    else:
        if not args.dataset:
            parser.error("either --config or --dataset is required")
        from cacheverifier.config import DatasetConfig

        cfg = DatasetConfig(
            name=Path(args.dataset).stem,
            processed_path=args.dataset,
            max_samples=args.max_samples,
            embedder=args.embedder,
            embedder_model=args.embedder_model,
        )

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
    tau_low_grid = args.tau_low_grid or DEFAULT_TAU_LOW_GRID
    # Oracle's score is binary (1.0/0.0): any threshold in (0, 1) is
    # equivalent, so sweeping it would just repeat identical work.
    threshold_grid = (
        [0.5]
        if args.verifier == "oracle"
        else (args.cross_encoder_threshold_grid or DEFAULT_CROSS_ENCODER_THRESHOLD_GRID)
    )

    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}.trace.json"

    if not args.no_cache and trace_cache_path.exists():
        log(f"Loading cached match trace from {trace_cache_path}...")
        trace = load_match_trace(trace_cache_path)
    else:
        log("Building match trace (single pass over the stream)...")
        t0 = time.time()
        trace = build_match_trace(records, embedder)
        log(f"  done in {time.time() - t0:.1f}s")
        save_match_trace(trace, trace_cache_path)
        log(f"Cached match trace to {trace_cache_path}")

    verifier_key = args.verifier if args.verifier != "cross_encoder" else f"cross_encoder_{_slug(args.cross_encoder_model)}"
    scored_cache_path = (
        CACHE_DIR
        / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__{verifier_key}"
          f"__lo{min(tau_low_grid)}__hi{args.tau_high}.scored.json"
    )

    if not args.no_cache and scored_cache_path.exists():
        log(f"Loading cached gray-zone scores from {scored_cache_path}...")
        scored = load_scored(scored_cache_path)
    else:
        log(f"Scoring gray-zone candidates once with verifier={args.verifier!r} "
            f"(similarity in [{min(tau_low_grid)}, {args.tau_high}))...")
        verifier = build_verifier(args.verifier, args.cross_encoder_model)
        scored = score_gray_zone(records, trace, verifier, gray_zone_lo=min(tau_low_grid), gray_zone_hi=args.tau_high)
        save_scored(scored, scored_cache_path)
        log(f"Cached gray-zone scores to {scored_cache_path}")

    if scored:
        raw_scores = np.array([s.score for s in scored.values()])
        log(f"Gray-zone score stats: n={len(raw_scores)}, "
            f"min={raw_scores.min():.3f}, mean={raw_scores.mean():.3f}, max={raw_scores.max():.3f} "
            f"-- sanity-check threshold_grid against this range")
    else:
        log("No candidates fell in the gray zone for this tau_low/tau_high range.")

    total_points = len(tau_low_grid) * len(threshold_grid)
    log(f"Sweeping {len(tau_low_grid)} tau_low x {len(threshold_grid)} threshold = {total_points} grid points "
        f"(each does a cheap replay + {args.bootstrap_resamples}-resample bootstrap CI)...")
    t_sweep_start = time.time()
    results = []
    point_i = 0
    for tau_low in tau_low_grid:
        for threshold in threshold_grid:
            point_i += 1
            t0 = time.time()
            outcomes = replay(trace, scored, tau_low=tau_low, tau_high=args.tau_high, threshold=threshold)
            row = summarize_verified(
                outcomes,
                extra={
                    "group": args.group,
                    "policy": f"synchronous_verified[{args.verifier}]",
                    "verifier": args.verifier,
                    "tau_low": tau_low,
                    "tau_high": args.tau_high,
                    "threshold": threshold,
                },
                n_resamples=args.bootstrap_resamples,
                confidence=args.confidence,
            )
            results.append(row)
            elapsed = time.time() - t_sweep_start
            rate = point_i / elapsed if elapsed > 0 else 0
            eta = (total_points - point_i) / rate if rate > 0 else float("inf")
            log(f"[{point_i}/{total_points}] tau_low={tau_low} threshold={threshold}  "
                f"hit_rate={row['hit_rate']:.4f}  error_rate={row['error_rate']:.4f}  "
                f"({time.time() - t0:.1f}s this point, eta={eta / 60:.1f}m)")

    for row in results:
        print(row)

    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"{dataset_path.stem}_group{args.group}_{args.verifier}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Wrote {len(results)} result rows to {output_path}")


if __name__ == "__main__":
    main()
