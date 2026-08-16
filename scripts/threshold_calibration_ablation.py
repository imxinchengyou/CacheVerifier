"""Direction 1 (RESEARCH_PROPOSAL.md / memory): does Group D's reported
"best point" survive an honest calibration/test split, or is it inflated by
grid-search optimism?

The current pipeline (`cacheverifier.experiments.run_verified`) sweeps a
whole (tau_low, threshold) grid and reports the single best point against
Group A's frontier -- PAPER.md Section 7 already flags this as a limitation
("the verifier's score threshold grid was calibrated after the fact ... not
fixed in advance on a held-out set"). This script fixes that specifically:
for each tau_low in the grid, the gray-zone stream at that tau_low is split
chronologically into a calibration half and a test half; `threshold` is
picked via Youden's J (ported from `cacheverifier-service`'s
`verifier_core.finetune.select_threshold`) on the CALIBRATION half only,
then hit_rate/error_rate is measured on the TEST half only -- a threshold
the pipeline never got to see final performance before committing to.

Reuses the cached match trace + gray-zone scores from
`cacheverifier.experiments.run_verified` (same on-disk cache), so this is
cheap to run for datasets already scored.

Usage:
    python scripts/threshold_calibration_ablation.py --config configs/lmarena.yaml \
        --output results/threshold_calibration_lmarena.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.run_verified import DEFAULT_TAU_LOW_GRID, _slug, CACHE_DIR
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
from cacheverifier.metrics.core import error_rate, hit_rate
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

DEFAULT_TAU_HIGH = 0.97


def select_threshold(scores: np.ndarray, labels: np.ndarray, min_test: int = 10) -> float | None:
    """Youden's J threshold selection -- ported verbatim (same formula) from
    `cacheverifier-service/verifier_core/finetune.py::select_threshold`, so
    this ablation uses the exact selection rule already shipped in the
    commercial product, not a reimplementation that merely resembles it."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0 or (n_pos + n_neg) < min_test:
        return None
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr, fpr = tp / n_pos, fp / n_neg
    youden_j = tpr - fpr
    return float(sorted_scores[int(np.argmax(youden_j))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tau-high", type=float, default=DEFAULT_TAU_HIGH)
    parser.add_argument("--tau-low-grid", default=None, help="Comma-separated; default DEFAULT_TAU_LOW_GRID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    args = parser.parse_args()

    def log(msg: str) -> None:
        import time

        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cfg = load_dataset_config(args.config)
    tau_low_grid = (
        [float(x) for x in args.tau_low_grid.split(",")] if args.tau_low_grid else DEFAULT_TAU_LOW_GRID
    )

    dataset_path = Path(cfg.processed_path)
    log(f"Loading records from {dataset_path}...")
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    log(f"Loaded {len(records)} records")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}.trace.json"

    if trace_cache_path.exists():
        log(f"Loading cached match trace from {trace_cache_path}...")
        trace = load_match_trace(trace_cache_path)
    else:
        log("Building match trace...")
        trace = build_match_trace(records, embedder)
        save_match_trace(trace, trace_cache_path)
        log(f"Cached to {trace_cache_path}")

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    verifier_key = f"cross_encoder_{_slug(cross_encoder_model)}"
    scored_cache_path = (
        CACHE_DIR
        / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__{verifier_key}"
          f"__lo{min(tau_low_grid)}__hi{args.tau_high}.scored.json"
    )
    if scored_cache_path.exists():
        log(f"Loading cached gray-zone scores from {scored_cache_path}...")
        scored = load_scored(scored_cache_path)
    else:
        log("Scoring gray-zone candidates with the off-the-shelf verifier...")
        verifier = CrossEncoderVerifier(cross_encoder_model)
        scored = score_gray_zone(records, trace, verifier, gray_zone_lo=min(tau_low_grid), gray_zone_hi=args.tau_high)
        save_scored(scored, scored_cache_path)
        log(f"Cached to {scored_cache_path}")

    results = []
    for tau_low in tau_low_grid:
        # Indices in the gray zone for THIS tau_low (a subset of `scored`,
        # which was computed at the widest tau_low already).
        gz_indices = [
            i for i, t in enumerate(trace)
            if t.similarity is not None and tau_low <= t.similarity < args.tau_high and i in scored
        ]
        if len(gz_indices) < 20:
            log(f"tau_low={tau_low}: only {len(gz_indices)} gray-zone points, skipping")
            continue

        split = len(gz_indices) // 2
        calib_idx, test_idx = gz_indices[:split], gz_indices[split:]

        calib_scores = np.array([scored[i].score for i in calib_idx])
        calib_labels = np.array([1 if t.would_be_correct else 0 for t, i in [(trace[i], i) for i in calib_idx]])
        threshold = select_threshold(calib_scores, calib_labels)
        if threshold is None:
            log(f"tau_low={tau_low}: calibration split too small/single-class, skipping")
            continue

        # Honest eval: replay only the TEST half of the stream at this
        # (tau_low, threshold) -- the calibration half never contributes to
        # this metric, so `threshold` genuinely never saw it.
        test_index_set = set(test_idx)
        full_outcomes = replay(trace, scored, tau_low=tau_low, tau_high=args.tau_high, threshold=threshold)
        # `replay` walks the WHOLE stream (all records, not just gray-zone
        # ones) to produce per-request outcomes in original order; restrict
        # to requests whose gray-zone candidate index is in the test half,
        # plus every non-gray-zone request (tau_low/tau_high shortcut
        # branches -- included for a fair hit_rate/error_rate denominator,
        # exactly as run_verified.py's normal replay does).
        test_outcomes = [
            o for idx, o in enumerate(full_outcomes)
            if trace[idx].similarity is None
            or not (tau_low <= trace[idx].similarity < args.tau_high)
            or idx in test_index_set
        ]

        hr = hit_rate(test_outcomes)
        er = error_rate(test_outcomes)
        hr_ci = bootstrap_ci(test_outcomes, hit_rate, n_resamples=args.bootstrap_resamples)
        er_ci = bootstrap_ci(test_outcomes, error_rate, n_resamples=args.bootstrap_resamples)

        log(f"tau_low={tau_low}  calibrated_threshold={threshold:.3f}  "
            f"n_calib={len(calib_idx)}  n_test={len(test_idx)}  hit_rate={hr:.4f}  error_rate={er:.4f}")

        results.append({
            "tau_low": tau_low,
            "tau_high": args.tau_high,
            "calibrated_threshold": threshold,
            "n_calibration": len(calib_idx),
            "n_test": len(test_idx),
            "hit_rate": hr,
            "error_rate": er,
            "hit_rate_ci": [hr_ci.ci_low, hr_ci.ci_high],
            "error_rate_ci": [er_ci.ci_low, er_ci.ci_high],
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
