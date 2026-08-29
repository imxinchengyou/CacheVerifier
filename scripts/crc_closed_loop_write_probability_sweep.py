"""Locates the write-probability threshold at which online recalibration
fully offsets self-selection's harm on LmArena -- the one dataset (of the
three tested in `scripts/crc_closed_loop_self_selection.py`, see PAPER.md
5.16) where recalibration mitigated but did NOT fully close the gap back to
the original insert-always baseline (self_select_recal vs. baseline:
diff=+0.0065, 95% CI [+0.0038,+0.0093], still significantly worse).

Two earlier design ideas for locating this threshold were considered and
rejected before this one:

  1. Sweep tau_high to artificially raise/lower a dataset's direct-hit
     rate. Rejected: tau_high isn't just "how much redundancy exists in
     the stream" -- it's also the boundary at which similarity alone is
     trusted without the verifier. Moving it confounds "more self-selection
     exposure" with "the direct-hit shortcut itself becomes less reliable"
     (this is exactly what PAPER.md 5.11 already found -- moving tau_high
     is never a neutral operation). Not a clean single-variable test.

  2. Inject synthetic exact-duplicate records into a low-redundancy dataset
     (e.g. Quora) to dial its direct-hit rate up toward LmArena's. Rejected
     as more complex than necessary and adding a new "is synthetic
     duplication realistic" question the result would have to defend,
     without touching the actual dataset that already showed the effect.

This script instead generalizes `run_regime`'s insert_on_hit_probability
(0.0=self_select, 1.0=baseline, already generalized to a continuous
parameter for exactly this use) and sweeps it directly on LmArena itself,
with everything else (embeddings, verifier, tau_low/tau_high, threshold
calibration protocol) held exactly as already validated. This is both a
cleaner single-variable manipulation (no data construction, no realism
argument needed) AND more directly product-relevant: an intermediate
insert_on_hit_probability models a cache that occasionally rewrites a
popular entry anyway (e.g. a TTL-driven refresh) -- a concrete, actionable
mitigation a real cache could implement, not just a diagnostic knob.

Only re-runs regime 1 (baseline, to get its frozen threshold AND raw
post-warmup observations for the bootstrap comparison -- the main script's
results JSON only persisted chunk-level summaries, not raw observations)
plus one run per swept probability, in "recalibrating" mode (matching
self_select_recal's protocol exactly, just with insert_on_hit_probability
varied instead of held at 0.0).

Usage:
    python scripts/crc_closed_loop_write_probability_sweep.py --dataset lmarena
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

from crc_closed_loop_self_selection import WARMUP_N, bootstrap_risk_difference, run_regime

DEFAULT_PROBABILITIES = [0.1, 0.25, 0.5, 0.75]
"""0.0 and 1.0 are already covered by the main script's self_select_recal
and baseline regimes -- no need to re-run either endpoint here."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="lmarena", choices=["quora", "lmarena", "search_queries_corrected"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--probabilities", default=",".join(str(p) for p in DEFAULT_PROBABILITIES), help="Comma-separated list"
    )
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    probabilities = [float(p) for p in args.probabilities.split(",") if p.strip()]

    cfg = load_dataset_config(args.config or f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    print(f"Loaded {len(records)} records from {cfg.processed_path}")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    print("Embedding records once (shared across every probability run)...")
    embeddings = embedder.embed(records)
    verifier = CrossEncoderVerifier()

    print("\n=== Baseline (p=1.0, frozen-after-warmup threshold) -- re-run for raw post-warmup observations ===")
    baseline = run_regime(
        "baseline", records, embeddings, verifier, insert_on_hit_probability=1.0, threshold_mode="frozen_after_warmup"
    )
    if baseline.post_warmup_threshold is None:
        raise RuntimeError(f"baseline never reached WARMUP_N={WARMUP_N} gray-zone observations")
    frozen_threshold = baseline.post_warmup_threshold
    print(f"frozen threshold: {frozen_threshold}")
    baseline_post_warmup = baseline.observations[WARMUP_N:]

    results = {
        "dataset": args.dataset,
        "warmup_n": WARMUP_N,
        "frozen_threshold": frozen_threshold,
        "baseline_n_gray_zone": len(baseline.observations),
        "sweep": [],
    }

    for p in probabilities:
        print(f"\n=== insert_on_hit_probability={p} (recalibrating) ===")
        r = run_regime(
            f"p{p}",
            records,
            embeddings,
            verifier,
            insert_on_hit_probability=p,
            threshold_mode="recalibrating",
            fixed_threshold=frozen_threshold,
        )
        cmp = bootstrap_risk_difference(
            baseline_post_warmup, r.observations[WARMUP_N:], n_resamples=args.n_resamples, seed=0
        )
        sig = "SIGNIFICANT" if cmp["significant"] else "not significant"
        direction = "residual harm" if cmp["diff"] > 0 else ("fully compensates" if cmp["significant"] else "borderline")
        print(
            f"  n_gray_zone={len(r.observations)}  final_store_size={r.final_store_size}  "
            f"risk {cmp['risk_a']:.4f} -> {cmp['risk_b']:.4f}  diff={cmp['diff']:+.4f}  "
            f"95% CI=[{cmp['diff_ci_low']:+.4f}, {cmp['diff_ci_high']:+.4f}]  ({sig}, {direction})"
        )
        results["sweep"].append(
            {
                "insert_on_hit_probability": p,
                "n_gray_zone": len(r.observations),
                "final_store_size": r.final_store_size,
                "comparison_vs_baseline": cmp,
                "direction": direction,
            }
        )

    print(f"\n{'p':>6} | {'risk':>8} {'diff vs baseline':>18} {'95% CI':>24} {'verdict':>18}")
    for row in results["sweep"]:
        cmp = row["comparison_vs_baseline"]
        ci = f"[{cmp['diff_ci_low']:+.4f},{cmp['diff_ci_high']:+.4f}]"
        print(f"{row['insert_on_hit_probability']:>6} | {cmp['risk_b']:>8.4f} {cmp['diff']:>+18.4f} {ci:>24} {row['direction']:>18}")

    out_path = (
        Path(args.output) if args.output else Path(f"results/crc_closed_loop_write_probability_sweep_{args.dataset}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
