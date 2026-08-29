"""CRC closed-loop self-selection experiment (PAPER.md §8 future-work item
11(d) / RESEARCH_PROPOSAL.md): the one open question in the CRC line of
work (§5.13) that Protocol R and Protocol T don't answer. Both of those
protocols split a fixed, already-completed historical trace (random vs.
chronological order) -- neither tests a true online closed loop where the
verifier's own accept/reject decisions change what future queries can even
match against.

Verified before writing this (not assumed): EVERY other experiment in this
project -- Groups A-E, CRC Protocol R/T, the Top-K cascade -- shares one
convention inherited from vCache's own benchmark harness
(`cacheverifier/experiments/verified_sweep.py::build_match_trace`,
`cacheverifier/experiments/runner.py::ExperimentRunner.run`): every record
gets inserted into the cache regardless of whether it was a hit or a miss.
That means the candidate pool available at any point in the stream has
never, in any prior experiment in this project, depended on any policy's
decisions -- the self-selection mechanism this script tests has never been
simulated here before. This script does NOT modify verified_sweep.py or
runner.py (every other experiment in this project relies on their current
insert-always behavior) -- it's a standalone closed-loop simulator.

Three regimes, same record stream, same embeddings, isolating two variables:

  1. baseline    -- insert-always semantics (matches every existing result
                     in this project) + a threshold frozen once from an
                     initial warm-up slice. Sanity check: should
                     approximately reproduce Protocol T's already-published
                     Quora exceedance finding (scripts/crc_protocol_t_chronological.py).
  2. self_select  -- write-on-miss-only semantics (a "hit" reuses and does
                     NOT write a duplicate entry; only a miss/reject writes
                     a fresh one) + the SAME frozen threshold as (1),
                     transplanted as a fixed numeric value. Isolates
                     whether cache-content feedback alone, with the
                     threshold never changing, shifts realized risk.
  3. self_select_recal -- write-on-miss-only semantics + the CRC threshold
                     periodically re-derived from a sliding window of the
                     most recent gray-zone (score, label) pairs observed so
                     far in THIS live run. The full hypothesized failure
                     mode: the calibration pool itself is shaped by what
                     this policy has been approving/rejecting.

Bootstrap policy before the first calibration exists (all three regimes):
treat every gray-zone candidate as rejected (miss) -- this mirrors
cacheverifier-service's own real `cold_start_mode="fail_closed"` default
(see app/db/models.py in that sibling repo), not an arbitrary choice.

No precomputed match trace / scored-candidates cache is used: the candidate
pool differs by regime, so gray-zone scoring needs live cross-encoder
inference during the loop, unlike every other experiment in this project.

Also computes an unpaired bootstrap CI on the realized-risk difference
between regimes (post-warm-up observations only, see
`bootstrap_risk_difference`) -- 2026-08-28, added after the first Quora
pass showed the naive "overall risk" comparison was confounded by regime
1's own warm-up period, and to put a confidence interval (not just a
single-run point estimate) on whether self-selection has a detectable
effect at all.

Usage:
    python scripts/crc_closed_loop_self_selection.py --dataset quora
    python scripts/crc_closed_loop_self_selection.py --dataset lmarena
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.cache.store import CacheEntry, VectorCacheStore
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.data.schema import QueryRecord
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.metrics.core import crc_select_threshold
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

TAU_LOW = 0.80
TAU_HIGH = 0.97
ALPHA = 0.02
"""One level from Protocol R/T's existing grid ([0.05, 0.02, 0.01, 0.005])
-- not sweeping all four, to keep this first pass tractable."""

WARMUP_N = 1500
"""Gray-zone observations collected (all treated as rejected -- see module
docstring) before the first real calibration runs. Same order of magnitude
as RECAL_WINDOW below, not derived from data -- a modeling choice, flagged
as such in the writeup."""

RECAL_EVERY = 500
"""Regime 3 only: re-derive the threshold every this-many new gray-zone
observations after warm-up."""

RECAL_WINDOW = 2000
"""Regime 3 only: recalibrate from the most recent this-many gray-zone
(score, label) pairs -- a SLIDING window, not an expanding pool. An
expanding pool would dilute any endogenous shift with more and more
history, which is the wrong direction for testing this specific
hypothesis (see module docstring)."""

N_CHUNKS = 10


@dataclass
class GrayZoneObservation:
    stream_position: int
    score: float
    correct: bool
    threshold_used: float
    query_id: str | None = None
    query_text: str | None = None
    candidate_query_id: str | None = None
    candidate_query_text: str | None = None
    candidate_answer_text: str | None = None
    """Only populated when `run_regime(..., capture_text=True)` -- default
    runs leave these None to keep the main quantitative results JSON lean.
    Added for `scripts/crc_closed_loop_spot_check.py`'s qualitative audit:
    is the "correct near-duplicate candidate gets silently crowded out"
    mechanism hypothesized for LmArena's self-selection effect actually
    what's happening in specific cases, not just a statistical pattern."""


@dataclass
class RegimeResult:
    name: str
    insert_on_hit_probability: float
    observations: list[GrayZoneObservation] = field(default_factory=list)
    n_direct_hit: int = 0
    n_direct_miss: int = 0
    n_gray_hit: int = 0
    n_gray_miss: int = 0
    final_store_size: int = 0
    post_warmup_threshold: float | None = None
    """Only set for threshold_mode="frozen_after_warmup" (regime 1) -- the
    value calibrated from the warm-up slice, meant to be read out and
    transplanted as `fixed_threshold` into regimes 2/3."""
    outcomes: list[str] = field(default_factory=list)
    """"hit"/"miss" per stream position, in order -- NOT restricted to
    gray-zone positions (unlike `observations`). Cheap to keep for every
    run (one short string per record); lets `scripts/crc_closed_loop_spot_check.py`
    check, for any earlier record j, whether it was ever actually written
    into THIS regime's store (miss) or was itself served from an existing
    entry without writing one (hit) -- the direct test of whether a
    specific "correct candidate" was crowded out."""


def run_regime(
    name: str,
    records: list[QueryRecord],
    embeddings: np.ndarray,
    verifier: CrossEncoderVerifier,
    insert_on_hit_probability: float,
    threshold_mode: str,
    fixed_threshold: float | None = None,
    capture_text: bool = False,
    seed: int = 0,
) -> RegimeResult:
    """One full pass over the stream.

    `capture_text`: when True, each GrayZoneObservation also carries the
    query/candidate ids and text (default False, to keep normal runs'
    memory/JSON footprint lean) -- for `scripts/crc_closed_loop_spot_check.py`'s
    qualitative audit of specific self-selection cases, not needed for the
    quantitative risk numbers this module reports by default.

    `insert_on_hit_probability`: on a hit (direct or gray-zone-approved),
    the probability the cache still writes a fresh entry anyway, instead of
    purely reusing the existing one. 1.0 reproduces the insert-always
    convention every other experiment in this project uses (regime 1). 0.0
    is write-on-miss-only (regimes 2/3): a hit never writes a duplicate;
    only a miss (direct or gray-zone-rejected) triggers a fresh generation
    that gets cached. Values in between model a cache that occasionally
    rewrites a popular entry anyway (e.g. a TTL-driven refresh) -- added
    2026-08-29 to test whether a partial rewrite policy is enough to fully
    offset self-selection's harm on a dataset where a pure 0.0/1.0
    comparison (see PAPER.md 5.16) left a residual gap online recalibration
    alone didn't close (LmArena). A miss always inserts regardless of this
    probability -- only what happens on a HIT is being varied; that was
    never the part any regime tested here changes.

    `threshold_mode`: "frozen_after_warmup" (regime 1) calibrates once from
    the first WARMUP_N gray-zone observations and never changes again.
    "fixed" (regime 2) uses `fixed_threshold` from the very first gray-zone
    observation, unchanged throughout -- this is regime 1's OWN resulting
    threshold, transplanted as a plain number, so regimes 1 and 2 differ
    ONLY in insert_on_hit_probability, nothing else. "recalibrating"
    (regime 3) behaves like "fixed" during warm-up (same transplanted
    value, so all three regimes are identical until they're designed to
    diverge), then switches to periodic sliding-window recalibration once
    WARMUP_N gray-zone observations have been collected.
    """
    assert threshold_mode in ("frozen_after_warmup", "fixed", "recalibrating")
    if threshold_mode in ("fixed", "recalibrating"):
        assert fixed_threshold is not None
    assert 0.0 <= insert_on_hit_probability <= 1.0

    store = VectorCacheStore(dim=embeddings.shape[1])
    result = RegimeResult(name=name, insert_on_hit_probability=insert_on_hit_probability)
    insert_rng = np.random.default_rng(seed)

    current_threshold = fixed_threshold if fixed_threshold is not None else 0.0
    calibrated = threshold_mode in ("fixed", "recalibrating")  # both start pre-calibrated (transplanted value)
    frozen_after_warmup_done = False
    since_last_recal = 0

    t0 = time.time()
    for i, (record, embedding) in enumerate(zip(records, embeddings)):
        match = store.query(embedding)

        if match is None or match.similarity < TAU_LOW:
            outcome = "miss"
            result.n_direct_miss += 1
        elif match.similarity >= TAU_HIGH:
            outcome = "hit"
            result.n_direct_hit += 1
        else:
            score, _latency_ms = verifier.score(record, match.entry)
            correct = match.entry.equivalence_id == record.equivalence_id

            if not calibrated:
                # Bootstrap: fail closed, same as cacheverifier-service's
                # real cold_start_mode default -- see module docstring.
                # Recorded threshold is +inf so a later `score >
                # threshold_used` re-derivation (chunk_summary) reproduces
                # this forced-reject decision exactly, not just usually.
                approved = False
                observed_threshold = float("inf")
            else:
                approved = score > current_threshold  # strict, matching gray_zone_risk_curve's accept rule
                observed_threshold = current_threshold

            result.observations.append(
                GrayZoneObservation(
                    stream_position=i,
                    score=score,
                    correct=correct,
                    threshold_used=observed_threshold,
                    query_id=record.query_id if capture_text else None,
                    query_text=record.query if capture_text else None,
                    candidate_query_id=match.entry.query_id if capture_text else None,
                    candidate_query_text=match.entry.query if capture_text else None,
                    candidate_answer_text=match.entry.answer if capture_text else None,
                )
            )

            if approved:
                outcome = "hit"
                result.n_gray_hit += 1
            else:
                outcome = "miss"
                result.n_gray_miss += 1

            n_obs = len(result.observations)

            if threshold_mode == "frozen_after_warmup" and not frozen_after_warmup_done and n_obs >= WARMUP_N:
                cal_scores = np.array([o.score for o in result.observations[-WARMUP_N:]])
                cal_labels = np.array([1 if o.correct else 0 for o in result.observations[-WARMUP_N:]])
                picked = crc_select_threshold(cal_scores, cal_labels, ALPHA)
                current_threshold = picked if picked is not None else 0.0
                result.post_warmup_threshold = current_threshold
                calibrated = True
                frozen_after_warmup_done = True

            elif threshold_mode == "recalibrating":
                since_last_recal += 1
                if n_obs >= WARMUP_N and since_last_recal >= RECAL_EVERY:
                    window = result.observations[-RECAL_WINDOW:]
                    cal_scores = np.array([o.score for o in window])
                    cal_labels = np.array([1 if o.correct else 0 for o in window])
                    picked = crc_select_threshold(cal_scores, cal_labels, ALPHA)
                    if picked is not None:
                        current_threshold = picked
                    since_last_recal = 0

        result.outcomes.append(outcome)

        # A miss always inserts (unchanged across every regime this module
        # has ever tested); only whether a HIT also inserts is governed by
        # insert_on_hit_probability -- see run_regime's docstring.
        should_insert = outcome == "miss" or insert_rng.random() < insert_on_hit_probability
        if should_insert:
            store.insert(
                CacheEntry(
                    query_id=record.query_id,
                    query=record.query,
                    answer=record.answer,
                    equivalence_id=record.equivalence_id,
                ),
                embedding,
            )

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{name}] {i + 1}/{len(records)} records, "
                f"{len(result.observations)} gray-zone so far, {elapsed:.0f}s elapsed",
                flush=True,
            )

    result.final_store_size = len(store)
    return result


def bootstrap_risk_difference(
    obs_a: list[GrayZoneObservation],
    obs_b: list[GrayZoneObservation],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Unpaired percentile-bootstrap CI on realized_risk(obs_b) -
    realized_risk(obs_a). Same pattern `cacheverifier.metrics.bootstrap
    .bootstrap_ci` uses throughout this project's Go/No-Go comparisons
    ("an improvement only counts if its CI does not overlap the
    baseline's"), adapted to GrayZoneObservation and made unpaired: once
    self-selection kicks in, two regimes' match sequences diverge, so
    there's no natural 1:1 pairing between a baseline observation and a
    self-select observation at the same gray-zone index -- obs_a and obs_b
    are resampled independently, not element-for-element.

    Callers should pass observations with the warm-up period already
    excluded (see WARMUP_N) -- regime 1 alone goes through a fail-closed
    bootstrap before its first calibration, which regimes 2/3 skip (they
    start from a transplanted already-calibrated threshold), so including
    it would compare a fail-closed period against an already-calibrated
    one rather than isolating the effect actually being tested.
    """

    def accepted_incorrect(obs: list[GrayZoneObservation]) -> tuple[np.ndarray, np.ndarray]:
        accepted = np.array([o.score > o.threshold_used for o in obs])
        incorrect = np.array([not o.correct for o in obs])
        return accepted, incorrect

    acc_a, inc_a = accepted_incorrect(obs_a)
    acc_b, inc_b = accepted_incorrect(obs_b)
    n_a, n_b = len(obs_a), len(obs_b)

    def risk(accepted: np.ndarray, incorrect: np.ndarray) -> float:
        return float((accepted & incorrect).mean()) if len(accepted) else 0.0

    point_a = risk(acc_a, inc_a)
    point_b = risk(acc_b, inc_b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        diffs[i] = risk(acc_b[idx_b], inc_b[idx_b]) - risk(acc_a[idx_a], inc_a[idx_a])

    alpha = 1.0 - confidence
    ci_low, ci_high = (float(x) for x in np.quantile(diffs, [alpha / 2, 1.0 - alpha / 2]))
    return {
        "n_a": n_a,
        "n_b": n_b,
        "risk_a": point_a,
        "risk_b": point_b,
        "diff": point_b - point_a,
        "diff_ci_low": ci_low,
        "diff_ci_high": ci_high,
        "significant": bool(ci_low > 0 or ci_high < 0),
    }


def chunk_summary(observations: list[GrayZoneObservation], n_chunks: int = N_CHUNKS) -> list[dict]:
    n = len(observations)
    if n == 0:
        return []
    chunk_size = max(1, n // n_chunks)
    chunks = [observations[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]
    if n % n_chunks:
        chunks[-1] = observations[(n_chunks - 1) * chunk_size :]
    chunks = [c for c in chunks if c]

    rows = []
    for idx, chunk in enumerate(chunks):
        accepted = np.array([o.score > o.threshold_used for o in chunk])
        incorrect = np.array([not o.correct for o in chunk])
        n_c = len(chunk)
        rows.append(
            {
                "chunk": idx,
                "n": n_c,
                "reuse_rate": float(accepted.mean()),
                "realized_risk": float((accepted & incorrect).mean()),
                "exceeds_alpha": bool((accepted & incorrect).mean() > ALPHA),
                "mean_threshold_used": float(np.mean([o.threshold_used for o in chunk])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="quora", choices=["quora", "lmarena", "search_queries_corrected"])
    parser.add_argument("--config", default=None, help="Defaults to configs/{dataset}.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-resamples", type=int, default=2000, help="Bootstrap resamples for risk-difference CIs")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config or f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    print(f"Loaded {len(records)} records from {cfg.processed_path}")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    print("Embedding records once (shared across all three regimes -- embeddings don't depend on policy)...")
    embeddings = embedder.embed(records)

    verifier = CrossEncoderVerifier()

    print("\n=== Regime 1: baseline (insert-always, frozen-after-warmup threshold) ===")
    r1 = run_regime(
        "baseline", records, embeddings, verifier, insert_on_hit_probability=1.0, threshold_mode="frozen_after_warmup"
    )
    if r1.post_warmup_threshold is None:
        raise RuntimeError(
            f"regime 1 never reached WARMUP_N={WARMUP_N} gray-zone observations "
            f"(only got {len(r1.observations)}) -- reduce WARMUP_N or use a larger dataset"
        )
    frozen_threshold = r1.post_warmup_threshold
    print(f"Regime 1 frozen threshold (post-warmup): {frozen_threshold}")

    print("\n=== Regime 2: self-selection only (write-on-miss-only, threshold fixed at regime 1's value) ===")
    r2 = run_regime(
        "self_select",
        records,
        embeddings,
        verifier,
        insert_on_hit_probability=0.0,
        threshold_mode="fixed",
        fixed_threshold=frozen_threshold,
    )

    print("\n=== Regime 3: self-selection + online recalibration (write-on-miss-only, sliding-window recal) ===")
    r3 = run_regime(
        "self_select_recal",
        records,
        embeddings,
        verifier,
        insert_on_hit_probability=0.0,
        threshold_mode="recalibrating",
        fixed_threshold=frozen_threshold,
    )

    print(f"\n{'regime':>20} | {'n_gray_zone':>11} {'n_direct_hit':>12} {'n_direct_miss':>13} {'final_cache_n':>13}")
    for r in (r1, r2, r3):
        print(
            f"{r.name:>20} | {len(r.observations):>11} {r.n_direct_hit:>12} {r.n_direct_miss:>13} {r.final_store_size:>13}"
        )

    results = {
        "dataset": args.dataset,
        "alpha": ALPHA,
        "tau_low": TAU_LOW,
        "tau_high": TAU_HIGH,
        "warmup_n": WARMUP_N,
        "recal_every": RECAL_EVERY,
        "recal_window": RECAL_WINDOW,
        "frozen_threshold_from_regime1": frozen_threshold,
        "regimes": {},
    }
    for r in (r1, r2, r3):
        chunks = chunk_summary(r.observations)
        overall_accepted = np.array([o.score > o.threshold_used for o in r.observations])
        overall_incorrect = np.array([not o.correct for o in r.observations])
        overall_risk = float((overall_accepted & overall_incorrect).mean()) if len(r.observations) else float("nan")
        print(f"\n--- {r.name}: overall realized risk = {overall_risk:.4f} (target alpha = {ALPHA}) ---")
        for row in chunks:
            flag = " *** EXCEEDS ALPHA ***" if row["exceeds_alpha"] else ""
            print(
                f"  chunk={row['chunk']} n={row['n']} reuse_rate={row['reuse_rate']:.4f} "
                f"realized_risk={row['realized_risk']:.4f} mean_threshold={row['mean_threshold_used']:.3f}{flag}"
            )
        results["regimes"][r.name] = {
            "insert_on_hit_probability": r.insert_on_hit_probability,
            "n_gray_zone": len(r.observations),
            "n_direct_hit": r.n_direct_hit,
            "n_direct_miss": r.n_direct_miss,
            "final_store_size": r.final_store_size,
            "overall_realized_risk": overall_risk,
            "chunks": chunks,
        }

    # Bootstrap comparisons, post-warm-up observations only (see
    # bootstrap_risk_difference's docstring for why regime 1's own
    # WARMUP_N-observation fail-closed bootstrap period must be excluded
    # from all three regimes before comparing them).
    print(f"\n=== Bootstrap risk-difference comparisons (post-warmup only, n_resamples={args.n_resamples}) ===")
    post_warmup = {r.name: r.observations[WARMUP_N:] for r in (r1, r2, r3)}
    comparisons = {
        "self_select_vs_baseline": ("baseline", "self_select"),
        "self_select_recal_vs_self_select": ("self_select", "self_select_recal"),
        "self_select_recal_vs_baseline": ("baseline", "self_select_recal"),
    }
    results["bootstrap_comparisons"] = {}
    for label, (name_a, name_b) in comparisons.items():
        cmp = bootstrap_risk_difference(
            post_warmup[name_a], post_warmup[name_b], n_resamples=args.n_resamples, seed=0
        )
        results["bootstrap_comparisons"][label] = cmp
        sig = "SIGNIFICANT" if cmp["significant"] else "not significant"
        print(
            f"  {label}: risk {cmp['risk_a']:.4f} -> {cmp['risk_b']:.4f}  "
            f"diff={cmp['diff']:+.4f}  95% CI=[{cmp['diff_ci_low']:+.4f}, {cmp['diff_ci_high']:+.4f}]  ({sig})"
        )

    out_path = Path(args.output) if args.output else Path(f"results/crc_closed_loop_self_selection_{args.dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
