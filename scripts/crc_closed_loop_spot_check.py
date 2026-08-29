"""Qualitative spot-check for `scripts/crc_closed_loop_self_selection.py`'s
LmArena finding: self-selection alone (write-on-miss-only, fixed threshold)
significantly increased realized risk vs. insert-always (diff=+0.0305, 95%
CI [+0.0274,+0.0337], results/crc_closed_loop_self_selection_lmarena.json).
The hypothesized mechanism (matching PAPER.md §8 future-work item 11(d)'s
original wording): a candidate that keeps getting correctly matched and
approved never triggers a fresh write, so if THAT candidate's own record is
the true match for some later query, the later query can only match
against some other, imperfect substitute already in the (sparser)
self-selecting cache -- the correct candidate was silently crowded out by
its own earlier success, not literally removed.

This is directly testable without guessing: for each false-accept in the
self-selecting regime (approved a candidate that was wrong), find the true
match (an earlier record with the same equivalence_id) and check whether
THAT record's own occurrence was ever written into the self-selecting
store (RegimeResult.outcomes[j] == "miss") or was itself served from an
existing entry without writing one (outcomes[j] == "hit" -- crowded out).
An off-the-shelf ANN recall failure (the true match WAS written but HNSW's
approximate search just didn't surface it) is the alternative explanation
this same check also distinguishes: outcomes[j] == "miss" but the false
accept still happened points there instead.

Only reruns regime 1 (baseline) and regime 2 (self_select) -- regime 3
(online recalibration) isn't needed for this specific mechanism check.

Usage:
    python scripts/crc_closed_loop_spot_check.py --dataset lmarena --n-examples 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

from crc_closed_loop_self_selection import WARMUP_N, run_regime


def find_prior_equivalents(records, position: int) -> list[int]:
    """Every earlier record sharing this query's equivalence_id, most
    recent first -- any one of them being in the self-selecting store
    would have given a correct match to surface. Checking every prior
    equivalent, not just the nearest, avoids overstating "crowded out": if
    even one earlier equivalent record WAS written, the correct answer was
    still reachable in principle (a different explanation -- e.g. HNSW's
    approximate search just didn't surface it -- is more likely than
    crowding-out for that case)."""
    target = records[position].equivalence_id
    return [j for j in range(position - 1, -1, -1) if records[j].equivalence_id == target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="lmarena", choices=["quora", "lmarena", "search_queries_corrected"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-examples", type=int, default=15)
    args = parser.parse_args()

    cfg = load_dataset_config(args.config or f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    print(f"Loaded {len(records)} records from {cfg.processed_path}")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embeddings = embedder.embed(records)
    verifier = CrossEncoderVerifier()

    print("=== Regime 1: baseline (capture_text=True) ===")
    r1 = run_regime(
        "baseline",
        records,
        embeddings,
        verifier,
        insert_on_hit_probability=1.0,
        threshold_mode="frozen_after_warmup",
        capture_text=True,
    )
    frozen_threshold = r1.post_warmup_threshold
    print(f"frozen threshold: {frozen_threshold}")

    print("=== Regime 2: self_select (capture_text=True) ===")
    r2 = run_regime(
        "self_select",
        records,
        embeddings,
        verifier,
        insert_on_hit_probability=0.0,
        threshold_mode="fixed",
        fixed_threshold=frozen_threshold,
        capture_text=True,
    )

    false_accepts = [
        o for o in r2.observations if o.stream_position >= WARMUP_N and o.score > o.threshold_used and not o.correct
    ]
    print(f"\n{len(false_accepts)} false-accepts in self_select (post-warmup)")

    crowded_out = 0
    ann_recall_miss = 0
    no_earlier_equivalent = 0
    shown = 0

    for obs in false_accepts:
        prior = find_prior_equivalents(records, obs.stream_position)
        if not prior:
            no_earlier_equivalent += 1
            continue

        written_flags = [r2.outcomes[j] == "miss" for j in prior]
        any_written = any(written_flags)
        nearest_j = prior[0]

        if not any_written:
            crowded_out += 1
            category = f"CROWDED OUT (all {len(prior)} prior equivalent(s) were self_select hits, never written)"
        else:
            ann_recall_miss += 1
            category = (
                f"{sum(written_flags)}/{len(prior)} prior equivalent(s) WERE written -- not a crowding-out case"
            )

        if shown < args.n_examples:
            shown += 1
            print(f"\n--- example {shown}: {category} ---")
            print(f"  query (pos {obs.stream_position}): {obs.query_text!r}")
            print(f"  self_select matched instead (score={obs.score:.2f}, threshold={obs.threshold_used:.2f}):")
            print(f"    candidate query: {obs.candidate_query_text!r}")
            print(f"    candidate answer: {obs.candidate_answer_text!r}")
            print(f"  nearest true match (pos {nearest_j}, self_select outcome there: {r2.outcomes[nearest_j]}):")
            print(f"    query: {records[nearest_j].query!r}")
            print(f"    answer: {records[nearest_j].answer!r}")

    print(f"\n=== Summary over all {len(false_accepts)} post-warmup false-accepts ===")
    print(f"  no earlier equivalent record at all (first occurrence): {no_earlier_equivalent}")
    print(f"  true match WAS written to self_select's store (not crowding-out): {ann_recall_miss}")
    print(f"  CROWDED OUT (true match was itself a hit, never written): {crowded_out}")
    denom = ann_recall_miss + crowded_out
    if denom:
        print(f"  crowded-out share of explainable false-accepts: {crowded_out / denom:.1%}")


if __name__ == "__main__":
    main()
