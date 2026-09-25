"""PAPER_B_FORMAL_RISK_CONTROL.md §4.4: which calibration data source keeps a CRC
gate at alpha inside the self-selection closed loop of PAPER.md §5.16, when
labels are only available for part of the traffic?

Why this exists. §5.16's regime 3 ("self_select_recal",
scripts/crc_closed_loop_self_selection.py) recalibrates from the TRUE label
of EVERY gray-zone request, including approved ones -- which in production
have no label (the cached answer was served, the LLM never ran). Several
readers of the §5.16 write-up suggested a random "shadow holdout" stream as
the fix. In this simulator a holdout is just a random subset of what
regime 3 already sees, so it cannot beat regime 3 (hypothesis H1); its real
value is under realistic labels. This script compares, on the same
write-on-miss closed loop and the same recalibration schedule:

  S0   oracle: true labels of every gray-zone request (== §5.16 regime 3;
       the --regression-check flag asserts identical observations)
  S1   reject-only: labels only for gray-zone requests the gate rejected
       (they went to the LLM anyway, so comparing answers is free)
  S2   reject labels + random audit of approved gray-zone requests at rate p
       (audited = served fresh, labelled, written like a miss), weighted 1/p
       in a weighted CRC (cacheverifier.metrics.core.crc_select_threshold_weighted)
       -- this is the random audit cacheverifier-service implements
  S3a  shadow holdout: each request joins the holdout with probability h;
       holdout requests read the cache (candidate, score, label) but never go
       through the gate and never write; calibration uses holdout samples only
  S3b  as S3a, but a request only joins if its (normalized) query text has not
       been in the holdout before -- the per-distinct-query variant

Calibration pools. S0 and S1/S2 recalibrate from the most recent RECAL_WINDOW
pool entries, like §5.16. S3's pool grows by only h per request, so a
same-length window would hold a few dozen points -- below CRC's nontrivial
sample size at alpha=0.02, forcing the reject-everything fallback. To give
the holdout its best shot, S3 calibrates on EVERY holdout sample so far
(more data, but older). S1-S3 wait for MIN_POOL labelled points before their
first recalibration and keep regime 1's transplanted threshold until then,
exactly as S0 does during warm-up.

Risk metrics (post-warm-up gray-zone observations, as in §5.16):
  policy_risk  share of gray-zone requests the GATE approved while wrong
               (audited requests still count -- the gate did approve them;
               holdout requests are excluded -- the gate never saw them)
  served_risk  share actually served a wrong cached answer (audits served fresh)
Plus hit rate over all requests and extra LLM calls (audits, and holdout
requests that the live gate would have served from cache).

Verifier scores are memoized per (query_id, candidate query_id) across all
runs of one invocation -- they are deterministic, and most pairs recur.

Usage:
    python scripts/crc_closed_loop_calibration_sources.py --dataset quora --regression-check
    python scripts/crc_closed_loop_calibration_sources.py --dataset lmarena
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crc_closed_loop_self_selection as base  # noqa: E402
from cacheverifier.cache.store import CacheEntry, VectorCacheStore  # noqa: E402
from cacheverifier.config import load_dataset_config  # noqa: E402
from cacheverifier.data.loaders import load_jsonl  # noqa: E402
from cacheverifier.experiments.run_baselines import build_embedder  # noqa: E402
from cacheverifier.metrics.core import crc_select_threshold, crc_select_threshold_weighted  # noqa: E402

ALPHA, TAU_LOW, TAU_HIGH = base.ALPHA, base.TAU_LOW, base.TAU_HIGH
WARMUP_N, RECAL_EVERY, RECAL_WINDOW = base.WARMUP_N, base.RECAL_EVERY, base.RECAL_WINDOW
MIN_POOL = 200
"""S1-S3: labelled points needed before the first recalibration. Well above
CRC's nontrivial floor at alpha=0.02 (~49), so an early recalibration isn't
just the lambda_max fallback."""
RATES = (0.01, 0.02, 0.05)


class ScoreCache:
    def __init__(self, verifier):
        self.verifier = verifier
        self.memo: dict[tuple[str, str], float] = {}
        self.calls = 0

    def score(self, record, entry) -> float:
        key = (record.query_id, entry.query_id)
        if key not in self.memo:
            self.memo[key], _ = self.verifier.score(record, entry)
            self.calls += 1
        return self.memo[key]


@dataclass
class PolicyObs:
    stream_position: int
    score: float
    correct: bool
    threshold_used: float
    approved: bool
    audited: bool = False


@dataclass
class SourceResult:
    name: str
    source: str
    rate: float | None
    observations: list[PolicyObs] = field(default_factory=list)
    pool_size_final: int = 0
    n_recalibrations: int = 0
    n_fallback_recalibrations: int = 0
    n_served_hits: int = 0
    n_requests: int = 0
    n_audits: int = 0
    n_holdout: int = 0
    n_holdout_gray: int = 0
    n_holdout_would_hit: int = 0
    thresholds: list[float] = field(default_factory=list)


def run_source(name, source, rate, records, embeddings, scorer, frozen_threshold, seed=0) -> SourceResult:
    assert source in ("S0", "S1", "S2", "S3a", "S3b")
    store = VectorCacheStore(dim=embeddings.shape[1])
    res = SourceResult(name=name, source=source, rate=rate)
    insert_rng = np.random.default_rng(seed)  # same stream as §5.16's run_regime (only consumed on hits)
    side_rng = np.random.default_rng(seed + 1_000_003)  # audit / holdout coins, independent of insert_rng
    threshold = frozen_threshold
    since_last_recal = 0
    pool_scores: list[float] = []
    pool_labels: list[int] = []
    pool_weights: list[float] = []
    holdout_queries: set[str] = set()

    t0 = time.time()
    for i, (record, embedding) in enumerate(zip(records, embeddings)):
        res.n_requests += 1
        holdout = False
        if source in ("S3a", "S3b") and side_rng.random() < rate:
            key = " ".join(record.query.lower().split())
            if source == "S3a" or key not in holdout_queries:
                holdout = True
                holdout_queries.add(key)

        match = store.query(embedding)

        if holdout:
            res.n_holdout += 1
            if match is not None and match.similarity >= TAU_HIGH:
                res.n_holdout_would_hit += 1
            elif match is not None and match.similarity >= TAU_LOW:
                s = scorer.score(record, match.entry)
                correct = match.entry.equivalence_id == record.equivalence_id
                pool_scores.append(s)
                pool_labels.append(1 if correct else 0)
                pool_weights.append(1.0)
                res.n_holdout_gray += 1
                if s > threshold:
                    res.n_holdout_would_hit += 1
            continue  # served fresh, never written, never gated

        if match is None or match.similarity < TAU_LOW:
            outcome = "miss"
        elif match.similarity >= TAU_HIGH:
            outcome = "hit"
        else:
            s = scorer.score(record, match.entry)
            correct = match.entry.equivalence_id == record.equivalence_id
            approved = s > threshold
            audited = source == "S2" and approved and side_rng.random() < rate
            res.observations.append(PolicyObs(i, s, correct, threshold, approved, audited))
            if audited:
                res.n_audits += 1
            outcome = "hit" if approved and not audited else "miss"

            label = 1 if correct else 0
            if source == "S0":
                pool_scores.append(s), pool_labels.append(label), pool_weights.append(1.0)
            elif source in ("S1", "S2") and not approved:
                pool_scores.append(s), pool_labels.append(label), pool_weights.append(1.0)
            elif source == "S2" and audited:
                pool_scores.append(s), pool_labels.append(label), pool_weights.append(1.0 / rate)

            since_last_recal += 1
            n_obs = len(res.observations)
            ready = n_obs >= WARMUP_N if source == "S0" else (n_obs >= WARMUP_N and len(pool_scores) >= MIN_POOL)
            if ready and since_last_recal >= RECAL_EVERY:
                if source in ("S3a", "S3b"):
                    sl = slice(None)  # every holdout sample so far -- see module docstring
                else:
                    sl = slice(-RECAL_WINDOW, None)
                sc, lb, wt = np.array(pool_scores[sl]), np.array(pool_labels[sl]), np.array(pool_weights[sl])
                if source == "S0":
                    picked = crc_select_threshold(sc, lb, ALPHA)  # byte-for-byte the §5.16 call
                else:
                    picked = crc_select_threshold_weighted(sc, lb, wt, ALPHA)
                if picked is not None:
                    if picked >= sc.max():
                        res.n_fallback_recalibrations += 1
                    threshold = picked
                    res.n_recalibrations += 1
                    res.thresholds.append(threshold)
                since_last_recal = 0

        if outcome == "hit":
            res.n_served_hits += 1
        should_insert = outcome == "miss" or insert_rng.random() < 0.0
        if should_insert:
            store.insert(
                CacheEntry(query_id=record.query_id, query=record.query, answer=record.answer, equivalence_id=record.equivalence_id),
                embedding,
            )
        if (i + 1) % 10000 == 0:
            print(f"  [{name}] {i + 1}/{len(records)}  gray={len(res.observations)}  pool={len(pool_scores)}  "
                  f"thr={threshold:.3f}  scorer_calls={scorer.calls}  {time.time() - t0:.0f}s", flush=True)
    res.pool_size_final = len(pool_scores)
    return res


def summarize(r: SourceResult) -> dict:
    post = r.observations[WARMUP_N:]
    acc = np.array([o.approved for o in post])
    aud = np.array([o.audited for o in post])
    inc = np.array([not o.correct for o in post])
    return {
        "source": r.source,
        "rate": r.rate,
        "n_gray_zone_policy": len(r.observations),
        "policy_risk": float((acc & inc).mean()) if len(post) else None,
        "served_risk": float((acc & ~aud & inc).mean()) if len(post) else None,
        "reuse_rate_gray": float(acc.mean()) if len(post) else None,
        "hit_rate_all_requests": r.n_served_hits / r.n_requests,
        "extra_llm_calls": r.n_audits + r.n_holdout_would_hit,
        "extra_llm_call_share": (r.n_audits + r.n_holdout_would_hit) / r.n_requests,
        "n_audits": r.n_audits,
        "n_holdout": r.n_holdout,
        "n_holdout_gray": r.n_holdout_gray,
        "pool_size_final": r.pool_size_final,
        "n_recalibrations": r.n_recalibrations,
        "n_fallback_recalibrations": r.n_fallback_recalibrations,
        "threshold_median": float(np.median(r.thresholds)) if r.thresholds else None,
        "threshold_final": r.thresholds[-1] if r.thresholds else None,
        "chunks": base.chunk_summary([base.GrayZoneObservation(o.stream_position, o.score, o.correct, o.threshold_used) for o in r.observations]),
    }


def bootstrap_vs(a: SourceResult, b: SourceResult, n_resamples: int) -> dict:
    to_base = lambda r: [base.GrayZoneObservation(o.stream_position, o.score, o.correct, o.threshold_used) for o in r.observations[WARMUP_N:]]  # noqa: E731
    return base.bootstrap_risk_difference(to_base(a), to_base(b), n_resamples=n_resamples, seed=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="quora", choices=["quora", "lmarena", "search_queries_corrected"])
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--sources", default="S0,S1,S2,S3a,S3b")
    ap.add_argument("--regression-check", action="store_true",
                    help="also run §5.16's original regime 3 and assert S0 reproduces it observation-for-observation")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    rates = [float(x) for x in args.rates.split(",")]
    sources = args.sources.split(",")

    cfg = load_dataset_config(f"configs/{args.dataset}.yaml")
    records = load_jsonl(cfg.processed_path)
    if cfg.max_samples:
        records = records[: cfg.max_samples]
    embeddings = build_embedder(cfg.embedder, cfg.embedder_model).embed(records)
    verifier = base.CrossEncoderVerifier()  # sentence-transformers picks CUDA automatically when available
    scorer = ScoreCache(verifier)

    print("=== Regime 1 (§5.16 baseline) to get the transplanted threshold ===", flush=True)
    r1 = base.run_regime("baseline", records, embeddings, scorer_as_verifier(scorer), 1.0, "frozen_after_warmup")
    frozen = r1.post_warmup_threshold
    published = json.loads(Path(f"results/crc_closed_loop_self_selection_{args.dataset}.json").read_text())
    print(f"frozen threshold {frozen} (published {published['frozen_threshold_from_regime1']})", flush=True)

    out = {"dataset": args.dataset, "alpha": ALPHA, "warmup_n": WARMUP_N, "recal_every": RECAL_EVERY,
           "recal_window": RECAL_WINDOW, "min_pool": MIN_POOL, "frozen_threshold": frozen,
           "published_frozen_threshold": published["frozen_threshold_from_regime1"], "runs": {}, "vs_S0": {}}

    runs: dict[str, SourceResult] = {}
    for src in sources:
        for rate in ([None] if src in ("S0", "S1") else rates):
            name = src if rate is None else f"{src}@{rate}"
            print(f"\n=== {name} ===", flush=True)
            runs[name] = run_source(name, src, rate, records, embeddings, scorer, frozen)
            out["runs"][name] = summarize(runs[name])
            s = out["runs"][name]
            print(f"  policy_risk={s['policy_risk']:.4f} served_risk={s['served_risk']:.4f} hit={s['hit_rate_all_requests']:.4f} "
                  f"extra_calls={s['extra_llm_call_share']:.4f} pool={s['pool_size_final']} recals={s['n_recalibrations']} "
                  f"fallbacks={s['n_fallback_recalibrations']} thr_med={s['threshold_median']}", flush=True)

    if args.regression_check and "S0" in runs:
        print("\n=== Regression check: §5.16 regime 3 vs S0 ===", flush=True)
        r3 = base.run_regime("self_select_recal", records, embeddings, scorer_as_verifier(scorer), 0.0, "recalibrating", fixed_threshold=frozen)
        same = len(r3.observations) == len(runs["S0"].observations) and all(
            a.stream_position == b.stream_position and a.score == b.score and a.threshold_used == b.threshold_used
            for a, b in zip(r3.observations, runs["S0"].observations))
        r3_risk = float(np.mean([(o.score > o.threshold_used) and not o.correct for o in r3.observations]))
        out["regression_check"] = {"identical_observations": bool(same), "regime3_overall_risk_now": r3_risk,
                                   "regime3_overall_risk_published": published["regimes"]["self_select_recal"]["overall_realized_risk"]}
        print(f"  identical={same}  regime3 risk now={r3_risk:.5f} published={out['regression_check']['regime3_overall_risk_published']:.5f}")
        assert same, "S0 does not reproduce §5.16 regime 3 -- stop and investigate before trusting anything else"

    if "S0" in runs:
        for name, r in runs.items():
            if name != "S0":
                out["vs_S0"][name] = bootstrap_vs(runs["S0"], r, args.n_resamples)
                c = out["vs_S0"][name]
                print(f"  {name} vs S0: policy risk {c['risk_a']:.4f} -> {c['risk_b']:.4f}  diff {c['diff']:+.4f} "
                      f"[{c['diff_ci_low']:+.4f}, {c['diff_ci_high']:+.4f}]{' *' if c['significant'] else ''}", flush=True)

    out["scorer_calls"] = scorer.calls
    path = Path(args.output or f"results/crc_closed_loop_calibration_sources_{args.dataset}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


def scorer_as_verifier(scorer: ScoreCache):
    """Adapter so §5.16's own run_regime shares the memoized scores."""

    class _V:
        def score(self, record, entry):
            return scorer.score(record, entry), 0.0

    return _V()


if __name__ == "__main__":
    main()
