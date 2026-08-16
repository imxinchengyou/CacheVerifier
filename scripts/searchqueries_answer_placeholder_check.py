"""Sanity check: is SearchQueries' Group D near-random AUC (PAPER.md 0.4881)
explained by a data issue rather than an "axis problem" or distribution
shift?

`data/processed/search_queries.jsonl`'s `answer` field is the literal
constant string "Not required for the benchmark because of the id_set" for
ALL 150,000 records (verified: only one unique value in the file) -- carried
over verbatim from the upstream `vCache/SemBenchmarkSearchQueries` HF
dataset's `response_llama_3_8b` column, which was apparently never
populated because vCache's own harness only needs `id_set` equality, never
answer content. `cacheverifier.verifiers.cross_encoder_verifier
.CrossEncoderVerifier.score()` feeds this constant string as
`candidate.answer` into the cross-encoder for every single gray-zone pair,
so the verifier's second input carries zero information about which
candidate was actually matched.

This script re-scores a sample of the real gray-zone pairs with the SAME
off-the-shelf, un-fine-tuned verifier under two conditions:
  - "placeholder": candidate_answer = the constant string actually on disk
    (reproduces what Group D actually measured)
  - "echoed_query": candidate_answer = the matched candidate's own query
    text (the same substitute Quora already uses per PAPER.md Section 6.2,
    chosen here only as a quick, already-precedented way to give the
    verifier SOME per-record distinguishing text, not because it's the
    "correct" fix)

If AUC jumps sharply under "echoed_query" relative to "placeholder", that's
strong evidence the near-random Group D number is a placeholder-answer
artifact, not evidence about embeddings/verifiers failing on short queries.

Usage:
    python scripts/searchqueries_answer_placeholder_check.py --n-sample 3000
"""

import argparse
import random
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

TAU_LOW = 0.80
TAU_HIGH = 0.97
SEED = 0


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n_resamples: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = roc_auc(scores[idx], labels[idx])
    lo, hi = np.nanquantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/search_queries.yaml")
    parser.add_argument("--n-sample", type=int, default=3000, help="Gray-zone pairs to re-score (subsample for speed)")
    parser.add_argument("--verifier-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    args = parser.parse_args()

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cacheverifier.config import load_dataset_config
    from cacheverifier.data.loaders import load_jsonl
    from cacheverifier.experiments.run_baselines import build_embedder
    from cacheverifier.experiments.verified_sweep import build_match_trace, resolve_candidate

    cfg = load_dataset_config(args.config)
    records = load_jsonl(Path(cfg.processed_path))
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    print(f"Loaded {len(records)} records from {cfg.processed_path}")

    unique_answers = {r.answer for r in records}
    print(f"Unique 'answer' strings in this dataset: {len(unique_answers)}")
    if len(unique_answers) <= 3:
        for a in unique_answers:
            print(f"  {a!r}")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    print("Building match trace (this reuses the exact policy-independent trace Group A/C/D share)...")
    t0 = time.time()
    trace = build_match_trace(records, embedder)
    print(f"  done in {time.time() - t0:.1f}s")

    gray_zone = []
    for i, (record, t) in enumerate(zip(records, trace)):
        if t.similarity is not None and TAU_LOW <= t.similarity < TAU_HIGH:
            candidate = resolve_candidate(records, t)
            gray_zone.append((record, candidate, t.would_be_correct))
    print(f"Gray zone [{TAU_LOW}, {TAU_HIGH}): {len(gray_zone)} pairs total")

    rng = random.Random(SEED)
    sample = gray_zone if len(gray_zone) <= args.n_sample else rng.sample(gray_zone, args.n_sample)
    n_pos = sum(1 for _, _, label in sample if label)
    print(f"Scoring a sample of {len(sample)} pairs ({n_pos} correct, {len(sample) - n_pos} incorrect)")

    print(f"Loading verifier {args.verifier_model!r}...")
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.verifier_model)

    labels = np.array([1 if label else 0 for _, _, label in sample])

    print("Scoring with the answer field AS STORED ON DISK ('placeholder' condition)...")
    t0 = time.time()
    placeholder_scores = np.array(
        model.predict([(r.query, c.answer) for r, c, _ in sample], show_progress_bar=False)
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("Scoring with candidate_answer replaced by the matched query's own text ('echoed_query' condition)...")
    t0 = time.time()
    echoed_scores = np.array(
        model.predict([(r.query, c.query) for r, c, _ in sample], show_progress_bar=False)
    )
    print(f"  done in {time.time() - t0:.1f}s")

    for name, scores in [("placeholder (what Group D actually measured)", placeholder_scores),
                          ("echoed_query (Quora-style substitute)", echoed_scores)]:
        auc = roc_auc(scores, labels)
        lo, hi = bootstrap_auc_ci(scores, labels)
        print(f"\n{name}")
        print(f"  AUC = {auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"  score std = {scores.std():.4f}  (near-zero std would itself confirm 'no signal')")


if __name__ == "__main__":
    main()
