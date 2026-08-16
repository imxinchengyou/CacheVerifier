"""Diagnose why Quora's honest-calibration ablation (direction 1) shows an
almost-invariant Youden's J threshold across tau_low (~3.7-4.2, barely
moving) and a small net loss vs Group A, unlike LmArena/SearchQueries where
honest calibration reversed a worse original result. Reuses the cached
gray-zone scores from the direction-1 run -- no new scoring needed.

Hypothesis under test: Quora's `answer` field is `answer = query` of the
matched historical question (not a real LLM-generated response, per
scripts/convert_quora_dataset.py), so the cross-encoder is scoring
(question, other_question) pairs -- structurally different from the
(query, informative-passage) relevance task ms-marco cross-encoders are
trained for. If that compresses the score distribution or weakens
label separation, it would explain both symptoms directly.

Usage:
    python scripts/diagnose_quora_threshold.py --config configs/quora.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.run_verified import _slug, CACHE_DIR
from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored, resolve_candidate

TAU_LOW = 0.80
TAU_HIGH = 0.97


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
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
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    dataset_path = Path(cfg.processed_path)
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}.trace.json"
    trace = load_match_trace(trace_cache_path)

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    verifier_key = f"cross_encoder_{_slug(cross_encoder_model)}"
    scored_cache_path = (
        CACHE_DIR
        / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__{verifier_key}"
          f"__lo{TAU_LOW}__hi{TAU_HIGH}.scored.json"
    )
    scored = load_scored(scored_cache_path)

    gz_indices = [i for i in scored if trace[i].similarity is not None and TAU_LOW <= trace[i].similarity < TAU_HIGH]
    scores = np.array([scored[i].score for i in gz_indices])
    labels = np.array([1 if trace[i].would_be_correct else 0 for i in gz_indices])

    print(f"n gray-zone pairs: {len(gz_indices)}")
    print(f"positive rate: {labels.mean():.4f}  (n_pos={int(labels.sum())}, n_neg={int((1-labels).sum())})")
    print(f"overall AUC: {roc_auc(scores, labels):.4f}")
    print()
    print("Score distribution, overall:")
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{q:3d}: {np.percentile(scores, q):+.3f}")
    print()
    print("Score distribution, positives (would_be_correct=True):")
    pos_scores = scores[labels == 1]
    for q in [0, 10, 25, 50, 75, 90, 100]:
        print(f"  p{q:3d}: {np.percentile(pos_scores, q):+.3f}")
    print()
    print("Score distribution, negatives (would_be_correct=False):")
    neg_scores = scores[labels == 0]
    for q in [0, 10, 25, 50, 75, 90, 100]:
        print(f"  p{q:3d}: {np.percentile(neg_scores, q):+.3f}")
    print()
    print(f"median(pos) - median(neg) = {np.median(pos_scores) - np.median(neg_scores):+.3f}")
    print(f"overlap check: pos p25={np.percentile(pos_scores,25):+.3f}  neg p75={np.percentile(neg_scores,75):+.3f}"
          f"  (if pos_p25 < neg_p75, the middle 50% of each class overlaps)")

    # Sample a few example (query, candidate) pairs across the score range to inspect qualitatively.
    print()
    print("=== Example pairs across the score range ===")
    order = np.argsort(scores)
    picks = {
        "lowest score": order[0],
        "median score": order[len(order) // 2],
        "highest score": order[-1],
    }
    for label, idx_in_order in picks.items():
        i = gz_indices[idx_in_order]
        record = records[i]
        candidate = resolve_candidate(records, trace[i])
        print(f"\n[{label}] score={scored[i].score:+.3f}  would_be_correct={trace[i].would_be_correct}")
        print(f"  query:     {record.query!r}")
        print(f"  candidate: {candidate.query!r}")
        print(f"  candidate.answer (=candidate.query per convert_quora_dataset.py): {candidate.answer!r}")

    # Check whether "high score but labeled wrong" is systematic (mislabeled
    # ground truth) rather than a one-off from the single highest-score pick.
    print()
    print("=== Top 15 highest-scoring pairs, by label ===")
    top15 = order[-15:][::-1]
    for rank, idx_in_order in enumerate(top15, 1):
        i = gz_indices[idx_in_order]
        record = records[i]
        candidate = resolve_candidate(records, trace[i])
        print(f"{rank:2d}. score={scored[i].score:+.3f}  correct={trace[i].would_be_correct}  "
              f"q={record.query!r}  c={candidate.query!r}")

    n_top100_correct = sum(1 for idx_in_order in order[-100:] for i in [gz_indices[idx_in_order]] if trace[i].would_be_correct)
    print(f"\nTop 100 highest-scoring pairs: {n_top100_correct}/100 labeled correct "
          f"(vs {labels.mean()*100:.1f}% base rate over the whole gray zone)")


if __name__ == "__main__":
    main()
