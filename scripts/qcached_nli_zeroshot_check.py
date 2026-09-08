"""Tests whether an NLI-pretrained cross-encoder does better than the
MS-MARCO-ranking-pretrained one (cross-encoder/ms-marco-MiniLM-L6-v2, this
project's default) at catching the negation/entity-swap/quantity-swap
adversarial axes, WITH ZERO FINE-TUNING -- the sharpest possible test of
"is the persistent entity_swap/negation weakness a pretraining-objective
mismatch, not a data problem," since §5.12 already ruled out "just needs a
bigger/broader-trained ranking model" (RESEARCH_PROPOSAL.md / PAPER.md) and
RESEARCH_PROPOSAL.md's own list of untested architecture directions named
NLI models specifically.

Model: cross-encoder/nli-MiniLM2-L6-H768 -- 6 layers, matching our default
verifier's depth, so any difference is attributable to the pretraining
OBJECTIVE (entailment/contradiction/neutral on SNLI+MultiNLI) rather than
model capacity, controlling for the confound §5.12 already tested and ruled
out for capacity/training-distribution within the ranking-model family.

Score construction: NLI models emit 3 logits (id2label: 0=contradiction,
1=entailment, 2=neutral). This script uses softmax probabilities and defines
score = P(entailment) - P(contradiction) as the natural analog of this
project's single relevance score (positive = looks safe to reuse, negative =
looks contradictory), decision threshold 0.

Evaluates:
  1. Natural LmArena held-out gray-zone test set (AUC vs would_be_correct).
  2. The 306-sample adversarial red-team set (false-accept rate by category,
     matching qcached_redteam_eval.py's off-the-shelf number of 84.0% for
     direct comparison).

Usage:
    python scripts/qcached_nli_zeroshot_check.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def proportion_bootstrap_ci(successes: int, n: int, n_resamples: int = 5000, seed: int = 0):
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array([1] * successes + [0] * (n - successes))
    resample_means = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(resample_means, [0.025, 0.975])
    return float(lo), float(hi)


def nli_scores(model, pairs, batch_size=64):
    """score = P(entailment) - P(contradiction), softmax over the 3 raw logits."""
    raw = np.array(model.predict(pairs, batch_size=batch_size, show_progress_bar=False, apply_softmax=False))
    exp = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    # id2label: 0=contradiction, 1=entailment, 2=neutral
    return probs[:, 1] - probs[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nli-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    parser.add_argument("--natural-stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--tau-low", type=float, default=0.80)
    parser.add_argument("--output", default="results/qcached_nli_zeroshot_check.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    log(f"Loading NLI model {args.nli_model!r}...")
    model = CrossEncoder(args.nli_model, device=device)
    log(f"  num_labels={model.config.num_labels}  id2label={model.config.id2label}")

    # --- Part 1: natural LmArena held-out AUC ---
    log("\n=== Part 1: natural LmArena held-out test set ===")
    natural = json.loads(Path(args.natural_stash).read_text(encoding="utf-8"))
    test_rows = natural["test"]
    labels = np.array([1 if row[-1] else 0 for row in test_rows])
    pairs = [(row[1], row[3]) for row in test_rows]  # (query, answer)
    t0 = time.time()
    scores = nli_scores(model, pairs)
    auc = roc_auc(scores, labels)
    log(f"  NLI zero-shot natural AUC = {auc:.4f} (scored {len(pairs)} pairs in {time.time() - t0:.1f}s)")

    # --- Part 2: adversarial red-team set, false-accept rate by category ---
    log("\n=== Part 2: adversarial red-team set (306-sample, off-the-shelf comparison: 84.0%) ===")
    data = json.loads(Path(args.redteam_results).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= args.tau_low]
    log(f"  {len(triples)} triples clear tau_low={args.tau_low}")
    adv_pairs = [(t["query_a"], t["answer_b"]) for t in triples]
    t0 = time.time()
    adv_scores = nli_scores(model, adv_pairs)
    fa = adv_scores >= 0.0
    by_category = {}
    for t, f in zip(triples, fa):
        c = t["category"]
        by_category.setdefault(c, {"n": 0, "fa": 0})
        by_category[c]["n"] += 1
        by_category[c]["fa"] += int(f)
    total_n, total_fa = len(triples), int(fa.sum())
    ci = proportion_bootstrap_ci(total_fa, total_n)
    log(f"  TOTAL false-accept: {total_fa}/{total_n} = {total_fa / total_n * 100:.1f}% "
        f"(95% CI [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%])  (scored in {time.time() - t0:.1f}s)")
    for cat, s in sorted(by_category.items()):
        log(f"    {cat:>14}: {s['fa']:>3}/{s['n']:<3} = {s['fa'] / s['n'] * 100:5.1f}%")

    output = {
        "nli_model": args.nli_model,
        "natural_auc": auc,
        "adversarial_total_fa_pct": total_fa / total_n * 100,
        "adversarial_total_fa_ci95": ci,
        "adversarial_by_category": by_category,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
