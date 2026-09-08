"""Direction 16 follow-up to the forgetting-sweep (qcached_nli_forgetting_sweep.py):
that experiment showed entity_swap's regression from the zero-shot NLI score's
5.1% false-accept rate to the fully fine-tuned model's 25.6% happens FAST (by
~25% into the single training epoch) and then plateaus -- early stopping
within the same single-output-head training run cannot recover it, because
the loss is a fast overwrite, not a gradual erosion.

This script tests a different idea, flagged as an open limitation there:
instead of replacing the original 3-way entailment head with a fine-tuned
single-output regression head, KEEP BOTH scores and fuse them post-hoc:
  - score_zero: P(entailment) - P(contradiction) from the UNTOUCHED
    cross-encoder/nli-MiniLM2-L6-H768 (query, answer) pair, no fine-tuning
    at all -- this is what gets entity_swap to 5.1% but has weak natural-data
    AUC (0.5889) and a weak spot on direction (40.8%).
  - score_ft: the sigmoid output of the already-trained diff+adversarial
    checkpoint (results/finetuned_verifier_model_lmarena_nli_qcached_adv_
    ablation_qcached_adv_diff) on the diff-mode (query+diff, answer) pair --
    this is what gets natural AUC to 0.86 and four of five axes to <5%, but
    entity_swap only to 25.6%.

Two fusion strategies, both requiring ZERO additional training (both models
already exist, this script only does inference + score combination):
  1. Weighted average: score_zero is rescaled to [0,1] via the fixed,
     data-independent transform (score_zero + 1) / 2 (mathematically valid
     since P(entailment) - P(contradiction) in [-1, 1]), then
     score_fusion = w * score_ft + (1 - w) * score_zero_scaled, decision
     threshold 0.5, swept over w in {0.0, 0.1, ..., 1.0}.
  2. AND-gate veto: accept only if score_ft >= 0.5 AND score_zero >= 0 --
     lets the zero-shot entailment signal veto an accept regardless of how
     confident the fine-tuned score is, instead of averaging it away.

Usage:
    python scripts/qcached_nli_fusion_eval.py
"""

import argparse
import difflib
import json
import time
from pathlib import Path

import numpy as np

TAU_LOW = 0.80
DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"
FUSION_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
    a, b = cached_query.split(), current_query.split()
    sm = difflib.SequenceMatcher(None, a, b)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(a[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(b[j1:j2])
    if not removed and not added:
        return "identical"
    parts = []
    if removed:
        parts.append("removed: " + " ".join(removed[:max_words]))
    if added:
        parts.append("added: " + " ".join(added[:max_words]))
    return "; ".join(parts)


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


def nli_zero_shot_scores(model, pairs, batch_size=64):
    raw = np.array(model.predict(pairs, batch_size=batch_size, show_progress_bar=False, apply_softmax=False))
    exp = np.exp(raw - raw.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1] - probs[:, 0]  # id2label: 0=contradiction, 1=entailment, 2=neutral


def by_category_fa(categories, approved):
    by_category = {}
    for c, is_fa in zip(categories, approved):
        by_category.setdefault(c, {"n": 0, "fa": 0})
        by_category[c]["n"] += 1
        by_category[c]["fa"] += int(is_fa)
    return {c: {"n": v["n"], "fa": v["fa"], "fa_pct": v["fa"] / v["n"] * 100 if v["n"] else float("nan")}
            for c, v in sorted(by_category.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zero-shot-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    parser.add_argument("--finetuned-checkpoint",
                         default="results/finetuned_verifier_model_lmarena_nli_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--natural-stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--output", default="results/qcached_nli_fusion_eval_lmarena.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    natural = json.loads(Path(args.natural_stash).read_text(encoding="utf-8"))
    test_rows = natural["test"]
    test_labels = np.array([1 if row[-1] else 0 for row in test_rows])
    natural_zero_pairs = [(row[1], row[3]) for row in test_rows]  # (query, answer), matches zero-shot check script
    natural_ft_pairs = [(DIFF_TEMPLATE.format(query=row[1], diff=word_diff_summary(row[2], row[1])), row[3])
                         for row in test_rows]
    log(f"Natural test: {len(test_rows)} pairs, {int(test_labels.sum())} positive")

    data = json.loads(Path(args.redteam_results).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= TAU_LOW]
    redteam_categories = [t["category"] for t in triples]
    redteam_zero_pairs = [(t["query_a"], t["answer_b"]) for t in triples]
    redteam_ft_pairs = [(DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"])),
                          t["answer_b"]) for t in triples]
    log(f"Redteam probes: {len(triples)} (tau_low={TAU_LOW})")

    log(f"Loading zero-shot model {args.zero_shot_model!r}...")
    zero_model = CrossEncoder(args.zero_shot_model, device=device)
    log(f"Loading fine-tuned checkpoint {args.finetuned_checkpoint!r}...")
    ft_model = CrossEncoder(args.finetuned_checkpoint, device=device)

    log("Scoring natural test set (zero-shot)...")
    natural_score_zero_raw = nli_zero_shot_scores(zero_model, natural_zero_pairs)
    log("Scoring natural test set (fine-tuned)...")
    natural_score_ft = np.array(ft_model.predict(natural_ft_pairs, batch_size=64, show_progress_bar=False))
    log("Scoring redteam set (zero-shot)...")
    redteam_score_zero_raw = nli_zero_shot_scores(zero_model, redteam_zero_pairs)
    log("Scoring redteam set (fine-tuned)...")
    redteam_score_ft = np.array(ft_model.predict(redteam_ft_pairs, batch_size=64, show_progress_bar=False))

    # Fixed, data-independent rescale: P(entailment)-P(contradiction) in [-1,1] -> [0,1], preserving the 0 decision
    # threshold at 0.5 so it lines up with score_ft's own 0.5 sigmoid threshold. No calibration on held-out data.
    natural_score_zero = (natural_score_zero_raw + 1.0) / 2.0
    redteam_score_zero = (redteam_score_zero_raw + 1.0) / 2.0

    log(f"Reference points -- zero-shot alone: natural AUC={roc_auc(natural_score_zero_raw, test_labels):.4f}, "
        f"redteam FA={100 * (redteam_score_zero_raw >= 0.0).mean():.1f}%")
    log(f"Reference points -- fine-tuned alone: natural AUC={roc_auc(natural_score_ft, test_labels):.4f}, "
        f"redteam FA={100 * (redteam_score_ft >= 0.5).mean():.1f}%")

    results = {"weighted_average": [], "and_gate_veto": None}

    log("\n=== Weighted average fusion sweep ===")
    for w in FUSION_WEIGHTS:
        fused_natural = w * natural_score_ft + (1 - w) * natural_score_zero
        fused_redteam = w * redteam_score_ft + (1 - w) * redteam_score_zero
        auc = roc_auc(fused_natural, test_labels)
        approved = fused_redteam >= 0.5
        cats = by_category_fa(redteam_categories, approved)
        total_n, total_fa = len(triples), int(approved.sum())
        entry = {
            "w_finetuned": w,
            "natural_auc": auc,
            "redteam_total_fa_pct": total_fa / total_n * 100,
            "redteam_by_category": cats,
        }
        results["weighted_average"].append(entry)
        ent = cats.get("entity_swap", {}).get("fa_pct", float("nan"))
        log(f"  w_ft={w:.1f}  natural_auc={auc:.4f}  redteam_fa={entry['redteam_total_fa_pct']:5.1f}%  "
            f"entity_swap={ent:5.1f}%")

    log("\n=== AND-gate veto (accept only if BOTH score_ft>=0.5 AND score_zero_raw>=0) ===")
    approved_and = (redteam_score_ft >= 0.5) & (redteam_score_zero_raw >= 0.0)
    cats_and = by_category_fa(redteam_categories, approved_and)
    total_fa_and = int(approved_and.sum())
    # For natural AUC under an AND-gate there's no single continuous score; report the natural false-reject
    # cost instead -- how many of the natural test's POSITIVE (would_be_correct) rows the veto newly rejects
    # relative to the fine-tuned-alone 0.5 threshold.
    ft_alone_approved_natural = natural_score_ft >= 0.5
    and_approved_natural = (natural_score_ft >= 0.5) & (natural_score_zero_raw >= 0.0)
    newly_rejected = ft_alone_approved_natural & (~and_approved_natural)
    newly_rejected_true_positive_rate = float((newly_rejected & (test_labels == 1)).sum()) / max(1, int((test_labels == 1).sum()))
    results["and_gate_veto"] = {
        "redteam_total_fa_pct": total_fa_and / len(triples) * 100,
        "redteam_by_category": cats_and,
        "natural_newly_rejected_count": int(newly_rejected.sum()),
        "natural_newly_rejected_true_positive_rate": newly_rejected_true_positive_rate,
        "natural_ft_alone_approve_rate": float(ft_alone_approved_natural.mean()),
        "natural_and_gate_approve_rate": float(and_approved_natural.mean()),
    }
    ent_and = cats_and.get("entity_swap", {}).get("fa_pct", float("nan"))
    log(f"  redteam_fa={results['and_gate_veto']['redteam_total_fa_pct']:5.1f}%  entity_swap={ent_and:5.1f}%  "
        f"natural approve-rate {ft_alone_approved_natural.mean():.4f} -> {and_approved_natural.mean():.4f}  "
        f"(newly-rejected true-positive rate {newly_rejected_true_positive_rate:.4f})")

    output = {
        "zero_shot_model": args.zero_shot_model,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "n_natural_test": len(test_rows),
        "n_redteam": len(triples),
        "reference_zero_shot_alone": {
            "natural_auc": roc_auc(natural_score_zero_raw, test_labels),
            "redteam_total_fa_pct": 100 * float((redteam_score_zero_raw >= 0.0).mean()),
        },
        "reference_finetuned_alone": {
            "natural_auc": roc_auc(natural_score_ft, test_labels),
            "redteam_total_fa_pct": 100 * float((redteam_score_ft >= 0.5).mean()),
        },
        **results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
