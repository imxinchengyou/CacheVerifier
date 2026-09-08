"""Content attribution: which part of `answer` does the ALREADY-TRAINED
qcached-diff+adversarial model (results/finetuned_verifier_model_lmarena_
qcached_adv_ablation_qcached_adv_diff, trained on FULL-length LmArena
answers) actually use when it makes its judgment?

Motivation (RESEARCH_PROPOSAL.md direction 16 "再续之六"): retraining on
front-truncated vs back-truncated LmArena answers gave very different
improvement magnitudes (13.7pp vs 25.5pp) at the SAME truncated length,
showing that WHICH part of the answer survives matters more than how long
it is. That was a training-time ablation (different models, different
data). This script instead takes the ONE existing model trained on full
answers and asks, at inference time, which slice of a held-out answer the
model's score actually depends on -- a cleaner form of attribution than
retraining, since the model's weights never change across the comparison.

Method: for every held-out natural test row (from finetune_verifier_
qcached_experiment.examples.json, i.e. real LmArena gray-zone pairs, not
the synthetic adversarial set), build several truncated-`answer` variants
of the SAME text_a (query + diff-vs-cached_query, unaffected by any of
this) and score all variants with the SAME fixed model:
  - full          : the untouched answer
  - front_half    : first ceil(len/2) words
  - back_half     : last ceil(len/2) words
  - front50/back50: first/last 50 words (matches the fixed-length
                    truncation used in the retraining ablations, for
                    direct comparability)
  - middle50      : the 50 words centered in the answer

Two attribution signals are reported per variant:
  - AUC vs ground truth (`would_be_correct`) -- does this slice alone
    preserve the model's overall discriminative power?
  - Pearson r between this variant's raw scores and the full-answer's raw
    scores -- does this slice alone reproduce what the model concludes
    from the full text, request by request? (a segment can have plausible
    AUC from a totally different scoring pattern; the correlation catches
    that this is a different kind of failure than "same conclusions, less
    signal.")

Usage:
    python scripts/qcached_diff_answer_attribution.py
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"


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


def build_answer_variants(answer: str, fixed_n: int) -> dict:
    words = answer.split()
    n = len(words)
    half = -(-n // 2)  # ceil
    mid_start = max(0, (n - fixed_n) // 2)
    return {
        "full": answer,
        "front_half": " ".join(words[:half]),
        "back_half": " ".join(words[-half:]) if half > 0 else "",
        f"front{fixed_n}": " ".join(words[:fixed_n]),
        f"back{fixed_n}": " ".join(words[-fixed_n:]) if n > 0 else "",
        f"middle{fixed_n}": " ".join(words[mid_start:mid_start + fixed_n]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--natural-stash", default="results/finetune_verifier_qcached_experiment.examples.json")
    parser.add_argument("--checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--fixed-n", type=int, default=50)
    parser.add_argument("--output", default="results/qcached_diff_answer_attribution.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    natural = json.loads(Path(args.natural_stash).read_text(encoding="utf-8"))
    test_rows = natural["test"]
    log(f"Loaded {len(test_rows)} natural held-out test rows from {args.natural_stash}")

    labels = np.array([1 if label else 0 for _, _, _, _, label in test_rows])
    answer_word_counts = [len(a.split()) for _, _, _, a, _ in test_rows]
    log(f"Answer length: mean={np.mean(answer_word_counts):.1f} words, "
        f"median={np.median(answer_word_counts):.0f}, min={min(answer_word_counts)}, max={max(answer_word_counts)}")

    log(f"Loading fixed model from {args.checkpoint} (weights never change across variants)...")
    model = CrossEncoder(args.checkpoint, device=device)

    variant_names = ["full", "front_half", "back_half", f"front{args.fixed_n}", f"back{args.fixed_n}", f"middle{args.fixed_n}"]
    variant_texts_b = {name: [] for name in variant_names}
    texts_a = []
    for _, q, cq, a, _ in test_rows:
        texts_a.append(DIFF_TEMPLATE.format(query=q, diff=word_diff_summary(cq, q)))
        variants = build_answer_variants(a, args.fixed_n)
        for name in variant_names:
            variant_texts_b[name].append(variants[name])

    results = {}
    full_scores = None
    for name in variant_names:
        pairs = list(zip(texts_a, variant_texts_b[name]))
        t0 = time.time()
        scores = np.array(model.predict(pairs, batch_size=64, show_progress_bar=False))
        auc = roc_auc(scores, labels)
        if name == "full":
            full_scores = scores
            corr = 1.0
        else:
            corr = float(np.corrcoef(scores, full_scores)[0, 1])
        results[name] = {
            "auc": auc,
            "pearson_r_vs_full_scores": corr,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        log(f"  [{name:>12}] AUC={auc:.4f}  r_vs_full={corr:.4f}  "
            f"(scored {len(pairs)} pairs in {time.time() - t0:.1f}s)")

    output = {
        "checkpoint": args.checkpoint,
        "n_test": len(test_rows),
        "fixed_n": args.fixed_n,
        "answer_word_count_mean": float(np.mean(answer_word_counts)),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\n{'variant':>12} | {'AUC':>8} | {'r vs full':>10}")
    for name in variant_names:
        r = results[name]
        log(f"{name:>12} | {r['auc']:>8.4f} | {r['pearson_r_vs_full_scores']:>10.4f}")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
