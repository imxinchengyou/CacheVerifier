"""Same content-attribution method as qcached_diff_answer_attribution.py,
applied to the ADVERSARIAL red-team held-out set instead of the natural
gray-zone test set -- the direct follow-up RESEARCH_PROPOSAL.md direction 16
"再续之七" flagged as the missing check: the natural-set attribution found
`answer`'s FRONT half carries the fixed qcached-diff model's judgment (front_half
AUC 0.8559 / r=0.95 vs back_half 0.8050 / r=0.64), the opposite of what the
retraining ablation's "back-truncated training data generalizes better to the
adversarial set" result would suggest if the same "important segment" applied
to both natural and adversarial distributions. This script checks whether the
SAME fixed model's segment-dependence looks different on the adversarial set
itself.

Each of the 306 held-out triples (query_a, query_b, answer_b) becomes two
labeled rows, matching build_adversarial_rows()'s convention exactly:
  - (query_a, cached_query=query_b, answer_b) -> label False (the false-accept case)
  - (query_b, cached_query=query_b, answer_b) -> label True  (contrasting correct case)

Unlike the natural set (mean answer ~272 words), these answers are short
(mean ~16.5 words, median 15, max 51) -- fixed 50-word windows would just
reproduce "full" for nearly every row, so this script uses front_half/
back_half/middle_third (each row's own length) plus a small fixed_n=5 (still
meaningful at this length) instead of fixed_n=50.

Reports, per answer-segment variant, using the SAME fixed qcached-diff model
(no retraining):
  - AUC over both labels (612 rows)
  - false-accept rate on the label=False rows ONLY (306 rows) -- this is the
    metric the retraining ablation actually reported, so it's the most
    directly comparable number
  - Pearson r vs the full-answer variant's raw scores (over all 612 rows)

Usage:
    python scripts/qcached_diff_answer_attribution_redteam.py
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
TAU_LOW = 0.80


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
    third = max(1, n // 3)
    mid_start = max(0, (n - third) // 2)
    return {
        "full": answer,
        "front_half": " ".join(words[:half]),
        "back_half": " ".join(words[-half:]) if half > 0 else "",
        f"front{fixed_n}": " ".join(words[:fixed_n]),
        f"back{fixed_n}": " ".join(words[-fixed_n:]) if n > 0 else "",
        "middle_third": " ".join(words[mid_start:mid_start + third]),
    }


def detect_decision_threshold(model) -> float:
    s_same = float(model.predict([("what is the capital of France", "Paris is the capital of France.")])[0])
    s_diff = float(model.predict([("what is the capital of France", "Mix flour, water, salt, and starter.")])[0])
    if 0.0 <= s_diff <= 1.0 and 0.0 <= s_same <= 1.0:
        return 0.5
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--fixed-n", type=int, default=5)
    parser.add_argument("--output", default="results/qcached_diff_answer_attribution_redteam.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    data = json.loads(Path(args.redteam_results).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= TAU_LOW]
    log(f"Loaded {len(triples)}/{len(data['all_triples'])} triples clearing tau_low={TAU_LOW}")

    lens = [len(t["answer_b"].split()) for t in triples]
    log(f"answer_b length: mean={np.mean(lens):.1f} median={np.median(lens):.0f} min={min(lens)} max={max(lens)}")

    # rows: (query, cached_query, answer, label)
    rows = []
    for t in triples:
        rows.append((t["query_a"], t["query_b"], t["answer_b"], False))
        rows.append((t["query_b"], t["query_b"], t["answer_b"], True))
    labels = np.array([1 if r[3] else 0 for r in rows])
    neg_mask = labels == 0
    log(f"Built {len(rows)} labeled rows ({int(labels.sum())} positive / {int((~labels.astype(bool)).sum())} negative)")

    log(f"Loading fixed model from {args.checkpoint} (weights never change across variants)...")
    model = CrossEncoder(args.checkpoint, device=device)
    thr = detect_decision_threshold(model)
    log(f"Decision threshold: {thr}")

    variant_names = ["full", "front_half", "back_half", f"front{args.fixed_n}", f"back{args.fixed_n}", "middle_third"]
    variant_texts_b = {name: [] for name in variant_names}
    texts_a = []
    for q, cq, a, _ in rows:
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
        fa_rate = float((scores[neg_mask] >= thr).mean())
        if name == "full":
            full_scores = scores
            corr = 1.0
        else:
            corr = float(np.corrcoef(scores, full_scores)[0, 1])
        results[name] = {
            "auc": auc,
            "false_accept_rate": fa_rate,
            "pearson_r_vs_full_scores": corr,
        }
        log(f"  [{name:>12}] AUC={auc:.4f}  FA_rate={fa_rate:.4f}  r_vs_full={corr:.4f}  "
            f"(scored {len(pairs)} pairs in {time.time() - t0:.1f}s)")

    output = {
        "checkpoint": args.checkpoint,
        "n_triples": len(triples),
        "n_rows": len(rows),
        "fixed_n": args.fixed_n,
        "answer_word_count_mean": float(np.mean(lens)),
        "decision_threshold": thr,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\n{'variant':>12} | {'AUC':>8} | {'FA_rate':>8} | {'r vs full':>10}")
    for name in variant_names:
        r = results[name]
        log(f"{name:>12} | {r['auc']:>8.4f} | {r['false_accept_rate']:>8.4f} | {r['pearson_r_vs_full_scores']:>10.4f}")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
