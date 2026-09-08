"""Deep follow-up to the two answer-attribution analyses (natural set:
front_half dominates; adversarial set: front/back tie, but ANY truncation
beats full). This script tries to nail down the mechanism behind "full
answer is more easily fooled on the adversarial set" instead of leaving it
as an unverified one-line hypothesis, along three angles:

1. Is this specific to the diff-trained model, or does ANY verifier show
   it? Runs the same full-vs-front_half-vs-back_half comparison across
   THREE checkpoints: off-the-shelf, text_only_adv (this dataset's own
   fine-tuned-without-qcached control), and qcached_adv_diff.

2. Does word overlap between cached_query (query_b) and the answer variant
   predict false-accept? Computes Jaccard word overlap for each row/variant
   and correlates it with that variant's score, to quantitatively test the
   "more answer content -> more surface overlap with cached_query -> more
   false accepts" hypothesis instead of asserting it narratively.

3. Concrete case inspection: lists the triples where the qcached_adv_diff
   model false-accepts on `full` but correctly rejects on `back_half` (or
   vice versa), so the mechanism can be checked by actually reading a
   handful of real examples instead of only looking at aggregate rates.

Usage:
    python scripts/qcached_diff_answer_attribution_deep.py
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


def jaccard(a: str, b: str) -> float:
    sa, sb = set(w.lower().strip(".,!?;:\"'") for w in a.split()), set(w.lower().strip(".,!?;:\"'") for w in b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_variants(answer: str) -> dict:
    words = answer.split()
    half = -(-len(words) // 2)
    return {"full": answer, "front_half": " ".join(words[:half]), "back_half": " ".join(words[-half:]) if half > 0 else ""}


def detect_decision_threshold(model) -> float:
    s_same = float(model.predict([("what is the capital of France", "Paris is the capital of France.")])[0])
    s_diff = float(model.predict([("what is the capital of France", "Mix flour, water, salt, and starter.")])[0])
    if 0.0 <= s_diff <= 1.0 and 0.0 <= s_same <= 1.0:
        return 0.5
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--off-the-shelf-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--text-only-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_text_only_adv")
    parser.add_argument("--qcached-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--output", default="results/qcached_diff_answer_attribution_deep.json")
    parser.add_argument("--n-cases", type=int, default=10)
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

    # ---- Part 1: three checkpoints x three variants, negative rows only (the FA-relevant ones) ----
    neg_texts_a_qcached = [DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"])) for t in triples]
    neg_texts_a_plain = [t["query_a"] for t in triples]  # for off-the-shelf / text-only (no diff input)

    variant_texts_b = {name: [] for name in ["full", "front_half", "back_half"]}
    for t in triples:
        variants = build_variants(t["answer_b"])
        for name in variant_texts_b:
            variant_texts_b[name].append(variants[name])

    checkpoints = {
        "off_the_shelf": (args.off_the_shelf_model, neg_texts_a_plain),
        "text_only_adv": (args.text_only_checkpoint, neg_texts_a_plain),
        "qcached_adv_diff": (args.qcached_checkpoint, neg_texts_a_qcached),
    }

    part1 = {}
    per_model_scores = {}
    for model_name, (ckpt, texts_a) in checkpoints.items():
        log(f"\n=== Loading {model_name} ({ckpt}) ===")
        model = CrossEncoder(ckpt, device=device)
        thr = detect_decision_threshold(model)
        part1[model_name] = {"threshold": thr, "variants": {}}
        per_model_scores[model_name] = {}
        for variant_name, texts_b in variant_texts_b.items():
            pairs = list(zip(texts_a, texts_b))
            scores = np.array(model.predict(pairs, batch_size=64, show_progress_bar=False))
            fa_rate = float((scores >= thr).mean())
            part1[model_name]["variants"][variant_name] = fa_rate
            per_model_scores[model_name][variant_name] = scores
            log(f"  [{model_name:>16}][{variant_name:>10}] FA_rate={fa_rate:.4f}")
        del model

    # ---- Part 2: word overlap (cached_query vs answer variant) vs score, qcached_adv_diff only ----
    log("\n=== Part 2: cached_query/answer word overlap vs score (qcached_adv_diff) ===")
    overlap_by_variant = {name: [] for name in variant_texts_b}
    for name, texts_b in variant_texts_b.items():
        overlap_by_variant[name] = [jaccard(t["query_b"], ans) for t, ans in zip(triples, texts_b)]

    part2 = {}
    qd_scores = per_model_scores["qcached_adv_diff"]
    for name in variant_texts_b:
        overlaps = np.array(overlap_by_variant[name])
        scores = qd_scores[name]
        r = float(np.corrcoef(overlaps, scores)[0, 1]) if overlaps.std() > 0 and scores.std() > 0 else float("nan")
        part2[name] = {"mean_jaccard_overlap": float(overlaps.mean()), "pearson_r_overlap_vs_score": r}
        log(f"  [{name:>10}] mean_overlap={overlaps.mean():.4f}  r(overlap, score)={r:.4f}")

    # ---- Part 3: concrete cases where full false-accepts but back_half correctly rejects ----
    log(f"\n=== Part 3: concrete cases (qcached_adv_diff, full FA but back_half correctly rejects) ===")
    thr_qd = part1["qcached_adv_diff"]["threshold"]
    full_scores = qd_scores["full"]
    back_scores = qd_scores["back_half"]
    front_scores = qd_scores["front_half"]
    flip_cases = []
    for i, t in enumerate(triples):
        if full_scores[i] >= thr_qd and back_scores[i] < thr_qd:
            flip_cases.append({
                "category": t["category"],
                "query_a": t["query_a"],
                "query_b": t["query_b"],
                "answer_b": t["answer_b"],
                "score_full": float(full_scores[i]),
                "score_front_half": float(front_scores[i]),
                "score_back_half": float(back_scores[i]),
                "overlap_full": overlap_by_variant["full"][i],
                "overlap_back_half": overlap_by_variant["back_half"][i],
            })
    log(f"  {len(flip_cases)}/{len(triples)} cases flip from false-accept (full) to correct-reject (back_half)")
    for c in flip_cases[: args.n_cases]:
        log(f"  --- [{c['category']}] score full={c['score_full']:.3f} back_half={c['score_back_half']:.3f} "
            f"(overlap full={c['overlap_full']:.2f} back={c['overlap_back_half']:.2f})")
        log(f"      query_a: {c['query_a']}")
        log(f"      query_b (cached): {c['query_b']}")
        log(f"      answer_b: {c['answer_b']}")

    output = {
        "part1_fa_rate_by_model_and_variant": part1,
        "part2_overlap_vs_score": part2,
        "part3_n_flip_cases": len(flip_cases),
        "part3_flip_cases_sample": flip_cases[: args.n_cases],
        "part3_all_flip_cases": flip_cases,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
