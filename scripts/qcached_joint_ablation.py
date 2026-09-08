"""Joint ablation: is the query-side shortcut (raw concat of current_query +
cached_query, RESEARCH_PROPOSAL.md direction 16's original finding) the SAME
underlying mechanism as the answer-side shortcut (answer/cached_query surface
overlap, direction 16 "再续之九"), or two separate, additive problems?

Design: a 2x2 factorial, crossing the two independent fixes tested so far in
this line of work, holding everything else (base model, training data,
hyperparameters) fixed:

  - query-side representation: "concat" (raw cached_query text next to the
    current query -- the mode found to backfire, 97.7%/69.0% FA rate) vs
    "diff" (word-level diff summary -- the fix, 12.4% FA rate on full answers)
  - answer-side content: "full" (the whole answer_b) vs "back_half" (last
    ceil(len/2) words -- found to roughly halve FA rate regardless of query
    mode or training)

Both checkpoints being compared were trained identically except for this one
input-construction difference (finetune_verifier_qcached_adversarial_train_
eval.py --input-mode concat vs diff), so the 2x2 isolates exactly these two
variables. If fixing either one alone gets most of the benefit and fixing
both barely adds more (sub-additive / interaction), that supports "same
underlying mechanism." If the combination is close to the product of the two
individual reductions (additive in log-odds / roughly multiplicative in FA
rate), that supports "two separate, independently-fixable problems."

Usage:
    python scripts/qcached_joint_ablation.py
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QCACHED_TEMPLATE = "{query}\n[cached_query] {cached_query}"
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


def build_answer_variants(answer: str) -> dict:
    words = answer.split()
    half = -(-len(words) // 2)
    return {
        "full": answer,
        "front_half": " ".join(words[:half]),
        "back_half": " ".join(words[-half:]) if half > 0 else "",
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
    parser.add_argument("--concat-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_qcached_adv")
    parser.add_argument("--diff-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--output", default="results/qcached_joint_ablation.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    data = json.loads(Path(args.redteam_results).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= TAU_LOW]
    log(f"Loaded {len(triples)}/{len(data['all_triples'])} triples clearing tau_low={TAU_LOW} (negative/FA rows only)")

    answer_variants = {"full": [], "front_half": [], "back_half": []}
    for t in triples:
        v = build_answer_variants(t["answer_b"])
        for name in answer_variants:
            answer_variants[name].append(v[name])

    query_modes = {
        "concat": (args.concat_checkpoint, [QCACHED_TEMPLATE.format(query=t["query_a"], cached_query=t["query_b"]) for t in triples]),
        "diff": (args.diff_checkpoint, [DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"])) for t in triples]),
    }

    results = {}
    for mode_name, (ckpt, texts_a) in query_modes.items():
        log(f"\n=== query mode: {mode_name} ({ckpt}) ===")
        model = CrossEncoder(ckpt, device=device)
        thr = detect_decision_threshold(model)
        results[mode_name] = {"threshold": thr, "answer_variants": {}}
        for answer_name, texts_b in answer_variants.items():
            pairs = list(zip(texts_a, texts_b))
            scores = np.array(model.predict(pairs, batch_size=64, show_progress_bar=False))
            fa_rate = float((scores >= thr).mean())
            results[mode_name]["answer_variants"][answer_name] = fa_rate
            log(f"  [{mode_name:>6}][{answer_name:>10}] FA_rate={fa_rate:.4f}")
        del model

    # 2x2 summary + additivity check (in log-odds space, since FA rate is a probability)
    def logit(p, eps=1e-4):
        p = min(max(p, eps), 1 - eps)
        return np.log(p / (1 - p))

    fa_concat_full = results["concat"]["answer_variants"]["full"]
    fa_concat_back = results["concat"]["answer_variants"]["back_half"]
    fa_diff_full = results["diff"]["answer_variants"]["full"]
    fa_diff_back = results["diff"]["answer_variants"]["back_half"]

    query_effect_on_full = logit(fa_diff_full) - logit(fa_concat_full)
    query_effect_on_back = logit(fa_diff_back) - logit(fa_concat_back)
    answer_effect_on_concat = logit(fa_concat_back) - logit(fa_concat_full)
    answer_effect_on_diff = logit(fa_diff_back) - logit(fa_diff_full)
    predicted_diff_back_additive = logit(fa_concat_full) + query_effect_on_full + answer_effect_on_concat
    actual_diff_back = logit(fa_diff_back)
    interaction = actual_diff_back - predicted_diff_back_additive

    log("\n=== 2x2 table (false-accept rate) ===")
    log(f"{'':>12} | {'full':>8} | {'back_half':>10}")
    log(f"{'concat':>12} | {fa_concat_full:>8.4f} | {fa_concat_back:>10.4f}")
    log(f"{'diff':>12} | {fa_diff_full:>8.4f} | {fa_diff_back:>10.4f}")
    log(f"\nquery-fix effect (logit scale): on full answer = {query_effect_on_full:+.3f}, on back_half = {query_effect_on_back:+.3f}")
    log(f"answer-fix effect (logit scale): on concat query = {answer_effect_on_concat:+.3f}, on diff query = {answer_effect_on_diff:+.3f}")
    log(f"interaction term (actual diff+back_half logit - additive prediction) = {interaction:+.3f}")
    log("(interaction near 0 => the two fixes are additive/independent; "
        "interaction far from 0, esp. positive/sub-additive, => shared mechanism, diminishing returns from combining)")

    output = {
        "fa_rate_2x2": {
            "concat_full": fa_concat_full, "concat_back_half": fa_concat_back,
            "diff_full": fa_diff_full, "diff_back_half": fa_diff_back,
        },
        "logit_effects": {
            "query_fix_on_full": query_effect_on_full,
            "query_fix_on_back_half": query_effect_on_back,
            "answer_fix_on_concat": answer_effect_on_concat,
            "answer_fix_on_diff": answer_effect_on_diff,
            "interaction": interaction,
        },
        "full_results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
