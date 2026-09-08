"""Evaluates the q_cached ablation's checkpoints on the EXISTING adversarial
red-team held-out set (results/llm_redteam_results.json, Section 5.18/5.19)
instead of generating new adversarial data.

Each triple in that set already has (query_a, query_b, answer_b): query_a is
the current query, answer_b is only correct for query_b, and query_b is
exactly the "cached query" concept this ablation adds -- so no new data
construction is needed to test the paper's own anchor hypothesis ("does
adding q_cached let the verifier catch a high-similarity but non-transferable
answer that a (query, answer)-only verifier false-accepts?").

Compares false-accept rate (overall and by the five failure-axis categories:
negation, action_verb, direction, entity_swap, quantity_swap) for:
  - off-the-shelf CrossEncoderVerifier, (query_a, answer_b)          [already published: 84.0%]
  - LmArena-finetuned, text-only, (query_a, answer_b)                [already published: 87.6%]
  - this ablation's text-only arm, (query_a, answer_b)               [freshly trained control]
  - this ablation's qcached arm, (query_a + "\\n[cached_query] " + query_b, answer_b)

Only the last row is new science; the first two are recomputed here (not
copied from the JSON) so all four numbers come from one script run under
identical conditions.

Usage:
    python scripts/qcached_redteam_eval.py
"""

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TAU_LOW = 0.80
QCACHED_TEMPLATE = "{query}\n[cached_query] {cached_query}"
DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
    """Word-level diff describing how cached_query differs from current_query,
    e.g. 'removed: pause; added: cancel', or 'identical'. Pure difflib -- must
    match the function of the same name in finetune_verifier_qcached_train_eval.py
    / finetune_verifier_qcached_adversarial_train_eval.py so evaluation uses
    exactly the input format the diff-mode checkpoint was trained on."""
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


def detect_decision_threshold(verifier):
    from cacheverifier.cache.store import CacheEntry
    from cacheverifier.data.schema import QueryRecord

    obviously_same = QueryRecord(query_id="probe1", query="what is the capital of France", answer="", equivalence_id="x")
    cand_same = CacheEntry(query_id="probe1c", query="what is the capital of France", answer="Paris is the capital of France.", equivalence_id="x")
    obviously_diff = QueryRecord(query_id="probe2", query="what is the capital of France", answer="", equivalence_id="y")
    cand_diff = CacheEntry(query_id="probe2c", query="how do I bake sourdough bread", answer="Mix flour, water, salt, and starter; let it ferment for 12-18 hours.", equivalence_id="z")
    s_same, _ = verifier.score(obviously_same, cand_same)
    s_diff, _ = verifier.score(obviously_diff, cand_diff)
    if 0.0 <= s_diff <= 1.0 and 0.0 <= s_same <= 1.0:
        return 0.5
    return 0.0


def proportion_bootstrap_ci(successes: int, n: int, n_resamples: int = 5000, seed: int = 0):
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array([1] * successes + [0] * (n - successes))
    resample_means = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(resample_means, [0.025, 0.975])
    return float(lo), float(hi)


def score_condition(name, triples, use_qcached, verifier, input_mode="concat"):
    from cacheverifier.cache.store import CacheEntry
    from cacheverifier.data.schema import QueryRecord

    thr = detect_decision_threshold(verifier)
    by_category = {}
    for t in triples:
        if not use_qcached:
            query_text = t["query_a"]
        elif input_mode == "concat":
            query_text = QCACHED_TEMPLATE.format(query=t["query_a"], cached_query=t["query_b"])
        else:
            query_text = DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"]))
        q = QueryRecord(query_id="q", query=query_text, answer="", equivalence_id="a")
        cand = CacheEntry(query_id="c", query=t["query_b"], answer=t["answer_b"], equivalence_id="b")
        score, _ = verifier.score(q, cand)
        false_accept = score >= thr
        c = t["category"]
        by_category.setdefault(c, {"n": 0, "fa": 0})
        by_category[c]["n"] += 1
        by_category[c]["fa"] += int(false_accept)

    total_n = sum(v["n"] for v in by_category.values())
    total_fa = sum(v["fa"] for v in by_category.values())
    ci = proportion_bootstrap_ci(total_fa, total_n)
    return {
        "name": name,
        "threshold": thr,
        "by_category": by_category,
        "total_n": total_n,
        "total_fa": total_fa,
        "total_fa_pct": total_fa / total_n * 100 if total_n else float("nan"),
        "total_fa_ci95": ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--redteam-results", default="results/llm_redteam_results.json")
    parser.add_argument("--text-only-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_ablation_text_only")
    parser.add_argument("--qcached-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_ablation_qcached")
    parser.add_argument("--output", default="results/qcached_redteam_comparison.json")
    parser.add_argument("--input-mode", choices=["concat", "diff"], default="concat",
                        help="Must match the --input-mode the --qcached-checkpoint was trained with.")
    args = parser.parse_args()

    from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

    data = json.loads(Path(args.redteam_results).read_text(encoding="utf-8"))
    triples = [t for t in data["all_triples"] if t["similarity"] >= TAU_LOW]
    print(f"Loaded {len(triples)}/{len(data['all_triples'])} triples clearing tau_low={TAU_LOW} "
          f"from {args.redteam_results}")

    conditions = [
        ("off_the_shelf", CrossEncoderVerifier(), False),
        ("finetuned_text_only_ablation", CrossEncoderVerifier(model_name=args.text_only_checkpoint), False),
        ("finetuned_qcached_ablation", CrossEncoderVerifier(model_name=args.qcached_checkpoint), True),
    ]

    all_results = {}
    for name, verifier, use_qcached in conditions:
        print(f"\nScoring condition: {name} (use_qcached={use_qcached}, input_mode={args.input_mode})")
        result = score_condition(name, triples, use_qcached, verifier, args.input_mode)
        all_results[name] = result
        print(f"  TOTAL false-accept: {result['total_fa']}/{result['total_n']} = {result['total_fa_pct']:.1f}% "
              f"(95% CI [{result['total_fa_ci95'][0] * 100:.1f}%, {result['total_fa_ci95'][1] * 100:.1f}%])")
        for cat, s in sorted(result["by_category"].items()):
            print(f"    {cat:>14}: {s['fa']:>3}/{s['n']:<3} = {s['fa'] / s['n'] * 100:5.1f}%")
        del verifier

    print(f"\n{'=' * 90}")
    print(f"{'category':>14} | {'off-the-shelf':>14} | {'ft text-only':>14} | {'ft qcached':>14}")
    categories = sorted(all_results["off_the_shelf"]["by_category"].keys())
    for cat in categories:
        row = []
        for name in ("off_the_shelf", "finetuned_text_only_ablation", "finetuned_qcached_ablation"):
            s = all_results[name]["by_category"][cat]
            row.append(f"{s['fa'] / s['n'] * 100:5.1f}% (n={s['n']})")
        print(f"{cat:>14} | {row[0]:>14} | {row[1]:>14} | {row[2]:>14}")
    print(f"{'TOTAL':>14} | {all_results['off_the_shelf']['total_fa_pct']:13.1f}% | "
          f"{all_results['finetuned_text_only_ablation']['total_fa_pct']:13.1f}% | "
          f"{all_results['finetuned_qcached_ablation']['total_fa_pct']:13.1f}%")

    Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
