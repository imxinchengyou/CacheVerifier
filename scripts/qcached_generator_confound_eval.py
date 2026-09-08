"""Scores a given adversarial triples file against off-the-shelf and
qcached_adv_diff, exactly like qcached_redteam_eval.py, but generalized to
accept ANY triples source (a plain list of {query_a, query_b, answer_b,
category} dicts with or without a precomputed `similarity`, or a dict with
an `all_triples` key in that shape -- both results/llm_redteam_results.json
and results/openai_redteam_generated.json parse fine). If `similarity` is
missing, it's computed here with the same embedder (all-MiniLM-L6-v2) and
tau_low=0.80 filter the rest of the project uses.

This is the generator-confound check for RESEARCH_PROPOSAL.md direction 16:
compares false-accept rate on a NEW adversarial set (not written by
DeepSeek, not seen during training) against the numbers already published
for the DeepSeek-written held-out set (84.0% off-the-shelf, 12.4%
qcached_adv_diff).

Usage:
    python scripts/qcached_generator_confound_eval.py --triples-file results/openai_redteam_generated.json
    python scripts/qcached_generator_confound_eval.py --triples-file results/generator_c_adversarial_triples.json
"""

import argparse
import difflib
import json
import sys
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


def cosine_sim(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def detect_decision_threshold(verifier) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triples-file", required=True)
    parser.add_argument("--off-the-shelf-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--qcached-checkpoint", default="results/finetuned_verifier_model_lmarena_qcached_adv_ablation_qcached_adv_diff")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = json.loads(Path(args.triples_file).read_text(encoding="utf-8"))
    triples = data["all_triples"] if isinstance(data, dict) and "all_triples" in data else data
    print(f"Loaded {len(triples)} triples from {args.triples_file}")

    if not triples or "similarity" not in triples[0]:
        print("No precomputed similarity found, embedding with all-MiniLM-L6-v2...")
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb_a = embedder.encode([t["query_a"] for t in triples], convert_to_numpy=True, show_progress_bar=False)
        emb_b = embedder.encode([t["query_b"] for t in triples], convert_to_numpy=True, show_progress_bar=False)
        for t, ea, eb in zip(triples, emb_a, emb_b):
            t["similarity"] = cosine_sim(ea, eb)

    qualifying = [t for t in triples if t["similarity"] >= TAU_LOW]
    print(f"{len(qualifying)}/{len(triples)} triples clear tau_low={TAU_LOW}")

    from cacheverifier.cache.store import CacheEntry
    from cacheverifier.data.schema import QueryRecord
    from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

    def score_condition(name, verifier, use_diff):
        thr = detect_decision_threshold(verifier)
        by_category = {}
        for t in qualifying:
            query_text = DIFF_TEMPLATE.format(query=t["query_a"], diff=word_diff_summary(t["query_b"], t["query_a"])) if use_diff else t["query_a"]
            q = QueryRecord(query_id="q", query=query_text, answer="", equivalence_id="a")
            cand = CacheEntry(query_id="c", query=t["query_b"], answer=t["answer_b"], equivalence_id="b")
            score, _ = verifier.score(q, cand)
            fa = score >= thr
            c = t["category"]
            by_category.setdefault(c, {"n": 0, "fa": 0})
            by_category[c]["n"] += 1
            by_category[c]["fa"] += int(fa)
        total_n = sum(v["n"] for v in by_category.values())
        total_fa = sum(v["fa"] for v in by_category.values())
        ci = proportion_bootstrap_ci(total_fa, total_n)
        print(f"\n[{name}] TOTAL false-accept: {total_fa}/{total_n} = {total_fa / total_n * 100:.1f}% "
              f"(95% CI [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%])")
        for cat, s in sorted(by_category.items()):
            print(f"    {cat:>14}: {s['fa']:>3}/{s['n']:<3} = {s['fa'] / s['n'] * 100:5.1f}%")
        return {"threshold": thr, "by_category": by_category, "total_n": total_n, "total_fa": total_fa,
                "total_fa_pct": total_fa / total_n * 100, "total_fa_ci95": ci}

    off_the_shelf = CrossEncoderVerifier(model_name=args.off_the_shelf_model)
    result_off = score_condition("off_the_shelf", off_the_shelf, use_diff=False)
    del off_the_shelf

    qcached = CrossEncoderVerifier(model_name=args.qcached_checkpoint)
    result_qcached = score_condition("qcached_adv_diff", qcached, use_diff=True)
    del qcached

    output = {
        "triples_file": args.triples_file,
        "n_triples_qualifying": len(qualifying),
        "off_the_shelf": result_off,
        "qcached_adv_diff": result_qcached,
    }
    out_path = args.output or (Path(args.triples_file).stem + "_confound_eval.json")
    out_path = Path("results") / out_path if not str(out_path).startswith("results/") else Path(out_path)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
