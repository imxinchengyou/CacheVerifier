"""LLM-driven automated red-teaming of the verifier -- method "J" from the
2026-08-29 research-methods brainstorm (see RESEARCH_PROPOSAL.md Sec 10,
"研究现状总览").

Every gray-zone false-accept measured elsewhere in this paper comes from
naturally-occurring near-duplicate pairs inside a fixed benchmark dataset.
That measures average-case robustness on whatever hard cases happen to
exist in LmArena/Quora/SearchQueries -- it says nothing about worst-case
robustness against pairs an adversary (or just an unlucky real user)
deliberately phrases to look like a near-duplicate while requiring a
different answer. This script uses an LLM (DeepSeek, same substitution
already used for Group C's oracle-latency measurement and the SearchQueries
answer-regeneration fix) to GENERATE such pairs on purpose, across five
categories chosen because this project's own prior work already flagged
each as a real failure axis, not picked arbitrarily:

  - negation:        "cancel X" vs "do not cancel X" / "keep X active"
  - action_verb:      the canonical "cancel my subscription" vs "pause my
                       subscription" example from RESEARCH_PROPOSAL.md Sec 10
                       direction 7's Reddit-sourced discussion
  - direction:        same verb, reversed object direction -- the specific
                       mechanism Section 5.10 found explains bucketing's
                       regression on LmArena ("convert audio to video" vs
                       "convert video to audio")
  - entity_swap:      different named entity, same sentence template -- the
                       mechanism Section 5.4's Quora honest-calibration
                       diagnosis names explicitly ("老布什" vs "小布什",
                       George H.W. Bush vs George W. Bush)
  - quantity_swap:    different numeric parameter in an otherwise identical
                       sentence

For each generated (query_a, query_b, answer_b) triple, query_a and
answer_b (an answer that is only actually correct for query_b) are embedded
and scored exactly as this project's own gray-zone pipeline would: live
sentence-transformer embedding (`all-MiniLM-L6-v2`, matching this project's
own Quora config, since these are synthetic pairs with no precomputed
vCache embedding), filtered to pairs whose cosine similarity clears
tau_low=0.80 (the floor used across every dataset config in this project --
below that, the pair would never reach the verifier at all in the real
pipeline), then scored with CrossEncoderVerifier -- both the off-the-shelf
checkpoint (Group D) and, for comparison, the LmArena fine-tuned checkpoint
(Group E) at results/finetuned_verifier_model. The fine-tuned checkpoint's
training domain does NOT match these synthetic general-topic adversarial
pairs, so this is an exploratory comparison, not a domain-matched retest of
Section 5.6 -- flagged as such in the summary.

Usage:
    python scripts/llm_redteam_verifier.py --n-per-category 15
    python scripts/llm_redteam_verifier.py --n-per-category 15 --output results/llm_redteam_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

CATEGORY_PROMPTS = {
    "negation": (
        "Generate {n} triples for testing a semantic-cache verifier's robustness to NEGATION. "
        "Each triple has: query_a (a customer support / general question), query_b (a query that looks "
        "almost identical to query_a in wording and topic, but differs by a negation -- e.g. adds 'not', "
        "'don't', 'never', or flips 'keep' vs 'cancel'), and answer_b (a specific, correct answer to "
        "query_b ONLY -- it must be WRONG if given in response to query_a). The two queries should be "
        "similar enough in surface wording that a similarity-based cache might confuse them, but the "
        "negation must make the correct answer genuinely different, not just phrased differently."
    ),
    "action_verb": (
        "Generate {n} triples for testing a semantic-cache verifier's robustness to ACTION-VERB SWAPS. "
        "Each triple has: query_a and query_b about the same object/service (e.g. a subscription, an "
        "order, an account) but with a DIFFERENT ACTION VERB (cancel vs pause vs downgrade vs renew vs "
        "refund vs delete), and answer_b, a specific correct answer to query_b ONLY that would be WRONG "
        "advice if given for query_a (e.g. steps to cancel are wrong if the user actually wanted to pause). "
        "Keep the object/service identical between query_a and query_b so only the verb differs."
    ),
    "direction": (
        "Generate {n} triples for testing a semantic-cache verifier's robustness to REVERSED DIRECTION. "
        "Each triple has: query_a and query_b that use the SAME VERB and the same two nouns/formats, but "
        "in OPPOSITE DIRECTION (e.g. 'convert audio to video' vs 'convert video to audio', 'translate "
        "English to French' vs 'translate French to English', 'import CSV to database' vs 'export database "
        "to CSV'), and answer_b, a specific correct answer to query_b ONLY that would be procedurally WRONG "
        "if given for query_a (reversed steps/tools)."
    ),
    "entity_swap": (
        "Generate {n} triples for testing a semantic-cache verifier's robustness to NAMED-ENTITY SWAPS. "
        "Each triple has: query_a and query_b that are IDENTICAL in sentence structure and topic but ask "
        "about a DIFFERENT specific named entity (different person, product model, city, or brand -- e.g. "
        "'George H.W. Bush' vs 'George W. Bush', 'iPhone 14' vs 'iPhone 15', 'Paris, Texas' vs 'Paris, "
        "France'), and answer_b, a specific factual answer about query_b's entity ONLY that would be "
        "factually WRONG if given for query_a's entity."
    ),
    "quantity_swap": (
        "Generate {n} triples for testing a semantic-cache verifier's robustness to QUANTITY/PARAMETER "
        "SWAPS. Each triple has: query_a and query_b that are identical except for a numeric quantity, "
        "percentage, date, or duration (e.g. 'save 10% on my bill' vs 'save 50% on my bill', 'refund within "
        "30 days' vs 'refund within 90 days'), and answer_b, a specific correct answer to query_b's exact "
        "number that would be WRONG (a different eligibility/outcome/amount) if given for query_a's number."
    ),
}

JSON_INSTRUCTION = (
    "\n\nReturn ONLY a JSON array (no markdown fences, no commentary) of exactly {n} objects, each with "
    'keys "query_a", "query_b", "answer_b" (all strings). Make the pairs realistic customer-support / '
    "general-Q&A style queries, similar in length and register to what a real user would type."
)

TAU_LOW = 0.80


def call_deepseek_generate(api_key: str, category: str, n: int, model: str) -> list[dict]:
    prompt = CATEGORY_PROMPTS[category].format(n=n) + JSON_INSTRUCTION.format(n=n)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 4000,
    }
    resp = requests.post(url=DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    triples = json.loads(content)
    for t in triples:
        t["category"] = category
    return triples


def cosine_sim(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-category", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=1,
                         help="Independent generation rounds to pool (temperature=0.9 means a single "
                              "round with a small n-per-category has high sampling variance; pooling "
                              "several rounds gives a stabler estimate with a real CI).")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--finetuned-checkpoint", default="ChengyouXin/cacheverifier-lmarena",
                         help="Local dir or HF Hub id. Defaults to the same HF-hosted checkpoint "
                              "PAPER.md Sec 5.4's honest-calibration table used, for provenance-matched "
                              "comparability.")
    parser.add_argument("--output", default="results/llm_redteam_results.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-triples", default=None,
                         help="Path to a prior run's results JSON (its 'all_triples' field) -- skips "
                              "generation/embedding entirely and re-scores the EXACT SAME triples with "
                              "whatever --finetuned-checkpoint is given. Use this to test a new checkpoint "
                              "(e.g. one trained on adversarial data) against an untouched, already-held-out "
                              "test set instead of generating fresh (and therefore not-quite-comparable) samples.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if args.load_triples:
        print(f"[{time.strftime('%H:%M:%S')}] Loading triples from {args.load_triples} "
              f"(skipping generation/embedding -- re-scoring an existing held-out set)...", flush=True)
        prior = json.loads(Path(args.load_triples).read_text(encoding="utf-8"))
        all_triples = prior["all_triples"]
    else:
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY not found (checked .env and environment).")

        print(f"[{time.strftime('%H:%M:%S')}] Generating {args.n_per_category} adversarial triples "
              f"per category x {args.rounds} round(s) across {len(CATEGORY_PROMPTS)} categories "
              f"via {args.model}...", flush=True)

        all_triples = []
        for round_i in range(1, args.rounds + 1):
            for category in CATEGORY_PROMPTS:
                try:
                    triples = call_deepseek_generate(api_key, category, args.n_per_category, args.model)
                except Exception as e:
                    print(f"  [round {round_i}][{category}] generation FAILED: {e}", flush=True)
                    continue
                print(f"  [round {round_i}][{category}] generated {len(triples)} triples", flush=True)
                all_triples.extend(triples)

        if not all_triples:
            raise SystemExit("No triples generated -- aborting before spending verifier/embedding compute.")

        print(f"\n[{time.strftime('%H:%M:%S')}] Embedding {len(all_triples)} query_a/query_b pairs "
              f"with all-MiniLM-L6-v2 (live encode, matches this project's Quora config)...", flush=True)
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts_a = [t["query_a"] for t in all_triples]
        texts_b = [t["query_b"] for t in all_triples]
        emb_a = embedder.encode(texts_a, convert_to_numpy=True, show_progress_bar=False)
        emb_b = embedder.encode(texts_b, convert_to_numpy=True, show_progress_bar=False)

        for t, ea, eb in zip(all_triples, emb_a, emb_b):
            t["similarity"] = cosine_sim(ea, eb)

    qualifying = [t for t in all_triples if t["similarity"] >= TAU_LOW]
    print(f"  {len(qualifying)}/{len(all_triples)} pairs cleared tau_low={TAU_LOW} "
          f"(would actually reach a verifier in this project's real pipeline)", flush=True)

    if not qualifying:
        print("No pairs cleared tau_low -- LLM-generated pairs weren't similar enough to be gray-zone "
              "candidates. This is itself a finding (see summary).", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Scoring {len(qualifying)} qualifying pairs with "
          f"off-the-shelf CrossEncoderVerifier...", flush=True)
    from cacheverifier.cache.store import CacheEntry
    from cacheverifier.data.schema import QueryRecord
    from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

    off_the_shelf = CrossEncoderVerifier()
    finetuned = None
    ft_path_str = args.finetuned_checkpoint
    try:
        finetuned = CrossEncoderVerifier(model_name=ft_path_str)
    except Exception as e:
        print(f"  (could not load fine-tuned checkpoint '{ft_path_str}': {e} -- skipping that comparison)", flush=True)

    def detect_decision_threshold(verifier) -> float:
        """Some checkpoints, when loaded in this environment, come back through
        CrossEncoder.predict() already sigmoid-squashed into [0,1] instead of the
        raw unbounded logits CrossEncoderVerifier.score() normally returns and
        this project's threshold=0.0 default assumes (a library-version /
        saved-activation-metadata mismatch, not a property of the checkpoint's
        actual training -- see module docstring). Since sigmoid is monotonic,
        logit>=0 is EXACTLY equivalent to sigmoid(logit)>=0.5, so probing with two
        extreme sanity pairs and checking whether both scores land in [0,1]
        recovers the right decision threshold without guessing exact logit values.
        """
        obviously_same = QueryRecord(query_id="probe1", query="what is the capital of France", answer="", equivalence_id="x")
        cand_same = CacheEntry(query_id="probe1c", query="what is the capital of France", answer="Paris is the capital of France.", equivalence_id="x")
        obviously_diff = QueryRecord(query_id="probe2", query="what is the capital of France", answer="", equivalence_id="y")
        cand_diff = CacheEntry(query_id="probe2c", query="how do I bake sourdough bread", answer="Mix flour, water, salt, and starter; let it ferment for 12-18 hours.", equivalence_id="z")
        s_same, _ = verifier.score(obviously_same, cand_same)
        s_diff, _ = verifier.score(obviously_diff, cand_diff)
        if 0.0 <= s_diff <= 1.0 and 0.0 <= s_same <= 1.0:
            return 0.5, True
        return 0.0, False

    off_thr, off_sigmoid = detect_decision_threshold(off_the_shelf)
    print(f"  off-the-shelf decision threshold: {off_thr} (sigmoid-space detected: {off_sigmoid})", flush=True)
    ft_thr, ft_sigmoid = (None, None)
    if finetuned is not None:
        ft_thr, ft_sigmoid = detect_decision_threshold(finetuned)
        print(f"  fine-tuned decision threshold: {ft_thr} (sigmoid-space detected: {ft_sigmoid})", flush=True)

    for i, t in enumerate(qualifying, start=1):
        q = QueryRecord(query_id=f"redteam_{i}", query=t["query_a"], answer="", equivalence_id="a")
        cand = CacheEntry(query_id=f"redteam_b_{i}", query=t["query_b"], answer=t["answer_b"], equivalence_id="b")
        score_raw, _ = off_the_shelf.score(q, cand)
        t["score_off_the_shelf"] = score_raw
        t["false_accept_off_the_shelf_thr0"] = score_raw >= off_thr
        if finetuned is not None:
            score_ft, _ = finetuned.score(q, cand)
            t["score_finetuned_lmarena"] = score_ft
            t["false_accept_finetuned_thr0"] = score_ft >= ft_thr
        if i % 10 == 0 or i == len(qualifying):
            print(f"  [{i}/{len(qualifying)}] scored", flush=True)

    # Summary
    by_category: dict[str, dict] = {}
    for t in qualifying:
        c = t["category"]
        by_category.setdefault(c, {"n": 0, "fa_offshelf": 0, "fa_finetuned": 0})
        by_category[c]["n"] += 1
        if t.get("false_accept_off_the_shelf_thr0"):
            by_category[c]["fa_offshelf"] += 1
        if t.get("false_accept_finetuned_thr0"):
            by_category[c]["fa_finetuned"] += 1

    if ft_sigmoid:
        print(
            "\nNote: this environment's sentence-transformers version returns this checkpoint's scores "
            "already sigmoid-squashed into [0,1] instead of the raw logits threshold=0.0 assumes -- a "
            "loading-library-version issue (older sentence-transformers doesn't honor newer checkpoints' "
            "saved activation_fn metadata), confirmed present on BOTH the local checkpoint and the HF-Hub "
            "checkpoint used for PAPER.md's honest calibration. Compensated by using the mathematically "
            "equivalent threshold=0.5 in probability space (sigmoid is monotonic, so logit>=0 iff "
            "sigmoid(logit)>=0.5) -- reproduces the same decision rule threshold=0.0 gives on raw logits, "
            "not a workaround that changes what's being measured.",
            flush=True,
        )

    print(f"\n{'=' * 70}")
    print("Red-team false-accept rate by category (verifier's naive-default decision threshold, "
          "auto-detected per verifier -- see note above):")
    print(f"{'category':>15} | {'n qualifying':>13} | {'off-the-shelf FA%':>18} | {'fine-tuned(LmArena) FA%':>24}")
    total_n = total_fa_off = total_fa_ft = 0
    for c, s in by_category.items():
        fa_off_pct = s["fa_offshelf"] / s["n"] * 100 if s["n"] else float("nan")
        fa_ft_pct = s["fa_finetuned"] / s["n"] * 100 if s["n"] and finetuned is not None else float("nan")
        print(f"{c:>15} | {s['n']:>13} | {fa_off_pct:>17.1f}% | {fa_ft_pct:>23.1f}%")
        total_n += s["n"]
        total_fa_off += s["fa_offshelf"]
        total_fa_ft += s["fa_finetuned"]
    def proportion_bootstrap_ci(successes: int, n: int, n_resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
        if n == 0:
            return (float("nan"), float("nan"))
        rng = np.random.default_rng(seed)
        arr = np.array([1] * successes + [0] * (n - successes))
        resample_means = rng.choice(arr, size=(n_resamples, n), replace=True).mean(axis=1)
        lo, hi = np.quantile(resample_means, [0.025, 0.975])
        return float(lo), float(hi)

    if total_n:
        off_ci = proportion_bootstrap_ci(total_fa_off, total_n)
        ft_ci = proportion_bootstrap_ci(total_fa_ft, total_n) if finetuned is not None else (float("nan"), float("nan"))
        print(f"{'TOTAL':>15} | {total_n:>13} | {total_fa_off / total_n * 100:>17.1f}% | "
              f"{(total_fa_ft / total_n * 100) if finetuned is not None else float('nan'):>23.1f}%")
        print(f"\nTOTAL 95% bootstrap CI -- off-the-shelf: [{off_ci[0] * 100:.1f}%, {off_ci[1] * 100:.1f}%]"
              + (f", fine-tuned: [{ft_ci[0] * 100:.1f}%, {ft_ci[1] * 100:.1f}%]" if finetuned is not None else ""))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n_per_category_requested": args.n_per_category,
                "rounds": args.rounds,
                "tau_low": TAU_LOW,
                "n_generated_total": len(all_triples),
                "n_qualifying_gray_zone": len(qualifying),
                "by_category_summary": by_category,
                "total_fa_off_the_shelf_pct": total_fa_off / total_n * 100 if total_n else None,
                "total_fa_off_the_shelf_ci95": off_ci if total_n else None,
                "total_fa_finetuned_pct": (total_fa_ft / total_n * 100) if total_n and finetuned is not None else None,
                "total_fa_finetuned_ci95": ft_ci if total_n and finetuned is not None else None,
                "all_triples": all_triples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
