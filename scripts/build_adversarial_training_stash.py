"""Builds a combined natural+adversarial training stash for the adversarial
fine-tuning experiment (follow-up to Section 5.18's finding that in-domain
fine-tuning provides no protection against LLM-generated adversarial
samples).

Combines:
  - the EXISTING natural LmArena gray-zone train/test split
    (results/finetune_verifier_experiment.examples.json -- the same stash
    Section 5.6's original LmArena fine-tuning used, reused unmodified so
    the natural-data test set stays untouched and comparable)
  - a FRESH adversarial training pool (results/llm_redteam_train_pool.json,
    generated separately from Section 5.18's 306-sample held-out test set
    -- a new, independent stochastic draw from the same 5-category prompt
    templates, not overlapping with the held-out set by construction)

Each qualifying adversarial triple (query_a, query_b, answer_b) becomes TWO
training rows: (query_a, answer_b) -> label 0 (the adversarial negative --
this is exactly the case Section 5.18 showed both off-the-shelf and
naturally-fine-tuned verifiers approve at 84-88%) and (query_b, answer_b) ->
label 1 (the correct pairing, giving the model a contrasting positive
example in the same style rather than only ever seeing hard negatives).

Output stash format matches scripts/finetune_verifier_train_eval.py's
expected input exactly ({"train": [...], "test": [...]}), so that script
runs UNMODIFIED on this stash -- same training procedure (epochs=1,
batch_size=16, off-the-shelf base) as Section 5.6's original run, so the
only variable between that run and this one is the training DATA (natural
only vs. natural + adversarial), not the training method.

Usage:
    python scripts/build_adversarial_training_stash.py
"""

import json
from pathlib import Path

NATURAL_STASH = "results/finetune_verifier_experiment.examples.json"
ADVERSARIAL_POOL = "results/llm_redteam_train_pool.json"
OUTPUT = "results/finetune_adversarial_stash.examples.json"
TAU_LOW = 0.80


def main() -> None:
    natural = json.loads(Path(NATURAL_STASH).read_text(encoding="utf-8"))
    natural_train, natural_test = natural["train"], natural["test"]
    print(f"Natural LmArena stash: {len(natural_train)} train, {len(natural_test)} test (unchanged)")

    pool = json.loads(Path(ADVERSARIAL_POOL).read_text(encoding="utf-8"))
    qualifying = [t for t in pool["all_triples"] if t.get("similarity", 0.0) >= TAU_LOW]
    print(f"Adversarial training pool: {len(qualifying)}/{len(pool['all_triples'])} triples cleared tau_low={TAU_LOW}")

    by_category: dict[str, int] = {}
    adversarial_rows = []
    for i, t in enumerate(qualifying):
        by_category[t["category"]] = by_category.get(t["category"], 0) + 1
        # negative: query_a paired with the WRONG answer (answer_b) -- the exact
        # adversarial case Section 5.18 showed both verifiers false-accept.
        adversarial_rows.append((f"adv_neg_{i}", t["query_a"], t["answer_b"], False))
        # positive: query_b paired with its own correct answer -- a contrasting
        # example in the same surface style, not just more hard negatives.
        adversarial_rows.append((f"adv_pos_{i}", t["query_b"], t["answer_b"], True))

    print("By category:", by_category)
    print(f"Adversarial training rows: {len(adversarial_rows)} ({len(qualifying)} triples x 2)")

    combined_train = natural_train + adversarial_rows
    combined = {"train": combined_train, "test": natural_test}

    n_adv_frac = len(adversarial_rows) / len(combined_train)
    print(f"Combined train set: {len(combined_train)} rows "
          f"({len(natural_train)} natural + {len(adversarial_rows)} adversarial, "
          f"{n_adv_frac:.1%} adversarial)")
    print(f"Test set: {len(natural_test)} rows (natural only, unchanged -- adversarial "
          f"robustness is evaluated separately against Section 5.18's held-out 306-sample set)")

    Path(OUTPUT).write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
