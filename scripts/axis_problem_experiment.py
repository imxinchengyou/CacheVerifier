"""Direction 5 (RESEARCH_PROPOSAL.md §10): does the off-the-shelf verifier's
SearchQueries failure track "raw query length" or a specific structural
axis -- confusable ACTION words on a shared OBJECT ("cancel" vs "pause my
subscription") -- independent of length?

Builds two controlled pair families from the same template pool and
vocabulary style (so query length is matched by construction, not just
asserted):

  Family A (fixed object, varied action): pairs share the object
      ("subscription") and differ only in the action verb ("cancel" vs
      "pause"). This is the structure of the cancel/pause example from the
      launch post's comment thread.
  Family B (fixed action, varied object): pairs share the action ("cancel")
      and differ only in the object ("subscription" vs "order"). Control
      group -- same template/vocabulary style, different axis of confusion.

For every pair, `candidate_query` is the paraphrase already "in the cache"
and `candidate_answer` is its canonical answer. Two signals are scored per
pair, matching how the rest of this codebase splits the two decisions:
  - similarity: cosine(embed(query), embed(candidate_query)) -- what a real
    semantic cache uses to decide/rank hits (cacheverifier.cache.store).
  - verifier: cross-encoder(query, candidate_answer) -- what
    cacheverifier.verifiers.cross_encoder_verifier.CrossEncoderVerifier
    scores (Group D in PAPER.md), same default model
    (cross-encoder/ms-marco-MiniLM-L6-v2).

Hypothesis under test: both signals' AUC (separating true "would be
correct" pairs from false ones) should be lower on Family A than Family B,
even though the two families are length-matched -- i.e. the failure is
about the confusable axis, not raw shortness.

Usage:
    python scripts/axis_problem_experiment.py --output results/axis_problem_experiment.json
"""

import argparse
import itertools
import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SEED = 0

TEMPLATES_LONG = [
    "how do I {action} my {object}",
    "I want to {action} my {object}",
    "can you help me {action} my {object}",
    "I need to {action} my {object}",
    "is there a way to {action} my {object}",
    "what's the process to {action} my {object}",
    "trying to {action} my {object}, not sure how",
    "can I {action} my {object} online",
    "{object} -- how do I {action} it",
    "please {action} my {object} asap",
    "hey, how do i {action} my {object}",  # casual/lowercase
    "wheres the option to {action} my {object}",  # typo (missing apostrophe)
]

# Keyword-dense, SearchQueries-style phrasing -- no filler words, matching
# the paper's own description of that dataset ("短查询、关键词密集").
TEMPLATES_SHORT = [
    "{action} {object}",
    "{action} my {object}",
    "{object} {action}",
    "{object} {action} help",
    "how {action} {object}",
    "{action} {object} now",
    "need to {action} {object}",
    "{object}: {action}",
    "{action} {object} online",
    "{object} {action} option",
]

ACTIONS_PER_OBJECT = {
    # v3: dropped near-synonym pairs found in v2's hardest cases --
    # "reactivate" (~= renew), "deactivate" (~= freeze), "restore" (~=
    # reactivate), "postpone" (~= reschedule) -- replaced with actions that
    # are topically related (same gray zone) but not literal synonyms.
    "subscription": ["cancel", "pause", "renew", "upgrade", "downgrade", "transfer"],
    "order": ["cancel", "return", "refund", "exchange", "track", "expedite"],
    "account": ["delete", "freeze", "verify", "restore", "merge", "transfer"],
    "appointment": ["cancel", "reschedule", "confirm", "extend", "transfer"],
    "shipment": ["track", "redirect", "hold", "expedite", "cancel"],
}

OBJECTS_PER_ACTION = {
    # v3: dropped near-synonym pairs found in v2's hardest cases --
    # "membership" (~= subscription), "reservation" (~= appointment),
    # "settings" (~= preferences) -- replaced with objects from clearly
    # different real-world domains, same fix as ACTIONS_PER_OBJECT above.
    "cancel": ["subscription", "order", "appointment", "flight", "insurance policy", "event ticket"],
    "update": ["email", "address", "password", "payment method", "phone number", "username"],
    "reset": ["password", "account", "device", "router", "printer", "voicemail"],
    "verify": ["email", "account", "identity", "payment method", "phone number"],
    "delete": ["account", "order", "review", "payment method", "address"],
}


def canonical_answer(action: str, obj: str) -> str:
    return f"Go to Settings > {obj.title()} > {action.title()}."


@dataclass(frozen=True)
class Pair:
    query: str
    candidate_query: str
    candidate_answer: str
    label: bool  # would serving candidate_answer for `query` be correct?
    family: str  # "A" (fixed object, varied action) or "B" (fixed action, varied object)
    length_condition: str  # "long" (full sentence) or "short" (keyword-dense)


def build_family_a(rng: random.Random, templates: list[str], length_condition: str) -> list[Pair]:
    pairs: list[Pair] = []
    for obj, actions in ACTIONS_PER_OBJECT.items():
        canon_template = {a: rng.choice(templates) for a in actions}
        canon = {
            a: (canon_template[a].format(action=a, object=obj), canonical_answer(a, obj))
            for a in actions
        }
        for action in actions:
            cq, ca = canon[action]
            positive_templates = [t for t in templates if t != canon_template[action]]
            for template in positive_templates:
                q = template.format(action=action, object=obj)
                pairs.append(Pair(q, cq, ca, True, "A", length_condition))
            for other in actions:
                if other == action:
                    continue
                template = rng.choice(templates)
                q = template.format(action=action, object=obj)
                other_cq, other_ca = canon[other]
                pairs.append(Pair(q, other_cq, other_ca, False, "A", length_condition))
    return pairs


def build_family_b(rng: random.Random, templates: list[str], length_condition: str) -> list[Pair]:
    pairs: list[Pair] = []
    for action, objects in OBJECTS_PER_ACTION.items():
        canon_template = {o: rng.choice(templates) for o in objects}
        canon = {
            o: (canon_template[o].format(action=action, object=o), canonical_answer(action, o))
            for o in objects
        }
        for obj in objects:
            cq, ca = canon[obj]
            positive_templates = [t for t in templates if t != canon_template[obj]]
            for template in positive_templates:
                q = template.format(action=action, object=obj)
                pairs.append(Pair(q, cq, ca, True, "B", length_condition))
            for other in objects:
                if other == obj:
                    continue
                template = rng.choice(templates)
                q = template.format(action=action, object=obj)
                other_cq, other_ca = canon[other]
                pairs.append(Pair(q, other_cq, other_ca, False, "B", length_condition))
    return pairs


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Plain rank-based AUC, no sklearn dependency surprises (matches
    scripts/finetune_verifier_train_eval.py::roc_auc)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = roc_auc(scores[idx], labels[idx])
    lo, hi = np.nanquantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--verifier-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output", default="results/axis_problem_experiment.json")
    args = parser.parse_args()

    rng = random.Random(SEED)
    pairs = (
        build_family_a(rng, TEMPLATES_LONG, "long")
        + build_family_b(rng, TEMPLATES_LONG, "long")
        + build_family_a(rng, TEMPLATES_SHORT, "short")
        + build_family_b(rng, TEMPLATES_SHORT, "short")
    )
    print(f"Built {len(pairs)} pairs across 4 cells (family x length_condition)")

    print(f"Loading embedder {args.embed_model!r}...")
    from sentence_transformers import SentenceTransformer, CrossEncoder

    embedder = SentenceTransformer(args.embed_model)
    unique_texts = sorted({p.query for p in pairs} | {p.candidate_query for p in pairs})
    vecs = embedder.encode(unique_texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    text_to_vec = dict(zip(unique_texts, vecs))

    similarity = np.array([float(np.dot(text_to_vec[p.query], text_to_vec[p.candidate_query])) for p in pairs])

    print(f"Loading verifier {args.verifier_model!r}...")
    cross_encoder = CrossEncoder(args.verifier_model)
    verifier_scores = np.array(
        cross_encoder.predict([(p.query, p.candidate_answer) for p in pairs], show_progress_bar=False)
    )

    labels_all = np.array([1 if p.label else 0 for p in pairs])
    families = np.array([p.family for p in pairs])
    length_conditions = np.array([p.length_condition for p in pairs])
    lengths = np.array([len(p.query.split()) for p in pairs])

    result = {"config": vars(args), "n_pairs": len(pairs), "cells": {}}
    print(f"\n{'cell':<10}{'n':<6}{'pos_rate':<10}{'mean_len(words)':<18}{'sim_AUC':<10}{'95% CI':<18}{'verifier_AUC':<14}{'95% CI'}")
    for length_cond in ("long", "short"):
        for fam in ("A", "B"):
            mask = (families == fam) & (length_conditions == length_cond)
            labels = labels_all[mask]
            sim_auc = roc_auc(similarity[mask], labels)
            sim_lo, sim_hi = bootstrap_auc_ci(similarity[mask], labels)
            ver_auc = roc_auc(verifier_scores[mask], labels)
            ver_lo, ver_hi = bootstrap_auc_ci(verifier_scores[mask], labels)
            mean_len = float(lengths[mask].mean())
            pos_rate = float(labels.mean())
            cell = f"{fam}-{length_cond}"
            print(f"{cell:<10}{int(mask.sum()):<6}{pos_rate:<10.2f}{mean_len:<18.2f}{sim_auc:<10.4f}"
                  f"[{sim_lo:.3f},{sim_hi:.3f}]{'':<3}{ver_auc:<14.4f}[{ver_lo:.3f},{ver_hi:.3f}]")
            result["cells"][cell] = {
                "family": fam,
                "length_condition": length_cond,
                "n": int(mask.sum()),
                "positive_rate": pos_rate,
                "mean_query_length_words": mean_len,
                "similarity_auc": sim_auc,
                "similarity_auc_ci": [sim_lo, sim_hi],
                "verifier_auc": ver_auc,
                "verifier_auc_ci": [ver_lo, ver_hi],
                "oracle_gap_similarity": 1.0 - sim_auc,
            }

    result["pairs"] = [
        {**asdict(p), "similarity": float(s), "verifier_score": float(v)}
        for p, s, v in zip(pairs, similarity, verifier_scores)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
