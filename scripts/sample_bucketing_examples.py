"""One-off: pull real examples from the search_queries_corrected bucketing
ablation run to ground Reddit post 4 (social_media/reddit/post_4_...) in
actual data instead of made-up illustrations. Reuses the same caches
bucketing_ablation.py already built (match trace, gray-zone scores) --
doesn't re-run anything expensive.

Samples:
  1. A few gray-zone pairs where action extraction failed on one or both
     sides (the 72.7% excluded from the mechanism entirely).
  2. A few gray-zone pairs where extraction succeeded but the two actions
     did NOT match (bucket would reject), split by whether the pair was
     actually correct or not.
  3. A few pairs where actions DID match (bucket would accept), split the
     same way.

Usage:
    python scripts/sample_bucketing_examples.py --config configs/search_queries_corrected.yaml
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.run_verified import _slug, CACHE_DIR
from cacheverifier.experiments.verified_sweep import (
    load_match_trace,
    load_scored,
    resolve_candidate,
)

TAU_LOW = 0.80
TAU_HIGH = 0.97


def extract_action(nlp, text: str) -> str | None:
    doc = nlp(text)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None and root.pos_ == "VERB":
        return root.lemma_.lower()
    verbs = [t for t in doc if t.pos_ == "VERB"]
    if len(verbs) == 1:
        return verbs[0].lemma_.lower()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-per-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    dataset_path = Path(cfg.processed_path)
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}.trace.json"
    trace = load_match_trace(trace_cache_path)

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    verifier_key = f"cross_encoder_{_slug(cross_encoder_model)}"
    scored_cache_path = (
        CACHE_DIR
        / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__{verifier_key}"
          f"__lo{TAU_LOW}__hi{TAU_HIGH}.scored.json"
    )
    scored = load_scored(scored_cache_path)

    gz_indices = [i for i in scored if trace[i].similarity is not None and TAU_LOW <= trace[i].similarity < TAU_HIGH]

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    except OSError:
        import glob
        import spacy as spacy_mod
        site_packages = Path(spacy_mod.__file__).resolve().parent.parent
        candidates = sorted(glob.glob(str(site_packages / "en_core_web_sm" / "en_core_web_sm-*")))
        nlp = spacy_mod.load(candidates[-1], disable=["ner", "textcat"])

    action_cache: dict[str, str | None] = {}

    def get_action(text: str) -> str | None:
        if text not in action_cache:
            action_cache[text] = extract_action(nlp, text)
        return action_cache[text]

    buckets = {
        "extraction_failed": [],
        "mismatch_but_correct": [],
        "mismatch_and_wrong": [],
        "match_but_wrong": [],
        "match_and_correct": [],
    }

    rng = random.Random(args.seed)
    shuffled = gz_indices[:]
    rng.shuffle(shuffled)

    for i in shuffled:
        record = records[i]
        candidate = resolve_candidate(records, trace[i])
        qa = get_action(record.query)
        ca = get_action(candidate.query)
        correct = trace[i].would_be_correct

        if qa is None or ca is None:
            key = "extraction_failed"
        elif qa != ca:
            key = "mismatch_but_correct" if correct else "mismatch_and_wrong"
        else:
            key = "match_but_wrong" if not correct else "match_and_correct"

        if len(buckets[key]) < args.n_per_bucket:
            buckets[key].append((record.query, candidate.query, qa, ca, correct))

        if all(len(v) >= args.n_per_bucket for v in buckets.values()):
            break

    for key, examples in buckets.items():
        print(f"\n=== {key} ({len(examples)} shown) ===")
        for q, c, qa, ca, correct in examples:
            print(f"  query:     {q!r}")
            print(f"  candidate: {c!r}")
            print(f"  actions:   query={qa!r}  candidate={ca!r}  would_be_correct={correct}")
            print()


if __name__ == "__main__":
    main()
