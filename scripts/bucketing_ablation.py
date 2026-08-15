"""Direction 7 (RESEARCH_PROPOSAL.md §10 / memory): structured pre-filtering
(action-verb bucketing) as an upstream complement to verification.

Design converged on through the Reddit thread documented in
RESEARCH_PROPOSAL.md: bucket by a deterministic, rule-based MORPHOLOGICAL
extraction of the query's action verb (never a semantic/embedding-based
normalization -- see the "converged design principle" in that section for
why: a step is safe to use *in front of* the bucket boundary iff it can
never map two different roots onto the same key, which inflectional
lemmatization satisfies and semantic clustering does not). Uses spaCy's
rule-based dependency parser + lemmatizer, not a learned/semantic model, to
extract one action verb per query:
  - the sentence's dependency-parse ROOT if it is a VERB, lemmatized;
  - else, the sole VERB-tagged token if there is exactly one;
  - else, extraction fails (None) -- this pair falls outside the
    "action/object schema plausible to extract" subset RESEARCH_PROPOSAL.md
    already flagged as the honest scope of this ablation, and is excluded,
    not forced into a bucket.

For the extraction-eligible subset of each dataset's gray zone (both the
query's own action AND the matched candidate's original query's action
extract successfully), compares three conditions:
  - verifier-only: today's Group D (AUC over the eligible subset)
  - bucketing-only: predict "correct" iff the two extracted actions match,
    ignoring the verifier score entirely
  - bucketing+verifier: reject outright on an action mismatch; verifier AUC
    measured ONLY on the bucket-matched remainder -- this is the pair count
    the mechanism is supposed to shrink, and the AUC that's supposed to
    rise as a result.

Usage:
    python scripts/bucketing_ablation.py --config configs/lmarena.yaml \
        --output results/bucketing_ablation_lmarena.json
"""

import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_baselines import build_embedder
from cacheverifier.experiments.run_verified import _slug, CACHE_DIR
from cacheverifier.experiments.verified_sweep import (
    build_match_trace,
    load_match_trace,
    load_scored,
    resolve_candidate,
    save_match_trace,
    save_scored,
    score_gray_zone,
)
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier

TAU_LOW = 0.80
TAU_HIGH = 0.97


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, n_resamples: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = roc_auc(scores[idx], labels[idx])
    lo, hi = np.nanquantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def extract_action(nlp, text: str) -> str | None:
    doc = nlp(text)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None and root.pos_ == "VERB":
        return root.lemma_.lower()
    verbs = [t for t in doc if t.pos_ == "VERB"]
    if len(verbs) == 1:
        return verbs[0].lemma_.lower()
    return None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    dataset_path = Path(cfg.processed_path)
    log(f"Loading records from {dataset_path}...")
    records = load_jsonl(dataset_path)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    log(f"Loaded {len(records)} records")

    embedder = build_embedder(cfg.embedder, cfg.embedder_model)
    embedder_key = cfg.embedder if cfg.embedder != "sentence-transformer" else f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace_cache_path = CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}.trace.json"
    if trace_cache_path.exists():
        log(f"Loading cached match trace from {trace_cache_path}...")
        trace = load_match_trace(trace_cache_path)
    else:
        log("Building match trace...")
        trace = build_match_trace(records, embedder)
        save_match_trace(trace, trace_cache_path)

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    verifier_key = f"cross_encoder_{_slug(cross_encoder_model)}"
    scored_cache_path = (
        CACHE_DIR
        / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__{verifier_key}"
          f"__lo{TAU_LOW}__hi{TAU_HIGH}.scored.json"
    )
    if scored_cache_path.exists():
        log(f"Loading cached gray-zone scores from {scored_cache_path}...")
        scored = load_scored(scored_cache_path)
    else:
        log("Scoring gray-zone candidates with the off-the-shelf verifier...")
        verifier = CrossEncoderVerifier(cross_encoder_model)
        scored = score_gray_zone(records, trace, verifier, gray_zone_lo=TAU_LOW, gray_zone_hi=TAU_HIGH)
        save_scored(scored, scored_cache_path)

    gz_indices = [i for i in scored if trace[i].similarity is not None and TAU_LOW <= trace[i].similarity < TAU_HIGH]
    log(f"Gray zone: {len(gz_indices)} pairs")

    log("Loading spaCy en_core_web_sm...")
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])  # keep tagger/parser/lemmatizer, drop the rest for speed
    except OSError:
        # Falls back to a direct path load when the model was extracted from
        # a tarball rather than pip-installed (no dist-info metadata for
        # spacy.util.is_package to find) -- e.g. an offline transfer to a
        # machine without PyPI access to the model's own release wheel.
        import glob

        site_packages = Path(spacy.__file__).resolve().parent.parent
        candidates = sorted(glob.glob(str(site_packages / "en_core_web_sm" / "en_core_web_sm-*")))
        if not candidates:
            raise
        nlp = spacy.load(candidates[-1], disable=["ner", "textcat"])

    # Cache action extraction per unique query text -- candidates repeat
    # across many gray-zone pairs, and re-parsing the same string is wasted
    # work at this scale (tens of thousands of pairs, far fewer unique
    # query strings among them).
    action_cache: dict[str, str | None] = {}

    def get_action(text: str) -> str | None:
        if text not in action_cache:
            action_cache[text] = extract_action(nlp, text)
        return action_cache[text]

    log("Extracting actions for gray-zone query/candidate pairs...")
    t0 = time.time()
    query_actions, candidate_actions, labels, verifier_scores = [], [], [], []
    for i in gz_indices:
        record = records[i]
        candidate = resolve_candidate(records, trace[i])
        qa = get_action(record.query)
        ca = get_action(candidate.query)
        query_actions.append(qa)
        candidate_actions.append(ca)
        labels.append(1 if trace[i].would_be_correct else 0)
        verifier_scores.append(scored[i].score)
    log(f"  done in {time.time() - t0:.1f}s ({len(action_cache)} unique query strings parsed)")

    labels = np.array(labels)
    verifier_scores = np.array(verifier_scores)
    eligible = np.array([qa is not None and ca is not None for qa, ca in zip(query_actions, candidate_actions)])
    bucket_match = np.array(
        [qa == ca if (qa is not None and ca is not None) else False for qa, ca in zip(query_actions, candidate_actions)]
    )

    n_total = len(gz_indices)
    n_eligible = int(eligible.sum())
    log(f"Extraction-eligible: {n_eligible}/{n_total} ({n_eligible / n_total:.1%})")

    elig_labels = labels[eligible]
    elig_scores = verifier_scores[eligible]
    elig_bucket_match = bucket_match[eligible]

    # verifier-only, on the eligible subset
    verifier_auc = roc_auc(elig_scores, elig_labels)
    verifier_auc_ci = bootstrap_auc_ci(elig_scores, elig_labels)

    # bucketing-only: bucket_match itself as a binary predictor
    tp = int(((elig_bucket_match == 1) & (elig_labels == 1)).sum())
    fp = int(((elig_bucket_match == 1) & (elig_labels == 0)).sum())
    tn = int(((elig_bucket_match == 0) & (elig_labels == 0)).sum())
    fn = int(((elig_bucket_match == 0) & (elig_labels == 1)).sum())
    bucketing_precision = tp / (tp + fp) if (tp + fp) else float("nan")
    bucketing_recall = tp / (tp + fn) if (tp + fn) else float("nan")
    bucketing_false_approve_rate = fp / (tp + fp) if (tp + fp) else float("nan")  # bucket said yes, was wrong
    bucketing_false_reject_rate = fn / (tn + fn) if (tn + fn) else float("nan")  # bucket said no, was actually fine

    # bucketing+verifier: verifier AUC restricted to bucket-matched pairs only
    matched_scores = elig_scores[elig_bucket_match]
    matched_labels = elig_labels[elig_bucket_match]
    if len(matched_labels) >= 20 and len(set(matched_labels.tolist())) == 2:
        combined_auc = roc_auc(matched_scores, matched_labels)
        combined_auc_ci = bootstrap_auc_ci(matched_scores, matched_labels)
    else:
        combined_auc, combined_auc_ci = float("nan"), (float("nan"), float("nan"))

    result = {
        "dataset": cfg.name,
        "n_gray_zone_total": n_total,
        "n_extraction_eligible": n_eligible,
        "extraction_eligible_fraction": n_eligible / n_total,
        "n_bucket_matched_among_eligible": int(elig_bucket_match.sum()),
        "bucket_matched_fraction_among_eligible": float(elig_bucket_match.mean()),
        "positive_rate_eligible": float(elig_labels.mean()),
        "verifier_only": {
            "auc": verifier_auc,
            "auc_ci": list(verifier_auc_ci),
            "n": int(eligible.sum()),
        },
        "bucketing_only": {
            "precision": bucketing_precision,
            "recall": bucketing_recall,
            "false_approve_rate": bucketing_false_approve_rate,
            "false_reject_rate": bucketing_false_reject_rate,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "bucketing_plus_verifier": {
            "auc_on_bucket_matched_subset": combined_auc,
            "auc_ci": list(combined_auc_ci),
            "n": int(elig_bucket_match.sum()),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
