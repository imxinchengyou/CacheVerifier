"""Convert the Kaggle "Customer Support on Twitter" dataset (Stuart
Axelbrooke, 2017; github/kaggle: thoughtvector/customer-support-on-twitter)
into CacheVerifier's JSONL schema — a fourth, independently-sourced dataset, and
the first that is genuinely REAL production customer-support traffic (not a
research benchmark) with real per-tweet timestamps, letting the drift
experiment run on an actual continuous stream instead of reslicing a static
benchmark into synthetic chunks (see the Limitations section of PAPER.md /
PRODUCT_EXPERIMENTS.md, which flags exactly this gap).

Raw schema (twcs/twcs.csv): tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id. Company support accounts are
`inbound == False` rows whose `author_id` is the brand handle itself (e.g.
"AmazonHelp", "sprintcare"); customers are anonymized numeric author_ids.

Like Quora, this dataset has no LLM-generated `answer` column and no
dataset-native equivalence-class label, so both are reconstructed:

  - (query, answer) pairs: for every reply tweet from the target company
    (`inbound == False`, `author_id == --company`), follow
    `in_response_to_tweet_id` back to the customer tweet it replied to. That
    customer tweet's text is `query`; the company's reply text is `answer`.
    Only the FIRST company reply to a given customer tweet is used (some
    threads have multiple back-and-forth turns; the pipeline models one-shot
    cache lookups, not multi-turn conversations).
  - Equivalence classes: unlike Quora (which has human-labeled duplicate
    pairs), there is no ground-truth signal for "these two customer
    questions are the same issue." Instead this script uses a real
    behavioral signal grounded in what support agents actually did: two
    customer queries are placed in the same equivalence class iff the
    company's replies to them are near-duplicate text, clustered by cosine
    similarity of sentence embeddings (default threshold 0.92) via
    sentence-transformers' `util.community_detection`, rather than requiring
    exact string equality. Support teams reuse canned/templated replies for
    genuinely equivalent issues (e.g. "please DM us your account info"), but
    those replies are frequently personalized (customer name, order number,
    word-order variation) — an earlier exact-string-match version of this
    script under-counted true duplicates so badly it produced a >99% false
    accept rate under Group A (static threshold), which is a labeling
    artifact, not a real product signal. Embedding-similarity clustering
    recovers near-duplicate canned replies that differ only in
    personalization while still requiring actual textual similarity, not a
    guess based on the customer query's content.
  - Stream order: customer tweets are sorted by their real `created_at`
    timestamp (parsed from Twitter's native format) — this is actual
    chronological order, not first-appearance-in-file order.

Usage:
    python scripts/convert_twitter_cs_dataset.py \\
        --csv twcs/twcs.csv --company AmazonHelp \\
        --out data/processed/twitter_amazon.jsonl
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

MENTION_RE = re.compile(r"@\S+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = MENTION_RE.sub("", text)
    text = URL_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def canonicalize(text: str) -> str:
    return clean_text(text).lower()


def parse_created_at(s: str) -> datetime:
    # Twitter native format: "Tue Oct 31 22:10:47 +0000 2017"
    return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="twcs/twcs.csv", help="Path to the raw twcs.csv")
    parser.add_argument("--company", required=True, help="Support account author_id, e.g. AmazonHelp / sprintcare / comcastcares")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on the number of query-answer pairs emitted")
    parser.add_argument("--similarity-threshold", type=float, default=0.92,
                        help="Cosine similarity threshold on reply-text embeddings for two records to share an equivalence class")
    parser.add_argument("--embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    print(f"Reading {args.csv} ...")
    tweets: dict[str, dict] = {}
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets[row["tweet_id"]] = row
    print(f"  {len(tweets)} tweets loaded")

    pairs = []  # (created_at, tweet_id, query_text, answer_text)
    n_company_replies = 0
    n_missing_parent = 0
    n_parent_not_inbound = 0
    for row in tweets.values():
        if row["inbound"] == "True":
            continue
        if row["author_id"] != args.company:
            continue
        n_company_replies += 1
        parent_id = row["in_response_to_tweet_id"]
        if not parent_id or parent_id not in tweets:
            n_missing_parent += 1
            continue
        parent = tweets[parent_id]
        if parent["inbound"] != "True":
            n_parent_not_inbound += 1
            continue
        query_text = clean_text(parent["text"])
        answer_text = clean_text(row["text"])
        if not query_text or not answer_text:
            continue
        try:
            created_at = parse_created_at(parent["created_at"])
        except ValueError:
            continue
        pairs.append((created_at, parent["tweet_id"], query_text, answer_text))

    print(f"  {n_company_replies} replies from {args.company}, "
          f"{n_missing_parent} with missing/absent parent, "
          f"{n_parent_not_inbound} whose parent wasn't an inbound customer tweet")

    # First company reply per customer tweet only (dedupe by parent tweet_id,
    # keeping the earliest-sent reply if somehow duplicated).
    by_parent: dict[str, tuple] = {}
    for created_at, parent_id, q, a in pairs:
        if parent_id not in by_parent or created_at < by_parent[parent_id][0]:
            by_parent[parent_id] = (created_at, parent_id, q, a)

    ordered = sorted(by_parent.values(), key=lambda x: x[0])
    if args.max_samples is not None:
        ordered = ordered[: args.max_samples]
    print(f"  {len(ordered)} unique (query, answer) pairs after dedup, sorted by real created_at")

    # Equivalence classes via embedding-similarity clustering on canonicalized
    # answer text (near-duplicate canned replies, not requiring exact match).
    print(f"Encoding {len(ordered)} reply texts with {args.embedder_model} for equivalence clustering...")
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(args.embedder_model)
    canon_answers = [canonicalize(a) for _, _, _, a in ordered]
    embeddings = model.encode(canon_answers, batch_size=256, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

    print(f"Clustering replies at cosine similarity >= {args.similarity_threshold} ...")
    clusters = util.community_detection(embeddings, threshold=args.similarity_threshold, min_community_size=2)

    # equivalence_id defaults to each record's own parent tweet_id (singleton
    # class); records assigned to a cluster share the cluster's first member's
    # tweet_id instead.
    equivalence_ids = [parent_id for _, parent_id, _, _ in ordered]
    for cluster in clusters:
        rep_id = equivalence_ids[cluster[0]]
        for idx in cluster:
            equivalence_ids[idx] = rep_id

    class_sizes: dict[str, int] = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, (created_at, parent_id, query_text, answer_text) in enumerate(ordered):
            equivalence_id = equivalence_ids[i]
            class_sizes[equivalence_id] = class_sizes.get(equivalence_id, 0) + 1
            record = {
                "query_id": str(i),
                "query": query_text,
                "answer": answer_text,
                "equivalence_id": equivalence_id,
                "created_at": created_at.isoformat(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_classes_multi = sum(1 for size in class_sizes.values() if size > 1)
    n_in_multi = sum(v for v in class_sizes.values() if v > 1)
    print(f"Wrote {len(ordered)} records to {out_path}")
    print(f"  {len(class_sizes)} equivalence classes total, {n_classes_multi} with 2+ members "
          f"({n_in_multi} records in a non-singleton class, {n_in_multi / len(ordered):.1%} of total)")
    if ordered:
        print(f"  time span: {ordered[0][0].isoformat()} .. {ordered[-1][0].isoformat()}")


if __name__ == "__main__":
    main()
