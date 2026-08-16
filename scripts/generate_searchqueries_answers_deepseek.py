"""Generate real per-query answers for SearchQueries records, replacing the
constant placeholder "Not required for the benchmark because of the id_set"
that the upstream `vCache/SemBenchmarkSearchQueries` HF dataset ships in its
`response_llama_3_8b` column (verified empty/unpopulated for all 150,000
rows -- see scripts/searchqueries_answer_placeholder_check.py).

Uses the DeepSeek API (OpenAI-compatible REST, chat completions) rather than
actually running Llama-3-8B locally -- no API access to that specific model
was available. PAPER.md's methodology text must say "DeepSeek" here, not
"Llama-3-8B", once this is wired into a corrected Group D/E run.

Only generates answers for the specific record indices that actually show
up as either the query-side or candidate-side of a gray-zone
([0.80, 0.97)) match -- that's the only place `.answer` text is ever read
by the verifier (see cacheverifier/verifiers/cross_encoder_verifier.py).
Reads that index list from
results/searchqueries_gray_zone_needed_indices.json (produced by
scripts/_count_gray_zone_candidates.py).

Resumable: writes one JSON line per completed record to the output file
immediately, and skips query_ids already present on a re-run, so an
interrupted run can just be restarted with the same command.

Usage:
    python scripts/generate_searchqueries_answers_deepseek.py \\
        --indices results/searchqueries_gray_zone_needed_indices.json \\
        --output data/processed/search_queries_answers_deepseek.jsonl
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = (
    "You are a helpful search assistant. Answer the user's short search "
    "query directly and concisely, the way a good search-result snippet "
    "or featured answer would. 1-3 sentences. Do not ask clarifying "
    "questions, do not add disclaimers -- just answer as best you can from "
    "the query text alone."
)


def call_deepseek(api_key: str, query: str, model: str, max_retries: int = 5) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            if resp.status_code in (429, 500, 502, 503, 529):
                time.sleep(min(2**attempt, 30))
                continue
            resp.raise_for_status()
        except (requests.RequestException, KeyError, IndexError):
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"DeepSeek call failed after {max_retries} retries for query={query!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/search_queries.yaml")
    parser.add_argument("--indices", default="results/searchqueries_gray_zone_needed_indices.json")
    parser.add_argument("--output", default="data/processed/search_queries_answers_deepseek.jsonl")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests")
    args = parser.parse_args()

    load_dotenv()
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found (checked .env and environment). Set it and retry.")

    cfg = load_dataset_config(args.config)
    records = load_jsonl(Path(cfg.processed_path))
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]

    needed_indices = json.loads(Path(args.indices).read_text())
    print(f"{len(needed_indices)} records need a real generated answer")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["query_id"])
        print(f"Resuming: {len(already_done)} already done in {out_path}")

    todo = [i for i in needed_indices if records[i].query_id not in already_done]
    print(f"{len(todo)} remaining to generate")

    lock = threading.Lock()
    write_f = out_path.open("a", encoding="utf-8")
    done_count = [len(already_done)]
    t0 = time.time()

    def worker(idx: int):
        record = records[idx]
        answer = call_deepseek(api_key, record.query, args.model)
        with lock:
            write_f.write(json.dumps({"query_id": record.query_id, "answer": answer}, ensure_ascii=False) + "\n")
            write_f.flush()
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                elapsed = time.time() - t0
                rate = (done_count[0] - len(already_done)) / elapsed if elapsed > 0 else 0
                remaining = len(todo) - (done_count[0] - len(already_done))
                eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                print(f"  {done_count[0]}/{len(needed_indices)}  {rate:.1f} req/s  eta={eta_min:.1f}m")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, idx) for idx in todo]
        for fut in as_completed(futures):
            fut.result()  # surface exceptions immediately

    write_f.close()
    print(f"Done. Wrote answers to {out_path}")


if __name__ == "__main__":
    main()
