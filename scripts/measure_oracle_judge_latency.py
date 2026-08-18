"""Real API-latency measurement to replace PAPER.md Sec 4.4/§7's modeled
70ms Oracle (Group C) latency assumption ("代表 Krites 自己会调用的 API 级 LLM
judge (GPT-4.1-nano) 的量级...这个数字是一个明确标注的建模假设,不是实测值").

No GPT-4.1-nano access is available (no OpenAI key), so this uses DeepSeek
(deepseek-chat) as a stand-in API-hosted judge model -- the same substitution
principle already used and documented for the SearchQueries erratum fix
(Llama-3-8B -> DeepSeek, "没有可用的 Llama-3-8B API 访问").

Samples real (query, candidate cached answer) pairs from the gray-zone-shaped
distribution of LmArena/Quora/SearchQueries, sends each as a real equivalence-
judgment prompt structurally similar to what Krites' own async judge or this
paper's Group C oracle stands in for, and times the wall-clock request
latency (sequential calls -- concurrency would understate single-call
latency, which is what Section 5.5's added-latency-per-request statistic
needs).

Usage:
    python scripts/measure_oracle_judge_latency.py --n-samples 150 \
        --output results/oracle_judge_latency_deepseek.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
JUDGE_SYSTEM_PROMPT = (
    "You are a cache-hit verifier for a semantic LLM cache. You will be given "
    "a NEW user query and a CANDIDATE cached answer (originally generated for "
    "a different, similar past query). Decide whether the candidate answer is "
    "still a correct, complete answer to the new query. Respond with exactly "
    "one word: YES or NO. No explanation."
)

DATASETS = [
    ("data/processed/lmarena.jsonl", 60000),
    ("data/processed/quora.jsonl", 60000),
    ("data/processed/search_queries_corrected.jsonl", 150000),
]


def load_pairs(n_per_dataset: int, seed: int) -> list[tuple[str, str]]:
    """(query, candidate_answer) pairs, candidate drawn from a different
    record than the query -- mirrors what a gray-zone judge call actually
    sees (query paired with a NEIGHBOR's cached answer, not its own)."""
    rng = random.Random(seed)
    pairs = []
    for path, n_total in DATASETS:
        p = Path(path)
        if not p.exists():
            continue
        indices = rng.sample(range(n_total), min(n_per_dataset, n_total))
        records = []
        with p.open() as f:
            for i, line in enumerate(f):
                if i > max(indices):
                    break
                if i in indices:
                    records.append(json.loads(line))
        for i in range(len(records)):
            candidate = records[(i + 1) % len(records)]
            pairs.append((records[i]["query"], candidate["answer"]))
    return pairs


def call_deepseek_timed(api_key: str, query: str, candidate_answer: str, model: str) -> tuple[str, float]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_content = f"NEW QUERY:\n{query}\n\nCANDIDATE CACHED ANSWER:\n{candidate_answer}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }
    t0 = time.perf_counter()
    resp = requests.post(headers=headers, json=payload, url=DEEPSEEK_URL, timeout=60)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    verdict = resp.json()["choices"][0]["message"]["content"].strip()
    return verdict, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=50, help="Samples per dataset (3 datasets)")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/oracle_judge_latency_deepseek.json")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found (checked .env and environment).")

    pairs = load_pairs(args.n_samples, args.seed)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(pairs)} (query, candidate_answer) pairs "
          f"across {len(DATASETS)} datasets", flush=True)

    latencies_ms: list[float] = []
    verdicts: list[str] = []
    for i, (query, candidate) in enumerate(pairs, start=1):
        try:
            verdict, latency_ms = call_deepseek_timed(api_key, query, candidate, args.model)
        except Exception as e:
            print(f"  [{i}/{len(pairs)}] FAILED: {e}", flush=True)
            continue
        latencies_ms.append(latency_ms)
        verdicts.append(verdict)
        if i % 10 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] last_latency={latency_ms:.1f}ms  "
                  f"running_mean={sum(latencies_ms) / len(latencies_ms):.1f}ms", flush=True)

    latencies_ms.sort()
    n = len(latencies_ms)
    summary = {
        "model": args.model,
        "n_calls": n,
        "n_requested": len(pairs),
        "mean_ms": sum(latencies_ms) / n,
        "median_ms": latencies_ms[n // 2],
        "p95_ms": latencies_ms[int(n * 0.95)],
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
        "verdict_counts": {v: verdicts.count(v) for v in set(verdicts)},
        "raw_latencies_ms": latencies_ms,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_latencies_ms"}, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
