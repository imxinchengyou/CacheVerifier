"""Direction 15 v2, Step 0 (cont.): is SearchQueries-corrected's Protocol-T
CRC violation a real calibration drift, or (like Quora) a benchmark-label
artifact?

crc_step0_violation_stability.py showed SearchQueries-corrected violates
Protocol T robustly (every split fraction, block bootstrap CI excludes 0)
with a FLAT gray-zone base rate -- unlike Quora, whose violation is a
monotone base-rate gradient that crc_step0_quora_label_noise_check.py traced
to QQP annotation-batch noise (gradient +0.16 under QQP labels ->
+0.01 under a fresh judge).

SearchQueries' violation decomposes into two parts (see
results/crc_step0_score_label_drift_diagnostic.json):
  - P(accepted) rises across the stream  (verifier score distribution drifts
                                          up: mean_s 5.57 -> 6.04) -- covariate
                                          shift, detectable
  - P(err | accepted) rises, mostly a    (score -> label separation degrades
    jump in the last decile               in the high-score tail) -- P(Y|s)
                                          drift, the insidious part

This script tests whether the P(err | accepted) climb survives a fresh
DeepSeek answer-correctness judgment. Sample per-decile from the ACCEPTED
subpopulation (score > first-half CRC lambda at alpha=0.05, the largest
acceptance set), judge "is this cached answer still correct for the new
query", compare per-decile judge-error vs benchmark-error.

  - judge-error also climbs (esp. last decile)  -> real calibration drift;
                                                   SearchQueries is a genuine
                                                   Protocol-T violation and
                                                   the surviving RQ1 anchor
  - judge-error flat                             -> benchmark id_set label
                                                   artifact; the one
                                                   remaining natural CRC
                                                   violation is also not
                                                   trustworthy

Note SearchQueries uses vCache's native `id_set` equivalence labels (not
reconstructed here), and its answers were regenerated post-erratum
(scripts/generate_searchqueries_answers_deepseek.py), so this is the proper
"is the cached answer correct" judge, same prompt as §5.20.

Usage:
    python scripts/crc_step0_searchqueries_label_check.py --n-per-decile 120 --workers 60
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_verified import CACHE_DIR, _slug
from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored
from cacheverifier.metrics.core import crc_select_threshold

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
JUDGE_SYSTEM_PROMPT = (
    "You are a cache-hit verifier for a semantic LLM cache. You will be given "
    "a NEW user query and a CANDIDATE cached answer (originally generated for "
    "a different, similar past query). Decide whether the candidate answer is "
    "still a correct, complete answer to the new query. Respond with exactly "
    "one word: YES or NO. No explanation."
)
N_DECILES = 10
ANCHOR_ALPHA = 0.05  # lambda whose acceptance set we sample from


def call_judge(api_key, query, cand, model, retries=4):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"NEW QUERY:\n{query}\n\nCANDIDATE CACHED ANSWER:\n{cand}"},
        ],
        "temperature": 0.0, "max_tokens": 5,
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=45)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/search_queries_corrected.yaml")
    ap.add_argument("--n-per-decile", type=int, default=120)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--output", default="results/crc_step0_searchqueries_label_check.json")
    args = ap.parse_args()

    load_dotenv("/Users/xin/PycharmProjects/PythonProject/CacheVerifier/.env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found")

    cfg = load_dataset_config(args.config)
    records = load_jsonl(Path(cfg.processed_path))
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    stem = _slug(Path(cfg.processed_path).stem)
    trace = load_match_trace(CACHE_DIR / f"{stem}__precomputed__n{len(records)}.trace.json")
    scored = load_scored(
        CACHE_DIR / (f"{stem}__precomputed__n{len(records)}"
                     f"__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json")
    )

    gz = [i for i in range(len(trace)) if i in scored]
    n = len(gz)
    s = np.array([scored[i].score for i in gz])
    y = np.array([1 if trace[i].would_be_correct else 0 for i in gz])
    half = n // 2
    lam = crc_select_threshold(s[:half], y[:half], alpha=ANCHOR_ALPHA)
    accepted = s > lam
    edges = np.linspace(0, n, N_DECILES + 1).astype(int)
    rng = np.random.default_rng(args.seed)

    print(f"SearchQueries-corrected gray zone n={n}; first-half CRC lambda "
          f"(alpha={ANCHOR_ALPHA})={lam:.3f}; sampling {args.n_per_decile} ACCEPTED pairs/decile")
    print(f"{'dec':>4} {'n_accepted':>11} {'bench_acc_err':>13}")
    plan = []
    for d in range(N_DECILES):
        lo, hi = edges[d], edges[d + 1]
        loc = np.arange(lo, hi)
        acc_loc = loc[accepted[lo:hi]]
        bench_err = float(np.mean(y[acc_loc] == 0)) if len(acc_loc) else float("nan")
        take = rng.choice(acc_loc, size=min(args.n_per_decile, len(acc_loc)), replace=False)
        plan += [(d, int(gz[j])) for j in take]
        print(f"{d:>4} {len(acc_loc):>11} {bench_err:>13.3f}")

    def judge_one(item):
        d, ti = item
        q = records[ti].query
        cand = records[trace[ti].candidate_index].answer
        y_bench = 1 if trace[ti].would_be_correct else 0
        verdict = call_judge(api_key, q, cand, args.model)
        jc = verdict.startswith("YES")
        return {"decile": d, "trace_idx": ti, "score": float(scored[ti].score),
                "y_bench": y_bench, "judge_verdict": verdict, "judge_correct": jc,
                "query": q[:300], "candidate_answer": cand[:500]}

    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, it): it for it in plan}
        for fut in as_completed(futs):
            done += 1
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                dd, ti = futs[fut]
                print(f"  [{done}/{len(plan)}] FAILED (decile {dd}): {e}", flush=True)
            if done % 60 == 0 or done == len(plan):
                print(f"  [{done}/{len(plan)}]  {done / max(1e-6, time.time() - t0):.1f}/s  ok={len(rows)}", flush=True)

    print(f"\n{'dec':>4} {'n':>5} | {'bench_err':>10} | {'judge_err':>10} {'judge_err_CI95':>18} | {'agree':>7}")
    per_decile = []
    for d in range(N_DECILES):
        dr = [r for r in rows if r["decile"] == d]
        m = len(dr)
        bench_err = np.mean([1 - r["y_bench"] for r in dr]) if m else float("nan")
        je_k = sum(0 if r["judge_correct"] else 1 for r in dr)
        judge_err = je_k / m if m else float("nan")
        lo_ci, hi_ci = wilson(je_k, m)
        agree = np.mean([1 if (r["judge_correct"] == bool(r["y_bench"])) else 0 for r in dr]) if m else float("nan")
        per_decile.append({"decile": d, "n": m, "bench_err": float(bench_err), "judge_err": float(judge_err),
                           "judge_err_ci95": [float(lo_ci), float(hi_ci)], "agreement": float(agree)})
        print(f"{d:>4} {m:>5} | {bench_err:>10.3f} | {judge_err:>10.3f} [{lo_ci:.3f},{hi_ci:.3f}] | {agree:>7.3f}")

    def blk(dl, key, err_of):
        vals = [err_of(r) for r in rows if r["decile"] in dl]
        return float(np.mean(vals)) if vals else float("nan")
    e_bench = blk([0, 1, 2], "y_bench", lambda r: 1 - r["y_bench"])
    l_bench = blk([7, 8, 9], "y_bench", lambda r: 1 - r["y_bench"])
    e_judge = blk([0, 1, 2], "judge_correct", lambda r: 0 if r["judge_correct"] else 1)
    l_judge = blk([7, 8, 9], "judge_correct", lambda r: 0 if r["judge_correct"] else 1)
    # last decile alone -- that's where the benchmark jump was
    d9_bench = blk([9], "y_bench", lambda r: 1 - r["y_bench"])
    d9_judge = blk([9], "judge_correct", lambda r: 0 if r["judge_correct"] else 1)

    print(f"\n--- accepted-pair error: early deciles (0-2) vs late (7-9), and decile 9 alone ---")
    print(f"  benchmark: {e_bench:.3f} -> {l_bench:.3f}  (delta {l_bench - e_bench:+.3f});  decile9={d9_bench:.3f}")
    print(f"  judge:     {e_judge:.3f} -> {l_judge:.3f}  (delta {l_judge - e_judge:+.3f});  decile9={d9_judge:.3f}")
    verdict = ("real calibration drift" if (l_judge - e_judge) > 0.06 or (d9_judge - e_judge) > 0.08
               else "benchmark-label artifact" if abs(l_judge - e_judge) <= 0.04 and abs(d9_judge - e_judge) <= 0.05
               else "partial / ambiguous")
    print(f"  => {verdict}")

    out = {
        "dataset": "search_queries_corrected", "anchor_alpha": ANCHOR_ALPHA, "first_half_lambda": float(lam),
        "n_per_decile": args.n_per_decile, "model": args.model, "seed": args.seed, "n_judge_calls": len(rows),
        "per_decile": per_decile,
        "gradient": {
            "bench_err_early": e_bench, "bench_err_late": l_bench, "bench_err_decile9": d9_bench,
            "judge_err_early": e_judge, "judge_err_late": l_judge, "judge_err_decile9": d9_judge,
            "verdict": verdict,
        },
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
