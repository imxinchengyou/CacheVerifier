"""Direction 15 v1, must-fix C: measure the ORACLE JUDGE's own error rate on
real gray-zone matched pairs, stratified by |s - theta| (near / far the
decision threshold), so the selective-deferral analysis can plug in a real
eps_fa / eps_fr instead of assuming a perfect oracle.

Why stratified: the DEFERRED set is a selection-biased hard subset (scores
near theta); the judge is plausibly also worse there, so a pooled eps
measured on a random gray-zone sample would understate the oracle error on
exactly the pairs that get deferred. We report eps_near (drives the gate) and
eps_far (metadata).

Uses DeepSeek (deepseek-chat) as the stand-in API judge -- same substitution
principle and prompt as scripts/measure_oracle_judge_latency.py and the
SearchQueries erratum fix. Compares the judge's YES/NO verdict against the
dataset's own equivalence-class ground truth (`would_be_correct` in the match
trace):

    would_be_correct == True,  judge == NO   -> false reject  (eps_fr)
    would_be_correct == False, judge == YES  -> false accept   (eps_fa)

Note (Quora): its ground truth is reconstructed / noisy (PAPER.md §7), so on
Quora "judge vs ground truth" disagreement conflates judge error with label
error -- reported, not resolved.

Usage:
    python scripts/measure_oracle_judge_accuracy.py --config configs/lmarena.yaml \
        --scored-cache results/.cache/<...>.scored.json \
        --n-per-cell 50 --output results/oracle_judge_accuracy_lmarena_groupD.json
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

from cacheverifier.config import load_dataset_config
from cacheverifier.data.loaders import load_jsonl
from cacheverifier.experiments.run_verified import CACHE_DIR, _slug
from cacheverifier.experiments.verified_sweep import load_match_trace, load_scored

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
JUDGE_SYSTEM_PROMPT = (
    "You are a cache-hit verifier for a semantic LLM cache. You will be given "
    "a NEW user query and a CANDIDATE cached answer (originally generated for "
    "a different, similar past query). Decide whether the candidate answer is "
    "still a correct, complete answer to the new query. Respond with exactly "
    "one word: YES or NO. No explanation."
)
TAU_HIGH = 0.97
TAU_LOW = 0.80


def youden_j(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    j = np.cumsum(y == 1) / n_pos - np.cumsum(y == 0) / n_neg
    return float(s[int(np.argmax(j))])


def call_judge(api_key: str, query: str, candidate_answer: str, model: str) -> tuple[str, float]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"NEW QUERY:\n{query}\n\nCANDIDATE CACHED ANSWER:\n{candidate_answer}"},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }
    t0 = time.perf_counter()
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip().upper(), latency_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scored-cache", required=True)
    ap.add_argument("--n-per-cell", type=int, default=50,
                    help="pairs per (near/far x correct/incorrect) cell -> 4x this many judge calls")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    def log(m: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found")

    cfg = load_dataset_config(args.config)
    records = load_jsonl(Path(cfg.processed_path))
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    embedder_key = (cfg.embedder if cfg.embedder != "sentence-transformer"
                    else f"sentence-transformer_{_slug(cfg.embedder_model)}")
    trace = load_match_trace(CACHE_DIR / f"{_slug(Path(cfg.processed_path).stem)}__{embedder_key}__n{len(records)}.trace.json")
    scored = {i: sc.score for i, sc in load_scored(Path(args.scored_cache)).items()}

    gz = [i for i, t in enumerate(trace)
          if t.similarity is not None and TAU_LOW <= t.similarity < TAU_HIGH and i in scored]
    half = len(gz) // 2
    calib, test = gz[:half], gz[half:]
    theta = youden_j(np.array([scored[i] for i in calib]),
                     np.array([1 if trace[i].would_be_correct else 0 for i in calib]))
    log(f"theta_youden={theta:.3f}  gray-zone test n={len(test)}")

    dist = np.array([abs(scored[i] - theta) for i in test])
    lab = np.array([1 if trace[i].would_be_correct else 0 for i in test])
    near_cut = np.quantile(dist, 1 / 3)
    far_cut = np.quantile(dist, 2 / 3)
    strata = {
        "near": [test[j] for j in range(len(test)) if dist[j] <= near_cut],
        "far": [test[j] for j in range(len(test)) if dist[j] >= far_cut],
    }

    rng = np.random.default_rng(args.seed)
    plan: list[tuple[str, int, int]] = []  # (stratum, label, trace_idx)
    for sname, idxs in strata.items():
        idxs_arr = np.array(idxs)
        lab_s = np.array([1 if trace[i].would_be_correct else 0 for i in idxs])
        for y in (0, 1):
            pool = idxs_arr[lab_s == y]
            if len(pool) == 0:
                continue
            take = rng.choice(pool, size=min(args.n_per_cell, len(pool)), replace=False)
            plan += [(sname, y, int(i)) for i in take]
    log(f"planned {len(plan)} judge calls across 4 cells")

    results = []
    for k, (sname, y, i) in enumerate(plan, 1):
        q = records[i].query
        cand = records[trace[i].candidate_index].answer
        try:
            verdict, lat = call_judge(api_key, q, cand, args.model)
        except Exception as e:  # noqa: BLE001
            log(f"  [{k}/{len(plan)}] FAILED: {e}")
            time.sleep(2.0)
            continue
        judge_yes = verdict.startswith("YES")
        results.append({
            "stratum": sname, "would_be_correct": y, "score": scored[i],
            "dist_to_theta": abs(scored[i] - theta),
            "judge_verdict": verdict, "judge_yes": judge_yes, "latency_ms": lat,
            "query": q[:400], "candidate_answer": cand[:600],
        })
        if k % 20 == 0 or k == len(plan):
            log(f"  [{k}/{len(plan)}] last={verdict} ({sname}, y={y})")

    def rates(rows: list[dict]) -> dict:
        pos = [r for r in rows if r["would_be_correct"] == 1]
        neg = [r for r in rows if r["would_be_correct"] == 0]
        fr = sum(1 for r in pos if not r["judge_yes"])
        fa = sum(1 for r in neg if r["judge_yes"])
        return {
            "n_pos": len(pos), "n_neg": len(neg),
            "eps_fr": (fr / len(pos)) if pos else None, "fr_count": fr,
            "eps_fa": (fa / len(neg)) if neg else None, "fa_count": fa,
        }

    summary = {
        "config": args.config, "dataset": cfg.name, "model": args.model,
        "scored_cache": Path(args.scored_cache).name, "theta_youden": theta,
        "n_calls": len(results),
        "near": rates([r for r in results if r["stratum"] == "near"]),
        "far": rates([r for r in results if r["stratum"] == "far"]),
        "pooled": rates(results),
        "rows": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
