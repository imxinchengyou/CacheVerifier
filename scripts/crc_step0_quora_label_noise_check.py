"""Direction 15 v2, Step 0 (cont.): is Quora's monotone gray-zone error-rate
gradient across stream position (0.45 -> 0.68, the driver of its Protocol-T
CRC violation) a REAL difficulty drift, or QQP reconstructed-label noise
concentrated in the later part of the file?

crc_step0_violation_stability.py established the Quora violation is robust
(every split fraction, block bootstrap CI excludes 0, directional). But
Quora's labels come from union-find over QQP's human `is_duplicate`
annotations (scripts/convert_quora_dataset.py) and PAPER.md §7 flags them as
noisy. If annotation quality itself drifts across the QQP file, the "error
rate rises across the stream" gradient could be a labeling artifact, not a
real shift in how hard the duplicates are.

Test: for a stream-position-decile-stratified sample of Quora gray-zone
pairs, get a fresh DeepSeek "are these the same question" verdict, and check
whether the per-decile error rate under JUDGE labels still shows the
gradient. Also report judge-vs-QQP agreement per decile (a drop in later
deciles => QQP label quality degraded there).

  - gradient persists under judge labels  -> real difficulty drift; Quora is
                                             a valid (if file-order-caveated)
                                             secondary RQ1 example
  - gradient flattens under judge labels   -> QQP label noise; Quora's
                                             Protocol-T violation is not
                                             trustworthy, drop it

Note this does NOT rehabilitate Quora's stream order (still QQP-file order,
not time) -- SearchQueries-corrected is the primary RQ1 benchmark regardless.

Usage:
    python scripts/crc_step0_quora_label_noise_check.py --n-per-decile 80
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

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
JUDGE_SYSTEM_PROMPT = (
    "You are judging whether two user questions are duplicates -- i.e. they "
    "ask the same thing and a good answer to one is a good answer to the "
    "other. Respond with exactly one word: YES or NO. No explanation."
)
TAU_LOW, TAU_HIGH = 0.80, 0.97
N_DECILES = 10


def call_judge(api_key: str, q_new: str, q_cand: str, model: str, retries: int = 4) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION A:\n{q_new}\n\nQUESTION B:\n{q_cand}"},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }
    last = None
    for attempt in range(retries):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=45)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quora.yaml")
    ap.add_argument("--n-per-decile", type=int, default=80)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--output", default="results/crc_step0_quora_label_noise_check.json")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found")

    cfg = load_dataset_config(args.config)
    records = load_jsonl(Path(cfg.processed_path))
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    embedder_key = f"sentence-transformer_{_slug(cfg.embedder_model)}"
    trace = load_match_trace(
        CACHE_DIR / f"{_slug(Path(cfg.processed_path).stem)}__{embedder_key}__n{len(records)}.trace.json"
    )
    scored = load_scored(
        CACHE_DIR / (f"{_slug(Path(cfg.processed_path).stem)}__{embedder_key}__n{len(records)}"
                     f"__cross_encoder_cross-encoder_ms-marco-MiniLM-L6-v2__lo0.8__hi0.97.scored.json")
    )

    gz = [i for i in range(len(trace)) if i in scored]
    n = len(gz)
    labels_qqp = np.array([1 if trace[i].would_be_correct else 0 for i in gz])
    edges = np.linspace(0, n, N_DECILES + 1).astype(int)
    rng = np.random.default_rng(args.seed)

    print(f"Quora gray zone n={n}; sampling {args.n_per_decile}/decile "
          f"({args.n_per_decile * N_DECILES} judge calls)")
    print(f"{'dec':>4} {'idx_range':>16} {'n_dec':>7} {'qqp_err':>8}")
    plan = []
    for d in range(N_DECILES):
        lo, hi = edges[d], edges[d + 1]
        pool = np.arange(lo, hi)
        qqp_err_dec = float(np.mean(labels_qqp[lo:hi] == 0))
        take = rng.choice(pool, size=min(args.n_per_decile, len(pool)), replace=False)
        plan += [(d, int(gz[j])) for j in take]
        print(f"{d:>4} {f'[{lo},{hi})':>16} {hi - lo:>7} {qqp_err_dec:>8.4f}")

    def judge_one(item):
        d, ti = item
        q_new = records[ti].query
        q_cand = records[trace[ti].candidate_index].answer  # answer == query for Quora
        y_qqp = 1 if trace[ti].would_be_correct else 0
        verdict = call_judge(api_key, q_new, q_cand, args.model)
        judge_dup = verdict.startswith("YES")
        return {
            "decile": d, "trace_idx": ti, "score": scored[ti].score,
            "y_qqp": y_qqp, "judge_verdict": verdict, "judge_duplicate": judge_dup,
            "q_new": q_new[:300], "q_cand": q_cand[:300],
        }

    rows = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, it): it for it in plan}
        for fut in as_completed(futs):
            done += 1
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                d, ti = futs[fut]
                print(f"  [{done}/{len(plan)}] FAILED (decile {d}, idx {ti}): {e}", flush=True)
            if done % 40 == 0 or done == len(plan):
                rate = done / max(1e-6, time.time() - t0)
                print(f"  [{done}/{len(plan)}]  {rate:.1f}/s  ok={len(rows)}", flush=True)

    # ---- per-decile analysis -------------------------------------------
    print(f"\n{'dec':>4} {'n':>5} | {'qqp_err':>8} | {'judge_err':>9} {'judge_err_CI95':>18} | "
          f"{'agree':>7} | {'qqp_dup':>8} {'judge_dup':>9}")
    per_decile = []
    for d in range(N_DECILES):
        dr = [r for r in rows if r["decile"] == d]
        m = len(dr)
        qqp_err = np.mean([1 - r["y_qqp"] for r in dr])
        judge_err = np.mean([0 if r["judge_duplicate"] else 1 for r in dr])  # judge NO = "not a real hit" = error
        je_lo, je_hi = wilson(sum(0 if r["judge_duplicate"] else 1 for r in dr), m)
        agree = np.mean([1 if (r["judge_duplicate"] == bool(r["y_qqp"])) else 0 for r in dr])
        qqp_dup = np.mean([r["y_qqp"] for r in dr])
        judge_dup = np.mean([1 if r["judge_duplicate"] else 0 for r in dr])
        per_decile.append({
            "decile": d, "n": m, "qqp_err": float(qqp_err), "judge_err": float(judge_err),
            "judge_err_ci95": [float(je_lo), float(je_hi)],
            "judge_qqp_agreement": float(agree),
            "qqp_dup_rate": float(qqp_dup), "judge_dup_rate": float(judge_dup),
        })
        print(f"{d:>4} {m:>5} | {qqp_err:>8.3f} | {judge_err:>9.3f} [{je_lo:.3f},{je_hi:.3f}] | "
              f"{agree:>7.3f} | {qqp_dup:>8.3f} {judge_dup:>9.3f}")

    # gradient tests: decile 0-2 vs decile 7-9
    def block(dlist, key):
        vals = [r[key] for r in rows if r["decile"] in dlist]
        return vals
    early_qqp_err = np.mean([1 - v for v in block([0, 1, 2], "y_qqp")])
    late_qqp_err = np.mean([1 - v for v in block([7, 8, 9], "y_qqp")])
    early_judge_err = np.mean([0 if v else 1 for v in block([0, 1, 2], "judge_duplicate")])
    late_judge_err = np.mean([0 if v else 1 for v in block([7, 8, 9], "judge_duplicate")])
    early_agree = np.mean([1 if (r["judge_duplicate"] == bool(r["y_qqp"])) else 0
                           for r in rows if r["decile"] in (0, 1, 2)])
    late_agree = np.mean([1 if (r["judge_duplicate"] == bool(r["y_qqp"])) else 0
                          for r in rows if r["decile"] in (7, 8, 9)])

    print(f"\n--- gradient: early deciles (0-2) vs late (7-9) ---")
    print(f"  QQP-label error:   {early_qqp_err:.3f} -> {late_qqp_err:.3f}   (delta {late_qqp_err - early_qqp_err:+.3f})")
    print(f"  judge-label error: {early_judge_err:.3f} -> {late_judge_err:.3f}   (delta {late_judge_err - early_judge_err:+.3f})")
    print(f"  judge-QQP agreement: {early_agree:.3f} -> {late_agree:.3f}   (delta {late_agree - early_agree:+.3f})")
    print()
    if late_judge_err - early_judge_err > 0.08:
        print("  => gradient PERSISTS under judge labels: real difficulty drift.")
    elif abs(late_judge_err - early_judge_err) <= 0.05:
        print("  => gradient FLATTENS under judge labels: likely QQP label-noise artifact.")
    else:
        print("  => partial: gradient shrinks but does not vanish; interpret with the agreement trend.")

    out = {
        "dataset": "quora", "n_per_decile": args.n_per_decile, "model": args.model, "seed": args.seed,
        "n_judge_calls": len(rows),
        "per_decile": per_decile,
        "gradient": {
            "early_deciles": [0, 1, 2], "late_deciles": [7, 8, 9],
            "qqp_err_early": float(early_qqp_err), "qqp_err_late": float(late_qqp_err),
            "judge_err_early": float(early_judge_err), "judge_err_late": float(late_judge_err),
            "agreement_early": float(early_agree), "agreement_late": float(late_agree),
        },
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
