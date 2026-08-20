"""PAPER.md §8 future-work item 9 ("E': rewrite instead of reject"), still
open until this script: when Group E's fine-tuned verifier rejects a
gray-zone candidate, the paper's existing binary gate converts that into a
plain `miss` (fall through to a fresh LLM call -- always correct in this
simulation, zero error risk, but zero cache benefit). TweakLLM
(arXiv:2507.23674, already cited in PAPER.md's related-work table) instead
asks a lightweight LLM to EDIT the rejected candidate's cached answer to fit
the new query. This script measures where "reject -> rewrite -> serve"
lands relative to "reject -> miss", stratified by whether the reject was a
false reject (would_be_correct=True -- the original candidate was already
right) or a true reject (would_be_correct=False -- genuinely a different
question, the harder case for a rewrite to salvage).

Reuses, rather than rebuilds:
  - `would_be_correct` ground truth from `verified_sweep.build_match_trace`
  - the honest calibration/test split + Youden's J threshold selection from
    `threshold_calibration_ablation_groupE.py` (imported directly)
  - the DeepSeek-as-API-judge-stand-in call mechanism from
    `measure_oracle_judge_latency.py` (same substitution principle already
    used twice in this project: SearchQueries erratum, Oracle latency)
  - `RequestOutcome`/`error_rate`/`bootstrap_ci` for the incorrect-rate CI,
    by representing every judged rewrite as `RequestOutcome(action="hit",
    correct=<judge verdict>)` -- no new stats code needed.

Sampled offline ablation, NOT a full-trace policy simulation: rewriting and
judging every gray-zone reject across all three ~60k-150k datasets would
mean tens of thousands of paid API calls. This samples up to
`--n-per-stratum` (default 100) from each of the two strata, on the
HELD-OUT TEST half only (never the half the threshold was calibrated on).

Supports splitting the GPU-bound part (match trace + gray-zone scoring,
needs a real model forward pass, no API key needed) from the API-bound part
(rewrite + judge, needs DEEPSEEK_API_KEY, no GPU needed) across two
machines, so the key never has to be copied onto a remote scoring box:

    # On the GPU box (no .env needed):
    python scripts/rewrite_vs_reject_experiment.py --mode sample \\
        --config configs/lmarena.yaml --model finetuned_verifier_model_lmarena \\
        --samples-output samples_lmarena.json --n-per-stratum 100
    # scp samples_lmarena.json back, then locally:
    python scripts/rewrite_vs_reject_experiment.py --mode judge \\
        --samples-output samples_lmarena.json --output results/rewrite_vs_reject_lmarena.json

Or run everything in one process (--mode full, the default) if GPU and the
API key are available in the same place:
    python scripts/rewrite_vs_reject_experiment.py \\
        --config configs/lmarena.yaml \\
        --model results/finetuned_verifier_model \\
        --output results/rewrite_vs_reject_lmarena.json \\
        --n-per-stratum 100
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threshold_calibration_ablation_groupE import select_threshold  # noqa: E402

from cacheverifier.config import load_dataset_config  # noqa: E402
from cacheverifier.data.loaders import load_jsonl  # noqa: E402
from cacheverifier.experiments.run_baselines import build_embedder  # noqa: E402
from cacheverifier.experiments.run_verified import DEFAULT_TAU_LOW_GRID, _slug, CACHE_DIR  # noqa: E402
from cacheverifier.experiments.verified_sweep import (  # noqa: E402
    build_match_trace,
    load_match_trace,
    load_scored,
    resolve_candidate,
    save_match_trace,
    save_scored,
    score_gray_zone,
)
from cacheverifier.metrics.bootstrap import bootstrap_ci  # noqa: E402
from cacheverifier.metrics.core import RequestOutcome, error_rate  # noqa: E402
from cacheverifier.verifiers.cross_encoder_verifier import CrossEncoderVerifier  # noqa: E402

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_TAU_HIGH = 0.97
DEFAULT_TAU_LOW = 0.80  # widest gray zone, same anchor CRC's Protocol R/T used (PAPER.md §5.13) -- avoids
                         # re-introducing the post-hoc tau_low grid-search optimism this project has
                         # already flagged and corrected for elsewhere (§5.11, §5.4).

REWRITE_SYSTEM_PROMPT = (
    "You are editing a cached answer so it correctly addresses a NEW user query. You will be given the "
    "ORIGINAL query the answer was written for, the ORIGINAL answer, and the NEW query. Edit the original "
    "answer so it correctly and completely answers the NEW query -- reuse whatever content is still "
    "accurate, and change only what needs to change. Do not write a fresh answer from scratch if editing "
    "suffices. Reply with ONLY the edited answer text, no preamble, no explanation."
)

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a REWRITTEN answer correctly addresses a QUERY, by comparing it against a "
    "REFERENCE answer known to be correct for that query. The rewritten answer does not need to match the "
    "reference word for word -- it needs to convey the same substantive answer. Respond with exactly one "
    "word: YES or NO. No explanation."
)


def call_deepseek(api_key: str, model: str, system_prompt: str, user_content: str, max_tokens: int = 300) -> tuple[str, float]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    resp = requests.post(headers=headers, json=payload, url=DEEPSEEK_URL, timeout=60)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    return content, latency_ms


def rewrite_answer(api_key: str, model: str, orig_query: str, orig_answer: str, new_query: str) -> tuple[str, float]:
    user_content = (
        f"ORIGINAL QUERY:\n{orig_query}\n\nORIGINAL ANSWER:\n{orig_answer}\n\nNEW QUERY:\n{new_query}"
    )
    return call_deepseek(api_key, model, REWRITE_SYSTEM_PROMPT, user_content, max_tokens=500)


def judge_rewrite(api_key: str, model: str, query: str, reference_answer: str, rewritten_answer: str) -> tuple[str, float]:
    user_content = (
        f"QUERY:\n{query}\n\nREFERENCE ANSWER (known correct):\n{reference_answer}\n\n"
        f"REWRITTEN ANSWER (to grade):\n{rewritten_answer}"
    )
    return call_deepseek(api_key, model, JUDGE_SYSTEM_PROMPT, user_content, max_tokens=5)


def summarize_latency(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "n": 0}
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": len(arr),
    }


def collect_sample(records, trace, indices: list[int], n_target: int, seed: int) -> list[dict]:
    """No API calls -- just picks the sample and dumps everything a later,
    possibly-remote `judge_stratum` call needs. Safe to run on a box with no
    DEEPSEEK_API_KEY."""
    rng = random.Random(seed)
    sample = rng.sample(indices, min(n_target, len(indices)))
    out = []
    for i in sample:
        record = records[i]
        candidate = resolve_candidate(records, trace[i])
        out.append({
            "index": i,
            "query": record.query,
            "orig_candidate_query": candidate.query,
            "orig_candidate_answer": candidate.answer,
            "reference_answer": record.answer,
            "would_be_correct": trace[i].would_be_correct,
        })
    return out


def _judge_one(api_key: str, model: str, ex: dict) -> dict | None:
    try:
        rewritten, rw_latency = rewrite_answer(
            api_key, model, ex["orig_candidate_query"], ex["orig_candidate_answer"], ex["query"]
        )
        verdict, _ = judge_rewrite(api_key, model, ex["query"], ex["reference_answer"], rewritten)
    except Exception as e:
        return {"index": ex["index"], "error": str(e)}
    correct = verdict.strip().upper().startswith("YES")
    return {
        **ex,
        "rewritten_answer": rewritten,
        "verdict": verdict,
        "judged_correct": correct,
        "rewrite_latency_ms": rw_latency,
    }


def judge_stratum(api_key: str, model: str, sample: list[dict], n_population: int, seed: int, log,
                   max_workers: int = 8) -> dict:
    """Bounded thread-pool concurrency across examples -- each call's own
    `time.perf_counter()` window still times only that call, so per-call
    rewrite_latency_ms stays a real measurement of that request; only the
    convenience of NOT waiting on one call before starting the next is
    parallelized. This deliberately differs from
    `measure_oracle_judge_latency.py`'s fully sequential design (that
    script's whole purpose was measuring latency itself; here latency is a
    secondary/informational stat, correctness is the primary outcome) --
    noted explicitly wherever these numbers get written up."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    outcomes: list[RequestOutcome] = []
    rewrite_latencies: list[float] = []
    examples: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, api_key, model, ex): ex for ex in sample}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is None or "error" in result:
                log(f"  [{done}/{len(sample)}] FAILED at index {result.get('index') if result else '?'}: "
                    f"{result.get('error') if result else 'unknown'}")
                continue
            rewrite_latencies.append(result["rewrite_latency_ms"])
            outcomes.append(RequestOutcome(action="hit", correct=result["judged_correct"], would_be_correct=result["would_be_correct"]))
            examples.append(result)
            if done % 20 == 0 or done == len(sample):
                log(f"  [{done}/{len(sample)}] judged_correct_so_far={sum(o.correct for o in outcomes)}/{len(outcomes)}")

    if not outcomes:
        return {
            "n_population": n_population, "n_sampled": 0, "incorrect_rate": None,
            "incorrect_rate_ci": None, "rewrite_latency_ms": summarize_latency([]), "examples": [],
        }

    er = error_rate(outcomes)
    er_ci = bootstrap_ci(outcomes, error_rate, n_resamples=1000, seed=seed)
    return {
        "n_population": n_population,
        "n_sampled": len(outcomes),
        "incorrect_rate": er,
        "incorrect_rate_ci": [er_ci.ci_low, er_ci.ci_high],
        "rewrite_latency_ms": summarize_latency(rewrite_latencies),
        "examples": examples[:10],  # keep a handful for manual spot-checking, not all of them
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["full", "sample", "judge"], default="full",
                         help="full: score+sample+judge in one process (needs GPU/model AND API key). "
                              "sample: score+sample only, writes --samples-output, no API key needed. "
                              "judge: reads --samples-output, runs rewrite+judge, no model/GPU needed.")
    parser.add_argument("--config", help="Required for --mode full/sample")
    parser.add_argument("--model", help="Fine-tuned checkpoint dir (Group E verifier); required for --mode full/sample")
    parser.add_argument("--tau-low", type=float, default=DEFAULT_TAU_LOW)
    parser.add_argument("--tau-high", type=float, default=DEFAULT_TAU_HIGH)
    parser.add_argument("--n-per-stratum", type=int, default=100)
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8, help="Thread-pool concurrency for --mode judge/full's DeepSeek calls")
    parser.add_argument("--samples-output", help="Where --mode sample writes / --mode judge reads raw samples")
    parser.add_argument("--output", help="Final results JSON; required for --mode full/judge")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.mode == "judge":
        samples = json.loads(Path(args.samples_output).read_text(encoding="utf-8"))
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY not found (checked .env and environment).")

        log("Judging false-reject stratum...")
        false_reject_result = judge_stratum(api_key, args.deepseek_model, samples["false_reject_sample"],
                                             samples["n_false_reject_population"], args.seed, log, args.max_workers)
        log("Judging true-reject stratum...")
        true_reject_result = judge_stratum(api_key, args.deepseek_model, samples["true_reject_sample"],
                                            samples["n_true_reject_population"], args.seed + 1, log, args.max_workers)

        output = {**{k: v for k, v in samples.items() if not k.endswith("_sample")},
                  "false_reject": false_reject_result, "true_reject": true_reject_result}
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        log(f"Wrote results to {out_path}")
        return

    if args.mode == "full":
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY not found (checked .env and environment).")

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
        log(f"Cached to {trace_cache_path}")

    verifier_key = f"cross_encoder_{_slug(args.model)}"
    scored_cache_path = (
        CACHE_DIR / f"{_slug(dataset_path.stem)}__{embedder_key}__n{len(records)}__{verifier_key}"
                     f"__lo{min(DEFAULT_TAU_LOW_GRID)}__hi{args.tau_high}.scored.json"
    )
    if scored_cache_path.exists():
        log(f"Loading cached gray-zone scores from {scored_cache_path}...")
        scored = load_scored(scored_cache_path)
    else:
        log(f"Scoring gray-zone candidates with the fine-tuned verifier ({args.model})...")
        verifier = CrossEncoderVerifier(args.model)
        scored = score_gray_zone(records, trace, verifier, gray_zone_lo=min(DEFAULT_TAU_LOW_GRID), gray_zone_hi=args.tau_high)
        save_scored(scored, scored_cache_path)
        log(f"Cached to {scored_cache_path}")

    gz_indices = [
        i for i, t in enumerate(trace)
        if t.similarity is not None and args.tau_low <= t.similarity < args.tau_high and i in scored
    ]
    split = len(gz_indices) // 2
    calib_idx, test_idx = gz_indices[:split], gz_indices[split:]
    log(f"tau_low={args.tau_low}: {len(gz_indices)} gray-zone points, {len(calib_idx)} calib / {len(test_idx)} test")

    calib_scores = np.array([scored[i].score for i in calib_idx])
    calib_labels = np.array([1 if trace[i].would_be_correct else 0 for i in calib_idx])
    threshold = select_threshold(calib_scores, calib_labels)
    if threshold is None:
        raise SystemExit("Calibration split too small/single-class -- cannot select a threshold.")
    log(f"Calibrated threshold (Youden's J, calib half only) = {threshold:.4f}")

    rejected_test_idx = [i for i in test_idx if scored[i].score < threshold]
    false_reject_idx = [i for i in rejected_test_idx if trace[i].would_be_correct]
    true_reject_idx = [i for i in rejected_test_idx if not trace[i].would_be_correct]
    log(f"Held-out test rejects: {len(rejected_test_idx)} total "
        f"({len(false_reject_idx)} false-reject, {len(true_reject_idx)} true-reject)")

    base_meta = {
        "dataset": dataset_path.stem,
        "model": args.model,
        "tau_low": args.tau_low,
        "tau_high": args.tau_high,
        "calibrated_threshold": threshold,
        "n_calibration": len(calib_idx),
        "n_test": len(test_idx),
    }

    if args.mode == "sample":
        false_reject_sample = collect_sample(records, trace, false_reject_idx, args.n_per_stratum, args.seed)
        true_reject_sample = collect_sample(records, trace, true_reject_idx, args.n_per_stratum, args.seed + 1)
        output = {
            **base_meta,
            "n_false_reject_population": len(false_reject_idx),
            "n_true_reject_population": len(true_reject_idx),
            "false_reject_sample": false_reject_sample,
            "true_reject_sample": true_reject_sample,
        }
        out_path = Path(args.samples_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        log(f"Wrote {len(false_reject_sample)} + {len(true_reject_sample)} raw samples to {out_path} "
            f"(no API calls made -- run --mode judge on this file where DEEPSEEK_API_KEY is available)")
        return

    log(f"Running false-reject stratum (target n={args.n_per_stratum})...")
    false_reject_sample = collect_sample(records, trace, false_reject_idx, args.n_per_stratum, args.seed)
    false_reject_result = judge_stratum(api_key, args.deepseek_model, false_reject_sample, len(false_reject_idx), args.seed, log, args.max_workers)
    log(f"Running true-reject stratum (target n={args.n_per_stratum})...")
    true_reject_sample = collect_sample(records, trace, true_reject_idx, args.n_per_stratum, args.seed + 1)
    true_reject_result = judge_stratum(api_key, args.deepseek_model, true_reject_sample, len(true_reject_idx), args.seed + 1, log, args.max_workers)

    output = {**base_meta, "false_reject": false_reject_result, "true_reject": true_reject_result}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
