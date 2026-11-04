"""Enterprise comparison, step 3: GPTCache's learned evaluator on the shared trace.

Runs gptcache's own `OnnxModelEvaluation` (GPTCache/albert-duplicate-onnx) on
every (new message, nearest cached message) pair, one pair per call exactly as
GPTCache's adapter does at serving time, and records the wall-clock latency of
each call.

Needs gptcache + onnxruntime, which conflict with the project's pinned
environment, so run it from a separate venv:
    python -m venv gptc_venv && gptc_venv/bin/pip install gptcache==0.1.44 onnxruntime transformers "numpy<2"
    gptc_venv/bin/python -u scripts/enterprise_compare_gptcache.py [GPU_SCORES_DIR]

Scores are computed in padded batches with the evaluator's own tokenizer and
ONNX session, taking the same output column as `OnnxModelEvaluation.inference`
and keeping its rule that case-insensitively identical questions score 1.
Before scoring, N_CHECK random pairs are compared against the official
`evaluation()` call and the run stops if they differ by more than 1e-4.
Latency is measured separately on N_LATENCY random pairs with GPTCache's
default evaluator, one pair per call.

Output: results/.cache/enterprise_{ds}_gptcache.npz (score[i]; latency_ms on
the sampled pairs, nan elsewhere and for i=0, which has no neighbour).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from gptcache.similarity_evaluation import OnnxModelEvaluation

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
N_LATENCY, N_CHECK, BATCH, SEED = 300, 300, 64, 0
GPU_SCORES = sys.argv[1] if len(sys.argv) > 1 else None  # directory with scores_{ds}.npy from the GPU run


def score_batched(evaluation: OnnxModelEvaluation, pairs: list[tuple[str, str]]) -> np.ndarray:
    tok = evaluation.tokenizer
    out = np.zeros(len(pairs), dtype=np.float32)
    enc = [tok(a, b) for a, b in pairs]
    order = np.argsort([len(e["input_ids"]) for e in enc])
    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        width = max(len(enc[k]["input_ids"]) for k in idx)
        feed = {}
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            arr = np.full((len(idx), width), tok.pad_token_id, dtype=np.int64)
            for r, k in enumerate(idx):
                arr[r, :len(enc[k][name])] = enc[k][name]
            feed[name] = arr
        out[idx] = evaluation.ort_session.run(None, feed)[0][:, 1]
    same = np.array([a.lower() == b.lower() for a, b in pairs])
    out[same] = 1.0
    return out


def main() -> None:
    evaluation = OnnxModelEvaluation()
    rng = np.random.default_rng(SEED)
    for ds, path in DATASETS.items():
        out = CACHE / f"enterprise_{ds}_gptcache.npz"
        if out.exists():
            print(f"{ds}: already scored", flush=True)
            continue
        nn = np.load(CACHE / f"enterprise_{ds}.npz")["nn"]
        rows = [json.loads(line)["query"] for line in open(ROOT / path, encoding="utf-8")]
        n = len(rows)
        score = np.full(n, np.nan, dtype=np.float32)
        lat = np.full(n, np.nan, dtype=np.float32)

        for i in rng.choice(np.arange(1, n), N_LATENCY, replace=False):
            s = time.perf_counter()
            evaluation.evaluation({"question": rows[i]}, {"question": rows[nn[i]]})
            lat[i] = (time.perf_counter() - s) * 1000.0

        pairs = [(rows[i], rows[nn[i]]) for i in range(1, n)]
        check = rng.choice(len(pairs), N_CHECK, replace=False)
        official = np.array([float(np.asarray(evaluation.evaluation({"question": pairs[k][0]}, {"question": pairs[k][1]})).ravel()[0])
                             for k in check])
        diff = np.abs(score_batched(evaluation, [pairs[k] for k in check]) - official).max()
        print(f"  {ds}: batched vs official evaluation(), max abs diff on {N_CHECK} pairs = {diff:.2e}", flush=True)
        if diff > 1e-4:
            raise SystemExit("batched scores do not match GPTCache's evaluation(); not saving")
        t0 = time.time()
        if GPU_SCORES:
            # scored on the T4 by enterprise_compare_gptcache_gpu.py (same batching, CUDA session);
            # this run still checks those scores against the official CPU evaluation() above
            score[1:] = np.load(Path(GPU_SCORES) / f"scores_{ds}.npy")
            gdiff = np.abs(score[1:][check] - official).max()
            print(f"  {ds}: GPU scores vs official evaluation(), max abs diff = {gdiff:.2e}", flush=True)
            if gdiff > 1e-3:
                raise SystemExit("GPU scores do not match GPTCache's evaluation(); not saving")
        else:
            score[1:] = score_batched(evaluation, pairs)
        np.savez_compressed(out, score=score, latency_ms=lat)
        print(f"{ds}: done {time.time() - t0:.0f}s, latency p50 {np.nanpercentile(lat, 50):.1f}ms "
              f"p95 {np.nanpercentile(lat, 95):.1f}ms, score>=0.8 share {np.nanmean(score >= 0.8):.4f}", flush=True)


if __name__ == "__main__":
    main()
