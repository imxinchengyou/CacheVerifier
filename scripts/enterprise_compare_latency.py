"""Enterprise comparison: one interleaved CPU latency benchmark for every component.

Timing each method in a separate run lets background load on the machine
(desktop apps etc.) land on one method and not another; earlier separate runs
differed by 2-6x. Here every sampled request goes through all components back
to back, in one process, batch size 1, on CPU:

  embed          query embedding (all-MiniLM-L6-v2)
  search         brute-force top-1 search over the cache as it was at that request
  cross_encoder  CacheVerifier's verifier on (new message, cached reply)
  gptcache_onnx  GPTCache's OnnxModelEvaluation.evaluation() on (new message, cached message)
  vcache_decide  vCache's decide(): one AdaptiveThresholdPolicy replay of the whole
                 stream (delta=0.02, judge labels), timing every call

Runs in the GPTCache venv (gptcache, onnxruntime, torch 2.2.2, sentence-transformers 3.4.1):
    HF_HUB_OFFLINE=1 gptc_venv/bin/python -u scripts/enterprise_compare_latency.py

Output: results/.cache/enterprise_latency.json (replaces the per-step measurements).
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from gptcache.similarity_evaluation import OnnxModelEvaluation
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cacheverifier.cache.adaptive_threshold import AdaptiveThresholdPolicy  # noqa: E402

CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
N, ROUNDS, SEED = 300, 2, 0


class _Entry:
    def __init__(self, qid):
        self.query_id = qid


class _Match:
    def __init__(self, qid, s):
        self.entry, self.similarity = _Entry(qid), s


def ms(fn):
    s = time.perf_counter()
    fn()
    return (time.perf_counter() - s) * 1000


def main() -> None:
    rng = np.random.default_rng(SEED)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
    gpt = OnnxModelEvaluation()
    out = {}
    for ds, path in DATASETS.items():
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn, emb = np.nan_to_num(z["sim"], nan=-1.0), z["nn"], z["emb"]
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        n = len(rows)
        lat = {k: [] for k in ("embed", "search", "cross_encoder", "gptcache_onnx")}
        for i in [rows[0]["query"]] * 3:  # warm-up
            embedder.encode([i], show_progress_bar=False)
        for _ in range(ROUNDS):
            for i in rng.choice(np.arange(n // 2, n), N, replace=False):
                q, j = rows[i]["query"], int(nn[i])
                lat["embed"].append(ms(lambda: embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)))
                lat["search"].append(ms(lambda: int(np.argmax(emb[:i] @ emb[i]))))
                lat["cross_encoder"].append(ms(lambda: ce.predict([(q, rows[j]["answer"])], show_progress_bar=False)))
                lat["gptcache_onnx"].append(ms(lambda: gpt.evaluation({"question": q}, {"question": rows[j]["query"]})))

        judge = np.zeros(n, dtype=bool)
        for line in open(CACHE / f"enterprise_{ds}_judge.jsonl"):
            r = json.loads(line)
            if r["kind"] == "above":
                judge[r["i"]] = r["verdict"] == "YES"
        random.seed(SEED)
        policy = AdaptiveThresholdPolicy(target_error_rate=0.02)
        decide = []
        for i in range(1, n):
            m = _Match(str(nn[i]), float(sim[i]))
            s = time.perf_counter()
            d = policy.decide(None, None, m)
            decide.append((time.perf_counter() - s) * 1000)
            policy.observe(None, m, bool(judge[i]), d.action)
        lat["vcache_decide"] = decide[n // 2:]
        out[ds] = lat
        print(f"{ds}: " + ", ".join(f"{k} p50 {np.percentile(v, 50):.1f} p95 {np.percentile(v, 95):.1f}" for k, v in lat.items()),
              flush=True)
    (CACHE / "enterprise_latency.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
