"""Enterprise comparison, step 4: CacheVerifier scores, fine-tuning, and latency.

For every gray-zone pair (0.80 <= sim < 0.97) on the shared trace:
  * `stock`: off-the-shelf cross-encoder/ms-marco-MiniLM-L6-v2 on (new message, cached reply)
  * `ft_judge` / `ft_same`: the same model fine-tuned on the calibration half's
    gray zone, once per label set (judge relabel / original template label).

Honest split (ENTERPRISE_COMPARISON.md section 4): the first
half of the stream is the calibration half. Fine-tuning uses only the first
70% of the calibration half's gray-zone pairs; the remaining 30% are left out
so the analysis step can pick thresholds on scores the model never trained
on. The test half is never used here.

Training matches the product since verifier-core 0.2.0: warmup 10% of total
steps (sentence-transformers 3.x `fit` defaults to warmup_steps=10000; see the
cacheverifier-python 0.3.1 changelog), 3 epochs, lr 2e-5.

Also measures batch-1 CPU latency (300 calls each) of the query embedding,
the brute-force nearest-neighbour search at the final cache size, and one
cross-encoder call.

JUDGE_TAG (default "judge") picks the judge label file, enterprise_{ds}_{JUDGE_TAG}.jsonl.
With another tag only the stock scores and ft_judge (trained on that tag's labels) are
produced, into enterprise_{ds}_verifier_{JUDGE_TAG}.npz.

Output: results/.cache/enterprise_{ds}_verifier[_{JUDGE_TAG}].npz,
results/.cache/enterprise_latency_verifier_step.json (per-step timings; the analysis uses
the interleaved benchmark in enterprise_latency.json instead)

Usage:
    .venv/bin/python -u scripts/enterprise_compare_verifier.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, InputExample, SentenceTransformer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
LO, HI, SIM_MIN = 0.80, 0.97, 0.6
TRAIN_FRACTION, EPOCHS, WARMUP_FRACTION, LR, BATCH = 0.7, 3, 0.1, 2e-5, 16
N_LATENCY, SEED = 300, 0
JUDGE_TAG = os.environ.get("JUDGE_TAG", "judge")
SUFFIX = "" if JUDGE_TAG == "judge" else f"_{JUDGE_TAG}"


def judge_labels(ds: str, n: int, sim: np.ndarray) -> np.ndarray:
    """True iff the judge said YES; pairs below SIM_MIN were not relabelled and count as False."""
    lab = np.zeros(n, dtype=bool)
    seen = np.zeros(n, dtype=bool)
    for line in open(CACHE / f"enterprise_{ds}_{JUDGE_TAG}.jsonl"):
        r = json.loads(line)
        if r["kind"] == "above" and r["verdict"] in ("YES", "PARTIAL", "NO"):  # failed calls (ERROR, unparsed) stay unjudged
            lab[r["i"]] = r["verdict"] == "YES"
            seen[r["i"]] = True
    missing = int(((sim >= SIM_MIN) & ~seen).sum())
    if missing:
        raise SystemExit(f"{ds}: {missing} pairs >= {SIM_MIN} still unjudged; finish enterprise_compare_judge.py first")
    return lab


def score(model: CrossEncoder, pairs: list[tuple[str, str]]) -> np.ndarray:
    return np.asarray(model.predict(pairs, batch_size=64, show_progress_bar=False), dtype=np.float32)


def finetune(pairs: list[tuple[str, str]], labels: np.ndarray) -> CrossEncoder:
    model = CrossEncoder(BASE, num_labels=1, device="cpu")
    examples = [InputExample(texts=[a, b], label=float(y)) for (a, b), y in zip(pairs, labels)]
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    warmup = max(1, int(WARMUP_FRACTION * len(loader) * EPOCHS))
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": LR}, show_progress_bar=False)
    return model


def main() -> None:
    rng = np.random.default_rng(SEED)
    stock = CrossEncoder(BASE, device="cpu")
    latency = {}
    for ds, path in DATASETS.items():
        if SUFFIX and (CACHE / f"enterprise_{ds}_verifier{SUFFIX}.npz").exists():
            print(f"{ds}: already done, skipping", flush=True)
            continue
        z = np.load(CACHE / f"enterprise_{ds}.npz")
        sim, nn, same, emb = z["sim"], z["nn"], z["same"], z["emb"]
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        n = len(rows)
        half = n // 2
        gray = np.where((sim >= LO) & (sim < HI))[0]
        pairs = [(rows[i]["query"], rows[nn[i]]["answer"]) for i in gray]

        t0 = time.time()
        out = {"gray": gray, "stock": score(stock, pairs)}
        print(f"{ds}: {len(gray)} gray-zone pairs, stock scored in {time.time() - t0:.0f}s", flush=True)

        calib = gray < half
        calib_idx = np.where(calib)[0]
        n_train = int(TRAIN_FRACTION * len(calib_idx))
        train_pos = calib_idx[:n_train]  # chronological: earliest 70% of the calibration half's gray zone
        out["train_until"] = np.array(gray[train_pos[-1]] if n_train else -1)
        label_sets = {"judge": judge_labels(ds, n, sim)}
        if not SUFFIX:
            label_sets = {"same": same, **label_sets}
        for name, lab in label_sets.items():
            y = lab[gray[train_pos]]
            t0 = time.time()
            model = finetune([pairs[k] for k in train_pos], y)
            out[f"ft_{name}"] = score(model, pairs)
            print(f"  ft_{name}: trained on {n_train} pairs ({y.mean():.3f} positive) in {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(CACHE / f"enterprise_{ds}_verifier{SUFFIX}.npz", **out)
        if SUFFIX:
            continue

        # batch-1 latency, the way a request is served
        pick = rng.choice(gray, N_LATENCY, replace=False)
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        lat = {"embed": [], "search": [], "cross_encoder": []}
        for i in pick:
            s = time.perf_counter()
            embedder.encode([rows[i]["query"]], normalize_embeddings=True, show_progress_bar=False)
            lat["embed"].append((time.perf_counter() - s) * 1000)
            s = time.perf_counter()
            int(np.argmax(emb @ emb[i]))
            lat["search"].append((time.perf_counter() - s) * 1000)
            s = time.perf_counter()
            stock.predict([(rows[i]["query"], rows[nn[i]]["answer"])], show_progress_bar=False)
            lat["cross_encoder"].append((time.perf_counter() - s) * 1000)
        latency[ds] = lat
        print(f"  latency p50 ms: " + ", ".join(f"{k} {np.percentile(v, 50):.1f}" for k, v in lat.items()), flush=True)
    if latency:
        (CACHE / "enterprise_latency_verifier_step.json").write_text(json.dumps(latency))


if __name__ == "__main__":
    main()
