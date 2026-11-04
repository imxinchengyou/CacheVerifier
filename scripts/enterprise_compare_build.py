"""Enterprise comparison, step 1: one shared nearest-neighbour trace per dataset.

Streams each Twitter customer-support dataset in created_at order through an
initially empty cache (every request is inserted after it is decided, as in
`ExperimentRunner`), so the top-1 neighbour of every request is independent of
the caching policy. Every method in ENTERPRISE_COMPARISON.md is
then replayed on this same trace.

Output: results/.cache/enterprise_{ds}.npz with
  sim[i]   cosine similarity to the nearest earlier request (nan for i=0)
  nn[i]    index of that request (-1 for i=0)
  same[i]  original label: same equivalence_id (template-identical reply)
  emb      L2-normalised all-MiniLM-L6-v2 query embeddings

Usage:
    python scripts/enterprise_compare_build.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ".cache"
DATASETS = {"twitter_amazon": "data/processed/twitter_amazon.jsonl",
            "twitter_comcast": "data/processed/twitter_comcast.jsonl"}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BLOCK = 1024


def main() -> None:
    model = SentenceTransformer(MODEL, device="cpu")
    for ds, path in DATASETS.items():
        rows = [json.loads(line) for line in open(ROOT / path, encoding="utf-8")]
        emb_file = CACHE / f"enterprise_{ds}_emb.npy"
        t0 = time.time()
        if emb_file.exists():
            emb = np.load(emb_file)
        else:
            emb = model.encode([r["query"] for r in rows], batch_size=256, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            np.save(emb_file, emb)
        t_emb = time.time() - t0

        # blocked search: for each request, the most similar strictly earlier request
        n = len(rows)
        sim = np.full(n, np.nan, dtype=np.float32)
        nn = np.full(n, -1, dtype=np.int64)
        t0 = time.time()
        for a in range(1, n, BLOCK):
            b = min(a + BLOCK, n)
            s = emb[a:b] @ emb[:b].T
            s[np.arange(b - a)[:, None] <= np.arange(b)[None, :] - a] = -np.inf  # only j < i
            j = np.argmax(s, axis=1)
            sim[a:b] = s[np.arange(b - a), j]
            nn[a:b] = j
        t_nn = time.time() - t0

        eq = np.array([r["equivalence_id"] for r in rows])
        same = np.zeros(n, dtype=bool)
        same[1:] = eq[1:] == eq[nn[1:]]
        np.savez_compressed(CACHE / f"enterprise_{ds}.npz", sim=sim, nn=nn, same=same, emb=emb)
        print(f"{ds}: n={n} embed {t_emb:.0f}s nn {t_nn:.0f}s | "
              f"sim>=0.6 {np.nanmean(sim >= 0.6):.3f} >=0.8 {np.nanmean(sim >= 0.8):.3f} >=0.97 {np.nanmean(sim >= 0.97):.3f} | "
              f"same-label rate among sim>=0.8: {same[sim >= 0.8].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
