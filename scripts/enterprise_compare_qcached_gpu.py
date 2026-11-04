"""GPU run of enterprise_compare_qcached.py's training: all three input arms.

Self-contained so it can run on the remote T4 with only qc_job.json.gz
(gray-zone texts, the chronological training positions and both label sets,
written locally from the same split as enterprise_compare_verifier.py). Same
recipe: cross-encoder/ms-marco-MiniLM-L6-v2, warmup 10% of steps, 3 epochs,
lr 2e-5, batch 16. Arms (PAPER §5.22 templates):

  text_only  text_a = new message
  concat     "{query}\\n[cached_query] {cached_query}"
  diff       "{query}\\n[diff vs cached_query] {word-level diff}"

Writes scores_{arm}_{tag}_{ds}.npy (one score per gray-zone pair, in qc_job order).
Environment: torch 2.2.2, sentence-transformers 3.4.1, transformers 4.57.6.

Usage (on the server, next to qc_job.json.gz):
    python -u enterprise_compare_qcached_gpu.py
"""

import difflib
import gzip
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
EPOCHS, WARMUP_FRACTION, LR, BATCH, SEED = 3, 0.1, 2e-5, 16, 0
QCACHED_TEMPLATE = "{query}\n[cached_query] {cached_query}"
DIFF_TEMPLATE = "{query}\n[diff vs cached_query] {diff}"


def word_diff_summary(cached_query: str, current_query: str, max_words: int = 12) -> str:
    """Copy of scripts/finetune_verifier_qcached_train_eval.py::word_diff_summary."""
    a, b = cached_query.split(), current_query.split()
    sm = difflib.SequenceMatcher(None, a, b)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(a[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(b[j1:j2])
    if not removed and not added:
        return "identical"
    parts = []
    if removed:
        parts.append("removed: " + " ".join(removed[:max_words]))
    if added:
        parts.append("added: " + " ".join(added[:max_words]))
    return "; ".join(parts)


ARMS = {"text_only": lambda q, cq: q,
        "concat": lambda q, cq: QCACHED_TEMPLATE.format(query=q, cached_query=cq),
        "diff": lambda q, cq: DIFF_TEMPLATE.format(query=q, diff=word_diff_summary(cq, q))}


def main() -> None:
    assert torch.cuda.is_available()
    job = json.load(gzip.open("qc_job.json.gz", "rt"))
    for tag in ("judge", "judge_thinklow"):
        for arm, fn in ARMS.items():
            for ds, d in job.items():
                out = f"scores_{arm}_{tag}_{ds}.npy"
                if os.path.exists(out):
                    continue
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                pairs = [(fn(q, cq), a) for q, cq, a in zip(d["q"], d["cq"], d["a"])]
                y = np.array(d["labels"][tag])
                tp = d["train_pos"]
                model = CrossEncoder(BASE, num_labels=1, device="cuda")
                loader = DataLoader([InputExample(texts=list(pairs[k]), label=float(y[k])) for k in tp], shuffle=True, batch_size=BATCH)
                t0 = time.time()
                model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=max(1, int(WARMUP_FRACTION * len(loader) * EPOCHS)),
                          optimizer_params={"lr": LR}, show_progress_bar=False)
                scores = np.asarray(model.predict(pairs, batch_size=128, show_progress_bar=False), dtype=np.float32)
                np.save(out, scores)
                print(f"{tag}/{arm}/{ds}: {len(tp)} train pairs ({y[tp].mean():.3f} positive), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
