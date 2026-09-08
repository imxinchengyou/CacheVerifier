"""Tests whether "truncating the candidate answer helps" (found on the
synthetic adversarial red-team set, RESEARCH_PROPOSAL.md direction 16 "再续
之九") also holds on NATURAL held-out gray-zone data across all three
datasets -- this was explicitly flagged as untested. If it holds here too,
answer truncation before verification is a training-free, near-zero-cost
robustness lever independent of the whole q_cached/diff research line.

For each dataset (LmArena, Quora, SearchQueries-corrected), scores the
natural held-out test split with TWO models (off-the-shelf, and that
dataset's own text_only_adv fine-tuned checkpoint -- no q_cached involved,
plain (query, answer) input) across THREE answer variants (full, front_half,
back_half), reporting AUC against the `would_be_correct` ground truth label
for each of the 3 x 2 x 3 = 18 conditions.

Usage:
    python scripts/qcached_natural_truncation_check.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def build_variants(answer: str) -> dict:
    words = answer.split()
    half = -(-len(words) // 2)
    return {"full": answer, "front_half": " ".join(words[:half]), "back_half": " ".join(words[-half:]) if half > 0 else ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off-the-shelf-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--output", default="results/qcached_natural_truncation_check.json")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    datasets = {
        "lmarena": {
            "stash": "results/finetune_verifier_qcached_experiment.examples.json",
            "text_only_adv_checkpoint": "results/finetuned_verifier_model_lmarena_qcached_adv_ablation_text_only_adv",
        },
        "quora": {
            "stash": "results/finetune_verifier_qcached_experiment_quora.examples.json",
            "text_only_adv_checkpoint": "results/finetuned_verifier_model_quora_qcached_adv_ablation_text_only_adv",
        },
        "searchqueries_corrected": {
            "stash": "results/finetune_verifier_qcached_experiment_searchqueries_corrected.examples.json",
            "text_only_adv_checkpoint": "results/finetuned_verifier_model_searchqueries_qcached_adv_ablation_text_only_adv",
        },
    }

    off_the_shelf = CrossEncoder(args.off_the_shelf_model, device=device)

    all_results = {}
    for ds_name, cfg in datasets.items():
        log(f"\n=== dataset: {ds_name} ===")
        stash = json.loads(Path(cfg["stash"]).read_text(encoding="utf-8"))
        test_rows = stash["test"]
        labels = np.array([1 if row[-1] else 0 for row in test_rows])
        answers = [row[3] for row in test_rows]  # (idx, query, cached_query, answer, label)
        queries = [row[1] for row in test_rows]
        log(f"  {len(test_rows)} test rows, mean answer words={np.mean([len(a.split()) for a in answers]):.1f}")

        variants = {"full": [], "front_half": [], "back_half": []}
        for a in answers:
            v = build_variants(a)
            for name in variants:
                variants[name].append(v[name])

        ds_results = {}
        models = {"off_the_shelf": off_the_shelf}
        text_only_model = CrossEncoder(cfg["text_only_adv_checkpoint"], device=device)
        models["text_only_adv"] = text_only_model

        for model_name, model in models.items():
            ds_results[model_name] = {}
            for variant_name, texts_b in variants.items():
                pairs = list(zip(queries, texts_b))
                t0 = time.time()
                scores = np.array(model.predict(pairs, batch_size=64, show_progress_bar=False))
                auc = roc_auc(scores, labels)
                ds_results[model_name][variant_name] = auc
                log(f"  [{model_name:>14}][{variant_name:>10}] AUC={auc:.4f} ({time.time() - t0:.1f}s)")
        del text_only_model
        all_results[ds_name] = ds_results

    Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log(f"\n=== summary ===")
    for ds_name, ds_results in all_results.items():
        for model_name, variants in ds_results.items():
            log(f"{ds_name:>24} | {model_name:>14} | full={variants['full']:.4f} "
                f"front={variants['front_half']:.4f} back={variants['back_half']:.4f}")
    log(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
