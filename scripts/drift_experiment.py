"""Drift / re-fine-tuning cadence experiment: split the gray-zone stream
into sequential time-ordered chunks, fine-tune a fresh verifier on each
chunk ("anchor"), then measure how much that frozen model's AUC decays when
evaluated on later chunks -- i.e. how many chunks of production drift a
one-time fine-tune survives before it needs to be redone.

Produces a full (anchor_chunk, eval_chunk) AUC matrix so the decay can be
averaged across every anchor at each temporal distance (eval_chunk -
anchor_chunk), rather than relying on a single before/after comparison.

Usage:
    python scripts/drift_experiment.py \\
        --stash finetune_verifier_experiment.examples.json \\
        --n-chunks 8 --epochs 3 --output drift_experiment.json
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores))
    pos_ranks = ranks[labels == 1]
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((pos_ranks.sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stash", default="finetune_verifier_experiment.examples.json")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-out-dir", default="drift_models")
    parser.add_argument("--output", default="drift_experiment.json")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import TrainerCallback

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    class VerboseLoggerCallback(TrainerCallback):
        def __init__(self):
            self.t_start = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.t_start = time.time()
            log(f"  [train] begin: max_steps={state.max_steps} epochs={args.num_train_epochs}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:
                return
            elapsed = time.time() - self.t_start
            step, total = state.global_step, state.max_steps
            log(f"  [train] step {step:>4}/{total} ({100 * step / total:5.1f}%)  loss={logs['loss']:.4f}  "
                f"elapsed={elapsed:.0f}s")

        def on_train_end(self, args, state, control, **kwargs):
            log(f"  [train] end: {state.global_step} steps in {time.time() - self.t_start:.0f}s")

    stash = json.loads(Path(args.stash).read_text(encoding="utf-8"))
    all_rows = stash["train"] + stash["test"]
    all_rows.sort(key=lambda r: r[0])
    log(f"Combined stream: {len(all_rows)} gray-zone examples, positions "
        f"{all_rows[0][0]}-{all_rows[-1][0]}")

    n_chunks = args.n_chunks
    chunk_size = len(all_rows) // n_chunks
    chunks = [all_rows[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]
    # fold any remainder into the last chunk
    if len(all_rows) % n_chunks:
        chunks[-1] = all_rows[(n_chunks - 1) * chunk_size:]
    for i, c in enumerate(chunks):
        n_pos = sum(1 for r in c if r[3])
        log(f"  chunk {i}: {len(c)} examples, positions {c[0][0]}-{c[-1][0]}, "
            f"{n_pos}/{len(c)} positive ({n_pos / len(c):.1%})")

    chunk_pairs = [[(q, a) for _, q, a, _ in c] for c in chunks]
    chunk_labels = [np.array([1 if lbl else 0 for _, _, _, lbl in c]) for c in chunks]

    log(f"Loading baseline (untuned) model {args.base_model!r} on {device}...")
    baseline = CrossEncoder(args.base_model, device=device)
    baseline_auc = []
    for i in range(n_chunks):
        scores = np.array(baseline.predict(chunk_pairs[i], batch_size=32, show_progress_bar=False))
        auc = roc_auc(scores, chunk_labels[i])
        baseline_auc.append(auc)
        log(f"  baseline AUC on chunk {i}: {auc:.4f}")
    del baseline

    matrix = {}  # (anchor, eval) -> auc
    for anchor in range(n_chunks):
        log(f"=== ANCHOR CHUNK {anchor} ===")
        t0 = time.time()
        tuned = CrossEncoder(args.base_model, device=device)
        train_dataset = Dataset.from_dict(
            {
                "query": [q for q, _ in chunk_pairs[anchor]],
                "response": [a for _, a in chunk_pairs[anchor]],
                "label": [float(lbl) for lbl in chunk_labels[anchor]],
            }
        )
        loss = BinaryCrossEntropyLoss(tuned)
        model_out = str(Path(args.model_out_dir) / f"anchor_{anchor}")
        training_args = CrossEncoderTrainingArguments(
            output_dir=model_out + "_checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            use_cpu=(device == "cpu"),
            report_to="none",
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=20,
            logging_first_step=True,
            disable_tqdm=True,
        )
        trainer = CrossEncoderTrainer(model=tuned, args=training_args, train_dataset=train_dataset, loss=loss)
        trainer.add_callback(VerboseLoggerCallback())
        trainer.train()
        log(f"  fine-tuned on chunk {anchor} in {time.time() - t0:.1f}s")

        for eval_chunk in range(n_chunks):
            scores = np.array(tuned.predict(chunk_pairs[eval_chunk], batch_size=32, show_progress_bar=False))
            auc = roc_auc(scores, chunk_labels[eval_chunk])
            matrix[f"{anchor},{eval_chunk}"] = auc
            distance = eval_chunk - anchor
            log(f"  anchor={anchor} eval={eval_chunk} distance={distance:+d}  auc={auc:.4f}  "
                f"(baseline={baseline_auc[eval_chunk]:.4f})")

        del tuned
        import shutil
        shutil.rmtree(model_out + "_checkpoints", ignore_errors=True)

    result = {
        "n_chunks": n_chunks,
        "chunk_size": chunk_size,
        "epochs": args.epochs,
        "baseline_auc_per_chunk": baseline_auc,
        "matrix": matrix,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Wrote drift matrix to {args.output}")


if __name__ == "__main__":
    main()
